import os
import copy
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms
from datetime import datetime

# import numpy as np # [FIX] Unused
import config
from model import MAML_YOLO_NAS # [FIX] Removed unused 'unwrap_predictions'
from prepare_data import build_dataset_info, get_annotations_for_image
from dataset import FSODDataset, fsod_collate_fn
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import cv2
import numpy as np

transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def dprint(*args, **kwargs):
    if False:
        print(*args, **kwargs)

def compute_ap(pred_boxes, pred_cls, conf, gt_boxes, gt_cls, iou_thresh=0.5):
    if len(pred_boxes) == 0:
        return 0.0

    # Sort by confidence
    idxs = np.argsort(-conf)
    pred_boxes = pred_boxes[idxs]
    pred_cls = pred_cls[idxs]
    conf = conf[idxs]

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = set()

    for i in range(len(pred_boxes)):
        best_iou = 0
        best_j = -1

        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue
            if pred_cls[i] != gt_cls[j]:
                continue

            iou = compute_iou(pred_boxes[i], gt_boxes[j])

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thresh:
            tp[i] = 1
            matched_gt.add(best_j)
        else:
            fp[i] = 1

    # cumulative
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / (len(gt_boxes) + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    # AP = area under PR curve
    ap = 0.0
    for i in range(len(recalls)):
        if i == 0:
            ap += precisions[i] * recalls[i]
        else:
            ap += precisions[i] * (recalls[i] - recalls[i-1])

    return ap

def visualize_predictions(image_path, pred_boxes, pred_cls, conf, gt_boxes=None, gt_cls=None, save_path="output.jpg"):
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))

    # --- Draw GT (RED) ---
    '''if gt_boxes is not None:
        for i in range(len(gt_boxes)):
            x1, y1, x2, y2 = map(int, gt_boxes[i])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if gt_cls is not None:
                cv2.putText(image, f"GT:{gt_cls[i]}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    '''                        

    # --- Draw Predictions (GREEN) ---
    for i in range(len(pred_boxes)):
        x1, y1, x2, y2 = map(int, pred_boxes[i])
        score = conf[i]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"P:{pred_cls[i]} {score:.2f}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    1)

    cv2.imwrite(save_path, image)
    print(f"Saved visualization → {save_path}")

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter + 1e-6

    return inter / union

def match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thresh=0.5):
    matched_gt = set()
    tp, fp = 0, 0

    for i in range(len(pred_boxes)):
        best_iou = 0
        best_j = -1

        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue

            if pred_cls[i] != gt_cls[j]:
                continue

            iou = compute_iou(pred_boxes[i], gt_boxes[j])

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn

def compute_metrics(pred_boxes, pred_cls, gt_boxes, gt_cls):
    tp, fp, fn = match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision, recall

def compute_esr(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thresh=0.5):
    tp, _, _ = match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thresh)

    return 1 if tp > 0 else 0

def normalize_pred_scores(pred_scores, num_classes, device):
    
    if pred_scores is None:
        return None
    B, N, C = tuple(pred_scores.shape)
    if C == num_classes:
        return pred_scores
    if C == num_classes + 1:
        # drop last channel (often background/conf)
        return pred_scores[:, :, :num_classes].to(device)
    if C > num_classes:
        return pred_scores[:, :, :num_classes].to(device)
    if C == 0:
        return torch.zeros((B, N, num_classes), device=device, dtype=pred_scores.dtype)
    # fallback: expand last dim to expected classes (very rare)
    return torch.zeros((B, N, num_classes), device=device, dtype=pred_scores.dtype)


def ensure_differentiable_loss(loss, predictions, device):
    """If loss doesn't require grad (e.g., scalar float), tie it to predictions so backward flows."""
    if not isinstance(loss, torch.Tensor):
        loss = torch.tensor(loss, dtype=torch.float32, device=device)
    if not loss.requires_grad:
        # If predictions is list/tuple, add tiny multiple of their means
        preds_for_grad = []
        if isinstance(predictions, (list, tuple)):
            for p in predictions:
                if torch.is_tensor(p):
                    preds_for_grad.append(p)
        elif torch.is_tensor(predictions):
            preds_for_grad = [predictions]

        if preds_for_grad:
            # make sure same device
            loss = loss.to(device)
            loss = loss + 0.0 * sum(p.mean() for p in preds_for_grad)
        else:
            loss = loss.clone().detach().requires_grad_(True)
    return loss


def convert_yolonas_to_ppyolo(pred_list, desired_num_classes):
        # Flatten input cooperatively (if user passed nested lists/tuples)
    flat = []
    for x in pred_list:
        if isinstance(x, (list, tuple)):
            for y in x:
                if torch.is_tensor(y):
                    flat.append(y)
        elif torch.is_tensor(x):
            flat.append(x)

    # initialize
    pred_scores = None
    pred_distri = None
    anchors = None
    anchor_points = None
    stride_tensor = None

    # heuristic matching: examine shapes
    for t in flat:
        if not torch.is_tensor(t):
            continue
        shape = tuple(t.shape)
        # classification scores: (B, N, C_out) where C_out usually small like 1..20
        if len(shape) == 3 and shape[2] in [0, 1, 2, 3, 4, 5, 10, 20]:
            pred_scores = t
        # distribution/regression: (B, N, 68) — typical YOLO-NAS reg output (4*(reg_max+1))
        elif len(shape) == 3 and shape[2] == 68:
            pred_distri = t
        # anchors: (N, 4)
        elif len(shape) == 2 and shape[1] == 4:
            anchors = t
        # anchor centers/points: (N, 2)
        elif len(shape) == 2 and shape[1] == 2:
            anchor_points = t
        # stride tensor: (N, 1)
        elif len(shape) == 2 and shape[1] == 1:
            stride_tensor = t

    # Basic device inference
    device = None
    for t in [pred_scores, pred_distri, anchors, anchor_points, stride_tensor]:
        if t is not None:
            device = t.device
            break
    if device is None:
        device = torch.device("cpu")

    # ensure pred_distri exists — PPYoloELoss needs it
    if pred_distri is None:
        # create a tiny zero distribution so we can still compute a valid loss (but will be small)
        # assume batch 1 and N=5376 fallback
        pred_distri = torch.zeros((1, 5376, 68), device=device)

    # ensure anchors/points/stride exist
    if anchors is None:
        anchors = torch.zeros((pred_distri.shape[1], 4), device=device)
    if anchor_points is None:
        anchor_points = torch.zeros((pred_distri.shape[1], 2), device=device)
    if stride_tensor is None:
        stride_tensor = torch.ones((pred_distri.shape[1], 1), device=device)

    # pred_scores adjust to desired_num_classes and device
    if pred_scores is None:
        # fallback zeros
        pred_scores = torch.zeros((pred_distri.shape[0], pred_distri.shape[1], desired_num_classes), device=device)
    else:
        pred_scores = pred_scores.to(device)
        pred_scores = normalize_pred_scores(pred_scores, desired_num_classes, device)

    # move other tensors to device
    pred_distri = pred_distri.to(device)
    anchors = anchors.to(device)
    anchor_points = anchor_points.to(device)
    stride_tensor = stride_tensor.to(device)

    num_anchors_list = [anchors.size(0)]


    return (pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor)


def flatten(x):
            out = []
            if isinstance(x, (list, tuple)):
                for e in x:
                    out.extend(flatten(e))
            elif torch.is_tensor(x):
                out.append(x)
            return out

def freeze_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()

def postprocess(outputs, conf_thresh=0.2, iou_thresh=0.5):
    (boxes, scores), _ = outputs

    boxes = boxes[0]        # [N, 4]
    scores = scores[0]      # [N, num_classes]

    # Get best class per box
    conf, cls = scores.max(dim=1)

    # Filter by confidence
    mask = conf > conf_thresh
    boxes = boxes[mask]
    conf = conf[mask]
    cls = cls[mask]

    # Apply NMS
    keep = nms(boxes, conf, iou_thresh)

    return boxes[keep], conf[keep], cls[keep]

def run_inference(model, image_path):
    model.eval()

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    return outputs


episode_classes = [0, 4]  # example: bus, car
label_map = {0: 0, 4: 1}

support_targets = []

support_paths = [
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000016_01352_d_0000069.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000068_02648_d_0000007.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000068_03714_d_0000012.jpg",
    #r"D:\fsod_project\fsod_dota\novel_classes\people\0000070_05880_d_0000005.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000076_04674_d_0000016.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000068_00460_d_0000002.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000079_01673_d_0000006.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000084_00000_d_0000001.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000098_00016_d_0000001.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000220_01000_d_0000003.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\people\0000261_00401_d_0000121.jpg",

    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000008_01999_d_0000040.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000008_03499_d_0000043.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000076_00616_d_0000003.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000076_02667_d_0000010.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000199_00000_d_0000162.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000007_04999_d_0000036.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000007_05499_d_0000037.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000008_03499_d_0000043.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000008_03999_d_0000044.jpg",
    r"D:\fsod_project\fsod_dota\novel_classes\truck\0000076_03930_d_0000013.jpg"
]

for path in support_paths:
    ann = get_annotations_for_image(path)

    boxes = ann["boxes"]
    labels = ann["labels"]

    # filter only episode classes
    mask = torch.isin(labels, torch.tensor(episode_classes, device=labels.device))
    boxes = boxes[mask]
    labels = labels[mask]

    # remap to 0..N-1
    labels = torch.tensor([label_map[int(l)] for l in labels], device=boxes.device, dtype=torch.long)

    support_targets.append({
        "boxes": boxes,
        "labels": labels
    })

dprint("\n--- SUPPORT SET DEBUG ---")
for i, t in enumerate(support_targets):
    dprint(f"{i}: boxes={t['boxes'].shape}, labels={t['labels']}")

#query_image = r"D:\fsod_project\fsod_dota\novel_classes\people\0000079_00761_d_0000002.jpg"
#query_image = r"D:\fsod_project\fsod_dota\novel_classes\truck\0000008_01999_d_0000040.jpg"
query_images = [r"D:\fsod_project\fsod_dota\novel_classes\people\0000079_00761_d_0000002.jpg", r"D:\fsod_project\fsod_dota\novel_classes\truck\0000008_01999_d_0000040.jpg", r"D:\fsod_project\0000076_04382_d_0000015.jpg"]
base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb0_s0_vis_06_03.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb0_s1_vis_03_03.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb0_s5_vis_18_02.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb0_s10_vis_27_02.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb10_s0_vis_06_03.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb10_s1_vis_05_03.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb10_s5_vis_19_02.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb10_s10_vis_26_02.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb25_s0_vis_06_03.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb25_s1_vis_03_03.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb25_s5_vis_17_02.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bb25_s10_vis_24_02.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bbf_s0_vis_09_03.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bbf_s1_vis_05_03.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bbf_s5_vis_23_02.pth"
#base_model = r"C:\Users\debsp\Downloads\best_checkpoint_dp_output_bbf_s10_vis_02_03.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
print(f"Using device: {device}")
test_model = MAML_YOLO_NAS(
                    model_arch=config.MODEL_ARCH,
                    num_classes=config.N_WAY,
                    checkpoint_path=None,   # IMPORTANT
                    verbose=False
                ).to(device)
# BEFORE loading
dprint("\n--- BEFORE LOADING ---")
out_before = run_inference(test_model, query_images[0])
dprint(out_before)
initial_weights = {}
for name, param in test_model.named_parameters():
    initial_weights[name] = param.clone()
#test_model.load_state_dict(base_model.state_dict())
#base_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
# [FIX] The final model is saved as a dictionary, so we extract the 'model_state_dict' key
#checkpoint = torch.load(base_model, map_location=device)
try:
    checkpoint = torch.load(base_model, map_location=device)
    print("Checkpoint loaded successfully")
except Exception as e:
    print("ERROR loading checkpoint:", e)
test_model.load_state_dict(checkpoint['model_state_dict'])

# AFTER loading → compare
same_count = 0
total = 0

for name, param in test_model.named_parameters():
    total += 1
    if torch.equal(param, initial_weights[name]):
        same_count += 1

print(f"Unchanged layers: {same_count}/{total}")

print("Successfully loaded MAML checkpoint.")

dprint("\n--- AFTER LOADING ---")
out_after = run_inference(test_model, query_images[0])
dprint(out_after)
boxes, conf, cls = postprocess(out_after)

dprint("Detections:", len(boxes))
dprint(conf[:10])
test_model.train()
#print("done")

NUM_EPISODES = 2

all_ap = []
all_precision = []
all_recall = []
all_esr = []
all_inference_time = []

for episode in range(NUM_EPISODES):

    print(f"\n=== Episode {episode} ===")
    #----Clone
    adapted_model = copy.deepcopy(test_model).to(device)
    adapted_model.train()
    for p in adapted_model.feature_extractor.parameters():
        p.requires_grad = False
    # Freeze BN (CRITICAL)

    adapted_model.apply(freeze_bn)
    optimizer = torch.optim.SGD(
        adapted_model.get_inner_loop_params(),  # 🔥 IMPORTANT
        lr=config.INNER_LR
    )
    before = copy.deepcopy(list(adapted_model.adaptation_head.parameters()))
    #----Loss
    criterion = PPYoloELoss(use_static_assigner=False, num_classes=config.N_WAY, reg_max=16).to(device)
    #---InnerLoop
    for step in range(config.INNER_UPDATE_STEPS):

        total_loss = torch.tensor(0.0, device=device)
        valid = False

        for s_path, s_target in zip(support_paths, support_targets):

            image = Image.open(s_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            boxes = s_target["boxes"]
            labels = s_target["labels"]

            if boxes.numel() == 0:
                continue

            # === SAME PREPROCESSING AS TRAINING ===
            img_w, img_h = image.size

            boxes_xywh = torch.zeros_like(boxes)
            boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
            boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
            boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0])
            boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1])

            # scale to 512
            boxes_xywh[:, [0, 2]] *= (512 / img_w)
            boxes_xywh[:, [1, 3]] *= (512 / img_h)

            batch_idx = torch.zeros((boxes_xywh.size(0), 1), device=device)
            labels = labels.unsqueeze(1).to(device)

            target = torch.cat([batch_idx, labels.float(), boxes_xywh.to(device)], dim=1)

            # === FORWARD ===
            raw_preds = adapted_model(img_tensor)

            # === FLATTEN (reuse your logic) ===
            

            preds_list = flatten(raw_preds)

            # === CONVERT ===
            pred_input = convert_yolonas_to_ppyolo(
                preds_list,
                desired_num_classes=config.N_WAY
            )

            # === LOSS ===
            loss, _ = criterion(pred_input, target)

            loss = ensure_differentiable_loss(loss, preds_list, device)

            total_loss = total_loss + loss
            valid = True

        if valid:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 5.0)
            optimizer.step()
            print(f"[Adapt Step {step}] Loss: {total_loss.item():.4f}")

        else:
            print(f"[Adapt Step {step}] No valid samples")

    #---Query
    adapted_model.eval()
    '''
    #query_out = run_inference(adapted_model, query_image)

    # --- Inference Timing ---
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        query_out = run_inference(adapted_model, query_image)
        end.record()

        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end)  # milliseconds
    else:
        import time
        t0 = time.time()
        query_out = run_inference(adapted_model, query_image)
        inference_time = (time.time() - t0) * 1000  # ms

    all_inference_time.append(inference_time)

    print(f"Inference Time: {inference_time:.2f} ms")

    boxes, conf, cls = postprocess(query_out)

    dprint("After Adaptation:")
    dprint("Detections:", len(boxes))
    dprint("Conf:", conf[:5])    
    after = list(adapted_model.adaptation_head.parameters())

    diff = sum((b - a).abs().sum() for b, a in zip(before, after))
    dprint("Head weight change:", diff.item())

    # 4. Get GT for query
    ann = get_annotations_for_image(query_image)

    gt_boxes = ann["boxes"].clone().to(device)
    gt_cls = ann["labels"].to(device)

    # filter episode classes
    mask = torch.isin(gt_cls, torch.tensor(episode_classes, device=device))
    gt_boxes = gt_boxes[mask]
    gt_cls = gt_cls[mask]

    # scale
    image = Image.open(query_image)
    img_w, img_h = image.size

    gt_boxes[:, [0, 2]] *= (512 / img_w)
    gt_boxes[:, [1, 3]] *= (512 / img_h)

    # REMAP labels
    gt_cls = torch.tensor([label_map[int(l)] for l in gt_cls], device=device)

    # convert to numpy (ONLY HERE)
    gt_boxes = gt_boxes.cpu().numpy()
    gt_cls = gt_cls.cpu().numpy()

    pred_boxes, conf, pred_cls = postprocess(query_out)
    pred_boxes = pred_boxes.detach().cpu().numpy()
    pred_cls = pred_cls.detach().cpu().numpy()

        # 5. Metrics
    ap = compute_ap(pred_boxes, pred_cls, conf.cpu().numpy(), gt_boxes, gt_cls)    
    precision, recall = compute_metrics(pred_boxes, pred_cls, gt_boxes, gt_cls)
    esr = compute_esr(pred_boxes, pred_cls, gt_boxes, gt_cls)

    all_ap.append(ap)
    all_precision.append(precision)
    all_recall.append(recall)
    all_esr.append(esr)

    # visualize
    visualize_predictions(
        query_image,
        pred_boxes,
        pred_cls,
        conf,
        gt_boxes,
        gt_cls,
        save_path=f"vis_episode_{episode}.jpg"
    )
    '''
    episode_ap = []
    episode_precision = []
    episode_recall = []
    episode_esr = []

    for q_idx, query_image in enumerate(query_images):

        # =====================
        # INFERENCE
        # =====================
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            query_out = run_inference(adapted_model, query_image)
            end.record()

            torch.cuda.synchronize()
            inference_time = start.elapsed_time(end)
        else:
            import time
            t0 = time.time()
            query_out = run_inference(adapted_model, query_image)
            inference_time = (time.time() - t0) * 1000

        all_inference_time.append(inference_time)

        # =====================
        # PREDICTIONS
        # =====================
        pred_boxes, conf, pred_cls = postprocess(query_out)

        pred_boxes = pred_boxes.detach().cpu().numpy()
        pred_cls = pred_cls.detach().cpu().numpy()
        conf_np = conf.detach().cpu().numpy()

        # =====================
        # GROUND TRUTH
        # =====================
        ann = get_annotations_for_image(query_image)

        gt_boxes = ann["boxes"].clone().to(device)
        gt_cls = ann["labels"].to(device)

        mask = torch.isin(gt_cls, torch.tensor(episode_classes, device=device))
        gt_boxes = gt_boxes[mask]
        gt_cls = gt_cls[mask]

        image = Image.open(query_image)
        img_w, img_h = image.size

        gt_boxes[:, [0, 2]] *= (512 / img_w)
        gt_boxes[:, [1, 3]] *= (512 / img_h)

        gt_cls = torch.tensor([label_map[int(l)] for l in gt_cls], device=device)

        gt_boxes = gt_boxes.cpu().numpy()
        gt_cls = gt_cls.cpu().numpy()

        # =====================
        # METRICS
        # =====================
        ap = compute_ap(pred_boxes, pred_cls, conf_np, gt_boxes, gt_cls)
        precision, recall = compute_metrics(pred_boxes, pred_cls, gt_boxes, gt_cls)
        esr = compute_esr(pred_boxes, pred_cls, gt_boxes, gt_cls)

        episode_ap.append(ap)
        episode_precision.append(precision)
        episode_recall.append(recall)
        episode_esr.append(esr)

        # =====================
        # VISUALIZATION
        # =====================
        visualize_predictions(
            query_image,
            pred_boxes,
            pred_cls,
            conf_np,
            gt_boxes,
            gt_cls,
            save_path=f"vis_episode_{episode}_q{q_idx}.jpg"
        )

    # =====================
    # EPISODE AVERAGE
    # =====================
    all_ap.append(np.mean(episode_ap))
    all_precision.append(np.mean(episode_precision))
    all_recall.append(np.mean(episode_recall))
    all_esr.append(np.mean(episode_esr))

print("\n===== FINAL RESULTS =====")
print(f"Inference Time (avg): {sum(all_inference_time)/len(all_inference_time):.2f} ms")
print(f"mAP@0.5: {sum(all_ap)/len(all_ap):.4f}")
print(f"Precision: {sum(all_precision)/len(all_precision):.4f}")
print(f"Recall@0.5: {sum(all_recall)/len(all_recall):.4f}")
print(f"ESR: {sum(all_esr)/len(all_esr):.4f}")
avg_time = sum(all_inference_time)/len(all_inference_time)
fps = 1000 / avg_time
print(f"FPS: {fps:.2f}")    