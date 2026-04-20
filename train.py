# ============================================================
# train_dp.py — MAML + YOLO-NAS (DataParallel, Debug-heavy, Fixed device & shape issues)
# Hardcoded checkpoint: /mnt/d/fsod_project/yolo_nas_l.pth
# ============================================================

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
from prepare_data import build_dataset_info
from dataset import FSODDataset, fsod_collate_fn



# -------------------------
# Backbone freeze helpers
# -------------------------
def freeze_backbone(model):
    for p in model.feature_extractor.parameters():
        p.requires_grad = False

def unfreeze_backbone(model):
    for p in model.feature_extractor.parameters():
        p.requires_grad = True


from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

import json

# -------------------------
# Logging / constants
# -------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/malformed_annotations.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


MIN_NORMALIZED_DIMENSION = 0.01

# <<< IMPORTANT: HARDCODED path added and used where needed >>>
#HARDCODED_CHECKPOINT_PATH = "/mnt/d/fsod_project/yolo_nas_l.pth"
HARDCODED_CHECKPOINT_PATH = r"D:\fsod_project\best_visdrone_nas_36.pth"
CHECKPOINT_PATH = "checkpoint_dp.pth"
CLAMP_MIN, CLAMP_MAX = -15.0, 15.0
# Visualization vs metric thresholds
VIS_SCORE_THRESH = 0.05    # what humans see
METRIC_SCORE_THRESH = 0.05 # what training/eval uses

# -------------------------
# Training visualization
# -------------------------
SAVE_TRAIN_VIS = True
VIS_EVERY_EPOCH = 1      # visualize every epoch
VIS_EPISODES_PER_EPOCH = 95  # visualize first 2 episodes only

# -------------------------
# Validation visualization
# -------------------------
SAVE_VAL_VIS = True
VAL_VIS_EPISODES = 3   # save first 3 validation episodes

# -------------------------
# META-TEST visualization
# -------------------------
SAVE_META_TEST_VIS = True
META_TEST_VIS_EPISODES = 3   # save only first 3 meta-test episodes
META_TEST_VIS_ROOT = "meta_test_vis"
META_TEST_VIS_SCORE_THRESH = 0.01  # visualization only

# -------------------------
# BatchNorm freeze helper (CRITICAL for MAML)
# -------------------------
def freeze_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()


# -------------------------
# Utility: safe flatten / unwrap
# -------------------------
def safe_unwrap(predictions):
    """Return a list of tensors flattened from nested YOLO-NAS output structures."""
    if predictions is None:
        return None

    def flatten(x):
        out = []
        if isinstance(x, (list, tuple)):
            for e in x:
                out.extend(flatten(e))
        elif torch.is_tensor(x):
            out.append(x)
        return out

    flat = flatten(predictions)
    return flat  #  may be empty list

# -------------------------
# Utility: normalize pred_scores channels to desired num_classes
# -------------------------
def normalize_pred_scores(pred_scores, num_classes, device):
    """
    pred_scores shape: (B, N, C_out)
    We want (B, N, num_classes). Several heuristics:
      - If C_out == num_classes: OK
      - If C_out == num_classes + 1: maybe includes background/conf => drop last
      - If C_out > num_classes: take first num_classes (heuristic)
      - If C_out == 0: return zeros
      - If None: return zeros (B,N,num_classes)
    """
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

# -------------------------
# convert_yolonas_to_ppyolo (robust & device-safe)
# -------------------------
def convert_yolonas_to_ppyolo(pred_list, desired_num_classes):
    """
    Convert flattened YOLO-NAS outputs (list of tensors) into PPYoloELoss-compatible tuple:
      (pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor)
    This function is robust to small structural differences and ensures all tensors live on same device.
    Also prints debug info.
    """
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

    # -------------------
    # [!!!] CRITICAL FIX [!!!]
    # REMOVED division: anchor_points = anchor_points / stride_tensor
    # We will pass PIXEL-SPACE anchor_points to both the loss and the decoder.
    # The PPYoloELoss accepts the stride_tensor and can handle pixel-space coords.
    # The decoder function `decode_yolonas_outputs` is already written to
    # expect PIXEL-SPACE anchor_points.
    # -------------------

    num_anchors_list = [anchors.size(0)]


    return (pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor)

# -------------------------
# ensure_differentiable_loss
# -------------------------
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

# -------------------------
# Decode for validation mAP (a small decoder; not perfect but OK)
# -------------------------
def decode_yolonas_outputs(pred_scores, pred_distri, anchor_points, stride, reg_max=16, score_thresh=0.25, iou_thresh=0.5):
    """
    Convert pred_scores, pred_distri, anchors, stride -> list of detections per image:
      each detection: [x1,y1,x2,y2,score,label]
    Boxes are decoded from distribution using expected reg_max.
    
    [FIXED]: Now uses anchor_points (N,2) for centers, not anchors (N,4).
    [NOTE]: This function expects anchor_points in PIXEL-SPACE.
    """
    device = pred_distri.device
    bs = pred_scores.shape[0]
    proj = torch.arange(reg_max + 1, dtype=torch.float, device=device)
    pred_dist = pred_distri.view(bs, -1, 4, reg_max + 1)
    pred_dist = pred_dist.softmax(dim=-1)
    pred_dist = (pred_dist * proj).sum(dim=-1)  #  (B, N, 4)

    # [FIX] Use anchor_points (N,2) as centers
    cx = anchor_points[:, 0].to(device)
    cy = anchor_points[:, 1].to(device)
    
    # stride may be (N,1)
    stride = stride.to(device)
    
    # form boxes: x1 = cx - l*stride, y1 = cy - t*stride, x2 = cx + r*stride, y2 = cy + b*stride
    # The PPYoloE prediction `pred_dist` is a distance in *pixel space* already,
    # because it's multiplied by stride *inside* the DFL-loss-like head.
    # The original YOLO-NAS decoder multiplies by stride.
    boxes = torch.zeros_like(pred_dist)
    boxes[..., 0] = cx - pred_dist[..., 0] * stride.squeeze(-1)
    boxes[..., 1] = cy - pred_dist[..., 1] * stride.squeeze(-1)
    boxes[..., 2] = cx + pred_dist[..., 2] * stride.squeeze(-1)
    boxes[..., 3] = cy + pred_dist[..., 3] * stride.squeeze(-1)

    pred_scores_sig = pred_scores.sigmoid()

    results = []
    for b in range(bs):
        scores, labels = pred_scores_sig[b].max(dim=-1)  #  per-anchor best class
        mask = scores > score_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 6), device=device))
            continue
        boxes_b = boxes[b][mask]
        scores_b = scores[mask]
        labels_b = labels[mask].float()
        keep = nms(boxes_b, scores_b, iou_thresh) if boxes_b.shape[0] > 0 else torch.tensor([], dtype=torch.long, device=device)
        if keep.numel() == 0:
            results.append(torch.zeros((0, 6), device=device))
            continue
        dets = torch.cat([boxes_b[keep], scores_b[keep, None], labels_b[keep, None]], dim=1)
        results.append(dets)
    return results

# -------------------------
# Main training function
# -------------------------
def main():
    print("\n--- Initializing Setup (DataParallel) ---")

    #--temp--
    TRAIN_VIS_ROOT = "train_vis"
    os.makedirs(TRAIN_VIS_ROOT, exist_ok=True)
    os.makedirs(META_TEST_VIS_ROOT, exist_ok=True)

    #--temp--

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Set up results logger ---
    results_logger = logging.getLogger('TrainingSummary')
    results_logger.setLevel(logging.INFO)
    # Prevent logs from propagating to the root logger
    results_logger.propagate = False 

    # Remove existing handlers (if any, from previous runs)
    if results_logger.hasHandlers():
        results_logger.handlers.clear()

    # Create file handler, mode='w' to overwrite file for each new run
    results_file_handler = logging.FileHandler("logs/training_summary.log", mode='w')
    results_file_handler.setLevel(logging.INFO)
    
    # Create formatter - simple message-only format for a clean CSV
    results_formatter = logging.Formatter('%(message)s')
    results_file_handler.setFormatter(results_formatter)
    
    # Add handler to the logger
    results_logger.addHandler(results_file_handler)
    
    # Write the CSV header
    results_logger.info(
        "timestamp,epoch,"
        "train_loss_box,train_loss_obj,train_loss_cls,train_mean_iou,"
        "train_recall_05,train_esr_05,"
        "val_loss,val_loss_box,val_loss_obj,val_loss_cls,val_map_50,"
        "val_recall_05,val_esr_05,"
        "test_map_50,test_recall_05,test_esr_05"
    )

    # --- End of results logger setup ---

    # Data transforms
    image_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset & loader
    base_dataset_info = build_dataset_info(config.BASE_CLASSES_PATH)
    train_dataset = FSODDataset(
        dataset_info=base_dataset_info,
        n_way=config.N_WAY,
        k_shot=config.K_SHOT,
        q_query=config.Q_QUERY,
        task_count=config.TASKS_PER_EPOCH
    )
    
    # -------------------------
    # META-TEST DATASET (NOVEL CLASSES)
    # -------------------------
    novel_dataset_info = build_dataset_info(config.NOVEL_CLASSES_PATH)

    meta_test_dataset = FSODDataset(
        dataset_info=novel_dataset_info,
        n_way=config.META_TEST_N_WAY,
        k_shot=config.META_TEST_K_SHOT,
        q_query=config.META_TEST_Q_QUERY,
        task_count=config.NUM_META_TEST_EPISODES
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.META_BATCH_SIZE,
        collate_fn=fsod_collate_fn,
        num_workers=0,
        shuffle=True
    )

    # Model
    print("Initializing model...")
    model = MAML_YOLO_NAS(
        model_arch="yolo_nas_l",
        num_classes=config.N_WAY,
        checkpoint_path=HARDCODED_CHECKPOINT_PATH
    ).to(device)
    '''print(f"🔁 Loading checkpoint: {config.BEST_CHECKPOINT_PATH}")
    #---Temp
    ckpt = torch.load(config.BEST_CHECKPOINT_PATH, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "net" in ckpt:
        model.load_state_dict(ckpt["net"])
    else:
        model.load_state_dict(ckpt)

    print("✅ Checkpoint loaded successfully")
    #---temp'''

    if torch.cuda.device_count() >= 1:
        device_ids = [0]
        model = nn.DataParallel(model, device_ids=device_ids)
        base_model = model.module
    else:
        base_model = model

    print("--- Setup Complete ---\nMETA-TRAINING LOOP_start")

    # Optimizer, loss, metrics
    meta_optimizer = optim.Adam(base_model.parameters(), lr=config.META_LR)

    # >>> ensure loss uses N_WAY (not config.NUM_CLASSES which may not exist)
    criterion = PPYoloELoss(use_static_assigner=False, num_classes=config.N_WAY, reg_max=16).to(device)

    post_processing_callback = PPYoloEPostPredictionCallback(
        score_threshold=0.25,
        nms_threshold=0.45,
        nms_top_k=1000,
        max_predictions=300
    )

    metrics = None
    try:
        import torchmetrics
        metrics = torchmetrics.detection.MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=[config.EVAL_IOU_THRESHOLD]
        )
    except Exception:
        metrics = None

    # --- NEW: Initialize Best Metric Tracking ---
    # -------------------------
    # BEST METRIC TRACKERS (META-VALIDATION)
    # -------------------------
    best_map_50 = -1.0
    best_val_loss = float("inf")
    best_val_esr = -1.0
    best_val_recall = -1.0

    patience_counter = 0

    # Minimum meaningful improvement thresholds
    MAP_EPS = 1e-3
    LOSS_EPS = 1e-4
    ESR_EPS = 1e-3
    RECALL_EPS = 1e-3

    BEST_CHECKPOINT_PATH = "best_checkpoint_dp.pth"

    # Early Stopping Settings
    EARLY_STOPPING_PATIENCE = 25  # Stop if no improvement for 15 epochs
    
    # --------------------------------------------

    # Training loop
    
    for epoch in range(0, config.NUM_EPOCHS):    
        
        # -------------------------
        # Freeze backbone early (meta-stability)
        # -------------------------
        if epoch < 0:
            freeze_backbone(base_model)
        else:
            unfreeze_backbone(base_model)
        
        '''# -------------------------
        # Backbone freeze schedule
        # -------------------------
        if epoch < 25:
            freeze_backbone(base_model)

        elif 25 <= epoch <= 35:
            unfreeze_backbone(base_model)

        else:
            freeze_backbone(base_model)'''


        meta_box_total, meta_obj_total, meta_cls_total, meta_iou_total = 0.0, 0.0, 0.0, 0.0
        meta_task_count = 0  #  counts number of query entries we aggregated for averages

        print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
        base_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")
        meta_optimizer.zero_grad()
        batch_query_losses = []
        malformed_counter = 0
        # ---- FSOD episodic metrics (per epoch) ----
        episode_recall_05_list = []
        episode_esr_05_list = []


        for episode in pbar:
                
                
            #for episode in task_batch:
                # episode is a batch list → extract tuple
                if isinstance(episode, list) or isinstance(episode, tuple):
                    episode = episode[0]
                
                #--temp--
                episode_idx = meta_task_count
                meta_task_count += 1
                #--uncomment below---

                do_visualize = (
                    SAVE_TRAIN_VIS
                    and (epoch % VIS_EVERY_EPOCH == 0)
                    and (episode_idx > VIS_EPISODES_PER_EPOCH)
                )
                #--temp--
                
                #meta_task_count += 1
                # ---- Reset FSOD metrics for this episode ----
                recall_hits_05 = 0
                total_gt_05 = 0
                episode_success_05 = False

                
                s_paths, s_targets, q_paths, q_targets, episode_classes = episode
                
                # ---- SAFE CLONE (NO deepcopy) ----
                fast_model = MAML_YOLO_NAS(
                    model_arch=config.MODEL_ARCH,
                    num_classes=config.N_WAY,
                    checkpoint_path=None,   # IMPORTANT: do NOT reload checkpoint
                    verbose=False
                ).to(device)

                fast_model.load_state_dict(base_model.state_dict())

                fast_model.train()
                fast_model.apply(freeze_bn)

                # make a fast copy for inner-loop
                #fast_model = copy.deepcopy(base_model).to(device)
                #fast_model.feature_extractor.train()
                #fast_model.train()
                #fast_model.apply(freeze_bn)


                '''for m in fast_model.feature_extractor.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.momentum = 0.01'''
                #inner_optimizer = optim.SGD(fast_model.get_inner_loop_params(), lr=config.INNER_LR)
                #inner_optimizer = optim.Adam(fast_model.get_inner_loop_params(), lr=config.INNER_LR)
                inner_optimizer = optim.SGD(
                    fast_model.get_inner_loop_params(),
                    lr=config.INNER_LR,
                    #momentum=0.9
                )#change


                # ---------- INNER (support) ----------
                for step in range(config.INNER_UPDATE_STEPS):
                    support_loss_task = torch.tensor(0.0, device=device)
                    valid_support_sample = False

                    for s_path, s_target in zip(s_paths, s_targets):
                        try:
                            image = Image.open(s_path).convert("RGB")
                            support_image = image_transforms(image).unsqueeze(0).to(device)

                            boxes, labels = s_target["boxes"], s_target["labels"]
                            if boxes.numel() == 0:
                                malformed_counter += 1
                                continue

                            # prefilter small boxes (pixel dims -> use original size)
                            img_w, img_h = image.size
                            widths = boxes[:, 2] - boxes[:, 0]
                            heights = boxes[:, 3] - boxes[:, 1]
                            valid_mask = ((widths / img_w) > MIN_NORMALIZED_DIMENSION) & ((heights / img_h) > MIN_NORMALIZED_DIMENSION)
                            if not valid_mask.any():
                                malformed_counter += 1
                                continue

                            boxes = boxes[valid_mask].float()
                            labels = labels[valid_mask].long()

                            # build cx,cy,w,h normalized relative to original image
                            boxes_xywh = torch.zeros_like(boxes)
                            boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
                            boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
                            boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]).clone()
                            boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]).clone()
                            # scale to model input (512x512) so targets are in same pixel space as anchors / stride
                            IN_W, IN_H = 512, 512
                            boxes_xywh[:, [0, 2]] = boxes_xywh[:, [0, 2]] * (IN_W / float(img_w))
                            boxes_xywh[:, [1, 3]] = boxes_xywh[:, [1, 3]] * (IN_H / float(img_h))


                            # ensure all on device before concatenation
                            batch_idx = torch.zeros((boxes_xywh.size(0), 1), device=device)
                            labels_dev = labels.unsqueeze(1).to(device)
                            boxes_dev = boxes_xywh.to(device)

                            s_target_tensor = torch.cat([batch_idx, labels_dev.float(), boxes_dev], dim=1)  #  (M,6)
                            # NOTE: PPYoloELoss expects float tensors; labels channel is sometimes float in their impl.

                            # forward
                            raw_preds = fast_model(support_image)
                            # ----------------------- NEW: recursive flatten fix -----------------------
                            if isinstance(raw_preds, (tuple, list)):
                                def flatten_all(x):
                                    out = []
                                    if isinstance(x, (list, tuple)):
                                        for e in x:
                                            out.extend(flatten_all(e))
                                    elif torch.is_tensor(x):
                                        out.append(x)
                                    return out

                                predictions_list = flatten_all(raw_preds)
                            else:
                                predictions_list = [raw_preds]

                            # sanity check
                            if predictions_list is None or len(predictions_list) == 0:
                                logging.info(f"[Epoch {epoch+1}] {s_path} - invalid model output structure.")
                                malformed_counter += 1
                                continue
                            
                            # convert
                            pred_input = convert_yolonas_to_ppyolo(predictions_list, desired_num_classes=config.N_WAY)
                            
                            # [FIX] Add clamping for stability
                            if isinstance(pred_input, (list, tuple)) and len(pred_input) >= 2:
                                ps, pd = pred_input[0], pred_input[1]
                                if torch.is_tensor(ps):
                                    ps = torch.clamp(ps, CLAMP_MIN, CLAMP_MAX)
                                if torch.is_tensor(pd):
                                    pd = torch.clamp(pd, CLAMP_MIN, CLAMP_MAX)
                                pred_input = (ps, pd, pred_input[2], pred_input[3], pred_input[4], pred_input[5])

                            # compute loss (PPYoloELoss wants (tuple, target) as we constructed)
                            support_loss, loss_parts = criterion(pred_input, s_target_tensor)
                            support_loss = ensure_differentiable_loss(support_loss, predictions_list, device)

                            # print loss parts safely
                            try:
                                lp = loss_parts.detach().cpu().numpy()
                            except Exception:
                                lp = str(loss_parts)
                            

                        except Exception as e:
                            malformed_counter += 1
                            logging.info(f"[Epoch {epoch+1}] {s_path} - support sample processing error: {e}")
                            print(f"[ERROR][Support] {s_path} - {e}")
                            continue

                        support_loss_task = support_loss_task + support_loss
                        valid_support_sample = True

                    # inner update
                    if not valid_support_sample or support_loss_task.item() == 0.0:
                        print(f"[Inner Step {step+1}] No valid support samples or zero loss, skipping update.")
                    else:
                        inner_optimizer.zero_grad()
                        support_loss_task.backward()
                        torch.nn.utils.clip_grad_norm_(fast_model.parameters(), max_norm=5.0)  #  <-- ADD THIS LINE
                        inner_optimizer.step()
                
                # ---------- QUERY loop ----------
                query_loss_task = torch.tensor(0.0, device=device)
                valid_query_sample = False
                #fast_model.eval()
                fast_model.train()
                fast_model.apply(freeze_bn)



                # [FIX #5] Define helper function once
                def xywh_to_xyxy_helper(boxes):
                    cx, cy, w, h = boxes.unbind(-1)
                    x1, y1 = cx - w / 2, cy - h / 2
                    x2, y2 = cx + w / 2, cy + h / 2
                    return torch.stack([x1, y1, x2, y2], dim=-1)

                for q_path, q_target in zip(q_paths, q_targets):
                    try:
                        image = Image.open(q_path).convert("RGB")
                        query_image = image_transforms(image).unsqueeze(0).to(device)

                        boxes, labels = q_target["boxes"], q_target["labels"]
                        if boxes.numel() == 0:
                            malformed_counter += 1
                            continue

                        img_w, img_h = image.size
                        widths = boxes[:, 2] - boxes[:, 0]
                        heights = boxes[:, 3] - boxes[:, 1]
                        valid_mask = ((widths / img_w) > MIN_NORMALIZED_DIMENSION) & ((heights / img_h) > MIN_NORMALIZED_DIMENSION)
                        if not valid_mask.any():
                            malformed_counter += 1
                            continue

                        boxes = boxes[valid_mask].float()
                        labels = labels[valid_mask].long()

                        boxes_xywh = torch.zeros_like(boxes)
                        boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
                        boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
                        boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]).clone()
                        boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]).clone()
                        # scale to model input (512x512) so targets are in same pixel space as anchors / stride
                        IN_W, IN_H = 512, 512
                        boxes_xywh[:, [0, 2]] = boxes_xywh[:, [0, 2]] * (IN_W / float(img_w))
                        boxes_xywh[:, [1, 3]] = boxes_xywh[:, [1, 3]] * (IN_H / float(img_h))


                        batch_idx = torch.zeros((boxes_xywh.size(0), 1), device=device)
                        labels_dev = labels.unsqueeze(1).to(device)
                        boxes_dev = boxes_xywh.to(device)
                        q_target_tensor = torch.cat([batch_idx, labels_dev.float(), boxes_dev], dim=1)

                        raw_preds = fast_model(query_image)
                        

                        # ----------------------- NEW: recursive flatten fix -----------------------
                        if isinstance(raw_preds, (tuple, list)):
                            

                            def flatten_all(x):
                                out = []
                                if isinstance(x, (list, tuple)):
                                    for e in x:
                                        out.extend(flatten_all(e))
                                elif torch.is_tensor(x):
                                    out.append(x)
                                return out

                            predictions_list = flatten_all(raw_preds)
                        else:
                            predictions_list = [raw_preds]

                        # sanity check
                        if predictions_list is None or len(predictions_list) == 0:
                            logging.info(f"[Epoch {epoch+1}] {q_path} - invalid model output structure.")
                            malformed_counter += 1
                            continue
# -------------------------------------------------------------------------


                        query_pred_input = convert_yolonas_to_ppyolo(predictions_list, desired_num_classes=config.N_WAY)
                        
                        # [FIX #3] Add clamping for stability
                        if isinstance(query_pred_input, (list, tuple)) and len(query_pred_input) >= 2:
                            ps, pd = query_pred_input[0], query_pred_input[1]
                            if torch.is_tensor(ps):
                                ps = torch.clamp(ps, CLAMP_MIN, CLAMP_MAX)
                            if torch.is_tensor(pd):
                                pd = torch.clamp(pd, CLAMP_MIN, CLAMP_MAX)
                            query_pred_input = (ps, pd, query_pred_input[2], query_pred_input[3], query_pred_input[4], query_pred_input[5])

                        #---temp---
                        # -------- VISUALIZE QUERY (AFTER CLAMPING) --------
                        if do_visualize:
                            import cv2
                            import numpy as np
                            #fast_model.eval()

                            with torch.no_grad():
                                pred_scores, pred_distri, _, anchor_points, _, stride_tensor = query_pred_input

                                decoded_vis = decode_yolonas_outputs(
                                    pred_scores,
                                    pred_distri,
                                    anchor_points,
                                    stride_tensor,
                                    score_thresh=VIS_SCORE_THRESH,   # debug visualization
                                    iou_thresh=0.5
                                )
                            #fast_model.train()    

                            dets = (
                                decoded_vis[0].detach().cpu().numpy()
                                if len(decoded_vis) > 0 and decoded_vis[0].numel() > 0
                                else np.zeros((0, 6), dtype=np.float32)
                            )

                            epoch_dir = os.path.join(TRAIN_VIS_ROOT, f"epoch_{epoch+1:03d}")
                            ep_dir = os.path.join(epoch_dir, f"episode_{episode_idx:03d}")
                            os.makedirs(ep_dir, exist_ok=True)

                            img = cv2.imread(q_path)
                            h, w = img.shape[:2]
                            sx, sy = w / 512.0, h / 512.0

                            for x1, y1, x2, y2, score, cls in dets:
                                x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # episodic -> global -> human-readable
                                global_cls = episode_classes[int(cls)]
                                class_name = config.INDEX_TO_CLASS.get(global_cls, f"id:{global_cls}")

                                label_text = f"{class_name} {score:.2f}"


                                cv2.putText(
                                    img,
                                    label_text,
                                    (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                    cv2.LINE_AA
                                )


                            out_path = os.path.join(ep_dir, os.path.basename(q_path))
                            cv2.imwrite(out_path, img)

                        #---temp---

                        

                        query_loss, loss_parts = criterion(query_pred_input, q_target_tensor)
                        query_loss = ensure_differentiable_loss(query_loss, predictions_list, device)
                        
                       
                        
                        try:
                            lp = loss_parts.detach().cpu().numpy()
                        except Exception:
                            lp = str(loss_parts)
                        

                    except Exception as e:
                        malformed_counter += 1
                        logging.info(f"[Epoch {epoch+1}] {q_path} - query sample processing error: {e}")
                        print(f"[ERROR][Query] {q_path} - {e}")
                        continue

                    query_loss_task = query_loss_task + query_loss
                    valid_query_sample = True

                    # ---------------- (NEW) META-TRAIN LOSS BREAKDOWN + IoU TRACKING ----------------
                    # ---------------- (FIXED) META-TRAIN LOSS BREAKDOWN + IoU TRACKING ----------------
                    try:
                        # Normalize the returned parts from PPYoloELoss (assume validation ordering: [box, cls, obj, total])
                        if isinstance(loss_parts, torch.Tensor) and loss_parts.numel() >= 4:
                            parts_arr = loss_parts.detach().cpu().numpy()
                            box_loss = float(parts_arr[0])
                            cls_loss = float(parts_arr[1])
                            obj_loss = float(parts_arr[2])
                            total_loss_reported = float(parts_arr[3])
                        elif isinstance(loss_parts, (list, tuple)) and len(loss_parts) >= 4:
                            box_loss = float(loss_parts[0].item() if torch.is_tensor(loss_parts[0]) else loss_parts[0])
                            cls_loss = float(loss_parts[1].item() if torch.is_tensor(loss_parts[1]) else loss_parts[1])
                            obj_loss = float(loss_parts[2].item() if torch.is_tensor(loss_parts[2]) else loss_parts[2])
                            total_loss_reported = float(loss_parts[3].item() if torch.is_tensor(loss_parts[3]) else loss_parts[3])
                        else:
                            # fallback if parts unavailable
                            total_loss_reported = float(query_loss.item())
                            box_loss = cls_loss = obj_loss = 0.0

                        meta_box_total += box_loss
                        meta_obj_total += obj_loss
                        meta_cls_total += cls_loss
                                            # after successfully aggregating losses / IoU for this query:
                        
                        print(f"[META-TRAIN][Query] reported_total={total_loss_reported:.4f} | box={box_loss:.4f} | obj={obj_loss:.4f} | cls={cls_loss:.4f}")
                    except Exception as e:
                        logging.info(f"[Meta-Train Loss Breakdown] Failed: {e}")

                    # --- (FIXED) IoU Computation: decode predictions and compare to GT ---
                    from torchvision.ops import box_iou

                    with torch.no_grad():
                        try:
                            pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor = query_pred_input


                            # decode predictions
                            decoded_list = decode_yolonas_outputs(
                                pred_scores, pred_distri, anchor_points, stride_tensor,
                                score_thresh=0.05, iou_thresh=0.5
                            )

                            if len(decoded_list) > 0 and decoded_list[0].numel() > 0:
                                dets = decoded_list[0]

                                # -------------------- NEW: Top-K filtering --------------------
                                MAX_DETECTIONS = 300
                                if dets.shape[0] > MAX_DETECTIONS:
                                    scores = dets[:, 4]
                                    _, topk_idx = torch.topk(scores, k=MAX_DETECTIONS)
                                    dets = dets[topk_idx]


                                preds_xyxy = dets[:, :4]
                                num_preds = preds_xyxy.shape[0]
                            else:
                                preds_xyxy = torch.zeros((0, 4), device=device)
                                num_preds = 0

                            # -------------------- CORRECTED: GT scaling --------------------
                            # q_target_tensor already contains 512x512 scaled xywh boxes
                            gt_xywh = q_target_tensor[:, 2:6] 

                            if gt_xywh.numel() > 0:
                                gt_xyxy = xywh_to_xyxy_helper(gt_xywh.to(device))
                            else:
                                gt_xyxy = torch.zeros((0, 4), device=device)

                            num_gts = gt_xyxy.shape[0]

                            # -------------------- IoU computation --------------------
                            if preds_xyxy.shape[0] > 0 and gt_xyxy.shape[0] > 0:
                                iou_m = box_iou(preds_xyxy.to(device), gt_xyxy.to(device))
                                best_per_gt, _ = iou_m.max(dim=0)
                                mean_iou = best_per_gt.mean().item()
                                meta_iou_total += mean_iou
                                print(f"[META-TRAIN][IoU] ✅ preds={num_preds} | gts={num_gts} | mean_best_iou={mean_iou:.4f}")
                                # ---- FSOD Recall@0.5 + ESR ----
                                hits = (best_per_gt >= 0.5).sum().item()
                                recall_hits_05 += hits
                                total_gt_05 += best_per_gt.numel()

                                if hits > 0:
                                    episode_success_05 = True

                            else:
                                if preds_xyxy.shape[0] == 0 and gt_xyxy.shape[0] == 0:
                                    print(f"[META-TRAIN][IoU] ⚠ Skipped (no preds *and* no GTs) for {os.path.basename(q_path)}")
                                elif preds_xyxy.shape[0] == 0:
                                    print(f"[META-TRAIN][IoU] ⚠ No preds (after decode/NMS) | GTs={num_gts} for {os.path.basename(q_path)}")
                                elif gt_xyxy.shape[0] == 0:
                                    print(f"[META-TRAIN][IoU] ⚠ No GT boxes (filtered out) | preds={num_preds} for {os.path.basename(q_path)}")

                        except Exception as e:
                            logging.info(f"[Meta-Train IoU Computation] Failed: {e}")
                            print(f"[META-TRAIN][IoU] ❌ Exception: {e}")

                # ---- Finalize FSOD metrics for this episode ----
                episode_recall = recall_hits_05 / max(total_gt_05, 1)
                episode_recall_05_list.append(episode_recall)
                episode_esr_05_list.append(float(episode_success_05))

                # ---------- outer/meta gradient accumulation ----------
                if valid_query_sample and query_loss_task.item() > 0.0:
                    #print(f"[Meta Grad] Epoch {epoch+1}: total query_loss_task={query_loss_task.item()}")
                    batch_query_losses.append(query_loss_task.item())

                    base_named_params = dict(base_model.named_parameters())

                    fast_named_params = {
                        name: p
                        for name, p in fast_model.named_parameters()
                        if p.requires_grad
                    }

                    assert set(fast_named_params.keys()).issubset(base_named_params.keys())


                    query_grads = torch.autograd.grad(
                        query_loss_task,
                        list(fast_named_params.values()),
                        allow_unused=True
                    )


                    # accumulate into base_model grads
                    

                    with torch.no_grad():
                        for (name, p_fast), g_query in zip(fast_named_params.items(), query_grads):
                            if g_query is None:
                                continue

                            p_base = base_named_params.get(name, None)
                            if p_base is None or not p_base.requires_grad:
                                continue

                            if p_base.grad is None:
                                p_base.grad = g_query.clone() / config.META_BATCH_SIZE
                            else:
                                p_base.grad += g_query.clone() / config.META_BATCH_SIZE

                    #---temp---
                    num_nonzero = sum(
                        1 for p in base_model.parameters()
                        if p.grad is not None and p.grad.abs().sum() > 0
                    )
                    print(f"[META] Non-zero base grads: {num_nonzero}")
                    #---temp---

                
                else:
                    print(f"[Meta Grad] Epoch {epoch+1}: No valid query samples or zero loss, skipping meta-gradient.")

                # meta optimizer step every META_BATCH_SIZE episodes
                if meta_task_count % config.META_BATCH_SIZE == 0:
                    if any(p.grad is not None for p in base_model.parameters()):
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=5.0)
                        
                        
                        
                        meta_optimizer.step()
                        meta_optimizer.zero_grad()
                        print(f"[Meta Update] Step applied")
                        if batch_query_losses:
                            mean_loss = sum(batch_query_losses) / len(batch_query_losses)
                            pbar.set_description(f"Epoch {epoch + 1} Meta-Loss: {mean_loss:.4f}")
                            batch_query_losses = []
                    else:
                        print(f"[Meta Update] No grads accumulated, skipping optimizer step.")

        print(f"[Summary] {malformed_counter} malformed/skipped samples in epoch {epoch + 1}")
        denom = meta_task_count if meta_task_count > 0 else 1
        # --- Store training metrics for logging ---
        train_mean_box = meta_box_total / denom
        train_mean_obj = meta_obj_total / denom
        train_mean_cls = meta_cls_total / denom
        train_mean_iou = meta_iou_total / denom

        print(f"[Meta-Train Breakdown] mean_box={meta_box_total/denom:.4f}, "
            f"mean_obj={meta_obj_total/denom:.4f}, "
            f"mean_cls={meta_cls_total/denom:.4f}")
        print(f"[Meta-Train IoU Summary] mean_iou={meta_iou_total/denom:.4f} (over {meta_task_count} queries)")
        mean_recall_05 = sum(episode_recall_05_list) / max(len(episode_recall_05_list), 1)
        mean_esr_05 = sum(episode_esr_05_list) / max(len(episode_esr_05_list), 1)

        print(f"[FSOD] Recall@0.5: {mean_recall_05:.4f}")
        print(f"[FSOD] ESR@0.5: {mean_esr_05:.4f}")


        logging.info(f"[Epoch {epoch+1} Summary] {malformed_counter} malformed/skipped samples.")
    
        # -------------------------
        # VALIDATION
        # -------------------------
        print("\n--- VALIDATION ---")
        # ---- FSOD VALIDATION METRICS (EPISODIC) ----
        val_episode_recall_05_list = []
        val_episode_esr_05_list = []

        if metrics is not None:
            metrics.reset()
        base_model.eval()

        val_loss_total = 0.0
        val_batches = 0

        val_cls_total, val_obj_total, val_box_total = 0.0, 0.0, 0.0

        val_count = min(config.VALIDATION_TASK_COUNT, len(train_dataset))
        val_pbar = tqdm(range(val_count), desc="Validation")

        for task_idx in val_pbar:
            val_episode_idx = task_idx
            try:
                episode = train_dataset.__getitem__(task_idx)
                #s_paths, s_targets, q_paths, q_targets, _ = episode
                s_paths, s_targets, q_paths, q_targets, episode_classes = episode

                # ---- Reset FSOD validation metrics for this episode ----
                val_recall_hits_05 = 0
                val_total_gt_05 = 0
                val_episode_success_05 = False

                # ---- SAFE CLONE (NO deepcopy) ----
                finetune_model = MAML_YOLO_NAS(
                    model_arch=config.MODEL_ARCH,
                    num_classes=config.N_WAY,
                    checkpoint_path=None,
                    verbose=False
                ).to(device)

                finetune_model.load_state_dict(base_model.state_dict())

                finetune_model.train()
                finetune_model.apply(freeze_bn)


                #finetune_model = copy.deepcopy(base_model).to(device)
                #finetune_model.train()
                #finetune_model.apply(freeze_bn)


                # IMPORTANT: freeze backbone BN during few-shot finetuning
                #finetune_model.feature_extractor.eval()
                #finetune_model.feature_extractor.train()

                '''for m in finetune_model.feature_extractor.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.momentum = 0.01'''

                do_val_visualize = (
                    SAVE_VAL_VIS
                    and val_episode_idx < VAL_VIS_EPISODES
                )


                optimizer = optim.Adam(finetune_model.get_inner_loop_params(), lr=config.INNER_LR)

                # ------------------ Fine-tune on Support ------------------
                for ft_step in range(config.FINETUNE_STEPS):
                    support_loss_task = torch.tensor(0.0, device=device)
                    valid_ft_sample = False
                    for s_path, s_target in zip(s_paths, s_targets):
                        try:
                            image = Image.open(s_path).convert("RGB")
                            support_image = image_transforms(image).unsqueeze(0).to(device)
                            boxes, labels = s_target["boxes"], s_target["labels"]
                            if boxes.numel() == 0:
                                continue
                            img_w, img_h = image.size
                            #print(f"[VAL][Support][{os.path.basename(s_path)}] unique class labels: {torch.unique(labels)}")


                            widths = boxes[:, 2] - boxes[:, 0]
                            heights = boxes[:, 3] - boxes[:, 1]
                            valid_mask = ((widths / img_w) > MIN_NORMALIZED_DIMENSION) & ((heights / img_h) > MIN_NORMALIZED_DIMENSION)
                            if not valid_mask.any():
                                continue
                            boxes = boxes[valid_mask].float()
                            labels = labels[valid_mask].long()
                            boxes_xywh = torch.zeros_like(boxes)
                            boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
                            boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
                            boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0])
                            boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1])
                            # scale to model input (512x512) so targets are in same pixel space as anchors / stride
                            IN_W, IN_H = 512, 512
                            boxes_xywh[:, [0, 2]] = boxes_xywh[:, [0, 2]] * (IN_W / float(img_w))
                            boxes_xywh[:, [1, 3]] = boxes_xywh[:, [1, 3]] * (IN_H / float(img_h))

                            batch_idx = torch.zeros((boxes_xywh.size(0), 1), device=device)
                            labels_dev = labels.unsqueeze(1).to(device)
                            boxes_dev = boxes_xywh.to(device)
                            target_tensor = torch.cat([batch_idx, labels_dev.float(), boxes_dev], dim=1)

                            raw_preds = finetune_model(support_image)

                            # ----------------------- NEW: recursive flatten fix -----------------------
                            if isinstance(raw_preds, (tuple, list)):

                                def flatten_all(x):
                                    out = []
                                    if isinstance(x, (list, tuple)):
                                        for e in x:
                                            out.extend(flatten_all(e))
                                    elif torch.is_tensor(x):
                                        out.append(x)
                                    return out

                                predictions_list = flatten_all(raw_preds)
                            else:
                                predictions_list = [raw_preds]

                            # sanity check
                            if predictions_list is None or len(predictions_list) == 0:
                                logging.info(f"[Validation Support] {s_path} - invalid model output structure.")
                                continue
                            # -------------------------------------------------------------------------


                            pred_input_tuple = convert_yolonas_to_ppyolo(predictions_list, desired_num_classes=config.N_WAY)
                            
                            # [FIX #2] Corrected NameError (was 'pred_input') and standardized clamp
                            if isinstance(pred_input_tuple, (list, tuple)) and len(pred_input_tuple) >= 2:
                                ps, pd = pred_input_tuple[0], pred_input_tuple[1]
                                if torch.is_tensor(ps):
                                    ps = torch.clamp(ps, CLAMP_MIN, CLAMP_MAX)
                                if torch.is_tensor(pd):
                                    pd = torch.clamp(pd, CLAMP_MIN, CLAMP_MAX)
                                # [FIX #2] Re-assign to pred_input_tuple
                                pred_input_tuple = (ps, pd, pred_input_tuple[2], pred_input_tuple[3], pred_input_tuple[4], pred_input_tuple[5])


                            pred_scores = pred_input_tuple[0]
                            #print(f"[VAL][Support][{os.path.basename(s_path)}] pred_scores range: min={pred_scores.min().item():.4f}, max={pred_scores.max().item():.4f}")

                            # [FIX] Removed redundant/conflicting clamp block
                            # pred_input_tuple = (
                            #     torch.clamp(pred_input_tuple[0], -10, 10), ...
                            # )

                            loss_ft, _ = criterion(pred_input_tuple, target_tensor)
                            loss_ft = ensure_differentiable_loss(loss_ft, predictions_list, device)
                            support_loss_task += loss_ft
                            valid_ft_sample = True
                        except Exception as e:
                            logging.info(f"[Validation FineTune Support] {s_path} - {e}")
                            continue

                    if valid_ft_sample and support_loss_task.item() > 0.0:
                        optimizer.zero_grad()
                        support_loss_task.backward()
                        torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), max_norm=5.0)
                        optimizer.step()

                # ------------------ Evaluate on Queries ------------------
                finetune_model.train()
                finetune_model.apply(freeze_bn)

                with torch.no_grad():
                    for q_path, q_target in zip(q_paths, q_targets):
                        try:
                            image = Image.open(q_path).convert("RGB")
                            query_image = image_transforms(image).unsqueeze(0).to(device)
                            raw_preds = finetune_model(query_image)

                            # ----------------------- NEW: recursive flatten fix -----------------------
                            if isinstance(raw_preds, (tuple, list)):
                                
                                def flatten_all(x):
                                    out = []
                                    if isinstance(x, (list, tuple)):
                                        for e in x:
                                            out.extend(flatten_all(e))
                                    elif torch.is_tensor(x):
                                        out.append(x)
                                    return out

                                predictions_list = flatten_all(raw_preds)
                            else:
                                predictions_list = [raw_preds]

                            # sanity check
                            if predictions_list is None or len(predictions_list) == 0:
                                logging.info(f"[Validation Query] {q_path} - invalid model output structure.")
                                continue
                            # -------------------------------------------------------------------------


                            pred_input_tuple = convert_yolonas_to_ppyolo(predictions_list, desired_num_classes=config.N_WAY)

                            # [FIX #4] Add clamping for stability
                            if isinstance(pred_input_tuple, (list, tuple)) and len(pred_input_tuple) >= 2:
                                ps, pd = pred_input_tuple[0], pred_input_tuple[1]
                                if torch.is_tensor(ps):
                                    ps = torch.clamp(ps, CLAMP_MIN, CLAMP_MAX)
                                if torch.is_tensor(pd):
                                    pd = torch.clamp(pd, CLAMP_MIN, CLAMP_MAX)
                                pred_input_tuple = (ps, pd, pred_input_tuple[2], pred_input_tuple[3], pred_input_tuple[4], pred_input_tuple[5])
                            pred_scores = pred_input_tuple[0]
                            #print(f"[VAL][Query][{os.path.basename(q_path)}] pred_scores range: min={pred_scores.min().item():.4f}, max={pred_scores.max().item():.4f}")



                            # Compute validation query loss
                            boxes, labels = q_target["boxes"], q_target["labels"]
                            if boxes.numel() == 0:
                                continue
                            img_w, img_h = image.size
                            #print(f"[VAL][Query][{os.path.basename(q_path)}] unique class labels: {torch.unique(labels)}")

                            boxes_xywh = torch.zeros_like(boxes)
                            boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
                            boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
                            boxes_xywh[:, 2] = (boxes[:, 2] - boxes[:, 0])
                            boxes_xywh[:, 3] = (boxes[:, 3] - boxes[:, 1])
                            # scale to model input (512x512) so targets are in same pixel space as anchors / stride
                            IN_W, IN_H = 512, 512
                            boxes_xywh[:, [0, 2]] = boxes_xywh[:, [0, 2]] * (IN_W / float(img_w))
                            boxes_xywh[:, [1, 3]] = boxes_xywh[:, [1, 3]] * (IN_H / float(img_h))

                            batch_idx = torch.zeros((boxes_xywh.size(0), 1), device=device)
                            labels_dev = labels.unsqueeze(1).to(device)
                            boxes_dev = boxes_xywh.to(device)
                            q_target_tensor = torch.cat([batch_idx, labels_dev.float(), boxes_dev], dim=1)

                            # Compute validation query loss with detailed breakdown
                            try:
                                val_loss, parts = criterion(pred_input_tuple, q_target_tensor)
                                val_loss_total += val_loss.item()
                                val_batches += 1

                                # --- Normalize parts to python floats ---
                                parts_list = None
                                try:
                                    if isinstance(parts, torch.Tensor):
                                        parts_list = parts.detach().cpu().numpy().tolist()
                                    elif isinstance(parts, (list, tuple)):
                                        parts_list = [p.item() if torch.is_tensor(p) else float(p) for p in parts]
                                    else:
                                        parts_list = [float(parts)]
                                except Exception:
                                    parts_list = [float(val_loss.item())]

                                # --- Print raw parts with indices so we can verify ordering ---
                                #print(f"[VAL][Parts] raw parts (idx:value): " + ", ".join([f"{i}:{v:.6f}" for i, v in enumerate(parts_list)]))

                                # --- Heuristic/default mapping ---
                                # Many PP-like losses return 4 items: [box, cls, obj, total] or [total, cls, obj, total] variants.
                                # We'll assume default order: [box, cls, obj, total] (index 0=box,1=cls,2=obj,3=total).
                                # If your printed parts show a different layout, change these indices accordingly.
                                if len(parts_list) >= 4:
                                    box_loss = float(parts_list[0])
                                    cls_loss = float(parts_list[1])
                                    obj_loss = float(parts_list[2])
                                    total_loss_reported = float(parts_list[3])
                                elif len(parts_list) == 3:
                                    # fallback: [box, cls, obj]
                                    box_loss, cls_loss, obj_loss = parts_list
                                    total_loss_reported = val_loss.item()
                                else:
                                    box_loss = cls_loss = obj_loss = 0.0
                                    total_loss_reported = val_loss.item()

                                # accumulate
                                val_box_total += box_loss
                                val_obj_total += obj_loss
                                val_cls_total += cls_loss

                                # print per-query breakdown (use both criterion total and reported total if available)
                                #print(f"[VAL][Query] {os.path.basename(q_path)}: total={val_loss.item():.6f} | reported_total={total_loss_reported:.6f} | cls={cls_loss:.6f} | obj={obj_loss:.6f} | box={box_loss:.6f}")

                            except Exception as e:
                                print(f"[VAL ERROR] {q_path} - {e}")
                                continue



                            # Decode for metric
                            pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor = pred_input_tuple
                            
                            # [FIX #6] Call decoder with 'anchor_points'
                            # --- (FIXED) Decode + IoU tracking + Metric update ---
                            from torchvision.ops import box_iou

                            try:
                                # decode predictions
                                decoded_preds = decode_yolonas_outputs(
                                    pred_scores, pred_distri, anchor_points, stride_tensor,
                                    score_thresh=0.01, iou_thresh=0.45
                                )

                                # -------- VISUALIZE VALIDATION QUERY --------
                                if do_val_visualize:
                                    import cv2
                                    import numpy as np

                                    dets = (
                                        decoded_preds[0].detach().cpu().numpy()
                                        if len(decoded_preds) > 0 and decoded_preds[0].numel() > 0
                                        else np.zeros((0, 6), dtype=np.float32)
                                    )

                                    if dets.shape[0] > 50:
                                        order = dets[:, 4].argsort()[::-1]
                                        dets = dets[order[:50]]

                                    VAL_VIS_ROOT = "val_vis"
                                    epoch_dir = os.path.join(VAL_VIS_ROOT, f"epoch_{epoch+1:03d}")
                                    ep_dir = os.path.join(epoch_dir, f"episode_{val_episode_idx:03d}")
                                    os.makedirs(ep_dir, exist_ok=True)

                                    img = cv2.imread(q_path)
                                    h, w = img.shape[:2]
                                    sx, sy = w / 512.0, h / 512.0

                                    for x1, y1, x2, y2, score, cls in dets:
                                        x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                                        # episodic -> global -> human-readable
                                        global_cls = episode_classes[int(cls)]
                                        class_name = config.INDEX_TO_CLASS.get(global_cls, f"id:{global_cls}")

                                        label_text = f"{class_name} {score:.2f}"


                                        cv2.putText(
                                            img,
                                            label_text,
                                            (x1, max(15, y1 - 5)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (255, 0, 0),
                                            1,
                                            cv2.LINE_AA
                                        )


                                    out_path = os.path.join(ep_dir, os.path.basename(q_path))
                                    cv2.imwrite(out_path, img)


                                if len(decoded_preds) > 0 and decoded_preds[0].numel() > 0:
                                    dets = decoded_preds[0]

                                    # --- NEW: Top-K filtering to stabilize metrics ---
                                    MAX_DETECTIONS = 300
                                    if dets.shape[0] > MAX_DETECTIONS:
                                        scores = dets[:, 4]
                                        _, topk_idx = torch.topk(scores, k=MAX_DETECTIONS)
                                        dets = dets[topk_idx]



                                    preds_xyxy = dets[:, :4]
                                    num_preds = preds_xyxy.shape[0]
                                else:
                                    preds_xyxy = torch.zeros((0, 4), device=device)
                                    num_preds = 0

                                # --- NEW: Scale GT boxes to model input (512x512) ---
                                gt_boxes = q_target["boxes"]
                                img_w, img_h = image.size
                                IN_W, IN_H = 512, 512
                                gt_xywh_scaled = torch.zeros_like(gt_boxes)
                                gt_xywh_scaled[:, 0] = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
                                gt_xywh_scaled[:, 1] = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
                                gt_xywh_scaled[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
                                gt_xywh_scaled[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]
                                gt_xywh_scaled[:, [0, 2]] *= IN_W / float(img_w)
                                gt_xywh_scaled[:, [1, 3]] *= IN_H / float(img_h)
                                gt_xyxy = torch.zeros((0, 4), device=device)
                                if gt_xywh_scaled.numel() > 0:
                                    gt_xywh_scaled = gt_xywh_scaled.to(device) # <-- ADD THIS LINE
                                    cx, cy, w, h = gt_xywh_scaled.unbind(-1)
                                    gt_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
                                num_gts = gt_xyxy.shape[0]

                                # --- IoU computation ---
                                if preds_xyxy.shape[0] > 0 and gt_xyxy.shape[0] > 0:
                                    iou_m = box_iou(preds_xyxy.to(device), gt_xyxy.to(device))
                                    best_per_gt, _ = iou_m.max(dim=0)
                                    mean_iou = best_per_gt.mean().item()
                                    # ---- FSOD Validation Recall@0.5 + ESR ----
                                    hits = (best_per_gt >= 0.5).sum().item()
                                    val_recall_hits_05 += hits
                                    val_total_gt_05 += best_per_gt.numel()

                                    if hits > 0:
                                        val_episode_success_05 = True

                                    print(f"[VAL][IoU] ✅ preds={num_preds} | gts={num_gts} | mean_best_iou={mean_iou:.4f}")
                                else:
                                    if preds_xyxy.shape[0] == 0 and gt_xyxy.shape[0] == 0:
                                        print(f"[VAL][IoU] ⚠ Skipped (no preds *and* no GTs) for {os.path.basename(q_path)}")
                                    elif preds_xyxy.shape[0] == 0:
                                        print(f"[VAL][IoU] ⚠ No preds (after decode/NMS) | GTs={num_gts} for {os.path.basename(q_path)}")
                                    elif gt_xyxy.shape[0] == 0:
                                        print(f"[VAL][IoU] ⚠ No GT boxes (filtered out) | preds={num_preds} for {os.path.basename(q_path)}")

                                # --- Metric computation ---
                                if metrics is not None:
                                    if preds_xyxy.shape[0] > 0:
                                        preds_for_metric = [{
                                            "boxes": preds_xyxy.detach().cpu(),
                                            "scores": dets[:, 4].detach().cpu(),
                                            "labels": dets[:, 5].long().detach().cpu(),
                                        }]
                                    else:
                                        preds_for_metric = [{
                                            "boxes": torch.zeros((0, 4)),
                                            "scores": torch.tensor([]),
                                            "labels": torch.tensor([]),
                                        }]

                                    

                                    targets_for_metric = [{
                                        "boxes": gt_xyxy.detach().cpu(),
                                        "labels": q_target["labels"].to("cpu"),
                                    }]

                                    try:
                                        metrics.update(preds_for_metric, targets_for_metric)
                                    except Exception as e:
                                        logging.info(f"[Validation Metric Update] error: {e}")

                            except Exception as e:
                                logging.info(f"[Validation Query Eval IoU Fix] {q_path} - {e}")



                        except Exception as e:
                            logging.info(f"[Validation Query Eval] {q_path} - {e}")
                            continue

                del finetune_model, optimizer
                val_episode_recall = val_recall_hits_05 / max(val_total_gt_05, 1)
                val_episode_recall_05_list.append(val_episode_recall)
                val_episode_esr_05_list.append(float(val_episode_success_05))

                torch.cuda.empty_cache()

            except Exception as e:
                logging.info(f"[Validation Episode {task_idx} Error] {e}")
                if 'finetune_model' in locals():
                    del finetune_model
                if 'optimizer' in locals():
                    del optimizer
                torch.cuda.empty_cache()
                continue
        if val_batches > 0:
            print(f"[Validation Breakdown] mean_box={val_box_total/val_batches:.4f}, mean_obj={val_obj_total/val_batches:.4f}, mean_cls={val_cls_total/val_batches:.4f}")

        # --- Store validation metrics for logging ---
        val_mean_box = 0.0
        val_mean_obj = 0.0
        val_mean_cls = 0.0
        
        if val_batches > 0:
            val_mean_box = val_box_total / val_batches
            val_mean_obj = val_obj_total / val_batches
            val_mean_cls = val_cls_total / val_batches
            print(f"[Validation Breakdown] mean_box={val_mean_box:.4f}, mean_obj={val_mean_obj:.4f}, mean_cls={val_mean_cls:.4f}")
        else:
             print(f"[Validation Breakdown] No valid validation batches.")

        # ------------------ Compute Metrics ------------------
        mean_val_loss = val_loss_total / max(val_batches, 1)
        map_50_val = -1.0

        if metrics is not None:
            try:
                # Safe mAP computation — only compute if we actually updated
                if hasattr(metrics, "_update_called") and getattr(metrics, "_update_called", False):
                    results = metrics.compute()
                    map_50 = results.get("map_50", None)
                    if map_50 is None:
                        map_50 = results.get("map", torch.tensor(-1.0))
                    map_50_val = map_50.item() if isinstance(map_50, torch.Tensor) else float(map_50)
                else:
                    map_50_val = -1.0  #  fallback if no updates happened

                print(f"✅ Epoch {epoch + 1} Validation | Loss: {mean_val_loss:.4f} | mAP@.50: {map_50_val:.4f}")
                mean_val_recall_05 = sum(val_episode_recall_05_list) / max(len(val_episode_recall_05_list), 1)
                mean_val_esr_05 = sum(val_episode_esr_05_list) / max(len(val_episode_esr_05_list), 1)

                print(f"[VAL][FSOD] Recall@0.5: {mean_val_recall_05:.4f}")
                print(f"[VAL][FSOD] ESR@0.5: {mean_val_esr_05:.4f}")


            except Exception as e:
                print(f"[ERROR] Failed to compute validation metrics: {e}")
                print(f"✅ Epoch {epoch + 1} Validation | Loss: {mean_val_loss:.4f}")
        else:
            print(f"✅ Epoch {epoch + 1} Validation | Loss: {mean_val_loss:.4f}")
        
        # -------------------------
        # META-TEST (NOVEL CLASSES)
        # -------------------------
        print("\n--- META-TEST (Novel Classes) ---")

        base_model.eval()

        test_episode_recall_05_list = []
        test_episode_esr_05_list = []

        test_metrics = None
        try:
            import torchmetrics
            test_metrics = torchmetrics.detection.MeanAveragePrecision(
                iou_type="bbox",
                iou_thresholds=[config.EVAL_IOU_THRESHOLD]
            )
        except Exception:
            test_metrics = None

        for episode_idx in range(config.NUM_META_TEST_EPISODES):
            try:
                s_paths, s_targets, q_paths, q_targets, episode_classes = meta_test_dataset[episode_idx]

                # ---- Reset episodic metrics ----
                test_recall_hits_05 = 0
                test_total_gt_05 = 0
                episode_success_05 = False

                # Copy θ → φ (NO gradient to θ)
                # ---- SAFE CLONE (NO deepcopy) ----
                test_model = MAML_YOLO_NAS(
                    model_arch=config.MODEL_ARCH,
                    num_classes=config.N_WAY,
                    checkpoint_path=None,   # IMPORTANT
                    verbose=False
                ).to(device)

                test_model.load_state_dict(base_model.state_dict())

                test_model.train()
                test_model.apply(freeze_bn)

                #test_model = copy.deepcopy(base_model).to(device)
                #test_model.train()
                #test_model.apply(freeze_bn)

                #test_model.feature_extractor.train()

                '''for m in test_model.feature_extractor.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.momentum = 0.01'''


                optimizer = optim.Adam(
                    test_model.get_inner_loop_params(),
                    lr=config.META_TEST_INNER_LR
                )

                # ---------- Adapt on support ----------
                for step in range(config.META_TEST_FINETUNE_STEPS):
                    support_loss_task = torch.tensor(0.0, device=device)
                    valid_sample = False

                    for s_path, s_target in zip(s_paths, s_targets):
                        image = Image.open(s_path).convert("RGB")
                        support_image = image_transforms(image).unsqueeze(0).to(device)

                        boxes, labels = s_target["boxes"], s_target["labels"]
                        if boxes.numel() == 0:
                            continue

                        img_w, img_h = image.size
                        boxes_xywh = torch.zeros_like(boxes)
                        boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
                        boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
                        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
                        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

                        boxes_xywh[:, [0, 2]] *= 512 / img_w
                        boxes_xywh[:, [1, 3]] *= 512 / img_h

                        batch_idx = torch.zeros((boxes_xywh.size(0), 1), device=device)
                        target = torch.cat(
                            [batch_idx, labels.unsqueeze(1).float().to(device), boxes_xywh.to(device)],
                            dim=1
                        )

                        preds = test_model(support_image)
                        preds = safe_unwrap(preds)
                        pred_input = convert_yolonas_to_ppyolo(preds, config.META_TEST_N_WAY)

                        loss, _ = criterion(pred_input, target)
                        loss = ensure_differentiable_loss(loss, preds, device)

                        support_loss_task += loss
                        valid_sample = True

                    if valid_sample:
                        optimizer.zero_grad()
                        support_loss_task.backward()
                        optimizer.step()

                # ---------- Evaluate on query ----------
                test_model.train()
                test_model.apply(freeze_bn)

                with torch.no_grad():
                    for q_path, q_target in zip(q_paths, q_targets):
                        image = Image.open(q_path).convert("RGB")
                        query_image = image_transforms(image).unsqueeze(0).to(device)

                        preds = test_model(query_image)
                        preds = safe_unwrap(preds)
                        pred_input = convert_yolonas_to_ppyolo(preds, config.META_TEST_N_WAY)

                        decoded = decode_yolonas_outputs(
                            pred_input[0], pred_input[1],
                            pred_input[3], pred_input[5],
                            score_thresh=METRIC_SCORE_THRESH
                        )

                        # -------------------------
                        # META-TEST VISUALIZATION
                        # -------------------------
                        if SAVE_META_TEST_VIS and episode_idx < META_TEST_VIS_EPISODES:
                            import cv2
                            import numpy as np

                            ep_dir = os.path.join(
                                META_TEST_VIS_ROOT,
                                f"epoch_{epoch+1:03d}",
                                f"episode_{episode_idx:03d}"
                            )
                            os.makedirs(ep_dir, exist_ok=True)

                            img = cv2.imread(q_path)
                            h, w = img.shape[:2]
                            sx, sy = w / 512.0, h / 512.0

                            dets = (
                                decoded[0].detach().cpu().numpy()
                                if len(decoded) > 0 and decoded[0].numel() > 0
                                else np.zeros((0, 6), dtype=np.float32)
                            )

                            MAX_VIS_BOXES = 10

                            if dets.shape[0] > MAX_VIS_BOXES:
                                scores = dets[:, 4]
                                topk_idx = np.argsort(scores)[-MAX_VIS_BOXES:]
                                dets = dets[topk_idx]



                            for x1, y1, x2, y2, score, cls in dets:
                                # ---- META-TEST VISUALIZATION FILTER (ONLY) ----
                                #if score < META_TEST_VIS_SCORE_THRESH:
                                    #continue    

                                x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                                # episodic -> global -> readable class
                                global_cls = episode_classes[int(cls)]
                                class_name = config.INDEX_TO_CLASS.get(global_cls, f"id:{global_cls}")
                                label_text = f"epi_{int(cls)} {score:.2f}"

                                cv2.putText(
                                    img,
                                    f"{class_name} {score:.2f}",
                                    (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    1,
                                    cv2.LINE_AA
                                )

                            out_path = os.path.join(ep_dir, os.path.basename(q_path))
                            cv2.imwrite(out_path, img)


                        if len(decoded) == 0 or decoded[0].numel() == 0:
                            preds_for_metric = [{
                                "boxes": torch.zeros((0, 4)),
                                "scores": torch.tensor([]),
                                "labels": torch.tensor([]),
                            }]
                        else:
                            preds_for_metric = [{
                                "boxes": decoded[0][:, :4].detach().cpu(),
                                "scores": decoded[0][:, 4].detach().cpu(),
                                "labels": decoded[0][:, 5].long().detach().cpu(),
                            }]

                        # --- FIX: scale GT boxes to 512x512 before IoU ---
                        gt_boxes = q_target["boxes"].to(device)

                        img_w, img_h = image.size
                        IN_W, IN_H = 512, 512

                        gt_xywh = torch.zeros_like(gt_boxes)
                        gt_xywh[:, 0] = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
                        gt_xywh[:, 1] = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
                        gt_xywh[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
                        gt_xywh[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]

                        # 🔴 THIS WAS MISSING
                        gt_xywh[:, [0, 2]] *= IN_W / float(img_w)
                        gt_xywh[:, [1, 3]] *= IN_H / float(img_h)

                        gt_xyxy = torch.stack(
                            [
                                gt_xywh[:, 0] - gt_xywh[:, 2] / 2,
                                gt_xywh[:, 1] - gt_xywh[:, 3] / 2,
                                gt_xywh[:, 0] + gt_xywh[:, 2] / 2,
                                gt_xywh[:, 1] + gt_xywh[:, 3] / 2,
                            ],
                            dim=1
                        )

                        assert gt_xyxy.max() <= 520, "GT boxes not in 512x512 model space"

                        from torchvision.ops import box_iou
                        iou = box_iou(decoded[0][:, :4], gt_xyxy)
                        best_iou, _ = iou.max(dim=0)

                        if test_metrics is not None:
                            if decoded and decoded[0].numel() > 0:
                                preds_for_metric = [{
                                    "boxes": decoded[0][:, :4].detach().cpu(),
                                    "scores": decoded[0][:, 4].detach().cpu(),
                                    "labels": decoded[0][:, 5].long().detach().cpu(),
                                }]
                            else:
                                preds_for_metric = [{
                                    "boxes": torch.zeros((0, 4)),
                                    "scores": torch.tensor([]),
                                    "labels": torch.tensor([]),
                                }]

                            targets_for_metric = [{
                                "boxes": gt_xyxy.detach().cpu(),
                                "labels": q_target["labels"].detach().cpu(),
                            }]

                            test_metrics.update(preds_for_metric, targets_for_metric)


                        hits = (best_iou >= 0.5).sum().item()
                        test_recall_hits_05 += hits
                        test_total_gt_05 += best_iou.numel()

                        if hits > 0:
                            episode_success_05 = True

                test_episode_recall_05_list.append(
                    test_recall_hits_05 / max(test_total_gt_05, 1)
                )
                test_episode_esr_05_list.append(float(episode_success_05))

                del test_model
                torch.cuda.empty_cache()

            except Exception as e:
                logging.info(f"[Meta-Test Episode {episode_idx}] Error: {e}")
                continue

        mean_test_recall_05 = sum(test_episode_recall_05_list) / max(len(test_episode_recall_05_list), 1)
        mean_test_esr_05 = sum(test_episode_esr_05_list) / max(len(test_episode_esr_05_list), 1)

        if hasattr(test_metrics, "_update_called") and test_metrics._update_called:
            results = test_metrics.compute()
            test_map_50 = results["map_50"].item()
        else:
            test_map_50 = -1.0


        print(
            f"[META-TEST] mAP@0.5={test_map_50:.4f} | "
            f"Recall@0.5={mean_test_recall_05:.4f} | "
            f"ESR@0.5={mean_test_esr_05:.4f}"
        )
            

        # --- <<< NEW: Log results to file >>> ---
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = (
                f"{timestamp},"
                f"{epoch + 1},"
                f"{train_mean_box:.6f},"
                f"{train_mean_obj:.6f},"
                f"{train_mean_cls:.6f},"
                f"{train_mean_iou:.6f},"
                f"{mean_recall_05:.6f},"
                f"{mean_esr_05:.6f},"
                f"{mean_val_loss:.6f},"
                f"{val_mean_box:.6f},"
                f"{val_mean_obj:.6f},"
                f"{val_mean_cls:.6f},"
                f"{map_50_val:.6f},"
                f"{mean_val_recall_05:.6f},"
                f"{mean_val_esr_05:.6f}"
                f",{test_map_50:.6f},"
                f"{mean_test_recall_05:.6f},"
                f"{mean_test_esr_05:.6f}"

            )

            results_logger.info(log_line)
        except Exception as e:
            print(f"[ERROR] Failed to write to results log: {e}")
        # --- <<< End of logging >>> ---

        # -------------------------
        # UPDATED EARLY STOPPING LOGIC (MAML-FSOD CORRECT)
        # -------------------------

        map_improved = map_50_val > (best_map_50 + MAP_EPS)
        loss_improved = mean_val_loss < (best_val_loss - LOSS_EPS)
        esr_improved = mean_val_esr_05 > (best_val_esr + ESR_EPS)
        recall_improved = mean_val_recall_05 > (best_val_recall + RECALL_EPS)

        is_best = False

        # ---- Primary improvement gate ----
        if map_improved or (map_50_val < 0 and loss_improved):
            is_best = True

        # ---- Update best trackers ----
        if map_improved:
            best_map_50 = map_50_val
        if loss_improved:
            best_val_loss = mean_val_loss
        if esr_improved:
            best_val_esr = mean_val_esr_05
        if recall_improved:
            best_val_recall = mean_val_recall_05

        # ---- Save best checkpoint ----
        if is_best:
            patience_counter = 0
            print(
                f"⭐ Improvement | "
                f"mAP@0.5={map_50_val:.4f}, "
                f"Loss={mean_val_loss:.4f}, "
                f"ESR={mean_val_esr_05:.4f}, "
                f"Recall={mean_val_recall_05:.4f}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": meta_optimizer.state_dict(),
                    "best_map_50": best_map_50,
                    "best_val_loss": best_val_loss,
                    "best_val_esr": best_val_esr,
                    "best_val_recall": best_val_recall,
                },
                BEST_CHECKPOINT_PATH,
            )

        else:
            patience_counter += 1
            print(
                f"⏳ No improvement | "
                f"patience {patience_counter}/{EARLY_STOPPING_PATIENCE} | "
                f"mAP={map_50_val:.4f}, "
                f"Loss={mean_val_loss:.4f}, "
                f"ESR={mean_val_esr_05:.4f}, "
                f"Recall={mean_val_recall_05:.4f}"
            )

            # ---- TRUE EARLY STOP CONDITION ----
            if (
                patience_counter >= EARLY_STOPPING_PATIENCE
                and not esr_improved
                and not recall_improved
            ):
                print(
                    f"[EarlyStop Check] "
                    f"map_improved={map_improved}, "
                    f"loss_improved={loss_improved}, "
                    f"esr_improved={esr_improved}, "
                    f"recall_improved={recall_improved}"
                )

                print("\n🛑 EARLY STOPPING TRIGGERED")
                print(f"   Best mAP@0.5 : {best_map_50:.4f}")
                print(f"   Best ValLoss: {best_val_loss:.4f}")
                print(f"   Best ESR@0.5: {best_val_esr:.4f}")
                print(f"   Best Recall : {best_val_recall:.4f}")
                break
        

        # checkpoint
        try:
            save_state = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": meta_optimizer.state_dict(),
            }
            torch.save(save_state, CHECKPOINT_PATH)
            print(f"--- Checkpoint saved at Epoch {epoch + 1} ---")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")

    # final save
    try:
        torch.save(base_model.state_dict(), "maml_yolo_nas_final.pth")
        
        print("✅ Meta-Training Complete! Final model saved to maml_yolo_nas_final_dp.pth")
    except Exception as e:
        print(f"[ERROR] Failed to save final model: {e}")


if __name__ == "__main__":
    main()