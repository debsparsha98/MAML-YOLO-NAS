import torch
import torch.nn as nn
from super_gradients.training import models


class MAML_YOLO_NAS(nn.Module):
    """
    A wrapper around YOLO-NAS for use in MAML-based Few-Shot Object Detection.
    """

    def __init__(self, model_arch: str, num_classes: int, checkpoint_path: str, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        print(f"Initializing {model_arch} with {num_classes} classes...")

        # 1. Load YOLO-NAS architecture
        self.yolo_nas = models.get(model_arch, num_classes=num_classes)

        # 2. Load checkpoint safely (local weights or SG training checkpoint)
        if checkpoint_path is not None:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Handle both raw state_dict or {'net': state_dict}
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                checkpoint_state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "net" in checkpoint:
                checkpoint_state_dict = checkpoint["net"]
            elif isinstance(checkpoint, dict):
                checkpoint_state_dict = checkpoint
            else:
                raise ValueError("Unsupported checkpoint format")

            # --------------------------------------------------
            # Handle DataParallel checkpoints
            # --------------------------------------------------
            if any(k.startswith("module.") for k in checkpoint_state_dict.keys()):
                checkpoint_state_dict = {
                    k.replace("module.", ""): v
                    for k, v in checkpoint_state_dict.items()
                }

            model_state_dict = self.yolo_nas.state_dict()

            # Filter to keep only matching keys
            filtered_state_dict = {
                k: v for k, v in checkpoint_state_dict.items()
                if k in model_state_dict and model_state_dict[k].shape == v.shape
            }

            missing = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
            if self.verbose:
                print(f"Loaded {len(filtered_state_dict)} / {len(model_state_dict)} layers from checkpoint.")
                print(f"Missing layers (new head or mismatched): {len(missing)}")

            self.yolo_nas.load_state_dict(filtered_state_dict, strict=False)
            print("Checkpoint loaded successfully with a new head.")
        else:
            print("No checkpoint loaded (initializing from scratch).")    

        # 3. Split model into backbone (outer-loop) and head (inner-loop)
        self.feature_extractor = nn.Sequential(
            self.yolo_nas.backbone,
            self.yolo_nas.neck
        )
        self.adaptation_head = self.yolo_nas.heads
        print("Model separated into feature extractor and adaptation head.")

    def forward(self, x):
        """
        Forward pass through YOLO-NAS.
        Returns full structured output tuple for flexible downstream use.
        """
        predictions = self.yolo_nas(x)
        
        # [CLEANUP] Commented out verbose debug prints
        # if self.verbose and isinstance(predictions, (tuple, list)):
        #     print(f"Model returned a tuple/list of length: {len(predictions)}")
        #     for i, p in enumerate(predictions):
        #         print(f"  Element {i}: Type={type(p)}")
        return predictions

    def get_inner_loop_params(self):
        """Return the adaptation head parameters (for MAML inner loop)."""
        return self.adaptation_head.parameters()

    def get_outer_loop_params(self):
        """Return the feature extractor parameters (for MAML outer loop)."""
        return self.feature_extractor.parameters()


def unwrap_predictions(preds):
    """
    Normalize YOLO-NAS / SuperGradients model outputs into a consistent tuple
    for PPYoloELoss. Handles multiple nested tuple structures.
    """
    while isinstance(preds, (list, tuple)) and len(preds) == 1:
        preds = preds[0]

    if isinstance(preds, (list, tuple)) and len(preds) == 2:
        if isinstance(preds[0], (list, tuple)):
            preds = preds[0]

    if isinstance(preds, dict):
        preds = tuple(preds.values())

    if not isinstance(preds, (tuple, list)):
        preds = (preds,)

    if len(preds) == 4:
        preds = tuple(preds) + (torch.empty(0), torch.empty(0))

    return preds
   