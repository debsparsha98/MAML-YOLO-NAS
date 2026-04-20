import os
from PIL import Image
import torch
from datetime import datetime

# ----------------------------- #
#   Helper: Logger setup
# ----------------------------- #
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "malformed_annotations.txt")
os.makedirs(LOG_DIR, exist_ok=True)

def log_warning(message):
    """Append warning message with timestamp to log file."""
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


# ----------------------------- #
#   Dataset Info Builder
# ----------------------------- #
def build_dataset_info(classes_path):
    """
    Builds a mapping: {class_idx: [image_path1, image_path2, ...]}.
    Automatically detects flat or nested annotation layout.
    Logs malformed or missing annotation files.
    """
    dataset_info = {}

    for class_folder in sorted(os.listdir(classes_path)):
        class_dir = os.path.join(classes_path, class_folder)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(class_dir, file)

                # Support both flat and nested 'labels/' structure
                label_path_flat = os.path.splitext(image_path)[0] + ".txt"
                label_path_nested = os.path.join(class_dir, "labels", os.path.splitext(file)[0] + ".txt")

                if os.path.exists(label_path_flat):
                    ann_path = label_path_flat
                elif os.path.exists(label_path_nested):
                    ann_path = label_path_nested
                else:
                    log_warning(f"[Missing Annotation] {label_path_flat}")
                    continue

                try:
                    classes_in_image = set()   # <-- collect unique class IDs FOR THIS IMAGE

                    with open(ann_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                log_warning(f"[Malformed Line] {ann_path} -> '{line.strip()}'")
                                continue

                            try:
                                cls_idx = int(float(parts[0]))
                                classes_in_image.add(cls_idx)
                            except Exception:
                                log_warning(f"[Invalid Class] {ann_path} -> '{line.strip()}'")
                                continue

                    # Add image_path ONCE per class in this image
                    for cls_idx in classes_in_image:
                        dataset_info.setdefault(cls_idx, []).append(image_path)

                except Exception as e:
                    log_warning(f"[Error Parsing] {ann_path}: {e}")


    return dataset_info


# ----------------------------- #
#   Annotation Parser
# ----------------------------- #
def get_annotations_for_image(image_path):
    """
    Reads YOLO-format annotations and converts them into absolute pixel boxes.
    Returns {'boxes': Tensor[N,4], 'labels': Tensor[N]}.
    Logs malformed lines or missing annotations.
    """
    # Handle both flat and nested structures
    base_dir = os.path.dirname(image_path)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    ann_path_flat = os.path.join(base_dir, filename + ".txt")
    ann_path_nested = os.path.join(base_dir, "labels", filename + ".txt")

    if os.path.exists(ann_path_flat):
        ann_path = ann_path_flat
    elif os.path.exists(ann_path_nested):
        ann_path = ann_path_nested
    else:
        log_warning(f"[Missing Annotation] {ann_path_flat}")
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.long)
        }

    boxes, labels = [], []

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        log_warning(f"[Image Open Error] {image_path}: {e}")
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.long)
        }

    malformed_count = 0
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                malformed_count += 1
                log_warning(f"[Malformed Line] {ann_path} -> '{line.strip()}'")
                continue

            try:
                cls, cx, cy, w, h = map(float, parts)
                cls = int(cls)
            except ValueError:
                malformed_count += 1
                log_warning(f"[Non-numeric Values] {ann_path} -> '{line.strip()}'")
                continue

            # Convert normalized xywh → absolute xyxy
            xmin = (cx - w / 2.0) * img_w
            ymin = (cy - h / 2.0) * img_h
            xmax = (cx + w / 2.0) * img_w
            ymax = (cy + h / 2.0) * img_h

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls)

    if malformed_count > 0:
        print(f"[Warning] {malformed_count} malformed lines skipped in {os.path.basename(ann_path)}")

    boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

    return {'boxes': boxes, 'labels': labels, 'img_size': (img_w, img_h)}
