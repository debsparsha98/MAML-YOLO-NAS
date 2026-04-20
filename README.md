# MAML-YOLO-NAS
MAML-YOLO-NAS training pipeline
# MAML-YOLO-NAS: Few-Shot Object Detection

## ⚠️ Important Note

This repository provides the **original research implementation** used in the paper.

Due to experimental design, some paths and configurations are defined inside the code.
Please follow the setup instructions carefully to reproduce results.

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

Update dataset paths inside:

```id="readme1"
your_original_code/config.py
```

Set:

```id="readme2"
BASE_CLASSES_PATH = "path/to/base_classes"
NOVEL_CLASSES_PATH = "path/to/novel_classes"
```

---

## 🧠 Pretrained Weights

Update:

```id="readme3"
PRETRAINED_CHECKPOINT_PATH = "path/to/yolo_nas_weights.pth"
```

---

## 🏋️ Training

```bash
bash run/train.sh
```

---

## 🧪 Meta Testing

```bash
bash run/meta_test.sh
```

---

## 📌 Notes

* Code uses episodic meta-learning (MAML)
* Detection backbone: YOLO-NAS
* Hardcoded configurations are kept for consistency with experiments

---
