
# ============================================================
# config_sanity_test.py — Quick verification setup
# ============================================================

# -------------------
# DATASET PATHS
# -------------------
#BASE_CLASSES_PATH = "/mnt/d/fsod_project/fsod_dota/base_classes"
BASE_CLASSES_PATH = r"D:\fsod_project\fsod_dota\base_classes"
#NOVEL_CLASSES_PATH = r"D:\fsod_project\fsod_dota\novel_classes"  # optional for meta-test
#BEST_CHECKPOINT_PATH = r"D:\fsod_project\best_checkpoint_dp.pth"

# -------------------
# CLASS MAPPING
# -------------------
CLASS_MAPPING = {
    'people': 0,
    'bicycle': 1,
    'car': 2,
    'van': 3,
    'truck': 4,
    'tricycle': 5,
    'awning-tricycle': 6,
    'bus': 7,
    'motor': 8,

    #Novel Class
    #'people': 0,
    #'van': 3,
    #'awning-tricycle': 6,    
}
INDEX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
#'tricycle': 5,
#'bus': 7,
#'motor': 8,
# -------------------
# MODEL & CHECKPOINT
# -------------------
MODEL_ARCH = "yolo_nas_l"
#PRETRAINED_CHECKPOINT_PATH = "/mnt/d/fsod_project/yolo_nas_l.pth"
PRETRAINED_CHECKPOINT_PATH = r"D:\fsod_project\best_visdrone_nas_36.pth"
PRETRAINED_NUM_CLASSES = 12


# -------------------
# FEW-SHOT SETUP
# -------------------
N_WAY = 2       # 2-way task for sanity test
K_SHOT = 1      # 1 support image per class
Q_QUERY = 2     # 1 query image per class
TASKS_PER_EPOCH = 100 #100change  # only 2 tasks per epoch to test flow


# -------------------
# META-LEARNING HYPERPARAMETERS
# -------------------
META_BATCH_SIZE = 1     # one task at a time
META_LR = 1e-4
INNER_LR = 0.005 #0.01change
INNER_UPDATE_STEPS = 5 #3change  # just 1 gradient step in inner loop
NUM_EPOCHS = 100         # single epoch sanity run
SAVE_INTERVAL = 10       # save checkpoint after one run


# -------- META TEST --------
META_TEST_INNER_LR = 0.01
META_TEST_N_WAY = 2
META_TEST_K_SHOT = 1
META_TEST_Q_QUERY = 2
META_TEST_FINETUNE_STEPS = 1 #10
NUM_META_TEST_EPISODES = 2 #15

NOVEL_CLASSES_PATH = r"D:\fsod_project\fsod_dota\novel_classes"
BEST_CHECKPOINT_PATH = r"D:\fsod_project\best_checkpoint_dp.pth"


# -------------------
# EVALUATION / VALIDATION
# -------------------
EVAL_IOU_THRESHOLD = 0.5
VALIDATION_TASK_COUNT = 3 #30   # only 1 episode for validation
FINETUNE_STEPS = 4        # minimal fine-tuning for test


# -------------------
# PROTOTYPE
# -------------------
FEATURE_DIM = 256
PROTO_TEMPERATURE = 10.0

# -------------------
# OTHER CONSTANTS
# -------------------
SEED = 42
MIN_NORMALIZED_DIMENSION = 0.005
