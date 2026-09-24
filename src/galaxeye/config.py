"""Central configuration constants for the GalaxEye take-home service."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Dataset locations (read-only input data)
DATASET_ROOT = PROJECT_ROOT / "be-mlsys-assignment-dataset"
CANDIDATE_TILES_DIR = DATASET_ROOT / "candidate_tiles"
EVAL_SET_DIR = DATASET_ROOT / "eval_set"
EVAL_LABELS_CSV = DATASET_ROOT / "eval_labels.csv"

# Image preprocessing (ImageNet stats, matching the frozen ResNet backbone)
IMAGE_SIZE = 64
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Model artifact (not downloaded at runtime; must be vendored)
MODEL_VERSION = "resnet18-7cls-v0.1.0"
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.pt"

# Classifier policy: flag a prediction LOW_CONFIDENCE when confidence < threshold.
# This is a configurable policy, NOT a calibrated probability.
CONFIDENCE_THRESHOLD = 0.8

# Storage
DB_PATH = PROJECT_ROOT / "data" / "galaxeye.db"
