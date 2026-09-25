"""Central paths and constants for the entity-resolution pipeline."""
from pathlib import Path

# Repo root = three levels up from this file (src/ -> business_entity_resolution/ -> code/ -> root)
ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "AmazonML" / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

WORK_DIR = ROOT / "work"
OUTPUT_DIR = ROOT / "output"
SUBMISSIONS_DIR = ROOT / "submissions"

for d in (WORK_DIR, OUTPUT_DIR, SUBMISSIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VAL_FRACTION = 0.10  # fraction of train S1 entities held out for local validation

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"  # MIT license, 118M params
EMBED_DIM = 384
MAX_SEQ_LEN = 64

TOP_K_CANDIDATES = 10  # per S2/S3 query record, before lexical union
