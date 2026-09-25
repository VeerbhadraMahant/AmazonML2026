# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + LightGBM pairwise matcher for resolving Source-1 business records
against Source-2/3 records across US, India, and (test-only, zero-shot)
France. See `../../METHODOLOGY.md` (repo root) for the full write-up and
`../../AmazonML/student_resource/Documentation_template.md` for the
challenge's own submission template.

## Pipeline

```
raw TSV -> normalize -> embed (GPU) -> candidate generation (blocking)
        -> pairwise features -> LightGBM matcher -> threshold decoding
        -> matching_results.tsv + candidate_pairs.tsv
```

| Stage | Script(s) | What it does |
|---|---|---|
| 0. Normalize | `normalize.py` | Transliteration, legal-suffix/honorific stripping, address abbreviation expansion, number extraction. Cached to `work/norm_*.parquet`. |
| 1. Blocking | `embed.py`, `retrieve.py`, `run_retrieve.py` | GPU embedding search (`multilingual-e5-small`) + two lexical safety-net joins, per country. Writes `work/candidates_{split}.parquet` and `work/candidate_pairs_{split}.tsv`. |
| 2. Features | `features.py`, `run_features.py` | 16 pairwise similarity/context features, computed out-of-core in batches (rapidfuzz, parallelized). Writes `work/pairs_features_{split}.parquet`. |
| 2b. Train | `split_and_sample.py`, `train_matcher.py` | 90/10 split by S1 entity, LightGBM binary classifier. Writes `work/lgbm_matcher.txt`. |
| 3. Decode | `decode.py`, `evaluate.py`, `score_matcher.py`, `run_test_finish.py` | Exclusive per-entity assignment above a tuned threshold; threshold was chosen by sweeping macro F0.5 on the held-out validation split. |

`config.py` holds all paths/constants; `io_utils.py` has the tab-separated
read/write helpers.

## Setup

```bash
pip install -r requirements.txt
```

`torch` needs a CUDA-enabled wheel to use the GPU (see the comment in
`requirements.txt`); a CPU-only install also works, just much slower for
this dataset's scale (~24M records total).

Place the challenge data so that, relative to the repo root:
```
AmazonML/student_resource/dataset/train/train_source{1,2,3}.tsv
AmazonML/student_resource/dataset/train/train_ground_truth.tsv
AmazonML/student_resource/dataset/test/test_source{1,2,3}.tsv
```
(`config.py` derives these paths automatically from its own location.)

## Reproducing the submission from scratch

Run each step as its own process (not chained in one script/notebook) --
this matters on machines with limited RAM/VRAM, since each stage's memory is
released when its process exits before the next stage starts. All commands
below are run from this directory (`code/business_entity_resolution/src/`).

```bash
# 1. Blocking (normalizes + embeds + generates candidates) for train, then test
python run_retrieve.py --split train
python run_retrieve.py --split test

# 2. Pairwise features for train, then test
python run_features.py --split train --n-workers 12
python run_features.py --split test  --n-workers 12

# 3. Split train into a training sample + a full held-out validation set
python split_and_sample.py

# 4. Train the LightGBM matcher, evaluate on validation (AUC/AP)
python train_matcher.py

# 5. Sweep the decoding threshold against the real competition metric
#    (macro F0.5, not AUC) and report per-country breakdown
python evaluate.py

# 6. Score the test candidates with the trained matcher, decode, and write
#    the two submission files into ../../../output/
python run_test_finish.py
```

Each script prints its own timing and is safe to re-run: normalization,
embeddings, and candidate generation are all cached under `../../../work/`
and skipped if already present.

Expected wall time on a machine with an 8GB-VRAM GPU, 20 CPU threads, 24GB
RAM (what this was developed/run on): roughly 45-60 minutes total end to end
for both train and test splits combined (dominated by GPU embedding of
~24M records and the two ~120M-row feature-computation passes).

## Design notes

- **Country is used only for blocking, never as a model feature** -- France
  appears only in the test set, so training a model that depends on country
  as a feature would make it unreliable exactly where it can't be checked.
  Blocking recall and macro F0.5 for France should be judged from the
  leaderboard, since there's no France data in training to validate against
  locally.
- **Each S2/S3 record is assigned to at most one S1 entity** (verified in
  training ground truth: zero records matched more than one S1), so
  decoding does an exclusive per-record argmax rather than independent
  per-pair thresholding.
- **The final model is LightGBM** (gradient-boosted trees, MIT-licensed,
  nowhere near the 8B-parameter cap) plus `multilingual-e5-small` (118M
  params, MIT-licensed) for retrieval embeddings -- comfortably inside the
  challenge's model-license/size constraints.
- No external data, APIs, or lookups are used anywhere in the pipeline --
  only the provided train/test files and fixed, in-code normalization
  rules.

## Validated results (held-out 10% of train, by S1 entity)

- Blocking recall: 97.4% overall (US 98.6%, India 95.5%)
- Matcher: validation AUC 0.9989, AP 0.9868
- **Macro F0.5 (the competition metric) at the chosen threshold (tau=0.91): 0.9359** (US 0.9483, India 0.9167)
