# Business Entity Resolution — Amazon ML Challenge 2026

This pipeline matches each Source-1 business record to its Source-2/3 records for the US, India and
France. France appears only in the test set and is handled zero-shot. The pipeline blocks candidates,
scores them with a 25-feature LightGBM matcher, re-scores them with a Stage-2 LightGBM stacker, and
decodes them with exclusive assignment. The methodology write-up is `Documentation_template.md` at the
zip root.

## Pipeline

```
raw TSV -> normalize -> e5 embeddings (GPU) -> blocking (kNN top-10 per country + 2 lexical joins)
        -> 16 pairwise features -> +9 context features -> Stage-1 LightGBM
        -> Stage-2 LightGBM stacker -> exclusive-assignment decode
        -> output/matching_results.tsv + output/candidate_pairs.tsv
```

| Stage | Script(s) | Output (under `<repo>/work/` unless noted) |
|---|---|---|
| 0. Normalize | `normalize.py` (called by `run_retrieve.py`) | `norm_{train,test}_s{1,2,3}.parquet` |
| 1. Blocking | `embed.py`, `retrieve.py`, `run_retrieve.py` | `emb_*.npy`, `candidates_{split}.parquet`, `candidate_pairs_{split}.tsv` |
| 2. Pairwise features | `features.py`, `run_features.py` | `candidates_augmented_{split}.parquet`, `pairs_features_{split}.parquet` |
| 2b. Context features | `add_context_features.py` | `pairs_features_{split}_ctx.parquet` |
| 3. Split | `split_and_sample.py` | `train_sample_ctx.parquet`, `val_full_ctx.parquet` |
| 4. Stage-1 matcher | `train_matcher.py` | `lgbm_matcher_ctx.txt`, `val_scored_ctx.parquet` |
| 5. Threshold sweep | `evaluate_fine.py` (uses `decode.py`, `evaluate.py`) | printed macro F0.5 per tau and per country |
| 6. Score test | `score_test.py` (uses `score_matcher.py`) | `test_scored_ctx.parquet` |
| 7. Stage-2 and decode | `stack_stage2.py` | `<repo>/output/matching_results.tsv` |
| 8. Candidate file | copy | `<repo>/output/candidate_pairs.tsv` |

`config.py` holds all paths and constants. `io_utils.py` has the TSV read/write helpers.

## Setup

We developed and ran the pipeline with Python 3.13.7 on Windows 11, using an 8 GB-VRAM NVIDIA GPU,
20 CPU threads and 24 GB RAM.

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA build of torch
pip install -r requirements.txt
```

A CPU-only torch also works (`embed.py` falls back to CPU), but embedding about 24M records will be
much slower. The first run downloads `intfloat/multilingual-e5-small` (MIT, 118M params) from the
Hugging Face Hub.

### Data layout

`config.py` finds the repository root as three levels above `src/`
(`src -> business_entity_resolution -> code -> <repo>`). It expects the following layout:

```
<repo>/AmazonML/student_resource/dataset/train/train_source1.tsv
<repo>/AmazonML/student_resource/dataset/train/train_source2.tsv
<repo>/AmazonML/student_resource/dataset/train/train_source3.tsv
<repo>/AmazonML/student_resource/dataset/train/train_ground_truth.tsv
<repo>/AmazonML/student_resource/dataset/test/test_source1.tsv
<repo>/AmazonML/student_resource/dataset/test/test_source2.tsv
<repo>/AmazonML/student_resource/dataset/test/test_source3.tsv
<repo>/work/        # intermediate artefacts (created automatically)
<repo>/output/      # final submission files (created automatically)
```

If you unpack the zip somewhere else, put the dataset at
`<unzip_root>/AmazonML/student_resource/dataset/...`, or edit `DATA_DIR` in `config.py`.

## Reproducing the submission end to end

Run every command from `code/business_entity_resolution/src/`. Run each step as a separate process, in
this order, so each stage's RAM/VRAM is freed before the next one starts; at 120M-row scale this
matters. Normalisation, embeddings and candidates are cached under `work/`, so re-running a step skips
work that is already done.

```bash
cd code/business_entity_resolution/src

# 1. Normalize + embed + block (writes candidates and candidate_pairs_{split}.tsv)
python run_retrieve.py --split train
python run_retrieve.py --split test

# 2. 16 pairwise features (train also joins ground-truth labels)
python run_features.py --split train --n-workers 12
python run_features.py --split test  --n-workers 12

# 3. 9 context/competition features -> pairs_features_{split}_ctx.parquet
python add_context_features.py train
python add_context_features.py test

# 4. 90/10 split by S1 entity (seed 42): train sample + full held-out validation
python split_and_sample.py ctx

# 5. Train Stage-1 LightGBM (25 features); prints val AUC/AP and writes val_scored_ctx.parquet
python train_matcher.py ctx

# 6. Sweep tau against macro F0.5 on validation (overall + per country)
python evaluate_fine.py ctx

# 7. Score the 120.8M test pairs with Stage 1 -> test_scored_ctx.parquet
python score_test.py ctx

# 8. Out-of-fold Stage-1 preds for ALL train pairs (2 folds by S1 id) + test preds
#    (average of the two fold models) -> work/oof_train_scored.parquet, work/test_scored_oof.parquet
python oof_stage1.py

# 9. Stage-2 stacker, trained on OOF preds with full competitor context, in the
#    "S1-dropped" (distractor-heavy, test-like) regime. GPU XGBoost (RTX 4060, CUDA):
python stack_stage2.py train_xgb 0.35 2000                  # -> work/xgb_stage2.json + stage2_ens_config.json
#    (LightGBM alternative, CPU: python stack_stage2.py train_drop 0.21 400 drop)

# 10. Apply Stage 2 to test + expected-F0.5 decoder -> output/matching_results.tsv
python stack_stage2.py train_xgb 0.45 2000                  # 2nd and 3rd GPU models (different seed / S1 sample) -> xgb_stage2_b.json, xgb_stage2_c.json
python stack_stage2.py apply_test ../../../output/matching_results.tsv test_scored_oof.parquet ens3   # average of the two

# 11. The candidate set fed to the matcher -> output/candidate_pairs.tsv
python -c "import shutil, config; shutil.copyfile(config.WORK_DIR/'candidate_pairs_test.tsv', config.OUTPUT_DIR/'candidate_pairs.tsv')"
```

**Stage-1-only fallback:** this replaces steps 7–9 without the stacker. It scores, decodes at the
given tau, writes `matching_results.tsv` and copies `candidate_pairs.tsv`:

```bash
python run_test_finish.py ctx 0.91
# or, if test_scored_ctx.parquet already exists:
python decode_test.py ctx 0.91 ../../../output/matching_results.tsv
```

**Validate the output format** with the challenge-provided script (its path is relative to the repo
root):

```bash
python AmazonML/student_resource/utils/validate_submission.py \
    -m output/matching_results.tsv -c output/candidate_pairs.tsv \
    -t AmazonML/student_resource/dataset/test --check-ids
```

### Expected wall time

End to end takes about 1.5–2 hours on the machine above. Most of the time goes to three steps:
- GPU embedding of about 24M records;
- the two passes that compute features for about 120M rows each;
- batched scoring.

Peak RAM stays under 24 GB because each stage streams parquet in batches.

## Design notes

- **Country is used for blocking only, never as a feature.** France is test-only, so a country feature
  would be unreliable exactly where it cannot be validated.
- **Exclusive assignment.** Each Source-2/3 record is assigned to at most one Source-1 entity. Training
  ground truth has zero records matched to more than one Source-1 entity.
- **`candidate_pairs.tsv` is the exact set the matcher runs inference on.** There is no pruning
  between blocking and scoring, so every id in `matching_results.tsv` also appears in it.
- **Compliance:**
  - Models: `multilingual-e5-small` (MIT, 118M params) and LightGBM (MIT, trees).
  - No external data, APIs or lookups: the only inputs are the provided train/test files and fixed
    normalisation rules in code.

## Validated results (held-out 10% of train S1 entities)

| | Val macro F0.5 | US | India | LB |
|---|---|---|---|---|
| Baseline (16 features) | 0.9359 | 0.948 | 0.917 | 0.922 |
| Stage-1 context matcher (25 features, tau=0.91; AUC 0.9995, AP 0.9928) | 0.9461 | 0.955 | 0.933 | |
| **Final (Stage 1 + Stage 2)** | **0.9631** | | | **see portal (single-XGBoost version: 0.938)** |

Blocking pair recall is 97.4% (US 98.6%, India 95.5%).
