# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Verbatim
**Team Members:** Veerbhadra Mahant, Meet Ramjiyani, Hari Chaudhari, Nirbhay Gajabi
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built a blocking-then-scoring entity resolution pipeline with four stages:

1. **Candidate generation.** A GPU embedding search (`intfloat/multilingual-e5-small`, MIT, 118M params)
   finds each record's top-10 Source-1 neighbours within its country, and two lexical exact-key joins
   are added on top as safety nets.
2. **Stage-1 matcher.** A LightGBM pairwise classifier scores every candidate pair on 25 features:
   16 pairwise similarity features and 9 context/competition features.
3. **Stage-2 stacker.** A GPU-trained XGBoost model re-scores every pair using out-of-fold Stage-1
   predictions of *all* competing pairs (per Source-2/3 record and per Source-1 entity), trained in
   a test-like regime with more ownerless records.
4. **Exclusive-assignment + expected-F0.5 decoder.** Each Source-2/3 record is assigned to at most
   one Source-1 entity, and each Source-1 entity keeps the number of matches that maximises its
   expected F0.5.

On a held-out 10% validation split (split by Source-1 entity), the Stage-1 context matcher alone scores
**macro F0.5 = 0.9461** (US 0.955, India 0.933). Our earlier 16-feature baseline scored 0.9359 on the
same split and 0.922 on the leaderboard (the Stage-2 LightGBM version, val 0.957, scored 0.936). The final pipeline with the Stage-2 stacker scores
**macro F0.5 = 0.9631** on validation and **see portal (single-XGBoost version: 0.938; Stage-2 LightGBM version: 0.936)** on the public leaderboard.

The main modelling idea comes from the training data: every Source-2/3 record matches at most one
Source-1 entity. We therefore treat matching as an assignment problem and add features that describe
how each candidate pair competes with the other pairs around it. This targets precision, which F0.5
weights more heavily than recall.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis of the training data (2.2M Source-1, 5.0M Source-2 and 5.3M Source-3 records)
found the following:

- **Scale.** All-pairs comparison is infeasible (2.2M x 10.3M, about 22 trillion pairs). Blocking is
  required.
- **Singletons are rare (5.6%).** Source-1 entities have 3.46 matches on average (maximum 11), so
  predicting "no match" too often is expensive even under a precision-weighted metric.
- **Each Source-2/3 record matches at most one Source-1 entity.** We checked this exactly on
  `train_ground_truth.tsv`: no Source-2/3 id is matched to more than one Source-1 row. This makes the
  task an assignment problem, not independent pairwise classification.
- **Country agrees in 100% of true matches.** Blocking within country is therefore safe. France, which
  appears only in the test set, is handled as just another country value.
- **Exact normalised-name equality holds in only 22% of true pairs** (73% share the first token).
  Blocking must be fuzzy or semantic.
- **Noise patterns:**
  - legal-suffix drift (Pvt/Private, LLC/Limited, French SAS/SARL/EURL/SCI)
  - honorifics added or dropped (Sri/Shri)
  - punctuation junk (`***`, `##`, `[..]`)
  - homoglyph typos (`m0tors`)
  - non-Latin scripts in Source-2/3 (Devanagari, Gujarati, Kannada)
  - truncated house numbers (`5014` vs `501`)
  - reordered or abbreviated addresses (Rd/Road, French R./Rue, Av./Avenue)
- **About 30% of Source-1 names repeat.** For example, "Primary Care Group" appears 253 times, each at
  a different address. Address and house-number evidence separate these records; name similarity
  cannot.
- **Diagnostic.** An oracle classifier over our candidate set scores about 0.99 macro F0.5. The
  remaining error after blocking is therefore mostly a precision/ranking problem inside the candidate
  set. This is why we invested in context features and stacking instead of more retrieval.

### 2.2 Solution Strategy

**Approach Type:** Blocking plus a two-stage classifier.
- Blocking: embedding retrieval and lexical exact-key joins.
- Scoring: a Stage-1 LightGBM pairwise matcher, then a Stage-2 LightGBM stacker.
- Decoding: exclusive assignment.

**Core Innovation:** The pipeline uses the verified "one owner per Source-2/3 record" structure in
three places:
- **Features.** Entity-side competition features describe how a candidate Source-1 compares with the
  *other* Source-1 candidates of the same Source-2/3 record.
- **Stage 2.** The stacker's inputs are Stage-1 predictions placed in context: rank, margin and gap
  among the record's own candidates and among the Source-1 entity's candidates.
- **Decoding.** Each Source-2/3 record goes to its single best Source-1 candidate, or to none if no
  candidate clears the threshold.

---

## 3. Candidate Generation (Blocking)

### 3.0 Normalisation (`src/normalize.py`)

These rules are fixed in code and use no external data:

- **Transliteration.** Non-Latin text is converted to Latin with `anyascii`. Only rows that contain
  non-ASCII characters are processed.
- **Cleaning.** Text is lowercased and punctuation noise is stripped.
- **Abbreviation expansion.** Address abbreviations are expanded, including St→street, Rd→road,
  R.→rue, Av.→avenue, Bd.→boulevard and Ch.→chemin.
- **Core name.** A `name_core` field is derived with English, Indian and French legal suffixes and
  honorifics removed.
- **Numbers.** House numbers and PIN/ZIP codes are extracted into `addr_numbers`.
- **Script flag.** The first name token and a "name was non-ASCII" flag are recorded.
- **Caching.** Output is cached to parquet.

### 3.1 Blocking keys

All three blocking methods are restricted to the same country. Their results are merged (union), and
duplicate pairs are removed, keeping the maximum cosine similarity.

1. **Semantic kNN (`src/embed.py`, `src/retrieve.py`).**
   - Model and input text: `intfloat/multilingual-e5-small` (MIT, 118M params, fp16 on GPU,
     max_seq_len 64) embeds `name_full , address_norm`.
   - Prefixes: Source-1 records use the E5 `passage:` prefix, and Source-2/3 records use `query:`.
   - Search: for each Source-2/3 record we take the top-K = 10 Source-1 records by cosine similarity.
   - GPU sizing: the search is a chunked GPU matmul followed by top-k. The chunk size depends on the
     size of the *passage* partition, so the similarity matrix fits in VRAM for the largest country.
2. **Lexical exact-key join on `(country, name_core)`.** This catches obvious matches. Keys shared by
   more than 40 Source-1 rows are skipped to avoid a blow-up on generic names; those cases are left to
   the embedding and number features.
3. **Lexical exact-key join on `(country, name_first_token, addr_number)`.** This separates generic
   names at different addresses, such as franchises and clinics.

### 3.2 Candidate volume and recall

- **Candidate pairs:**
  - Train: 119,684,711 pairs (about 54 per Source-1 entity).
  - Test: 120,781,918 pairs for 1,732,544 Source-1 entities (about 70 per entity).
- **Reduction:** more than 99.999% fewer comparisons than all pairs.
- **Blocking pair recall, measured on `train_ground_truth.tsv`:**

| Scope | Recall |
|---|---|
| Overall | **97.4%** |
| US | 98.6% |
| India | 95.5% |

Blocking recall is a hard ceiling on the final score, so we measured it before training any matcher.

- **`candidate_pairs.tsv`** contains exactly this candidate set, written by `run_retrieve.py`. There is
  no later pruning: every candidate is featurised, scored by Stage 1 and seen by Stage 2 and the
  decoder. Every id in `matching_results.tsv` therefore also appears in `candidate_pairs.tsv`.

---

## 4. Matching Model

### 4.1 Stage-1 features (25 total)

**Pairwise features (16, `src/features.py`).** These are computed out-of-core in 2M-row batches with
`rapidfuzz`, parallelised across CPU cores.

| # | Feature | Description |
|---|---|---|
| 1 | `sim` | e5 cosine similarity between the two records (null for lexical-only candidates) |
| 2 | `cand_rank` | Rank of this candidate by `sim` within the Source-1 entity's candidate list |
| 3 | `cand_margin` | `sim` minus the best `sim` in the Source-1 entity's candidate list |
| 4 | `s1_name_core_freq` | How often this Source-1 `name_core` occurs in Source-1 (generic-name signal) |
| 5 | `name_ratio` | `fuzz.ratio` on the normalised full names |
| 6 | `name_core_ratio` | `fuzz.ratio` on the suffix-stripped core names |
| 7 | `name_token_set_ratio` | `fuzz.token_set_ratio` on the full names |
| 8 | `addr_ratio` | `fuzz.ratio` on the normalised addresses |
| 9 | `addr_token_set_ratio` | `fuzz.token_set_ratio` on the addresses |
| 10 | `name_first_token_eq` | First name token equal |
| 11 | `number_exact` | First address number equal |
| 12 | `number_prefix` | One first address number is a prefix of the other (truncation, e.g. `5014`/`501`) |
| 13 | `number_jaccard` | Jaccard overlap of the address-number sets |
| 14 | `name_len_diff` | Absolute difference in name length |
| 15 | `addr_len_diff` | Absolute difference in address length |
| 16 | `either_non_ascii` | Either original name was in a non-Latin script (transliteration-noise flag) |

**Context/competition features (9, `src/add_context_features.py`).**
- These are window features over the full candidate table.
- `lex` = `name_core_ratio + addr_token_set_ratio`.
- *Entity-side* features look at the Source-2/3 record: how this Source-1 candidate compares with the
  other Source-1 candidates of the same record.
- *Source-1-side* features look at the Source-1 entity and its candidate list.

| # | Feature | Description |
|---|---|---|
| 17 | `ent_n_cands` | Number of Source-1 candidates for this Source-2/3 record |
| 18 | `ent_sim_rank` | Rank of this pair's `sim` among the record's candidates |
| 19 | `ent_sim_margin` | `sim` minus the record's best `sim` |
| 20 | `ent_sim_gap2` | Gap between the record's best and second-best `sim` |
| 21 | `ent_lex_rank` | Rank of `lex` among the record's candidates |
| 22 | `ent_lex_margin` | `lex` minus the record's best `lex` |
| 23 | `s1_n_cands` | Number of candidates for the Source-1 entity |
| 24 | `s1_lex_rank` | Rank of `lex` within the Source-1 entity's candidates |
| 25 | `s1_lex_margin` | `lex` minus the Source-1 entity's best `lex` |

**Country is never a model feature.** It is used only for blocking. France never appears in training,
so a country-dependent model would be least reliable exactly where we cannot validate it.

### 4.2 Stage-1 model

- **Model:** LightGBM binary classifier (`src/train_matcher.py`).
- **Hyperparameters:** objective `binary`, learning_rate 0.05, num_leaves 63, min_data_in_leaf 200,
  feature_fraction 0.9, bagging 0.8/5, seed 42, up to 500 rounds with early stopping (patience 30) on
  validation AP. The best iteration was 339.
- **Training data:** 21.9M rows. All positives are kept, plus a random 15% of negatives, from 1.99M
  training Source-1 entities.
- **Validation data:** 11.9M rows covering all candidates of 220,682 held-out Source-1 entities, with
  no subsampling.
- **Validation metrics:** AUC 0.9995 and AP 0.9928. The 16-feature baseline scored AUC 0.9989 and AP
  0.9868.
- **Most important features by gain:** `ent_lex_margin` and `ent_lex_rank` dominate, followed by
  `number_jaccard`, `name_len_diff`, `ent_sim_rank` and `addr_token_set_ratio`. The large gain of the
  entity-side competition features supports the assignment view of the problem.

### 4.3 Stage-2 stacker (`src/stack_stage2.py`)

The Stage-1 matcher scores every candidate pair independently, and the base context features
only see *similarity* scores of competitors. Stage 2 re-scores each pair knowing how the Stage-1
*model* scored every competing pair.

- **Out-of-fold Stage-1 predictions (`src/oof_stage1.py`).** Train Source-1 entities are split
  into 2 folds by a hash of `source1_entity_id`. A Stage-1 LightGBM (same 25 features, 339 rounds)
  is trained on each fold and scores the other fold, giving honest predictions for all 119.7M train
  pairs. Test predictions are the average of the two fold models, so train and test predictions come
  from models of the same strength.
- **Stage-2 features (48 total, no country):** the Stage-1 `pred`; the 25 Stage-1 features;
  per Source-2/3 record aggregates of `pred` over all its Source-1 candidates (max, second best,
  gap, sum, count > 0.5, is-argmax, margin to best and to best other); the same aggregates per
  Source-1 entity; and source features (is Source-3, the S1's best / count of confident candidates
  in the same vs the other source).
- **Full competitor context.** Aggregates are computed over *all* S1 entities (not just the ones a
  model is trained on), exactly as on test, where every S1 competes for every record.
- **Test-like training regime.** The test set has fewer S1 entities per S2/S3 record than train
  (1.73M S1 vs 10.0M S2/S3; train 2.2M vs 10.3M), i.e. more records whose owner is absent. We
  simulate this by dropping 21% of non-validation train S1 entities before computing aggregates, both
  for training and for the validation estimate.
- **Model:** XGBoost (Apache-2.0) trained on the GPU (RTX 4060, `device="cuda"`, `tree_method="hist"`)
  on ~28M rows from ~548K train S1 entities, depth 9, eta 0.05, up to 2000 rounds with early stopping
  on a held-out slice of *train* S1 entities. Validation S1 entities are never used for training.
- **Results on the 220,682 validation S1 entities (test-like simulation):** Stage-1 only 0.9451;
  Stage-2 LightGBM 0.9573; **Stage-2 single GPU XGBoost 0.9621; **final: average of two GPU XGBoost models (different seeds and train-S1 samples) 0.9631 (US 0.9703, India 0.9523)****.

### 4.4 Decoding and threshold selection (`src/decode.py`)

- **Exclusive assignment.** For each Source-2/3 record, keep only the highest-scoring Source-1
  candidate, and keep it only if its score is at least tau. The kept records are grouped per Source-1
  entity. Source-1 entities with no kept record are output as singletons (an empty list).
- **Guarantees.** By construction, no Source-2/3 id is assigned twice and no list contains duplicates.
- **Threshold choice.** tau is chosen by sweeping the competition metric (macro F0.5) on the
  validation split, not AUC/AP.
- **Stage-1 context matcher sweep:** the curve is flat at 0.9452–0.9461 for tau = 0.89–0.93. We chose
  **tau = 0.91** (0.9461; US 0.955, India 0.933) from the middle of this plateau. This makes the result
  robust to test singleton rates that differ from train.
- **Final decoder settings:** argmax per Source-2/3 record over all its Source-1 candidates, then a per-Source-1 expected-F0.5 choice of how many top candidates to keep (probability floor 0.2, gain 0.8); an empty list is chosen when P(no match) beats the best expected F0.5.

---

## 5. Validation Protocol

- **Split.** We use a 90/10 split **by Source-1 entity** (seed 42, `make_split_ids`), never by pair,
  so no entity's candidates leak across the split.
- **Validation set.** It contains *all* candidates of the held-out entities, with no negative
  subsampling. The decoder therefore sees realistic competition for each Source-2/3 record.
- **Metric.** Macro F0.5 is computed exactly as on the leaderboard: per-Source-1 F0.5, where an empty
  prediction for an empty gold list scores 1.0, averaged over entities. We report it overall and per
  country (`decode.f_beta_macro`, `evaluate_fine.py`).
- **Stage-2 evaluation.** The stacker is fitted and evaluated with internal folds or splits over the
  validation Source-1 entities, never on rows it was trained on.
- **Format check.** Every submission is checked with the challenge's `utils/validate_submission.py`
  (default and `--check-ids` modes) before upload.

---

## 6. Results & Error Analysis

| Model | Val macro F0.5 | US | India | Leaderboard |
|---|---|---|---|---|
| Baseline (16 features, tau=0.91) | 0.9359 | 0.948 | 0.917 | 0.922 |
| Stage-1 context matcher (25 features, tau=0.91) | 0.9461 | 0.955 | 0.933 | not submitted separately |
| **Final: Stage-1 + Stage-2 stacker + decoder** | **0.9631** | 0.9703 | 0.9523 | **see portal (single-XGBoost version: 0.938; Stage-2 LightGBM version: 0.936)** |

- **Test-set sanity check:**
  - The pipeline was run end-to-end on the test set: 1,732,544 Source-1 entities, including the
    unseen country France.
  - With the baseline, the singleton rate was 5.9% overall (France 4.2%) and the mean was 3.19
    matches per entity. Training ground truth has 5.6% and 3.46.
  - For the final submission, the singleton rate is 5.31% and the number of
    matched Source-2/3 ids is 5,866,073.
  - The match-count distribution peaks at k=3 and has a maximum of 11, which matches the training
    ground truth.
- **Common false positives:**
  - "Generic name, nearby address" cases, such as franchise and clinic chains sharing a name, where
    only the address separates the records.
  - Records with missing PIN codes or landmark-only addresses, where the address signal is weaker.
- **Common false negatives:**
  - *Blocking misses.* 2.6% of true pairs are never generated (India 4.5%, US 1.4%). The main causes
    are transliteration artifacts (for example Devanagari "Private" becoming "praivet") and heavily
    rewritten Indian free-text addresses.
  - *Decoder abstentions.* The high tau deliberately gives up some recall for precision.

---

## 7. France / Unseen-Country Handling

France appears only in the test set. Our design avoids depending on any country-specific learned
parameter:
- Country is used **only** to restrict blocking to the same country, which holds for 100% of true
  matches. It is never a model feature.
- Normalisation includes French legal suffixes (SAS, SARL, EURL, SCI) and French street abbreviations
  (R., Av., Bd., Ch.). These are rules written in code and learned from no data.
- The retrieval model is multilingual (e5 was pre-trained on 100+ languages), so French text embeds in
  the same space as English.
- All matcher features are language-agnostic string, number and rank statistics.

The only evidence available is indirect because France has no labels. With the baseline model,
France's predicted singleton rate (4.2%) and match-count distribution closely track the US/India and
training distributions.

---

## 8. Compliance

- **Models:**
  - `intfloat/multilingual-e5-small`: MIT license, 118M parameters, used only for retrieval
    embeddings.
  - LightGBM (Stage 1 and Stage 2): MIT license. These are gradient-boosted trees, not neural
    networks, and far below the 8B-parameter cap.
- **No external data lookup:** we use no external databases, APIs, geocoders, business registries or
  web data. The only inputs are the provided train/test TSVs and fixed normalisation rules in code.
  The e5 weights are downloaded once from the Hugging Face Hub as a pretrained model and are not used
  as a lookup source.
- **Other libraries:** polars, pyarrow, numpy, rapidfuzz, anyascii, scikit-learn, torch and
  sentence-transformers. All are open source under permissive licenses (MIT, BSD or Apache-2.0).

---

## 9. Conclusion

We built a blocking-then-scoring pipeline:
- blocking with multilingual embedding kNN plus lexical safety-net joins, at 97.4% recall;
- a 25-feature LightGBM matcher;
- a Stage-2 stacker over Stage-1 prediction context;
- an exclusive-assignment decoder.

It reaches validation macro F0.5 0.9631 (leaderboard see portal (single-XGBoost version: 0.938; Stage-2 LightGBM version: 0.936)), uses no external data
and stays well inside the license and size constraints.

The main lesson: once blocking recall was high, most of the remaining error was *ranking among
competing candidates*, not missing candidates. Features that describe competition between candidates
gave the largest single improvement (0.936 → 0.946). The next step would be to close India's blocking
gap, for example by fine-tuning the retriever on hard negatives or improving transliteration of legal
words.

---

## Appendix

### A. Code Artefacts

The full pipeline is in `code/business_entity_resolution/src/`. `README.md` lists the exact
reproduction commands, and `requirements.txt` pins the dependencies.

| File | Role |
|---|---|
| `config.py`, `io_utils.py` | Paths/constants, TSV I/O |
| `normalize.py` | Stage 0: normalisation |
| `embed.py`, `retrieve.py`, `run_retrieve.py` | Stage 1 blocking (entry point: `run_retrieve.py --split {train,test}`) |
| `features.py`, `run_features.py` | 16 pairwise features (entry point: `run_features.py --split {train,test}`) |
| `add_context_features.py` | 9 context/competition features (entry point: `add_context_features.py {train,test}`) |
| `split_and_sample.py`, `train_matcher.py` | Stage-1 matcher (entry point: `... ctx`) |
| `evaluate.py`, `evaluate_fine.py`, `decode.py` | Macro F0.5 metric, tau sweep, exclusive decoder |
| `score_matcher.py`, `score_test.py` | Batched Stage-1 scoring of the 120.8M test pairs |
| `stack_stage2.py` | Stage-2 stacker and final decode |
| `decode_test.py`, `run_test_finish.py` | Decode test scores into `output/` (Stage-1-only path) |
| `resample_hard_neg.py` | Optional hard-negative resampling experiment (not in the final path) |

### B. Additional Results

`METHODOLOGY.md` at the repository root has the narrative write-up and development timings. Validation
tau sweeps are logged in `work/log_eval_fine_ctx.txt`.
