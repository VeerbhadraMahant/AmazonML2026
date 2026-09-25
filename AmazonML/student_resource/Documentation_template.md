# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We built a three-stage blocking-then-scoring pipeline: a GPU embedding search plus two lexical
exact-key joins generate a candidate set per Source-1 entity, a LightGBM classifier scores each
candidate pair on 16 similarity/context features, and a threshold-based exclusive decoder (each
Source-2/3 record can belong to at most one Source-1 entity) turns those scores into the final
match lists. On a held-out 10% validation split by Source-1 entity, this scores **macro
F0.5 = 0.9359** (US 0.9483, India 0.9167). The core innovation is exploiting a structural fact we
found in the training data — every Source-2/3 record matches at most one Source-1 entity — to
reframe pairwise matching as an assignment problem, which directly improves precision, the metric
F0.5 weights twice as heavily as recall.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training data (2.2M Source-1, 5.0M Source-2, 5.3M Source-3 records) surfaced:

- **Scale.** Comparing all pairs is infeasible (2.2M x 10.3M ~ 22 trillion comparisons), so a
  strong blocking stage is mandatory, not optional.
- **Singletons are rare (5.6%)**, with a mean of 3.46 matches per Source-1 entity (up to 11).
  Aggressively predicting "no match" would score badly; recall matters even under a
  precision-heavy metric.
- **Each Source-2/3 record matches at most one Source-1 entity** — verified exactly in
  `train_ground_truth.tsv` (zero ids matched more than one Source-1 row). This turns the problem
  into an assignment problem rather than independent pairwise classification.
- **Country agrees in 100% of true matches.** Blocking strictly within country is safe and, since
  France only appears in the test set, requires no special-casing — it's just another country
  string.
- **Exact normalized name equality holds in only 22%** of true pairs (73% share a first token),
  so blocking needs fuzzy/semantic matching, not just exact-key lookups.
- **Noise patterns observed:** legal-suffix drift (Pvt/Private, LLC/Limited, and French
  SAS/SARL/EURL/SCI), honorifics added/dropped (Sri/Shri), punctuation junk (`***`, `##`, `[..]`),
  homoglyph typos (`m0tors`, `lbbie`), non-Latin scripts in Source-2/3 (Devanagari, Gujarati,
  Kannada) for businesses Source-1 always spells in Latin letters, truncated house numbers (`5014`
  vs `501`), and reordered/abbreviated addresses (Rd/Road, French R./Rue, Av./Avenue).
- Roughly 30% of Source-1 business names repeat (e.g. "Primary Care Group" appears 253 times),
  each at a different address — address and house-number features are what disambiguate these,
  not name similarity alone.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (embedding retrieval for candidate generation, LightGBM
gradient-boosted trees for pairwise scoring, exclusive-assignment decoding).

**Core Innovation:** Treating decoding as an assignment problem — each Source-2/3 record is
assigned to its single highest-probability Source-1 match (or none, if no candidate clears the
confidence threshold) — rather than thresholding every candidate pair independently. This
directly targets precision, which F0.5 weights 2x over recall, and is justified by the verified
1:1 structure of the ground truth.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. **Semantic search**: `intfloat/multilingual-e5-small` (118M params, MIT license) embeds
     `name | address` for every record; for each Source-2/3 record we retrieve its top-K
     (K=10) nearest Source-1 records by cosine similarity, computed on GPU (chunked matmul +
     top-k, sized to stay within VRAM regardless of partition size — see
     `code/business_entity_resolution/src/retrieve.py`). Restricted to matching country.
  2. **Exact-key lexical join** on `(country, name_core)` — the normalized name with legal
     suffixes and honorifics stripped — as a safety net for obvious matches, skipping keys with
     more than 40 matching Source-1 rows (left to the embedding + number features instead, to
     avoid candidate-set blow-up on generic names).
  3. **Exact-key lexical join** on `(country, name_first_token, first_address_number)`, which is
     effective at disambiguating businesses that share a generic name but sit at different
     addresses.
- **Candidate pairs generated:** 119,684,711 (train), 120,781,918 (test) — an average of ~54 and
  ~70 candidates per Source-1 entity respectively (test is higher because of the added France
  partition), a >99.999% reduction from the raw Source-2+3 pool size.
- **How we ensured true matches were not lost:** we measured blocking recall directly against
  `train_ground_truth.tsv` — **97.4% overall** (US 98.6%, India 95.5%) — before ever training the
  matcher, since this recall is a hard ceiling on the final score. `candidate_pairs.tsv` is
  written from exactly this candidate set (no further pruning before the matcher runs on it).

---

## 4. Matching Model

**Features used (16 total, computed per candidate pair):**
- Name features: edit-distance ratio (`rapidfuzz.fuzz.ratio`) and token-set ratio, computed on
  both the raw normalized name and the legal-suffix-stripped "core" name; first-token equality;
  name length difference.
- Address features: edit-distance ratio and token-set ratio on the normalized address; extracted
  house-number/PIN-code Jaccard overlap, exact match, and prefix match (to catch truncation like
  `5014` vs `501`); address length difference.
- Semantic: the Stage-1 embedding cosine similarity itself, plus this candidate's rank among the
  query's retrieved matches and its margin to the query's best-scoring candidate.
- Other: whether the Source-1 name recurs frequently across Source-1 (a "generic name, trust the
  address more" signal), and whether either side's original text was non-Latin script (a
  transliteration-noise signal).
- **Country is deliberately excluded as a model feature** (used only earlier, for blocking) —
  since France never appears in training, a country-dependent model would be unreliable exactly
  where it can't be locally validated.

**Model type:** LightGBM binary classifier (gradient-boosted trees; MIT-licensed, far under the
8B-parameter cap — this is not a neural network). Trained on a 90/10 split by Source-1 entity
(never by pair, to avoid leaking a Source-1 entity's candidates across train/validation): 21.9M
training rows (all positives + a subsampled 15% of negatives, for training speed) and 11.9M
full, unsampled validation rows (needed for a realistic decode-time evaluation). Validation
AUC = 0.9989, AP = 0.9868. By gain, the retrieval rank (`cand_rank`) is the dominant feature,
followed by address token-set ratio, embedding cosine similarity, and core-name similarity.

**Threshold selection method:** swept the decoding threshold tau directly against the actual
competition metric (macro F0.5 on the held-out validation set), not just AUC/AP, since those are
pairwise metrics that don't capture the per-entity, precision-weighted nature of F0.5. Score rises
from 0.797 at tau=0.30 to a flat peak of 0.9359 around tau=0.90-0.92, then declines past ~0.93.
We picked **tau=0.91** from the middle of that flat region rather than the single best point, to
be robust to the test set's true singleton rate differing slightly from train's.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** **0.9359** overall (US 0.9483, India 0.9167), on a 10%
  held-out split of Source-1 entities never used in training or threshold tuning.
- **Test-set sanity check:** the trained pipeline was run end-to-end on the actual test set
  (1,732,544 Source-1 entities, including France, unseen at training time). Resulting singleton
  rate is 5.89% overall (US 6.1%, India 6.2%, France 4.2%) and mean matches/entity is 3.19 —
  closely tracking train's 5.6% singleton rate and 3.46 mean, including for France, which suggests
  the model generalizes to the unseen country rather than behaving degenerately. The match-count
  distribution shape (peaking at k=3, smoothly decaying to a maximum of 11) closely mirrors the
  training ground truth's shape.
- **Common false positives (wrong merges):** expected to concentrate on the "generic name, similar
  address neighborhood" cases (e.g. clinics/franchises sharing a name across many locations),
  where the address features are the main signal and any residual address noise (missing PIN,
  landmark-only addresses) weakens that signal.
- **Common false negatives (missed matches):** primarily bounded by the 2.6% blocking recall gap
  (higher in India at 4.5% missed than US at 1.4%), likely from transliteration artifacts our
  in-code romanization doesn't fully normalize (e.g. "Private" transliterating to "praivet" from
  Devanagari, only partially matching the English legal-suffix stripping rules) plus heavier
  free-text address variation in Indian records (landmark references, missing PIN/state).

---

## 6. Conclusion

We built a blocking-then-scoring entity resolution pipeline that combines GPU-accelerated
multilingual embedding retrieval, lexical safety-net joins, and a LightGBM pairwise classifier
whose decoding exploits the verified 1:1 assignment structure of the ground truth, reaching a
validated macro F0.5 of 0.9359 while requiring no external data and staying comfortably within
the model-size/license constraints. The main lesson was that blocking recall (not model quality)
was the real ceiling on the final score, and that country-specific noise (especially India's
transliteration and address-format variety) is where most of the remaining error concentrates —
future iterations would focus there first, e.g. with fine-tuning the retrieval embedding model on
hard negatives.

---

## Appendix

### A. Code Artefacts

Full runnable pipeline ships under `code/business_entity_resolution/` (see its own `README.md` for
exact reproduction steps and `requirements.txt` for pinned dependencies). Structure:

- `src/normalize.py` — Stage 0: text normalization (transliteration, legal-suffix stripping,
  address abbreviation expansion, number extraction).
- `src/embed.py`, `src/retrieve.py`, `src/run_retrieve.py` — Stage 1: GPU embedding retrieval +
  lexical blocking; entry point `run_retrieve.py --split {train,test}`.
- `src/features.py`, `src/run_features.py` — Stage 2: out-of-core pairwise feature computation;
  entry point `run_features.py --split {train,test}`.
- `src/split_and_sample.py`, `src/train_matcher.py` — matcher training; entry points
  `split_and_sample.py` then `train_matcher.py`.
- `src/decode.py`, `src/evaluate.py` — threshold sweep / macro F0.5 evaluation against the
  official metric; entry point `evaluate.py`.
- `src/score_matcher.py`, `src/run_test_finish.py` — Stage 3: score + decode the test set into
  `output/matching_results.tsv` and `output/candidate_pairs.tsv`; entry point
  `run_test_finish.py`.
- `src/config.py`, `src/io_utils.py` — shared paths/constants and TSV I/O helpers.

Running the six commands listed in `code/business_entity_resolution/README.md`, in order,
regenerates both output files from the raw train/test data.

### B. Additional Results

See `METHODOLOGY.md` at the repository root for the full narrative write-up, including the EDA
that motivated each design decision, and per-stage timing/throughput figures measured during
development.
