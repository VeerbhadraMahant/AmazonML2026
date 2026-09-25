# Business Entity Resolution — Methodology

*A plain-language walkthrough of the pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. This is the working explainer; the formal submission write-up goes in `AmazonML/student_resource/Documentation_template.md`.*

## 1. The problem in one sentence

For every business record in **Source 1** (the clean reference list), find every record in **Source 2** and **Source 3** that describes the *same real business*, even though the name and address are spelled differently, abbreviated, transliterated, or noisy — and don't guess when unsure, because wrong merges are penalized twice as hard as missed matches (F0.5 scoring).

## 2. What the data actually looks like (from our EDA)

- **Scale:** 2.2M Source-1 records, ~5M each in Source 2 and Source 3 for training; test is similarly sized (1.7M / 4.9M / 5.1M). Comparing every pair is impossible — 2.2M × 10M would be ~22 trillion comparisons.
- **Only 5.6% of Source-1 entities are singletons** (no match anywhere); the average entity has ~3.5 matches. Aggressively predicting "no match" would score very badly — recall can't be sacrificed lazily.
- **Each Source-2/3 record belongs to at most one Source-1 entity** (verified in ground truth: zero records matched to more than one S1). This turns pairwise matching into an *assignment* problem: pick each record's single best Source-1 owner, or none.
- **Country always agrees within a true match** (100% in training data). Searching only within the same country string is safe, and it handles France (test-only) with no special-casing.
- Real noise found in samples: legal-suffix drift (`Pvt` vs `Private`, `LLC` vs `Limited`), honorifics added/dropped (`Sri`, `Shri`), punctuation junk (`***`, `##`, `[..]`), homoglyph typos (`m0tors`, `lbbie`), non-Latin scripts in Source 2/3 (Devanagari, Gujarati, Kannada) for businesses Source 1 always spells in Latin letters, truncated house numbers (`5014` vs `501`), and reordered/abbreviated addresses (`Rd` vs `Road`, French `R.` vs `Rue`).

## 3. The pipeline

```
raw TSV
  → Stage 0: Normalize (clean & standardize text)
  → Stage 1: Candidate Generation / "Blocking" (narrow millions down to dozens)
  → Stage 2: Pairwise Matching Model (score each candidate pair)
  → Stage 3: Decoding (turn scores into a final answer, respecting the rules)
  → matching_results.tsv + candidate_pairs.tsv
```

### Stage 0 — Normalize (`src/normalize.py`, implemented & tested)
- Transliterate non-Latin text (Hindi, Gujarati, Kannada, ...) to Latin letters, so a Devanagari business name can be compared to its English spelling. Only rows that actually contain non-ASCII characters pay the transliteration cost.
- Lowercase, strip punctuation noise, expand abbreviations (`St`→`street`, `Rd`→`road`, plus French `R.`→`rue`, `Av.`→`avenue`, `Bd.`→`boulevard`, `Ch.`→`chemin`).
- Produce a "core" name with legal suffixes and honorifics stripped (`Ram Investment Private Limited` → `ram investment`), so records differing only in legal wording still look similar.
- Extract numbers from the address (house numbers, PIN/ZIP) as a separate signal — often the most reliable disambiguator when many businesses share a generic name (e.g. "Primary Care Group" recurs 253 times in Source 1 alone, each at a different address).
- Runs at ~6–7 seconds per million rows; all ~24M records across train+test normalize in a few minutes and are cached to `work/*.parquet` so this never reruns unnecessarily.

### Stage 1 — Candidate Generation ("Blocking")
Three complementary techniques, unioned, all restricted to matching country:
1. **Semantic search** with `intfloat/multilingual-e5-small` (118M params, MIT license — well under the challenge's 8B-parameter cap). Every business's `name | address` becomes a GPU embedding; for each Source-2/3 record we find its nearest Source-1 records by cosine similarity. Catches heavily reworded or transliterated names.
2. **Exact-match lookup** on normalized core-name + country, as a safety net for obvious matches.
3. **House-number + first-name-token lookup**, which discriminates between businesses sharing a generic name but sitting at different addresses.

The union becomes `candidate_pairs.tsv` — the exact set fed to Stage 2. We choose how many candidates to keep per record by measuring **recall@K on a held-out validation slice**, targeting ≥98% before trusting the scoring stage, since blocking recall is a hard ceiling on the final score.

### Stage 2 — Pairwise Matching Model
Per (Source-1, candidate) pair, compute similarity features:
- **Name:** edit-distance ratio, token-set overlap, character n-gram overlap — on both full and core names.
- **Address:** token overlap, house-number/PIN exact-or-prefix match.
- **Semantic:** the Stage-1 embedding cosine similarity.
- **Context:** this candidate's rank among the query's matches, margin to the runner-up, how often this Source-1 name recurs (generic-name signal).

Country is **never** a model feature (only used to narrow the search) — this keeps the model well-behaved on France, which it never sees at train time.

Model: **LightGBM** classifier (MIT-licensed gradient-boosted trees, not a neural net, trivially inside the 8B-parameter rule), trained on labeled candidate pairs, producing a match probability per pair.

### Stage 3 — Decoding
Because each Source-2/3 record belongs to at most one Source-1 entity, we assign it to its single highest-probability match, kept only if confidence clears a threshold τ. τ is tuned to maximize **macro F0.5** on our own validation split — deliberately biasing toward "say nothing" when unsure, since false merges cost 2× more than missed matches. Source-1 records with no accepted matches are correctly left as empty (singleton) rows, which score full credit.

## 4. How we check our own work before submitting

We hold out 10% of training Source-1 entities as a private validation set (never trained on), score predictions with the exact macro-F0.5 formula the leaderboard uses, and run the challenge's own `utils/validate_submission.py` before every upload — so a submission is never wasted on a formatting rejection (we only get 5/day).

## 5. Why this approach, specifically

- **Blocking-then-scoring** is the standard, tractable way to do entity resolution at this scale, and it's literally what the `candidate_pairs.tsv` deliverable expects us to produce and document.
- **A small embedding model + gradient-boosted trees**, not a large LLM, stays safely inside the license/size rule, trains and runs fast enough to iterate many times in 72 hours, and is easy to explain and debug for the methodology write-up the organizers review.
- **Exploiting the "each record matches at most one Source-1 entity" structural fact** turns pairwise scoring into an assignment problem, which directly improves precision — the metric F0.5 weights most.

## 6. Status

- ✅ Stage 0 (normalization): implemented, tested against real data, all 6 train/test files cached as parquet.
- ⏭ Next: Stage 1 — load the embedding model on the GPU, generate the first candidate set, measure blocking recall.
