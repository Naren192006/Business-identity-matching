# Business Entity Resolution — PROGRESS LOG

**Working directory:** `6ab10eb3b23ba_student_resource/student_resource/`
**Last updated:** during run of final pipeline (see "Current status" at bottom)

---

## ✅ COMPLETED

### 1. Environment setup
- Python 3.11.9, 16 GB RAM, 22 CPUs (Windows).
- Installed: `lightgbm 4.7.0`, `rapidfuzz 3.14.6`, `doublemetaphone`, `indic-transliteration` (tested, rejected — Sanskrit-only).
- Note: **scikit-learn is blocked** on this machine (Windows Application Control policy blocks a DLL), so the pipeline uses **LightGBM** for the model and pure numpy/rapidfuzz everywhere else.

### 2. EDA (`src/eda.py`, `src/eda2.py`, `src/eda3.py`)
Key findings:
- Train: 2,206,821 S1 + 5,034,616 S2 + 5,285,603 S3 records; **7.64M positive pairs**; 5.6% singletons; most S1 entities have 2–5 matches.
- Test: 1,732,544 S1 + 4,887,273 S2 + 5,082,317 S3 records; countries US / India / **France** (test only).
- **Each S2/S3 record matches at most one S1 entity** (verified exhaustively) → one-to-one resolution is valid.
- **Zero cross-country matches** (0/7.64M) → country is a perfect blocking component.
- Noise: typos, accents, truncations, junk-token injections, DBA phrases, domain names (`wilkdeltaplus.com`), word-order transposition, address reordering, empty addresses (~170K in S2/S3).
- **~10% of matched S2/S3 names are in an Indic script** (Devanagari/Telugu/Tamil/Kannada/Malayalam/Bengali/Gujarati/Oriya/Gurmukhi) while S1 is always Latin.

### 3. Indic romanizer (`src/romanize.py`) — custom-built
- Rule-based, covers **all 9 Indic Unicode blocks** → business-style ASCII.
- Lookahead algorithm: consonant+virama → bare; consonant+matra → vowel; else inherent "a"; trailing inherent schwa deleted (शर्मा→`sharma`, लक्ष्मी→`laxmi`, प्राइवेट→`praivet`).
- Produces **two variants** per string (ph/f, sh/s, v/w ambiguities) — downstream takes max.
- Handles letter-names (एलएलपी→`llp`), conjuncts (क्ष→`x`), nukta, digits, ZWNJ.

### 4. Normalization (`src/textnorm.py`)
- NFKC + accent strip + lowercase + punctuation folding.
- Legal-suffix dictionary (60+ entries: ltd/pvt/corp/llp/sarl/sas/eurl/…) → "core name" views.
- Address word folding (rd/st/blvd/plot/soc/nagar/rue/allee/…), US+India+France state maps (both directions), PIN extraction, domain extraction, double-metaphone phone tokens (cached).

### 5. Preprocessing (`src/preprocess.py`) — checkpointed per source
- Streams TSVs → compact numpy arrays + vocab pickles in `artifacts/`.
- Token vocab (train: 2.05M tokens), phone vocab (596K), flat int32 token arrays, alt-romanization token arrays, address digits, PIN, domain, country ids, raw-text byte blob.
- **Train done**: `artifacts/train_data.npz` (12.53M records). **Test done**: `artifacts/test_data.npz` (11.70M records).

### 6. Validation split (`src/split_train.py`)
- Deterministic 25% holdout of S1 entities by id hash → 1,655,723 train S1 / 551,098 val S1.

### 7. Blocking (`src/blocking.py`) — 13 key families, all country-scoped
Exact token set / phone0+rest / PIN / digit+phone / phone set / domain / first+last phone / 3-phones+2-addr-tokens / alt-view token set / alt-view phone set / **2 rarest phones** / **rarest phone + first digit** / **2 rarest tokens**.
- Group-size caps per key; oversized exact-name groups keep only PIN-or-state-corroborated pairs.
- Fixed a critical numpy-npz performance bug (per-access decompression) by materializing arrays.
- **Train result: 57.0M unique candidate pairs; pair recall 70.1%; 40.7% of S1 entities have FULL gold recall (F0.5 ceiling).** (Was 54.9% before rare-token keys were added.)

### 8. Pair labels (`src/pairs_data.py`)
- `artifacts/train_pairdata.npz`: 5.35M positives / 57.0M pairs; val/train split flags per pair.

### 9. Feature extraction (`src/features.py`, `src/build_features.py`) — 43 features
- Exact set overlaps (name tokens, core tokens, phones, addr tokens, addr digits) via **vectorized explode+sort+intersect**.
- rapidfuzz similarities (Indel, JaroWinkler) on raw / normalized / core names + max over alt-romanization crosses + raw addresses.
- PIN / state / domain / country agreement, DF/rarity, subsequence, subset, nums-eq, emptiness flags.
- **Train features DONE**: `artifacts/train_feats.npy` (57M × 43 float32, 9.8 GB, 50 min).
- Debugged & fixed: npz re-decompression per access; `tok_rarity` indexing strings instead of ids; element-wise (not cross-product) fuzzy scoring.

### 10. Model training (`src/train_match.py`)
- LightGBM (127 leaves, lr 0.06, 1.5:1 neg subsample, early stopping) on 10.04M pairs.
- Threshold sweep on 14.18M val pairs; both plain and **one-to-one argmax** resolution.
- **RESULT: best validation macro-F0.5 = 0.78374 at τ=0.85, one_to_one mode.**
- Model saved: `artifacts/model.pkl` (τ=0.85, mode=one_to_one).

### 11. Inference + output writer (`src/predict_test.py`)
- Streams test pairs, applies threshold + 1-1 resolution, writes both output TSVs with **every test S1 entity exactly once** via merge-join (memory-bounded).

### 12. Docs
- `README.md` — full reproduction instructions.
- `requirements.txt` — pinned deps.
- `Documentation_template.md` — methodology write-up (F0.5 number now filled in).

---

## 🏁 FINAL STATUS — PIPELINE COMPLETE

Everything finished and validated:

- Test blocking: **29.77M candidate pairs** for 1.73M test S1 entities.
- Test features: 29.77M × 43 (≈25 min).
- Inference: 1,531,134 matches written (966,282 of 1,732,544 S1 rows have ≥1 match).
- **`utils/validate_submission.py` → `PASS — no blocking issues found. Safe to submit.`**
- Outputs on disk:
  - `output/matching_results.tsv` (42.8 MB — **upload this to the portal**)
  - `output/candidate_pairs.tsv` (406 MB)
- Final submission package built: **`team_name_submission.zip` (190 MB)** containing
  `output/`, `code/business_entity_resolution/` (src + README + requirements), and the
  filled methodology doc. Rename the zip with your real team name before uploading.

### Bugs fixed along the way (for the record)
- npz `__getitem__` decompressed the whole array on every access (materialize once).
- `src` array missing in assembled test data → derive from ID prefixes.
- `tok_rarity` indexed tok_df with strings → use token ids.
- LightGBM `evals_result` kwarg removed in 4.x → drop it.
- OverflowError hashing into int64 → mask to 62 bits.
- Pair-count explosions (300M) → per-key caps + pin/state corroboration for oversized exact-name groups.

### Optional improvements if time permits
- Tighten threshold near τ=0.85 (sweep 0.82–0.88 in 0.01 steps).
- Higher blocking recall via per-country caps or an S2↔S2 transitive-closure pass.
- Feature: country-specific legal-suffix handling for France (SARL/SAS/EURL already in dict).

---

## Validation metric reference
F0.5 = 1.25·P·R / (0.25·P + R), macro-averaged per S1 entity, singletons included.
**Final validation score: 0.7837** (τ=0.85, one-to-one mode).
