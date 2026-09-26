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

---

## Session 2 (2026-09-26): metric-faithful rescoring + ensemble/HPO experiments

### Scorer bug discovered & fixed (IMPORTANT)
The original validation scorer gave **0.0 to correctly-predicted singletons**; the official
metric gives **1.0** (predict-empty on a true singleton = perfect). The old 0.7837 was
therefore **understated**. With the official-faithful scorer (gold counts taken from the
full ground-truth file, all val S1 entities scored, singletons = 1.0 when predicted empty):

- **Honest baseline of the shipped model: val macro-F0.5 = 0.83670** (τ=0.87, one-to-one).
- Cross-checked against the old scorer per-entity (brute force over 547K entities) —
  old scorer reproduces 0.78374 at τ=0.85 exactly; the delta vs 0.8366 is the singleton term.

### Feature diagnostics (diag3.py)
- Zero-gain features (dropped for new models): `country_eq`, `dom_any`, `name_exact_norm`,
  `a_addr_empty` (and near-zero: `addr_exact_norm` gain≈391, `name_shorter_in_longer` 15K).
- **pin_eq mystery solved**: pin coverage is tiny (S1: 6.8%, India S2/S3: 1.2%, US: 12.1%)
  and blocking keys force pin_eq=1 for 92.8% of candidate pairs → near-constant feature.
- All rapidfuzz features are stored **100× too small** (`/100` on already-normalized scores)
  — harmless for trees (scale-invariant) but standardized for linear models.

### New experiment infrastructure
- `src/exp_lib.py` — vectorized val scorer (full-GT gold, singletons=1.0), fine tau sweeps,
  numpy IRLS logistic regression, AP metric, standardization helpers.
- `src/run_exp.py` — staged experiment driver with checkpoints in `artifacts/exp/`:
  1 baseline sweep ✓ (0.83670 @ τ=0.87 1-1) · 2 logistic-regression second model ·
  3 LightGBM HPO (AP early stopping, select by val macro-F0.5) · 4 avg/stack combos.
- `src/cand_tradeoff.py` — blocking KEY_CAPS presets (current/tight/loose) with cached key
  hashes; reports pairs, mean candidates/S1, pair recall, macro recall, full-recall fraction.

### Candidate-set-size vs recall tradeoff (task 5)
| preset | unique pairs | mean cand/S1 | median | p90 | pair recall | macro ceiling | full-recall entities |
|---|---|---|---|---|---|---|---|
| tight (caps≈½) | 45.4M | 20.6 | 9 | 24 | 67.75% | 67.88% | 38.02% |
| **current** | 57.0M | 25.8 | 13 | 39 | 70.07% | 70.14% | 40.71% |
| loose (caps≈2×) | 75.7M | 34.3 | 18 | 64 | (killed before scoring) | — | — |

Read: halving the caps cuts candidates 20% for ~2.3 pts of recall ceiling; doubling them buys
~33% more candidates. The **current caps sit near the knee** — recommended to keep for the
leaderboard run; use `tight` only if a smaller candidate file is explicitly preferred.

Note: `cap`/`cap_exact` args of `emit_pairs_from_keys` are effectively dead — every one of the
13 key families has an explicit `KEY_CAPS` entry, which is the real knob.

### Blocking bug found: alt-romanization keys (cols 8/9) could never fire
S1 records are always Latin and have no alt view, so they never got a col-8/9 hash; only
S2/S3 records (which have alts) landed there → keys 8/9 grouped S2/S3 with themselves only.
**Fixed**: records without an alt now fall back to their main-view hash in those columns, so a
Latin S1 record can meet an alt-romanized S2/S3 record. Recall upside to be measured.

### Experiment results (all on the official-metric scorer)
HPO: 4 randomized LightGBM configs (early stopping on average precision, AP≈0.9975–0.9979,
final selection by validation macro-F0.5). Winner: neg:pos 1.5, num_leaves 127, lr 0.1,
min_data_in_leaf 300, L1 2.0, L2 1.0, ff 0.7, bf 0.7.

| variant | tau | mode | val macro-F0.5 |
|---|---|---|---|
| **LGBM tuned (39 feats)** | **0.87** | **one-to-one** | **0.83669** ← shipped |
| stack(LGBM, LR) coefs=[10.74, 2.07, −8.99] | 0.77 | one-to-one | 0.83548 |
| avg(LGBM, LR) | 0.65 | one-to-one | 0.82788 |
| LogReg (numpy IRLS, standardized 39 feats) | 0.81 | one-to-one | 0.80010 |
| previous shipped model (43 feats, rescored) | 0.87 | one-to-one | 0.83670 |
| all 4 HPO configs | — | — | 0.8349–0.8367 |

**Conclusions**
1. The old 0.7837 was an artifact of the singleton bug; the honest shipped-model score is
   0.8367. Fine tau sweep (0.20–0.99 × plain/1-1): interior peak at τ=0.87 1-1 (0.90 was a
   boundary artifact of the old scorer; plain peaks later ~0.90 at 0.8348).
2. The classifier is NOT the bottleneck: every LGBM config lands within 0.002, AP > 0.9975.
   The 70.1% blocking recall ceiling caps macro-F0.5 ≈ 0.84 with this candidate set.
3. The ensemble loses to the single tree: stack coefficients put ~5× weight on LGBM
   (10.74 vs 2.07) and blending drags the calibrated LGBM toward the weaker LR
   (avg 0.8279 < 0.8367). Kept in the pipeline and re-evaluated automatically; selection
   stays metric-driven (single LGBM wins).
4. Feature cleanup: country_eq / dom_any / name_exact_norm / a_addr_empty have exactly 0
   gain in the shipped model (no splits) — provably redundant. New models train on 39 feats.
   pin_eq is near-constant by construction: pin coverage S1 6.8% / India S2S3 1.2% / US
   12.1%, and 92.8% of candidate pairs already have pin_eq=1 (blocking keys pin-partition).
5. rapidfuzz features were stored 100× too small (/100 on already-normalized scores) —
   harmless for trees, standardized for linear models; noted as a cosmetic bug.

### Code state after this session
- `src/train_match.py` — ensemble trainer: HPO (AP early stopping, F0.5 selection) → LR →
  avg/stack combos → fine tau sweeps (0.20–0.99, both modes) → picks winner by val F0.5 →
  saves model.pkl (variant-aware). Env knobs: N_HPO (default 8), LGBM_ONLY.
- `src/predict_test.py` — variant-aware inference (lgbm/lr/avg/stack), same output format.
- `src/exp_lib.py` — official-metric vectorized scorer, fine sweeps, IRLS LR, AP.
- `src/combos_from_cache.py` — rebuild combos from cached probs without retraining.
- `src/cand_tradeoff.py` — blocking recall-vs-size presets (see table above).
- `src/blocking.py` — alt-key fallback fix + allow_fallback toggle for oversized groups.

