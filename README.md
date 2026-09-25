# Business Entity Resolution — ML Challenge 2026

## Quick start (end-to-end reproduction)

All commands run from this directory (`student_resource/`). Python 3.10+ required.

```bash
pip install -r requirements.txt

# 1. Preprocess train + test into compact artifacts (~25 min total)
python3 -X utf8 src/preprocess.py train S1 dataset/train/train_source1.tsv
python3 -X utf8 src/preprocess.py train S2 dataset/train/train_source2.tsv
python3 -X utf8 src/preprocess.py train S3 dataset/train/train_source3.tsv
python3 -X utf8 src/preprocess.py train assemble
python3 -X utf8 src/preprocess.py test S1 dataset/test/test_source1.tsv
python3 -X utf8 src/preprocess.py test S2 dataset/test/test_source2.tsv
python3 -X utf8 src/preprocess.py test S3 dataset/test/test_source3.tsv
python3 -X utf8 src/preprocess.py test assemble

# 2. Deterministic validation split (25% of S1 entities, by id hash)
python3 -X utf8 src/split_train.py train

# 3. Blocking / candidate generation on train + recall report
python3 -X utf8 src/blocking.py train

# 4. Ground-truth pair labels
python3 -X utf8 src/pairs_data.py train

# 5. Feature extraction (train, then test)
python3 -X utf8 src/build_features.py train
python3 -X utf8 src/blocking.py test
python3 -X utf8 src/pairs_data.py test
python3 -X utf8 src/build_features.py test

# 6. Train LightGBM + calibrate F0.5 threshold on the validation split
python3 -X utf8 src/train_match.py

# 7. Score test pairs and write output/matching_results.tsv + candidate_pairs.tsv
python3 -X utf8 src/predict_test.py test

# 8. Validate output format
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline overview

1. **preprocess.py** — streams the TSVs; normalizes names/addresses
   (`textnorm.py`), romanizes Indic scripts (`romanize.py`), and packs
   everything into compact numpy arrays + vocab pickles under `artifacts/`.
2. **split_train.py** — deterministic 25% holdout of Source-1 entities.
3. **blocking.py** — multi-key blocking (country + 8 key families: exact
   token set, phone keys, pin code, address-digit+phone, domain, alt-view
   romanization keys) via 64-bit hashing + numpy grouping. Emits
   S1×S2/S3 candidate pairs; reports candidate recall on train.
4. **pairs_data.py** — labels pairs from the ground truth.
5. **build_features.py / features.py** — 43 features per pair: exact token /
   phone / digit overlaps, legal-suffix-stripped core overlaps, rapidfuzz
   similarities (Indel, JaroWinkler) on raw + normalized + core + alt-variant
   name strings and raw addresses, pin/state/domain agreement, DF/rarity.
6. **train_match.py** — LightGBM binary classifier (negatives subsampled
   1.5:1), early stopping, threshold sweep for macro-F0.5 on the val split,
   plus a one-to-one (argmax) resolution variant.
7. **predict_test.py** — scores test pairs, applies the calibrated threshold,
   writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Notes

- `country` is treated as an open string label: it is only used as part of
  blocking keys (countries never cross-match in training data) and as the
  `country_eq` feature. France (absent from training) flows through the same
  pipeline with no hard-coded lists.
- Every test S1 entity gets exactly one output row; singletons get an empty
  match list. Matched IDs are always a subset of candidate IDs by construction.
- Model: LightGBM (MIT license), ~1.3k trees at 127 leaves — well within the
  8B-parameter / MIT-Apache constraint.
