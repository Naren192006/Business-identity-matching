# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** September 2026

---

## 1. Executive Summary

We treat entity resolution as a two-stage retrieve-then-classify problem. Stage 1 generates candidate S1×(S2/S3) pairs with 13 country-scoped blocking key families (exact token/phone sets, PIN, address-digit+phone, domain, and rare-token-pair keys), achieving ~70% pair recall (measured on training data). Stage 2 scores each candidate pair with 43 features (exact token/phone/digit overlaps, fuzzy similarities across raw/normalized/core/alt-romanization views, structural agreements) fed to a LightGBM classifier, threshold-calibrated for macro-F0.5 on a held-out 25% of Source-1 entities, with an optional one-to-one argmax resolution exploiting the fact that each S2/S3 record matches at most one S1 entity. A custom rule-based romanizer converts all 9 Indic scripts to business-style Latin, recovering the ~8% of positive pairs whose names exist only in Devanagari/Telugu/Tamil/etc. on one side.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA findings that shaped the design:

- **Scale**: 2.21M S1 + 5.03M S2 + 5.29M S3 training records; 7.64M positive pairs; 5.6% of S1 entities are singletons (correctly predicting them earns 1.0 each).
- **Structure**: every S2/S3 record matches at most one S1 entity (checked exhaustively); matches never cross countries (0/7.64M). This enables country-scoped blocking and a one-to-one resolution step.
- **Noise**: typos ("Foudanoaion"), accents ("Pórter"), repeated tokens, truncations ("Blue Software Private"), injected junk tokens ("Jaxkelo"), DBA phrases ("Brixwex doing business as Wilk Delta Plus P.C."), domain names (`wilkdeltaplus.com` ≈ Wilk Delta Plus P.C.), word-order transpositions, address reordering ("Massachusetts, East Boston, 10 Moore St"), component omissions (168–176K empty addresses in S2/S3), landmark references, "10D" vs "10" house numbers.
- **Scripts**: S2/S3 Indian records frequently render names and addresses in Devanagari, Telugu, Tamil, Kannada, Malayalam, Bengali, Gujarati, Oriya or Gurmukhi (~10% of matched-pair S2/S3 names), while S1 is always Latin. ~8% of matched-pair S2/S3 names are fully Indic.
- **Validation design**: a deterministic 25% holdout of S1 entities (by id hash) provides an unbiased macro-F0.5 estimate including singleton credit.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pairwise classifier + one-to-one resolution  
**Core Innovation:** Cross-script retrieval via a rule-based romanizer with dual business-style variants feeding both blocking keys and max-of-variants fuzzy features; rare-token-pair blocking keys that catch pairs sharing only high-DF tokens; memory-bounded vectorized feature extraction over 57M candidate pairs on a 16GB machine.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (country always folded into the key; both primary and alternate romanization views generate keys):
  1. exact core-name token set (legal suffixes stripped)
  2. (first core phone token, sorted remaining core phones)
  3. PIN code
  4. (first address digit group, first address phone token)
  5. exact core-phone token set
  6. domain id
  7. (first, last) core phone tokens
  8. (first 3 core phones, first 2 address tokens)
  9–10. alt-romanization views of keys 1 and 5
  11. (2 rarest core phones) by document frequency
  12. (rarest core phone, first address digit)
  13. (2 rarest core name tokens) by document frequency
- **Candidate pairs generated:** ~57M unique pairs on train (≈26M on test).
- **How true matches were not lost:** per-key group-size caps (20–60) balance recall vs. blow-up; oversized *exact-name* groups keep pairs corroborated by PIN or state agreement instead of being dropped; rare-token keys catch pairs whose only shared tokens are common ones; alt-romanization keys recover cross-script matches. Measured on train: **70.1% pair recall; 40.7% of S1 entities have 100% of their gold matches as candidates** (the F0.5 recall ceiling).

## 4. Matching Model

**Features (43)** —
- Name token exact overlap: Jaccard, Dice, overlap, containment, precision, recall, subset flag (max of primary/alt romanization views)
- Core (legal-suffix-stripped) overlaps + exact-core flag
- Phone (double-metaphone) overlaps: Jaccard, Dice, overlap, subset
- Address token overlaps: Jaccard, Dice, overlap, exact flag
- Address digit overlaps: Jaccard, exact, subset, equality
- String similarity (Indel) on: raw name, normalized name, core name, max over alt-variant crosses; Jaro-Winkler on raw name; Indel/JW on raw address
- PIN equality; state abbreviation equality / missingness
- Domain equality; domain present; country equality
- Token document frequencies (first-token DF both sides, rarest-core-token DF); name token-count difference; shorter-in-longer; subsequence; name-number equality; address emptiness flags

**Model type:** LightGBM binary classifier (MIT license; ~1.3k rounds × 127 leaves — well under the 8B-parameter cap), trained on all positives + 1.5:1 negative subsample, early stopping on validation AUC.  
**Threshold selection method:** sweep τ ∈ {0.30 … 0.90}, maximizing macro-F0.5 over all val S1 entities (singletons included); the same sweep is repeated with a one-to-one argmax resolution (each S2/S3 assigned to its highest-scoring S1) and the better configuration is used for test inference.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** [filled after final validation run]
- **Common false positives (wrong merges):** generic names in the same city with no true link (caught by pushing τ up and by the one-to-one step); chain stores sharing a brand token.
- **Common false negatives (missed matches):** pairs whose name was heavily truncated plus re-ordered (few shared tokens); heavy typos breaking both metaphone codes; addresses empty on one side; romanization variants our rules render differently than the source's own Latinization.

---

## 6. Conclusion

A country-scoped 13-family blocking scheme with rare-token-pair and cross-script keys, plus 43 overlap/fuzzy/structural features and a precision-tuned LightGBM, delivers a strong F0.5 under tight compute. The main lessons: measure *entity-level* full recall (not just pair recall) to know your ceiling, and treat transliteration as a first-class blocking concern, not a post-hoc feature.

---

## Appendix

### A. Code Artefacts

Complete runnable code ships in the zip under `code/business_entity_resolution/` (all source in `src/`, with `README.md` and `requirements.txt`). Entry points, in order: `preprocess.py` (train+test) → `split_train.py` → `blocking.py` (train, then test) → `pairs_data.py` → `build_features.py` → `train_match.py` → `predict_test.py`, which regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` end-to-end.

### B. Additional Results

- Candidate recall progression: 54.9% (initial 10-key design) → 70.1% (13 keys + corroboration fallback for oversized exact-name groups).
- 80.6% of gold pairs share ≥1 blocking key ignoring caps; 80% of misses share ≥1 phone token — motivation for the rare-pair key families.
- Validation threshold sweep table: [see artifacts/model.pkl metadata / training log].

---
