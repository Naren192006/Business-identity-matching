"""Diagnostics for feature cleanup: LGBM importances, pin_eq coverage, ranges."""
import pickle

import numpy as np

ART = "artifacts"

with open(f"{ART}/model.pkl", "rb") as f:
    M = pickle.load(f)
booster = M["booster"]
names = M["feature_names"]
sp = booster.feature_importance(importance_type="split")
sg = booster.feature_importance(importance_type="gain")
order = np.argsort(-sg)
print("== LGBM feature importance (gain / split) ==")
for i in order:
    print(f"{names[i]:22s} gain={sg[i]:10.0f}  split={sp[i]:6d}")

pd_ = np.load(f"{ART}/train_pairdata.npz")
s1 = pd_["s1_idx"].astype(np.int64)
s2 = pd_["s2_idx"].astype(np.int64)
label = pd_["label"]
va = pd_["is_val"]
F = np.load(f"{ART}/train_feats.npy", mmap_mode="r")
va_idx = np.nonzero(va)[0]
rng = np.random.default_rng(0)
sub = rng.choice(va_idx, size=min(2_000_000, len(va_idx)), replace=False)

print("\n== per-feature stats on 2M random val pairs (overall / pos / neg) ==")
pos = label[sub] > 0
for j, nm in enumerate(names):
    x = np.asarray(F[sub, j])
    print(f"{nm:22s} mean={x.mean():7.4f} p10={np.percentile(x,10):6.3f} "
          f"p90={np.percentile(x,90):6.3f} pos_mean={x[pos].mean():7.4f} "
          f"neg_mean={x[~pos].mean():7.4f}")

# pin coverage: fraction of records with pin >= 0, by side and country
import sys
sys.path.insert(0, "src")
from blocking import load_data
d, meta = load_data("train")
pin = d["pin"]
src = d["src"]
cn = d["country"]
has_pin = pin >= 0
s1_m = src == 0
print("\n== pin coverage by source/country ==")
print(f"S1: {has_pin[s1_m].mean():.3%}")
for c in np.unique(cn[~s1_m]):
    m = (~s1_m) & (cn == c)
    print(f"S2/S3 country={c}: {has_pin[m].mean():.3%}")
# gold pairs pin agreement (both pins present & equal) on val subset
gold_mask = np.zeros(len(s1), dtype=bool)
gold_mask[pos] = True
j_pin = names.index("pin_eq")
x_pin = np.asarray(F[sub, j_pin])
print(f"\npin_eq on gold val pairs: {x_pin[pos].mean():.3%} "
      f"(i.e. {1-x_pin[pos].mean():.1%} of gold pairs have pin_eq=0)")
