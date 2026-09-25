"""Deterministic validation split: hold out 25% of S1 entities by id hash.

Saves artifacts/split.pkl with:
  is_val: bool array over train records (True = validation S1 or any record)
  val_s1_idx / train_s1_idx: indices of S1 records in each half
"""
import pickle
import sys

import numpy as np

from blocking import load_data

ART = "artifacts"


def main(tag="train", frac=0.25, seed=1234567):
    d, meta = load_data(tag)
    src = d["src"]
    eid = d["eid"]
    s1_idx = np.nonzero(src == 0)[0]
    vals = np.array([hash((seed, e)) % 10000 for e in eid[s1_idx]])
    is_val_s1 = vals < int(frac * 10000)
    val_s1_idx = s1_idx[is_val_s1]
    train_s1_idx = s1_idx[~is_val_s1]
    with open(f"{ART}/split.pkl", "wb") as f:
        pickle.dump(dict(val_s1_idx=val_s1_idx, train_s1_idx=train_s1_idx,
                         frac=frac, seed=seed), f, protocol=4)
    print(f"split: {len(train_s1_idx)} train S1, {len(val_s1_idx)} val S1")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
