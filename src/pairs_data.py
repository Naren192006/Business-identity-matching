"""Build pair datasets: ids + labels (train) and split membership.

Inputs: artifacts/{tag}_pairs.npz (from blocking.py), artifacts/split.pkl,
dataset/train/train_ground_truth.tsv.
Outputs: artifacts/{tag}_pairdata.npz with
  s1_idx, s2_idx : int64 arrays of aligned pair endpoints
  label          : float32 (train only) 1.0 = same business
  is_val         : bool array (train only) True = pair belongs to val S1s
"""
import csv
import pickle
import sys

import numpy as np

from blocking import load_data

ART = "artifacts"


def main(tag="train"):
    pairs = np.load(f"{ART}/{tag}_pairs.npz")
    s1 = pairs["s1"]
    s2 = pairs["s2"]
    print(f"pairs: {len(s1)}")

    if tag == "train":
        d, meta = load_data(tag)
        eid2idx = meta["eid2idx"]
        eid = d["eid"]
        # ground truth as sets of indices
        gt_idx = {}
        with open("dataset/train/train_ground_truth.tsv",
                  encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for row in rdr:
                i = eid2idx.get(row["source1_entity_id"])
                if i is None:
                    continue
                gt_idx[i] = [eid2idx[m] for m in
                             row["matched_entity_ids"].split(",") if m
                             if m in eid2idx]
        # label via searchsorted membership over sorted positive pair keys
        N = len(eid)
        pos_sorted = np.array([i * N + j for i, js in gt_idx.items() for j in js],
                              dtype=np.int64)
        pos_sorted.sort()
        pk = s1.astype(np.int64) * N + s2.astype(np.int64)
        pos_at = np.searchsorted(pos_sorted, pk)
        pos_at[pos_at >= len(pos_sorted)] = len(pos_sorted) - 1
        label = (pos_sorted[pos_at] == pk).astype(np.float32)
        npos = int(label.sum())
        print(f"labels: {npos} positive / {len(label)} "
              f"({npos/max(len(label),1):.3%})")

        with open(f"{ART}/split.pkl", "rb") as f:
            split = pickle.load(f)
        val_set = set(int(x) for x in split["val_s1_idx"])
        is_val = np.array([int(i) in val_set for i in s1], dtype=bool)
        print(f"val pairs: {int(is_val.sum())}, train pairs: {int((~is_val).sum())}")
        np.savez(f"{ART}/{tag}_pairdata.npz", s1_idx=s1, s2_idx=s2,
                 label=label, is_val=is_val)
    else:
        np.savez(f"{ART}/{tag}_pairdata.npz", s1_idx=s1, s2_idx=s2)
        print(f"saved {tag} pair ids")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
