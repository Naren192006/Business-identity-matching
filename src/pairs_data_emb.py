"""Pair dataset + label builder for any candidate source (keys, emb, union).

Mirrors src/pairs_data.py but parameterized by the candidate npz so the
embedding-blocking candidates (or the union) can flow into the SAME feature
builder / LightGBM pipeline without touching the original modules.

CLI:
  python3 src/pairs_data_emb.py --tag train --in train_pairs_all --suffix _all
  python3 src/pairs_data_emb.py --tag test  --in test_pairs_all  --suffix _all

Writes artifacts/{tag}_pairdata{suffix}.npz (train: + label, is_val) and
prints the blocking diagnostics that matter:
  candidate pair count, pair recall, macro recall (F0.5 ceiling proxy),
  full-recall entity fraction, candidates-per-S1 stats.
Also (train only) reports how much of the ceiling comes from val vs train.
"""
import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import load_data

ART = "artifacts"


def gold_pairs_idx(eid2idx):
    out_s1, out_s2 = [], []
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8",
              newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = eid2idx.get(row["source1_entity_id"])
            if i is None:
                continue
            for m in row["matched_entity_ids"].split(","):
                if m and m in eid2idx:
                    out_s1.append(i)
                    out_s2.append(eid2idx[m])
    return (np.asarray(out_s1, dtype=np.int64),
            np.asarray(out_s2, dtype=np.int64))


def report(tag, s1, s2, N, eid2idx, label=""):
    """Blocking metrics on the same definitions cand_tradeoff.py uses."""
    g1, g2 = gold_pairs_idx(eid2idx)
    N64 = np.int64(N)
    pk = np.unique(s1.astype(np.int64) * N64 + s2.astype(np.int64))
    gk = np.unique(g1 * N64 + g2)
    hit = np.searchsorted(pk, gk)
    hit[hit >= len(pk)] = len(pk) - 1
    found = pk[hit] == gk
    per_a_tot = np.bincount(g1, minlength=N)
    per_a_hit = np.bincount(g1[found], minlength=N)
    m = per_a_tot > 0
    mac = per_a_hit[m] / per_a_tot[m]
    d = np.load(f"{ART}/{tag}_data.npz")
    eid = d["eid"]
    s1_rows = np.nonzero([e.startswith("S1-") for e in eid])[0]
    cnt = np.bincount(s1, minlength=N)[s1_rows]
    print(f"[{label or tag}] pairs={len(pk)}  "
          f"pair recall={found.mean():.4%}  "
          f"macro recall (ceiling)={mac.mean():.4%}  "
          f"full-recall entities={(per_a_hit[s1_rows] == per_a_tot[s1_rows]).mean():.4%}",
          flush=True)
    print(f"  cand/S1: mean={cnt.mean():.2f} median={np.median(cnt):.0f} "
          f"p90={np.percentile(cnt, 90):.0f} max={cnt.max()}", flush=True)
    return dict(pairs=len(pk), pair_recall=float(found.mean()),
                macro_recall=float(mac.mean()),
                full_recall=float((per_a_hit[s1_rows]
                                   == per_a_tot[s1_rows]).mean()),
                mean_cand=float(cnt.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="train")
    ap.add_argument("--in", dest="cand", default="train_pairs_all",
                    help="candidate npz base name, e.g. train_pairs_all")
    ap.add_argument("--suffix", default="_all",
                    help="suffix for output pairdata file")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    tag = args.tag
    t0 = time.time()

    cand_path = f"{ART}/{args.cand}.npz"
    pairs = np.load(cand_path)
    s1 = pairs["s1"].astype(np.int64)
    s2 = pairs["s2"].astype(np.int64)
    print(f"{cand_path}: {len(s1)} pairs", flush=True)

    d, meta = load_data(tag)
    eid2idx = meta["eid2idx"]
    eid = d["eid"]
    N = len(eid)

    if args.report_only:
        report(tag, s1, s2, N, eid2idx, label=args.cand)
        return

    if tag == "train":
        # ---- gold pairs -> labels (same logic as pairs_data.py)
        gt_idx = {}
        with open("dataset/train/train_ground_truth.tsv", encoding="utf-8",
                  newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                i = eid2idx.get(row["source1_entity_id"])
                if i is None:
                    continue
                gt_idx[i] = [eid2idx[m] for m in
                             row["matched_entity_ids"].split(",") if m
                             if m in eid2idx]
        pos_sorted = np.array([i * N + j for i, js in gt_idx.items()
                               for j in js], dtype=np.int64)
        pos_sorted.sort()
        pk = s1 * N + s2
        pos_at = np.searchsorted(pos_sorted, pk)
        pos_at[pos_at >= len(pos_sorted)] = len(pos_sorted) - 1
        label = (pos_sorted[pos_at] == pk).astype(np.float32)
        npos = int(label.sum())
        print(f"labels: {npos} positive / {len(label)} "
              f"({npos/max(len(label),1):.3%})", flush=True)

        with open(f"{ART}/split.pkl", "rb") as f:
            split = pickle.load(f)
        val_set = set(int(x) for x in split["val_s1_idx"])
        is_val = np.array([int(i) in val_set for i in s1], dtype=bool)
        print(f"val pairs: {int(is_val.sum())}, train pairs: "
              f"{int((~is_val).sum())}", flush=True)
        out = f"{ART}/{tag}_pairdata{args.suffix}.npz"
        np.savez(out, s1_idx=s1, s2_idx=s2, label=label, is_val=is_val)
        print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)
        # ---- blocking report on both halves
        report(tag, s1[~is_val], s2[~is_val], N, eid2idx, label="train-half")
        report(tag, s1[is_val], s2[is_val], N, eid2idx, label="val-half")
    else:
        out = f"{ART}/{tag}_pairdata{args.suffix}.npz"
        np.savez(out, s1_idx=s1, s2_idx=s2)
        print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
