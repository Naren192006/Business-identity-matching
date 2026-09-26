"""Candidate-set-size vs recall tradeoff for the blocking stage.

Caches the 13-column key-hash matrix once (artifacts/train_keys.npz), then
for each KEY_CAPS preset emits pairs and reports:
  - total unique candidate pairs and per-S1 candidate-count stats
  - pair recall (gold pairs found among candidates)
  - macro candidate recall (mean over val+train S1 entities of the fraction
    of gold matches present in candidates) -- the recall ceiling for F0.5
  - full-recall entity fraction

Usage:  python3 src/cand_tradeoff.py --preset current|tight|loose|custom
        [--caps '{"0": 10, ...}']
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocking
from blocking import build_key_hashes, emit_pairs_from_keys, load_data

ART = "artifacts"

PRESETS = {
    "current": {0: 20, 1: 30, 2: 15, 3: 30, 4: 30, 5: 60, 6: 30, 7: 30,
                8: 20, 9: 30, 10: 30, 11: 30, 12: 30},
    "tight": {0: 10, 1: 15, 2: 8, 3: 15, 4: 15, 5: 30, 6: 15, 7: 15,
              8: 10, 9: 15, 10: 15, 11: 15, 12: 15},
    "loose": {0: 40, 1: 60, 2: 30, 3: 60, 4: 60, 5: 120, 6: 60, 7: 60,
              8: 40, 9: 60, 10: 60, 11: 60, 12: 60},
}


def get_keys(rd, state_arr):
    path = f"{ART}/train_keys.npz"
    if os.path.exists(path):
        print("loading cached keys", flush=True)
        return np.load(path)["keys"]
    t0 = time.time()
    keys = build_key_hashes(rd, pin=rd.pin, state_arr=state_arr)
    np.savez(path, keys=keys)
    print(f"keys hashed+cached ({time.time()-t0:.0f}s)", flush=True)
    return keys


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
    return np.asarray(out_s1, dtype=np.int64), np.asarray(out_s2,
                                                          dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="current")
    ap.add_argument("--caps", default=None)
    args = ap.parse_args()

    caps = dict(PRESETS[args.preset])
    if args.caps:
        caps.update({int(k): v for k, v in json.loads(args.caps).items()})
    print(f"preset={args.preset} caps={caps}", flush=True)

    blocking.KEY_CAPS.clear()
    blocking.KEY_CAPS.update(caps)

    d, meta = load_data("train")
    rd = blocking.RecData(d, meta)
    states = meta["states"]
    st_codes = {"": 0}
    state_arr = np.array([st_codes.setdefault(s, len(st_codes))
                          for s in states], dtype=np.int32)
    keys = get_keys(rd, state_arr)
    t0 = time.time()
    s1, s2 = emit_pairs_from_keys(keys, rd.src, pin=rd.pin,
                                  state_arr=state_arr)
    print(f"emitted {len(s1)} unique pairs ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- per-S1 candidate stats
    cnt = np.bincount(s1, minlength=rd.n)
    s1_rows = np.nonzero(rd.src == 0)[0]
    c = cnt[s1_rows]
    print(f"candidates per S1: mean={c.mean():.2f} median={np.median(c):.0f} "
          f"p90={np.percentile(c, 90):.0f} p99={np.percentile(c, 99):.0f} "
          f"max={c.max()}", flush=True)

    # ---- recall
    g1, g2 = gold_pairs_idx(meta["eid2idx"])
    N = rd.n
    pk = np.unique(s1.astype(np.int64) * N + s2.astype(np.int64))
    gk = g1 * N + g2
    hit = np.searchsorted(pk, gk)
    hit[hit >= len(pk)] = len(pk) - 1
    found = pk[hit] == gk
    print(f"pair recall: {found.mean():.4%}  "
          f"({int(found.sum())}/{len(gk)} gold pairs)", flush=True)

    # macro recall: per-entity fraction of gold found
    per_a_tot = np.bincount(g1, minlength=N)
    per_a_hit = np.bincount(g1[found], minlength=N)
    m = per_a_tot > 0
    mac = per_a_hit[m] / per_a_tot[m]
    print(f"macro candidate recall (S1 entities with gold): "
          f"{mac.mean():.4%}", flush=True)
    print(f"entities with all gold found (full recall): "
          f"{(per_a_hit[s1_rows] == per_a_tot[s1_rows]).mean():.4%}",
          flush=True)
    print(f"RECALL_CEILING={mac.mean():.5f} PAIRS={len(s1)} "
          f"MEAN_CAND={c.mean():.2f} PRESET={args.preset}", flush=True)


if __name__ == "__main__":
    main()
