"""Inference on test pairs -> output/matching_results.tsv + candidate_pairs.tsv.

Loads artifacts/model.pkl (booster, tau, mode), scores test pair features,
applies threshold + optional 1-1 resolution, writes the two output TSVs with
every test S1 entity present exactly once.  Outputs are written by merge-join
over the pair arrays (sorted by s1) so memory stays bounded.
"""
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

from blocking import load_data

ART = "artifacts"


def main(tag="test"):
    t0 = time.time()
    with open(f"{ART}/model.pkl", "rb") as f:
        M = pickle.load(f)
    booster = M["booster"]
    tau = M["tau"]
    mode = M["mode"]
    print(f"model: tau={tau} mode={mode} (val F0.5={M.get('val_f05')})",
          flush=True)

    pd_ = np.load(f"{ART}/{tag}_pairdata.npz")
    s1 = pd_["s1_idx"].astype(np.int64)
    s2 = pd_["s2_idx"].astype(np.int64)
    F = np.load(f"{ART}/{tag}_feats.npy", mmap_mode="r")
    print(f"{tag}: {len(s1)} pairs, feats {F.shape}", flush=True)

    probs = np.zeros(len(s1), dtype=np.float32)
    CH = 1000000
    for st in range(0, len(s1), CH):
        en = min(st + CH, len(s1))
        probs[st:en] = booster.predict(np.asarray(F[st:en]),
                                       num_iteration=booster.best_iteration)
        print(f"  predict {en}/{len(s1)} ({time.time()-t0:.0f}s)", flush=True)

    d, meta = load_data(tag)
    eid = d["eid"]
    if "src" in d:
        src = d["src"]
    else:
        src = np.fromiter(
            (0 if e.startswith("S1-") else 1 for e in eid),
            dtype=np.int8, count=len(eid))
    eid2idx = meta["eid2idx"]
    N = len(eid)

    # ---- 1-1 resolution: each S2/S3 goes to its argmax S1 (among kept pairs)
    if mode == "one_to_one":
        keep = np.nonzero(probs >= tau)[0]
        # sort kept pairs by (b, -p) so the first entry per b wins
        order = np.lexsort((-probs[keep], s2[keep]))
        kk = keep[order]
        b_seen = {}
        sel = np.zeros(len(kk), dtype=bool)
        bs = s2[kk]
        first = np.nonzero(np.diff(bs, prepend=bs[0] - 1) != 0)[0]
        sel[first] = True
        final_pairs = (s1[kk[sel]], s2[kk[sel]])
    else:
        keep = np.nonzero(probs >= tau)[0]
        final_pairs = (s1[keep], s2[keep])

    # sort final pairs by s1 for merge-join
    fo = np.lexsort((final_pairs[1], final_pairs[0]))
    f1 = final_pairs[0][fo]
    f2 = final_pairs[1][fo]

    # candidate pairs are already sorted by (s1, s2) from np.unique
    ci = 0
    fi = 0
    nf = len(f1)
    nc = len(s1)
    os.makedirs("output", exist_ok=True)
    n_match = n_cand = 0
    with open("output/matching_results.tsv", "w", encoding="utf-8",
              newline="") as fm, open("output/candidate_pairs.tsv", "w",
                                      encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        s1_rows = np.nonzero(src == 0)[0]
        for row_i in s1_rows:
            sid = eid[row_i]
            # candidates for this s1
            cands = []
            while ci < nc and s1[ci] == row_i:
                cands.append(eid[s2[ci]])
                ci += 1
            # advance over any pairs with s1 < row_i (S2/S3-side rows)
            while ci < nc and s1[ci] < row_i:
                ci += 1
            # matches for this s1
            matches = []
            while fi < nf and f1[fi] == row_i:
                matches.append(eid[f2[fi]])
                fi += 1
            while fi < nf and f1[fi] < row_i:
                fi += 1
            fm.write(f"{sid}\t{','.join(matches)}\n")
            fc.write(f"{sid}\t{','.join(cands)}\n")
            n_match += len(matches)
            n_cand += len(cands)
    print(f"written: {len(s1_rows)} S1 rows, {n_match} matches, "
          f"{n_cand} candidate pairs ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
