"""Inference on test pairs -> output/matching_results.tsv + candidate_pairs.tsv.

Loads artifacts/model.pkl (ensemble of LightGBM + logistic regression and the
selected combination variant: 'lgbm' | 'lr' | 'avg' | 'stack'), scores test
pair features, applies the calibrated threshold and (if selected) one-to-one
argmax resolution, then writes both TSVs with every test S1 entity exactly
once via a streaming merge-join over sorted pair arrays.
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import load_data
from features import FEATURE_NAMES

ART = "artifacts"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main(tag="test"):
    t0 = time.time()
    with open(f"{ART}/model.pkl", "rb") as f:
        M = pickle.load(f)
    booster = M["booster"]
    tau = M["tau"]
    mode = M["mode"]
    variant = M.get("variant", "lgbm")
    keep_names = set(M["feature_names"])
    keep_cols = np.array([n in keep_names for n in FEATURE_NAMES])
    print(f"model: variant={variant} tau={tau} mode={mode} "
          f"(val F0.5={M.get('val_f05'):.5f})", flush=True)

    pd_ = np.load(f"{ART}/{tag}_pairdata.npz")
    s1 = pd_["s1_idx"].astype(np.int64)
    s2 = pd_["s2_idx"].astype(np.int64)
    F = np.load(f"{ART}/{tag}_feats.npy", mmap_mode="r")
    print(f"{tag}: {len(s1)} pairs, feats {F.shape}", flush=True)

    # ---------------- LightGBM probs (chunked) ---------------------------
    probs_G = np.zeros(len(s1), dtype=np.float32)
    CH = 1_000_000
    for st in range(0, len(s1), CH):
        en = min(st + CH, len(s1))
        X = np.asarray(F[st:en][:, keep_cols], dtype=np.float32)
        probs_G[st:en] = booster.predict(X,
                                         num_iteration=booster.best_iteration)
        print(f"  lgbm {en}/{len(s1)} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- second model + combination --------------------------
    if variant == "lgbm":
        probs = probs_G
    else:
        ncol = int(keep_cols.sum())
        mu, sd = M["lr_mu"], M["lr_sd"]
        Xlr = np.empty((len(s1), ncol + 1), dtype=np.float32)
        for st in range(0, len(s1), CH):
            en = min(st + CH, len(s1))
            Xlr[st:en, :ncol] = np.asarray(F[st:en][:, keep_cols],
                                           dtype=np.float32)
            Xlr[st:en, ncol] = 1.0
        Xlr[:, :ncol] = (Xlr[:, :ncol] - mu) / sd
        p_lr = _sigmoid(Xlr @ M["lr_w"]).astype(np.float32)
        del Xlr
        if variant == "lr":
            probs = p_lr
        elif variant == "avg":
            probs = ((probs_G.astype(np.float64) +
                      p_lr.astype(np.float64)) / 2).astype(np.float32)
        else:  # stack: w = [w_g, w_lr, bias] on raw probs
            W = np.asarray(M["stack_w"], dtype=np.float64)
            probs = _sigmoid(W[0] * probs_G + W[1] * p_lr + W[2])
            probs = probs.astype(np.float32)
        del p_lr

    d, meta = load_data(tag)
    eid = d["eid"]
    if "src" in d:
        src = d["src"]
    else:
        src = np.fromiter(
            (0 if e.startswith("S1-") else 1 for e in eid),
            dtype=np.int8, count=len(eid))
    N = len(eid)

    # ---------------- threshold + optional 1-1 argmax ---------------------
    keep = np.nonzero(probs >= tau)[0]
    if mode == "one_to_one":
        order = np.lexsort((-probs[keep], s2[keep]))
        kk = keep[order]
        bs = s2[kk]
        first = np.nonzero(np.diff(bs, prepend=bs[0] - 1) != 0)[0]
        final_pairs = (s1[kk[first]], s2[kk[first]])
    else:
        final_pairs = (s1[keep], s2[keep])

    fo = np.lexsort((final_pairs[1], final_pairs[0]))
    f1 = final_pairs[0][fo]
    f2 = final_pairs[1][fo]

    ci = fi = 0
    nf, nc = len(f1), len(s1)
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
            cands = []
            while ci < nc and s1[ci] == row_i:
                cands.append(eid[s2[ci]])
                ci += 1
            while ci < nc and s1[ci] < row_i:
                ci += 1
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
