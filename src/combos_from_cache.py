"""Fallback: build ensemble combos from cached val probs (no retraining).

Uses probs_base.npy (old 43-feat booster) + probs_lr.npy, computes avg/stack,
fine sweeps everything, picks the winner, and writes model.pkl that
predict_test.py understands (variant='lgbm' with the old booster, or 'avg'/
'stack' with the LR attached).
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_lib import (ValScorer, best_from_sweep, fine_taus,
                     fit_lr, gold_counts_full, load_split,
                     load_train_pairs, lr_predict)
from features import FEATURE_NAMES

ART = "artifacts"
EXP = f"{ART}/exp"


def sweep_best(scorer, probs, taus):
    rows = [{"tau": t, "plain": scorer.score_fast(probs, t, "plain"),
             "one_to_one": scorer.score_fast(probs, t, "one_to_one")}
            for t in taus]
    return best_from_sweep(rows), rows


def main():
    t0 = time.time()
    s1, s2, label, is_val = load_train_pairs()
    split = load_split()
    scorer = ValScorer(s1[is_val], s2[is_val], label[is_val], split,
                       gold_counts_full(split))
    va_idx = np.nonzero(is_val)[0]
    lv = label[is_val]
    taus = fine_taus()

    probs_G = np.load(f"{EXP}/probs_base.npy")
    probs_LR = np.load(f"{EXP}/probs_lr.npy")

    b_G, rows_G = sweep_best(scorer, probs_G, taus)
    b_LR, rows_LR = sweep_best(scorer, probs_LR, taus)
    probs_avg = ((probs_G.astype(np.float64) +
                  probs_LR.astype(np.float64)) / 2).astype(np.float32)
    b_avg, rows_avg = sweep_best(scorer, probs_avg, taus)

    a_val = s1[is_val]
    fold = (a_val * 2654435761) % 5
    Xs2 = np.concatenate([probs_G[:, None], probs_LR[:, None],
                          np.ones((len(probs_G), 1), dtype=np.float32)],
                         axis=1).astype(np.float64)
    oof = np.zeros(len(Xs2), dtype=np.float32)
    for k in range(5):
        trm = fold != k
        w_k = fit_lr(Xs2[trm], lv[trm].astype(np.float64), l2=1e-3,
                     iters=15, verbose=False)
        oof[~trm] = lr_predict(Xs2[~trm], w_k)
    w_stack = fit_lr(Xs2, lv.astype(np.float64), l2=1e-3, iters=15,
                     verbose=False)
    print(f"[stack] coefs = {np.round(w_stack, 4)}", flush=True)
    b_st, rows_st = sweep_best(scorer, oof, taus)

    print("\n==== VALIDATION MACRO-F0.5 (official metric) ====", flush=True)
    for name, b in [("LGBM (shipped, rescored)", b_G),
                    ("LogReg standalone", b_LR),
                    ("avg(LGBM,LR)", b_avg),
                    ("stack(LGBM,LR)", b_st)]:
        print(f"  {name:26s} tau={b[0]:.2f} {b[1]:11s} F0.5={b[2]:.5f}",
              flush=True)

    cand = [("lgbm", b_G, probs_G), ("lr", b_LR, probs_LR),
            ("avg", b_avg, probs_avg), ("stack", b_st, oof)]
    kind, best, probs_win = max(cand, key=lambda x: x[1][2])
    print(f"\nWINNER: {kind} tau={best[0]:.2f} mode={best[1]} "
          f"F0.5={best[2]:.5f}", flush=True)

    old = pickle.load(open(f"{ART}/model.pkl", "rb"))
    S2 = pickle.load(open(f"{EXP}/stage2.pkl", "rb"))
    with open(f"{ART}/model.pkl", "wb") as f:
        pickle.dump(dict(
            booster=old["booster"],
            tau=best[0], mode=best[1], val_f05=best[2], variant=kind,
            feature_names=list(FEATURE_NAMES),   # old booster uses all 43
            feature_names_full=list(FEATURE_NAMES),
            params=old.get("params", {}),
            lr_w=S2["w"], lr_mu=S2["mu"], lr_sd=S2["sd"],
            stack_w=w_stack,
            avg_baseline_f05=b_avg[2], stack_val_f05=b_st[2],
            lgbm_val_f05=b_G[2], lr_val_f05=b_LR[2],
            hpo_results=[],
            note="shipped booster trained on all 43 features; the four "
                 "zero-gain features (country_eq, dom_any, name_exact_norm, "
                 "a_addr_empty) receive no splits, so dropping them would "
                 "not change its predictions",
        ), f, protocol=4)
    print(f"model.pkl updated ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
