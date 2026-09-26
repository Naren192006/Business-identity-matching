"""Train the pair matcher (LightGBM + logistic regression ensemble).

Pipeline (all from artifacts built by preprocess/build_features/pairs_data):
  1. load pair features; drop zero-gain features (country_eq, dom_any,
     name_exact_norm, a_addr_empty -- provably redundant: blocking never
     crosses countries, S1 always has an address, dom_any is ~always 0)
  2. randomized hyperparameter search over neg:pos ratio, num_leaves,
     learning_rate, min_data_in_leaf, L1/L2, sampling fractions;
     early stopping on average precision (rank metric), final selection by
     validation macro-F0.5 (official metric incl. singletons = 1.0)
  3. second model: logistic regression (numpy IRLS) on standardized features
  4. combinations: probability averaging + stacked LR on [p_lgbm, p_lr]
     (stack coefficients fit on the val split with 5-fold grouping by S1)
  5. fine tau sweep (0.20-0.99, step 0.01) in both plain and one-to-one
     resolution modes; pick the best variant overall
  6. save artifacts/model.pkl with everything predict_test.py needs

Set LGBM_ONLY=1 or N_HPO=0 to skip the search and train a single model.
"""
import csv
import os
import pickle
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_lib import (ValScorer, add_intercept, ap_feval, average_precision,
                     best_from_sweep, f05_vec, fine_taus, fit_lr,
                     gold_counts_full, load_split, load_train_pairs,
                     lr_predict)
from features import FEATURE_NAMES

ART = "artifacts"
EXP = f"{ART}/exp"
os.makedirs(EXP, exist_ok=True)

DROP = {"country_eq", "dom_any", "name_exact_norm", "a_addr_empty"}
KEEP = np.array([n not in DROP for n in FEATURE_NAMES])
KEPT_NAMES = [n for n in FEATURE_NAMES if n not in DROP]

N_HPO = int(os.environ.get("N_HPO", "8"))
BASE_PARAMS = dict(
    objective="binary",
    num_threads=22,
    verbosity=-1,
    seed=7,
)
HPO_GRID = dict(
    neg_pos_ratio=[1.5, 2.0, 3.0, 4.0],
    num_leaves=[31, 63, 127, 255],
    learning_rate=[0.05, 0.08, 0.1],
    min_data_in_leaf=[100, 300, 800],
    lambda_l1=[0.0, 0.5, 2.0],
    lambda_l2=[0.0, 1.0, 5.0],
    feature_fraction=[0.7, 0.85, 1.0],
    bagging_fraction=[0.7, 0.85, 1.0],
)
DROP4_PRINT = ", ".join(sorted(DROP))


def predict_chunked(booster, F, rows, chunk=1_000_000, tag=""):
    out = np.zeros(len(rows), dtype=np.float32)
    t0 = time.time()
    for st in range(0, len(rows), chunk):
        en = min(st + chunk, len(rows))
        X = np.asarray(F[rows[st:en]][:, KEEP], dtype=np.float32)
        out[st:en] = booster.predict(X, num_iteration=booster.best_iteration)
        if en % (chunk * 5) == 0 or en == len(rows):
            print(f"    predict {en}/{len(rows)} ({time.time()-t0:.0f}s)",
                  flush=True)
    return out


def sample_train(label, is_val, ratio, seed=42, cap=36_000_000):
    rng = np.random.default_rng(seed)
    tr_idx = np.nonzero(~is_val)[0]
    pos = tr_idx[label[tr_idx] > 0]
    neg = tr_idx[label[tr_idx] == 0]
    n_neg = min(len(neg), int(len(pos) * ratio))
    sel = rng.choice(neg, size=n_neg, replace=False)
    use = np.sort(np.concatenate([pos, sel]))
    if len(use) > cap:
        rng2 = np.random.default_rng(1)
        use = np.sort(rng2.choice(use, size=cap, replace=False))
    return use


def materialize(F, rows, mu=None, sd=None, intercept=False, chunk=2_000_000):
    """Materialize (standardized) feature rows for `rows` as float32."""
    extra = 1 if intercept else 0
    ncol = int(KEEP.sum())
    X = np.empty((len(rows), ncol + extra), dtype=np.float32)
    for st in range(0, len(rows), chunk):
        en = min(st + chunk, len(rows))
        X[st:en, :ncol] = np.asarray(F[rows[st:en]][:, KEEP],
                                     dtype=np.float32)
        if intercept:
            X[st:en, -1] = 1.0
        print(f"    materialize {en}/{len(rows)}", flush=True)
    if mu is not None:
        X[:, :ncol] = (X[:, :ncol] - mu) / sd
    return X


def train_one_lgbm(cfg, F, label, is_val, Xva_es, yva_es):
    tr_idx = sample_train(label, is_val, cfg["neg_pos_ratio"])
    Xtr = np.empty((len(tr_idx), int(KEEP.sum())), dtype=np.float32)
    CH = 2_000_000
    for st in range(0, len(tr_idx), CH):
        en = min(st + CH, len(tr_idx))
        Xtr[st:en] = np.asarray(F[tr_idx[st:en]][:, KEEP], dtype=np.float32)
    params = dict(BASE_PARAMS)
    for k in ("num_leaves", "learning_rate", "min_data_in_leaf", "lambda_l1",
              "lambda_l2", "feature_fraction", "bagging_fraction"):
        params[k] = cfg[k]
    params["bagging_freq"] = 1
    dtrain = lgb.Dataset(Xtr, label=label[tr_idx])
    booster = lgb.train(
        params, dtrain, num_boost_round=900,
        valid_sets=[lgb.Dataset(Xva_es, label=yva_es)], valid_names=["val"],
        feval=ap_feval,
        callbacks=[lgb.early_stopping(60, first_metric_only=True,
                                      verbose=False),
                   lgb.log_evaluation(0)])
    ap = booster.best_score["val"]["ap"]
    return booster, ap


def main():
    t0 = time.time()
    s1, s2, label, is_val = load_train_pairs()
    split = load_split()
    gold_full = gold_counts_full(split)
    scorer = ValScorer(s1[is_val], s2[is_val], label[is_val], split,
                       gold_full)
    va_idx = np.nonzero(is_val)[0]
    print(f"pairs={len(s1)}, val pairs={len(va_idx)}, feats kept="
          f"{int(KEEP.sum())}/43 (dropped: {DROP4_PRINT})", flush=True)
    F = np.load(f"{ART}/train_feats.npy", mmap_mode="r")

    rng = np.random.default_rng(99)
    vs = rng.choice(len(va_idx), size=min(500_000, len(va_idx)),
                    replace=False)
    Xva_es = np.asarray(F[va_idx[vs]][:, KEEP], dtype=np.float32)
    yva_es = label[va_idx[vs]].astype(np.float32)

    taus = fine_taus()

    def sweep_best(probs):
        rows = [{"tau": t, "plain": scorer.score_fast(probs, t, "plain"),
                 "one_to_one": scorer.score_fast(probs, t, "one_to_one")}
                for t in taus]
        return best_from_sweep(rows), rows

    # ---------------- LightGBM (HPO selected by val macro-F0.5) ----------
    if N_HPO <= 0:
        results = []
    best_G, rows_G = None, None
    if N_HPO > 0:
        results = pickle.load(open(f"{EXP}/stage3.pkl", "rb")) \
            if os.path.exists(f"{EXP}/stage3.pkl") else []
        if results:
            print(f"[hpo] reusing {len(results)} cached configs", flush=True)
        rng_hp = np.random.default_rng(1000 + len(results))
        seen = {tuple(sorted(r["cfg"].items())) for r in results}
        trials = []
        while len(results) + len(trials) < N_HPO:
            cfg = {k: HPO_GRID[k][int(rng_hp.integers(len(HPO_GRID[k])))]
                   for k in HPO_GRID}
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                trials.append(cfg)
        best_booster = None
        best_b = None
        best_cfg_b = None
        for j, cfg in enumerate(trials):
            i = len(results) + j
            tc = time.time()
            booster_i, ap = train_one_lgbm(cfg, F, label, is_val, Xva_es,
                                           yva_es)
            probs = predict_chunked(booster_i, F, va_idx)
            np.save(f"{EXP}/probs_hp{i}.npy", probs)
            b, rows = sweep_best(probs)
            results.append(dict(i=i, cfg=cfg, best_iter=booster_i.best_iteration,
                                ap=ap, f05=b[2], tau=b[0], mode=b[1]))
            pickle.dump(results, open(f"{EXP}/stage3.pkl", "wb"))
            print(f"[hpo] {len(results)}/{N_HPO}: F0.5={b[2]:.5f} AP={ap:.5f} "
                  f"({time.time()-tc:.0f}s) cfg={cfg}", flush=True)
            if best_b is None or b[2] > best_b[2]:
                best_b = b
                best_cfg_b = cfg
                if best_booster is not None:
                    del best_booster
                best_booster = booster_i
            else:
                del booster_i
            del probs
        if len(results) == 0:
            raise RuntimeError("no HPO results")
        results.sort(key=lambda r: -r["f05"])
        top = results[0]
        print(f"[hpo] winner cfg: F0.5={top['f05']:.5f} cfg={top['cfg']}",
              flush=True)
        probs_G = np.load(f"{EXP}/probs_hp{top['i']}.npy")
        best_G, rows_G = sweep_best(probs_G)
        best_G_cfg = top["cfg"]
        booster = best_booster
    else:
        cfg = dict(neg_pos_ratio=3.0, num_leaves=127, learning_rate=0.06,
                   min_data_in_leaf=300, lambda_l1=0.0, lambda_l2=1.0,
                   feature_fraction=0.85, bagging_fraction=0.85)
        booster, ap = train_one_lgbm(cfg, F, label, is_val, Xva_es, yva_es)
        probs_G = predict_chunked(booster, F, va_idx)
        best_G, rows_G = sweep_best(probs_G)
        best_G_cfg = cfg
        print(f"[lgbm] single model F0.5={best_G[2]:.5f}", flush=True)

    # ---------------- logistic regression (diverse second model) ---------
    if os.path.exists(f"{EXP}/stage2.pkl"):
        S2 = pickle.load(open(f"{EXP}/stage2.pkl", "rb"))
        mu, sd, w_lr = S2["mu"], S2["sd"], S2["w"]
        probs_LR = np.load(f"{EXP}/probs_lr.npy")
        print("[lr] reusing cached stage2 model", flush=True)
    else:
        tr_idx = sample_train(label, is_val, 3.0)
        sub = rng.choice(tr_idx, size=min(1_000_000, len(tr_idx)),
                         replace=False)
        Xs = np.asarray(F[sub][:, KEEP], dtype=np.float64)
        mu = Xs.mean(axis=0)
        sd = Xs.std(axis=0)
        sd[sd < 1e-8] = 1.0
        del Xs
        Xtr = materialize(F, tr_idx, mu, sd, intercept=True)
        w_lr = fit_lr(Xtr, label[tr_idx].astype(np.float64), l2=1.0, iters=8)
        del Xtr
        Xva = materialize(F, va_idx, mu, sd, intercept=True)
        probs_LR = lr_predict(Xva, w_lr)
        np.save(f"{EXP}/probs_lr.npy", probs_LR)
        pickle.dump(dict(w=w_lr, mu=mu, sd=sd, l2=1.0, ratio=3.0),
                    open(f"{EXP}/stage2.pkl", "wb"))
        del Xva
    best_LR, rows_LR = sweep_best(probs_LR)
    print(f"[lr] standalone F0.5={best_LR[2]:.5f} (tau={best_LR[0]:.2f} "
          f"mode={best_LR[1]})", flush=True)

    # ---------------- combinations ---------------------------------------
    probs_avg = ((probs_G.astype(np.float64) +
                  probs_LR.astype(np.float64)) / 2).astype(np.float32)
    best_avg, rows_avg = sweep_best(probs_avg)

    lv = label[is_val]
    a_val = s1[is_val]
    fold = (a_val * 2654435761) % 5  # group pairs by S1 entity
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
    print(f"[stack] coefs (p_lgbm, p_lr, bias) = {np.round(w_stack, 4)}",
          flush=True)
    best_st, rows_st = sweep_best(oof)

    # ---------------- report + pick winner --------------------------------
    base = None
    if os.path.exists(f"{ART}/model_prev_val.txt"):
        base = float(open(f"{ART}/model_prev_val.txt").read().strip())
    print("\n==== VALIDATION MACRO-F0.5 (official metric) ====", flush=True)
    print(f"  LGBM (tuned)      tau={best_G[0]:.2f} {best_G[1]:11s} "
          f"F0.5={best_G[2]:.5f}", flush=True)
    print(f"  LogReg standalone tau={best_LR[0]:.2f} {best_LR[1]:11s} "
          f"F0.5={best_LR[2]:.5f}", flush=True)
    print(f"  avg(LGBM,LR)      tau={best_avg[0]:.2f} {best_avg[1]:11s} "
          f"F0.5={best_avg[2]:.5f}", flush=True)
    print(f"  stack(LGBM,LR)    tau={best_st[0]:.2f} {best_st[1]:11s} "
          f"F0.5={best_st[2]:.5f}", flush=True)
    if base:
        print(f"  (previous shipped model: {base:.5f})", flush=True)

    cand = [("lgbm", best_G, probs_G, rows_G),
            ("lr", best_LR, probs_LR, rows_LR),
            ("avg", best_avg, probs_avg, rows_avg),
            ("stack", best_st, oof, rows_st)]
    kind, best, probs_win, rows_win = max(cand, key=lambda x: x[1][2])
    print(f"\nWINNER: {kind}  tau={best[0]:.2f} mode={best[1]} "
          f"F0.5={best[2]:.5f}", flush=True)

    # final LGBM is the winning-config booster; refit stack coefs saved
    with open(f"{ART}/model.pkl", "wb") as f:
        pickle.dump(dict(
            booster=booster,
            tau=best[0], mode=best[1], val_f05=best[2], variant=kind,
            feature_names=KEPT_NAMES,
            feature_names_full=FEATURE_NAMES,
            params={**BASE_PARAMS, **best_G_cfg},
            lr_w=w_lr, lr_mu=mu, lr_sd=sd,
            stack_w=w_stack,
            avg_baseline_f05=best_avg[2],
            stack_val_f05=best_st[2],
            lgbm_val_f05=best_G[2],
            lr_val_f05=best_LR[2],
            hpo_results=results,
        ), f, protocol=4)
    open(f"{ART}/model_prev_val.txt", "w").write(f"{best[2]:.6f}")
    print(f"saved model.pkl ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
