"""Staged experiment driver. Each stage checkpoint-pickles results into
artifacts/exp/ so the pipeline survives terminal timeouts -- rerun with
--stage N to resume from any checkpoint.

Stages:
  1  baseline probs (current model) + fine tau sweep, both modes
  2  train LR-A (standardized 41 features, neg:pos 3:1) -> val probs
  3  HPO random search (AP early stopping, select by val macro-F0.5)
  4  combos: probability averaging + stacked LR (grouped CV); fine sweeps;
     pick the winner and save artifacts/final_choice.pkl
"""
import argparse
import gc
import os
import pickle
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_lib import (ValScorer, average_precision, best_from_sweep,
                     fine_taus, fit_lr, gold_counts_full, load_split,
                     load_train_pairs, lr_predict, add_intercept, ap_feval)

ART = "artifacts"
EXP = f"{ART}/exp"
os.makedirs(EXP, exist_ok=True)

DROP = {"country_eq", "dom_any", "name_exact_norm", "a_addr_empty"}

BASE_PARAMS = dict(
    objective="binary",
    num_threads=22,
    verbosity=-1,
    seed=7,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
)

HPO_GRID = dict(
    neg_pos_ratio=[1.5, 2.0, 3.0, 4.0, 6.0],
    num_leaves=[31, 63, 127, 255],
    learning_rate=[0.05, 0.08, 0.1],
    min_data_in_leaf=[100, 300, 800],
    lambda_l1=[0.0, 0.5, 2.0],
    lambda_l2=[0.0, 1.0, 5.0],
    feature_fraction=[0.7, 0.85, 1.0],
    bagging_fraction=[0.7, 0.85, 1.0],
)


def keep_mask():
    from features import FEATURE_NAMES
    return np.array([n not in DROP for n in FEATURE_NAMES])


def load_feats(tag):
    F = np.load(f"{ART}/{tag}_feats.npy", mmap_mode="r")
    km = keep_mask()
    return F, km


def _ckpt(name, obj):
    with open(f"{EXP}/{name}", "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  [ckpt] {EXP}/{name}", flush=True)


def _load(name):
    with open(f"{EXP}/{name}", "rb") as f:
        return pickle.load(f)


def predict_chunked(booster, F, rows, km, chunk=1_000_000, tag=""):
    out = np.zeros(len(rows), dtype=np.float32)
    t0 = time.time()
    for st in range(0, len(rows), chunk):
        en = min(st + chunk, len(rows))
        X = np.asarray(F[rows[st:en]][:, km], dtype=np.float32)
        out[st:en] = booster.predict(X, num_iteration=booster.best_iteration)
        print(f"    predict {en}/{len(rows)} ({time.time()-t0:.0f}s)",
              flush=True)
    return out


def sample_train(label, is_val, ratio, seed=42):
    rng = np.random.default_rng(seed)
    tr_idx = np.nonzero(~is_val)[0]
    pos = tr_idx[label[tr_idx] > 0]
    neg = tr_idx[label[tr_idx] == 0]
    n_neg = min(len(neg), int(len(pos) * ratio))
    sel = rng.choice(neg, size=n_neg, replace=False)
    return np.sort(np.concatenate([pos, sel]))


# ------------------------------- stage 1 --------------------------------

def stage1():
    t0 = time.time()
    s1, s2, label, is_val = load_train_pairs()
    split = load_split()
    scorer = ValScorer(s1[is_val], s2[is_val], label[is_val], split,
                   gold_counts_full(split))
    del s1, s2, label, is_val
    gc.collect()
    F, km = load_feats("train")
    with open(f"{ART}/model.pkl", "rb") as f:
        M = pickle.load(f)
    if os.path.exists(f"{EXP}/meta.pkl"):
        va_idx = _load("meta.pkl")["va_idx"]
    else:
        va_idx = np.nonzero(_va_mask())[0]
        _ckpt("meta.pkl", dict(va_idx=va_idx))
    # the old booster was trained on all 43 columns -> no feature drop here
    if os.path.exists(f"{EXP}/probs_base.npy"):
        probs = np.load(f"{EXP}/probs_base.npy")
        print("    reusing existing probs_base.npy", flush=True)
    else:
        probs = predict_chunked(M["booster"], F, va_idx,
                                np.ones(F.shape[1], dtype=bool))
        print(f"    probs sanity: min={probs.min():.5f} "
              f"max={probs.max():.5f} mean={probs.mean():.5f}", flush=True)
    np.save(f"{EXP}/probs_base.npy", probs)
    taus = fine_taus()
    rows = []
    for tau in taus:
        rows.append({"tau": tau,
                     "plain": scorer.score_fast(probs, tau, "plain"),
                     "one_to_one": scorer.score_fast(probs, tau,
                                                     "one_to_one")})
    best = best_from_sweep(rows)
    _ckpt("stage1.pkl", dict(rows=rows, best=best))
    print(f"[stage1] base model fine sweep best: tau={best[0]} "
          f"mode={best[1]} F0.5={best[2]:.5f}  ({time.time()-t0:.0f}s)",
          flush=True)
    print(f"[stage1] note: old reported 0.7837 used a scorer that gave 0.0 "
          f"to correctly-predicted singletons; this scorer gives 1.0 per "
          f"the official metric", flush=True)
    return best


def _va_mask():
    s1, s2, label, is_val = load_train_pairs()
    return is_val


# ------------------------------- stage 2 --------------------------------

def stage2():
    t0 = time.time()
    s1, s2, label, is_val = load_train_pairs()
    F, km = load_feats("train")
    va_idx = np.nonzero(is_val)[0]
    _ckpt("meta.pkl", dict(va_idx=va_idx))
    tr_idx = sample_train(label, is_val, 3.0)
    print(f"[stage2] train rows: {len(tr_idx)}", flush=True)
    # standardization from 1M random train rows
    rng = np.random.default_rng(0)
    sub = rng.choice(tr_idx, size=min(1_000_000, len(tr_idx)),
                     replace=False)
    Xs = np.asarray(F[sub][:, km], dtype=np.float64)
    mu = Xs.mean(axis=0)
    sd = Xs.std(axis=0)
    sd[sd < 1e-8] = 1.0
    del Xs
    # materialize standardized train matrix (float32)
    Xtr = np.empty((len(tr_idx), int(km.sum()) + 1), dtype=np.float32)
    CH = 2_000_000
    for st in range(0, len(tr_idx), CH):
        en = min(st + CH, len(tr_idx))
        Xtr[st:en, :-1] = ((np.asarray(F[tr_idx[st:en]][:, km],
                                       dtype=np.float32) - mu) / sd)
        Xtr[st:en, -1] = 1.0
        print(f"    materialize {en}/{len(tr_idx)}", flush=True)
    ytr = label[tr_idx].astype(np.float64)
    ndim = Xtr.shape[1]
    w = fit_lr(Xtr, ytr, l2=1.0, iters=8)
    del Xtr
    gc.collect()
    # predict val
    Xva = np.empty((len(va_idx), ndim), dtype=np.float32)
    for st in range(0, len(va_idx), CH):
        en = min(st + CH, len(va_idx))
        Xva[st:en, :-1] = ((np.asarray(F[va_idx[st:en]][:, km],
                                       dtype=np.float32) - mu) / sd)
        Xva[st:en, -1] = 1.0
        print(f"    val materialize {en}/{len(va_idx)}", flush=True)
    probs = lr_predict(Xva, w)
    np.save(f"{EXP}/probs_lr.npy", probs)
    _ckpt("stage2.pkl", dict(w=w, mu=mu, sd=sd, l2=1.0, ratio=3.0))
    print(f"[stage2] LR-A done ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------- stage 3 --------------------------------

def sample_configs(n, seed=123):
    rng = np.random.default_rng(seed)
    keys = list(HPO_GRID)
    seen = set()
    out = []
    while len(out) < n:
        c = {k: HPO_GRID[k][rng.integers(len(HPO_GRID[k]))] for k in keys}
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def stage3(n_configs=20):
    t0 = time.time()
    s1, s2, label, is_val = load_train_pairs()
    split = load_split()
    scorer = ValScorer(s1[is_val], s2[is_val], label[is_val], split,
                   gold_counts_full(split))
    del s1, s2
    gc.collect()
    F, km = load_feats("train")
    meta = _load("meta.pkl") if os.path.exists(f"{EXP}/meta.pkl") else None
    va_idx = meta["va_idx"] if meta else np.nonzero(is_val)[0]
    if meta is None:
        _ckpt("meta.pkl", dict(va_idx=va_idx))
    # fixed val subsample for early stopping
    rng = np.random.default_rng(99)
    vs = rng.choice(len(va_idx), size=min(2_000_000, len(va_idx)),
                    replace=False)
    Xva_es = np.asarray(F[va_idx[vs]][:, km], dtype=np.float32)
    yva_es = label[va_idx[vs]].astype(np.float32)

    results = []
    if os.path.exists(f"{EXP}/stage3.pkl"):
        results = _load("stage3.pkl")
        print(f"[stage3] resuming with {len(results)} done", flush=True)
    taus = fine_taus()
    for i, cfg in enumerate(sample_configs(n_configs)):
        if i < len(results):
            continue
        tc = time.time()
        tr_idx = sample_train(label, is_val, cfg["neg_pos_ratio"])
        n_rows = len(tr_idx)
        if n_rows > 36_000_000:  # RAM guard for extreme ratios
            rng2 = np.random.default_rng(1)
            tr_idx = rng2.choice(tr_idx, size=36_000_000, replace=False)
            tr_idx.sort()
        Xtr = np.empty((len(tr_idx), int(km.sum())), dtype=np.float32)
        CH = 2_000_000
        for st in range(0, len(tr_idx), CH):
            en = min(st + CH, len(tr_idx))
            Xtr[st:en] = np.asarray(F[tr_idx[st:en]][:, km],
                                    dtype=np.float32)
        params = dict(BASE_PARAMS)
        for k in ("num_leaves", "learning_rate", "min_data_in_leaf",
                  "lambda_l1", "lambda_l2", "feature_fraction",
                  "bagging_fraction"):
            params[k] = cfg[k]
        dtrain = lgb.Dataset(Xtr, label=label[tr_idx])
        try:
            booster = lgb.train(
                params, dtrain, num_boost_round=1500,
                valid_sets=[lgb.Dataset(Xva_es, label=yva_es)],
                valid_names=["val"],
                feval=ap_feval,
                callbacks=[lgb.early_stopping(60, first_metric_only=True,
                                              verbose=False),
                           lgb.log_evaluation(0)],
            )
        except MemoryError:
            print(f"[stage3] cfg {i}: OOM, skipping", flush=True)
            results.append(dict(i=i, cfg=cfg, best_iter=0, ap=0.0,
                                f05=0.0, tau=0.5, mode="plain", error="oom"))
            _ckpt("stage3.pkl", results)
            continue
        del Xtr, dtrain
        gc.collect()
        probs = predict_chunked(booster, F, va_idx, km)
        np.save(f"{EXP}/probs_hp{i}.npy", probs)
        rows = [{"tau": t,
                 "plain": scorer.score_fast(probs, t, "plain"),
                 "one_to_one": scorer.score_fast(probs, t, "one_to_one")}
                for t in taus]
        best = best_from_sweep(rows)
        ap = booster.best_score["val"]["ap"]
        results.append(dict(i=i, cfg=cfg, best_iter=booster.best_iteration,
                            ap=ap, f05=best[2], tau=best[0], mode=best[1]))
        _ckpt("stage3.pkl", results)
        print(f"[stage3] cfg {i+1}/{n_configs}: F0.5={best[2]:.5f} "
              f"AP={ap:.5f} iters={booster.best_iteration} "
              f"({time.time()-tc:.0f}s) cfg={cfg}", flush=True)
        del booster, probs
        gc.collect()
    results.sort(key=lambda r: -r["f05"])
    print("[stage3] top 5 configs by val macro-F0.5:", flush=True)
    for r in results[:5]:
        print(f"  F0.5={r['f05']:.5f} AP={r['ap']:.5f} "
              f"tau={r['tau']} mode={r['mode']} cfg={r['cfg']}", flush=True)


# ------------------------------- stage 4 --------------------------------

def stage4():
    t0 = time.time()
    s1, s2, label, is_val = load_train_pairs()
    split = load_split()
    scorer = ValScorer(s1[is_val], s2[is_val], label[is_val], split,
                   gold_counts_full(split))
    va_idx = _load("meta.pkl")["va_idx"]
    taus = fine_taus()

    results = _load("stage3.pkl")
    bestcfg = results[0]  # sorted by f05 in stage3
    probs_G = np.load(f"{EXP}/probs_hp{bestcfg['i']}.npy")
    probs_LR = np.load(f"{EXP}/probs_lr.npy")
    probs_avg = ((probs_G.astype(np.float64) +
                  probs_LR.astype(np.float64)) / 2).astype(np.float32)

    # ---- stacking LR on [pG, pLR], grouped 5-fold CV by S1 entity
    a_val = s1[is_val]
    fold = (a_val * 2654435761) % 5  # deterministic hash-ish fold
    Xs = np.concatenate([probs_G[:, None], probs_LR[:, None],
                         np.ones((len(probs_G), 1), dtype=np.float32)],
                        axis=1)
    oof = np.zeros(len(Xs), dtype=np.float32)
    for k in range(5):
        trm = fold != k
        tem = ~trm
        w = fit_lr(Xs[trm], label[is_val][trm], l2=1e-3, iters=15,
                   verbose=False)
        oof[tem] = lr_predict(Xs[tem], w)
        print(f"    stack fold {k}: coefs={np.round(w, 3)}", flush=True)
    w_full = fit_lr(Xs, label[is_val], l2=1e-3, iters=15, verbose=False)
    print(f"    stack full-val coefs={np.round(w_full, 3)}", flush=True)
    probs_stack = oof

    def sweep_best(probs):
        rows = [{"tau": t,
                 "plain": scorer.score_fast(probs, t, "plain"),
                 "one_to_one": scorer.score_fast(probs, t, "one_to_one")}
                for t in taus]
        return best_from_sweep(rows), rows

    b_G, rows_G = sweep_best(probs_G)
    b_LR, rows_LR = sweep_best(probs_LR)
    b_avg, rows_avg = sweep_best(probs_avg)
    b_st, rows_st = sweep_best(probs_stack)
    b_base, rows_base = sweep_best(np.load(f"{EXP}/probs_base.npy"))

    table = dict(
        base_single=b_base, G_single=b_G, LR_single=b_LR,
        avg=b_avg, stack=b_st,
        rows=dict(G=rows_G, LR=rows_LR, avg=rows_avg, stack=rows_st,
                  base=rows_base),
        stack_coefs=w_full.tolist(),
        best_cfg=bestcfg,
    )
    _ckpt("stage4.pkl", table)
    print("\n==== FINAL COMPARISON (validation macro-F0.5) ====", flush=True)
    for name, b in [("old model (43 feats)", b_base),
                    ("tuned LGBM single", b_G),
                    ("logistic regression", b_LR),
                    ("avg(LGBM,LR)", b_avg),
                    ("stack(LGBM,LR)", b_st)]:
        print(f"  {name:24s} tau={b[0]:.2f} mode={b[1]:11s} "
              f"F0.5={b[2]:.5f}", flush=True)
    cand = [("single", b_G, probs_G), ("avg", b_avg, probs_avg),
            ("stack", b_st, probs_stack)]
    winner = max(cand, key=lambda x: x[1][2])
    print(f"\n  WINNER: {winner[0]} (F0.5={winner[1][2]:.5f})", flush=True)
    _ckpt("final_choice.pkl", dict(
        kind=winner[0], tau=winner[1][0], mode=winner[1][1],
        val_f05=winner[1][2],
        best_cfg=bestcfg, stack_coefs=w_full.tolist()))
    print(f"[stage4] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--n-configs", type=int, default=20)
    args = ap.parse_args()
    if args.stage == 1:
        stage1()
    elif args.stage == 2:
        stage2()
    elif args.stage == 3:
        stage3(args.n_configs)
    elif args.stage == 4:
        stage4()
    else:
        print("usage: run_exp.py --stage {1,2,3,4}")
