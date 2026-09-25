"""Train the pair matcher + calibrate threshold on the validation split.

Steps:
  1. load pairdata (train): s1/s2 idx, label, is_val
  2. features: memmap built by build_features.py at artifacts/train_feats.npy
  3. subsample negatives (keep all positives) for training
  4. LightGBM binary classifier, early stopping on val AUC
  5. predict val pairs in chunks; sweep threshold; macro-F0.5 on ALL val S1s
  6. also evaluate the "each S2/S3 matched to argmax S1" 1-1 resolution
  7. save model + tau + metadata to artifacts/model.pkl
"""
import pickle
import sys
import time

import numpy as np
import lightgbm as lgb

from blocking import load_data

ART = "artifacts"


def f05(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 1.25 * prec * rec / (0.25 * prec + rec)


def macro_f05_from_pairs(val_s1, val_s2, probs, tau, gold_by_idx, all_val_s1,
                         one_to_one=False, N=None):
    """Build predictions from val pair scores at threshold tau; score macro F0.5.

    gold_by_idx: dict s1_idx -> set(s23_idx) (ground truth, val S1s only)
    all_val_s1: iterable of every val S1 idx (for singleton scoring)
    """
    pred = {int(a): [] for a in all_val_s1}
    keep = probs >= tau
    for a, b, p in zip(val_s1[keep], val_s2[keep], probs[keep]):
        pred[int(a)].append((float(p), int(b)))
    if one_to_one and N:
        best = {}
        for a, lst in pred.items():
            for p, b in lst:
                if b not in best or p > best[b][0]:
                    best[b] = (p, a)
        pred2 = {a: [] for a in pred}
        for b, (p, a) in best.items():
            pred2[a].append(b)
        pred = {a: set(v) for a, v in pred2.items()}
    else:
        pred = {a: {b for _, b in v} for a, v in pred.items()}

    total = 0.0
    n = 0
    for a in all_val_s1:
        a = int(a)
        gold = gold_by_idx.get(a, set())
        pset = pred.get(a, set())
        tp = len(pset & gold)
        fp = len(pset - gold)
        fn = len(gold - pset)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        total += f05(prec, rec)
        n += 1
    return total / max(n, 1)


def main():
    t0 = time.time()
    tag = "train"
    pd_ = np.load(f"{ART}/{tag}_pairdata.npz")
    s1 = pd_["s1_idx"]
    s2 = pd_["s2_idx"]
    label = pd_["label"]
    is_val = pd_["is_val"]
    F = np.load(f"{ART}/{tag}_feats.npy", mmap_mode="r")
    print(f"pairs={len(s1)}, feats={F.shape}", flush=True)

    tr = ~is_val
    va = is_val

    # subsample negatives in the train portion for RAM efficiency
    rng = np.random.default_rng(42)
    tr_idx = np.nonzero(tr)[0]
    pos_tr = tr_idx[label[tr_idx] > 0]
    neg_tr = tr_idx[label[tr_idx] == 0]
    keep_neg = min(len(neg_tr), int(len(pos_tr) * 1.5))
    neg_sel = rng.choice(neg_tr, size=keep_neg, replace=False)
    use = np.sort(np.concatenate([pos_tr, neg_sel]))
    print(f"train rows: {len(use)} ({len(pos_tr)} pos / {keep_neg} neg)",
          flush=True)

    Xtr = np.asarray(F[use])
    ytr = label[use]
    Xva = None  # predicted in chunks

    params = dict(
        objective="binary",
        learning_rate=0.06,
        num_leaves=127,
        min_data_in_leaf=300,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l2=1.0,
        num_threads=22,
        verbosity=-1,
        seed=7,
    )
    dtrain = lgb.Dataset(Xtr, label=ytr)
    with open(f"{ART}/split.pkl", "rb") as f:
        split = pickle.load(f)
    all_val_s1 = split["val_s1_idx"]

    booster = lgb.train(
        params, dtrain, num_boost_round=1200,
        valid_sets=[lgb.Dataset(np.asarray(F[np.nonzero(va)[0][:1000000]]),
                                label=label[np.nonzero(va)[0][:1000000]])],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(60, verbose=True),
                   lgb.log_evaluation(100)],
    )
    print(f"best iter: {booster.best_iteration}", flush=True)

    # ---- predict all val pairs in chunks
    va_idx = np.nonzero(va)[0]
    probs = np.zeros(len(va_idx), dtype=np.float32)
    CH = 500000
    for st in range(0, len(va_idx), CH):
        en = min(st + CH, len(va_idx))
        probs[st:en] = booster.predict(np.asarray(F[va_idx[st:en]]),
                                       num_iteration=booster.best_iteration)
        print(f"  val predict {en}/{len(va_idx)}", flush=True)

    # ---- ground truth by index for val S1s
    d, meta = load_data(tag)
    eid2idx = meta["eid2idx"]
    import csv
    gt_idx = {}
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8",
              newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = eid2idx.get(row["source1_entity_id"])
            if i is None:
                continue
            gt_idx[i] = {eid2idx[m] for m in
                         row["matched_entity_ids"].split(",") if m
                         if m in eid2idx}
    val_s1_pairs = s1[va_idx]
    val_s2_pairs = s2[va_idx]
    val_set = set(int(x) for x in all_val_s1)
    gold_val = {a: g for a, g in gt_idx.items() if a in val_set}

    # ---- threshold sweep
    N = len(d["eid"])
    best = (None, -1, None)
    for tau in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                0.75, 0.80, 0.85, 0.90]:
        s_plain = macro_f05_from_pairs(val_s1_pairs, val_s2_pairs, probs, tau,
                                       gold_val, all_val_s1)
        s_o2o = macro_f05_from_pairs(val_s1_pairs, val_s2_pairs, probs, tau,
                                     gold_val, all_val_s1, one_to_one=True, N=N)
        print(f"tau={tau:.2f}  F0.5={s_plain:.5f}  (1-1: {s_o2o:.5f})",
              flush=True)
        if s_o2o >= s_plain:
            if s_o2o > best[1]:
                best = (tau, s_o2o, "one_to_one")
        else:
            if s_plain > best[1]:
                best = (tau, s_plain, "plain")
    print(f"BEST: tau={best[0]} mode={best[2]} F0.5={best[1]:.5f}", flush=True)

    with open(f"{ART}/model.pkl", "wb") as f:
        pickle.dump(dict(
            booster=booster,
            tau=best[0], mode=best[2], val_f05=best[1],
            feature_names=__import__("features").FEATURE_NAMES,
            params=params,
        ), f, protocol=4)
    print(f"saved model ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
