"""Shared experiment utilities: fast validation scoring, tau sweeps, LR, blends.

Scoring matches the official metric exactly:
  - gold counts come from the FULL ground-truth file (not just blocked pairs),
    so blocking misses count as false negatives;
  - every val S1 entity is scored, singletons included;
  - a singleton (no gold) predicted empty scores 1.0.
Per-S1 TP/kept counts are bincounts; the one-to-one mode sorts pairs by
(b, -p) once per probability array and scans for the first kept pair per b.
"""
import csv
import pickle

import numpy as np

ART = "artifacts"


def load_train_pairs():
    z = np.load(f"{ART}/train_pairdata.npz")
    return (z["s1_idx"].astype(np.int64), z["s2_idx"].astype(np.int64),
            z["label"], z["is_val"])


def gold_counts_full(split):
    """Per-val-entity gold match counts from the FULL ground-truth file.

    Returns int64[na] aligned with sorted split['val_s1_idx'].  This is the
    official recall denominator: gold matches that blocking never generated
    as candidates still count against recall.
    """
    with open(f"{ART}/train_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    eid2idx = meta["eid2idx"]
    val_s1_sorted = np.sort(split["val_s1_idx"])
    na = len(val_s1_sorted)
    counts = np.zeros(na, dtype=np.int64)
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8",
              newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = eid2idx.get(row["source1_entity_id"])
            if i is None:
                continue
            pos = np.searchsorted(val_s1_sorted, i)
            if pos < na and val_s1_sorted[pos] == i:
                counts[pos] = sum(
                    1 for m in row["matched_entity_ids"].split(",")
                    if m and m in eid2idx)
    return counts


def load_split():
    with open(f"{ART}/split.pkl", "rb") as f:
        return pickle.load(f)


def f05_vec(prec, rec):
    num = 1.25 * prec * rec
    den = 0.25 * prec + rec
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    return out


class ValScorer:
    """Official-metric-faithful macro-F0.5 scorer over all val S1 entities.

    gold_counts_full: int64[na] aligned with sorted val_s1_idx -- number of
    ground-truth matches per val entity FROM THE GT FILE (may exceed the
    gold pairs present in the candidate set).
    """

    def __init__(self, s1v, s2v, labelv, split, gold_counts_full):
        self.s1 = s1v
        self.s2 = s2v
        self.gold = labelv > 0
        val_s1_idx = split["val_s1_idx"]
        self.val_s1_sorted = np.sort(val_s1_idx)
        na = len(self.val_s1_sorted)
        self.na = na
        assert len(gold_counts_full) == na
        self.gold_count = np.asarray(gold_counts_full, dtype=np.int64)
        # every val S1 entity is scored (singletons included)
        self.has_entity = np.ones(na, dtype=bool)
        # position of each pair's S1 in the val-S1 array (na = dump bucket for
        # pairs whose S1 is not a val entity)
        pos = np.searchsorted(self.val_s1_sorted, s1v)
        pos_clipped = np.minimum(pos, na - 1)
        ok = self.val_s1_sorted[pos_clipped] == s1v
        self.apos = np.where(ok, pos_clipped, na).astype(np.int64)
        self._cache = {}

    def _order_for(self, p):
        k = id(p)
        ent = self._cache.get(k)
        if ent is None:
            if len(self._cache) >= 1:  # keep only the latest (RAM)
                self._cache.clear()
            ord_ = np.lexsort((-p.astype(np.float64), self.s2))
            s2o = self.s2[ord_]
            newblock = np.concatenate(
                ([True], s2o[1:] != s2o[:-1]))
            ent = (ord_, self.apos[ord_], self.gold[ord_], newblock)
            self._cache[k] = ent
        return ent

    def score_fast(self, p, tau, mode):
        keep = p >= tau
        na = self.na
        if mode == "plain":
            a = self.apos
            g = self.gold & (a < na)
            tp = np.bincount(a[keep & g], minlength=na + 1)[:na]
            kept = np.bincount(a[keep & (a < na)], minlength=na + 1)[:na]
        else:
            ord_, ao, go, newblock = self._order_for(p)
            ks = keep[ord_]
            prev = np.concatenate(([False], ks[:-1]))
            win = ks & (newblock | ~prev)  # first kept pair per b-block
            aa = ao[win]
            gg = go[win]
            tp = np.bincount(aa[gg & (aa < na)], minlength=na + 1)[:na]
            kept = np.bincount(aa[aa < na], minlength=na + 1)[:na]
        prec = np.where(kept > 0, tp / np.maximum(kept, 1), 1.0)
        rec = np.where(self.gold_count > 0,
                       tp / np.maximum(self.gold_count, 1), 1.0)
        s = f05_vec(prec, rec)
        return float(s[self.has_entity].mean())


def sweep(scorer, p, taus, modes=("plain", "one_to_one")):
    rows = []
    for tau in taus:
        row = {"tau": tau}
        for m in modes:
            row[m] = scorer.score_fast(p, tau, m)
        rows.append(row)
    return rows


def best_from_sweep(rows):
    best = None
    for r in rows:
        for m in ("plain", "one_to_one"):
            if best is None or r[m] > best[2]:
                best = (r["tau"], m, r[m])
    return best  # (tau, mode, score)


def fine_taus(lo=0.20, hi=0.99, step=0.01):
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def print_sweep(rows, front=None):
    print(f"{'tau':>6} | {'plain':>8} {'1-1':>8}", flush=True)
    for r in rows:
        best_here = max(r["plain"], r["one_to_one"])
        mark = "  <-- BEST" if front and best_here >= front[2] - 1e-12 else ""
        print(f"{r['tau']:6.2f} | {r['plain']:8.5f} "
              f"{r['one_to_one']:8.5f}{mark}", flush=True)


# ---------------- logistic regression (IRLS, numpy only) ----------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def standardize_stats(X, seed=0, n=1_000_000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xs = np.asarray(X[idx], dtype=np.float64)
    mu = Xs.mean(axis=0)
    sd = Xs.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def add_intercept(X):
    return np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)


def fit_lr(X, y, l2=1.0, iters=8, chunk=2_000_000, verbose=True):
    """IRLS logistic regression; X already standardized + intercept column."""
    d = X.shape[1]
    w = np.zeros(d, dtype=np.float64)
    reg = np.eye(d) * l2
    reg[-1, -1] = 0.0  # no penalty on intercept
    y = y.astype(np.float64)
    for it in range(iters):
        grad = np.zeros(d)
        H = np.zeros((d, d))
        ll = 0.0
        for st in range(0, len(X), chunk):
            Xc = X[st:st + chunk]
            yc = y[st:st + chunk]
            m = Xc @ w
            p = _sigmoid(m)
            Wc = p * (1 - p)
            grad += Xc.T @ (yc - p)
            H += (Xc * Wc[:, None]).T @ Xc
            ll += -np.sum(yc * np.log(np.maximum(p, 1e-12)) +
                          (1 - yc) * np.log(np.maximum(1 - p, 1e-12)))
        step = np.linalg.solve(H + reg, grad)
        w += step
        gn = float(np.max(np.abs(step)))
        if verbose:
            print(f"  lr iter {it}: nll={ll/len(X):.5f} maxstep={gn:.2e}",
                  flush=True)
        if gn < 1e-7:
            break
    return w


def lr_predict(Xw, w, chunk=2_000_000):
    out = np.zeros(len(Xw), dtype=np.float32)
    for st in range(0, len(Xw), chunk):
        out[st:st + chunk] = _sigmoid(Xw[st:st + chunk] @ w)
    return out


def average_precision(y, p):
    """Numpy average precision (no sklearn)."""
    y = y.astype(np.float64)
    order = np.argsort(-p, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / (np.arange(len(y)) + 1)
    rec = tp / max(y.sum(), 1)
    d = np.diff(np.concatenate([[0.0], rec]))
    return float(np.sum(prec * d))


def ap_feval(preds, dataset):
    y = dataset.get_label()
    return "ap", average_precision(y, preds), True
