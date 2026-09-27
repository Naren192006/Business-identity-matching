"""Two-level IVF (inverted-file) approximate nearest-neighbor search, numpy.

Built for the embedding blocking stage: index all S2/S3 record embeddings,
then for each S1 query return the top-k nearest records by cosine.

Structure (classic IVF, two levels so that both the assign pass and the
per-query probe stay cheap at N ~ 10M records):
  level 1: k1 k-means centroids over all indexed records
  level 2: within each level-1 cluster, k2 sub-centroids
           (leaf id = c1 * k2 + c2)
Query: top-n1 level-1 centroids -> within each, top-n2 leaves -> gather leaf
members -> exact cosine refine -> keep top `topk` per query.

All heavy ops are chunked matmuls (numpy BLAS); per-query work is vectorized
ragged gathers, no python loops over records.

Used by src/train_emb.py (negative mining during training) and by its own CLI
  python3 src/emb_candidates.py --build --tag train
  python3 src/emb_candidates.py --run   --tag train|--tag test [--merge]
which emits artifacts/{tag}_pairs_emb.npz (embedding-only candidates) and,
with --merge, artifacts/{tag}_pairs_all.npz (union with key-based pairs).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocking
from blocking import load_data

ART = "artifacts"


def l2n(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------ k-means
def kmeans(X, k, iters=6, seed=5, verbose=True, tag=""):
    """Batched k-means on L2-normalized rows; returns normalized centroids."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = len(X)
    C = X[rng.choice(n, size=min(k, n), replace=False)].astype(np.float32)
    C = l2n(C)
    for it in range(iters):
        s = np.zeros_like(C)
        cnt = np.zeros(len(C), dtype=np.int64)
        for lo in range(0, n, 200000):
            Xc = X[lo:lo + 200000]
            asg = np.argmax(Xc @ C.T, axis=1)
            order = np.argsort(asg, kind="stable")
            a_sorted = asg[order]
            starts = np.searchsorted(a_sorted, np.arange(k))
            ends = np.searchsorted(a_sorted, np.arange(k), side="right")
            nz = np.nonzero(ends > starts)[0]
            for c in nz:
                s[c] += Xc[order[starts[c]:ends[c]]].sum(axis=0)
            cnt += np.bincount(asg, minlength=k)
        empty = cnt == 0
        if empty.any():
            C[empty] = X[rng.choice(n, size=int(empty.sum()), replace=False)]
        C = l2n(np.where(cnt[:, None] > 0, s / np.maximum(cnt, 1)[:, None], C))
        if verbose:
            print(f"  kmeans{tag} iter {it}: empty={int(empty.sum())} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return C.astype(np.float32)


def _assign_chunked(E, C, rows, chunk=1000000):
    """argmax_c E[rows] @ C[c] for all rows (E may be an fp16 memmap)."""
    out = np.zeros(len(rows), dtype=np.int32)
    for lo in range(0, len(rows), chunk):
        r = rows[lo:lo + chunk]
        X = E[r].astype(np.float32)
        out[lo:lo + chunk] = np.argmax(X @ C.T, axis=1).astype(np.int32)
    return out


# ------------------------------------------------------------------ IVF build
def build_ivf(E, kept_rows, out_path, k1=1024, k2=64, sample=200000,
              iters1=6, iters2=3, seed=5, verbose=True):
    """Two-level IVF over E[kept_rows].  Saves npz with:
    cent1 [k1,D], cent2 [k1*k2, D], leaf_rows int32[], leaf_ptr int64[k1*k2+1]
    (leaf rows index RECORD space, i.e. original record ids).
    """
    t0 = time.time()
    kept_rows = np.asarray(kept_rows, dtype=np.int64)
    n = len(kept_rows)
    rng = np.random.default_rng(seed)
    if sample and sample < n:
        samp = kept_rows[rng.choice(n, size=sample, replace=False)]
    else:
        samp = kept_rows
    Xs = l2n(E[samp].astype(np.float32))
    if verbose:
        print(f"ivf: kmeans1 on {len(samp)} rows, k={k1}", flush=True)
    cent1 = kmeans(Xs, k1, iters=iters1, seed=seed, verbose=verbose, tag="1")

    if verbose:
        print(f"ivf: assign level 1 for {n} rows", flush=True)
    c1 = _assign_chunked(E, cent1, kept_rows)

    if verbose:
        print("ivf: kmeans2 per level-1 cluster", flush=True)
    D = E.shape[1]
    cent2 = np.zeros((k1 * k2, D), dtype=np.float32)
    leaf_parts = []
    for c in range(k1):
        rows_c = kept_rows[c1 == c]
        if len(rows_c) == 0:
            continue
        if len(rows_c) > 4000:
            sub = rows_c[rng.choice(len(rows_c), size=4000, replace=False)]
        else:
            sub = rows_c
        Xs2 = l2n(E[sub].astype(np.float32))
        kk = min(k2, max(1, len(rows_c) // 8))
        C2 = kmeans(Xs2, kk, iters=iters2, seed=seed + c, verbose=False)
        if kk < k2:  # pad unused sub-centroids (their leaves stay empty)
            pad = np.zeros((k2 - kk, D), dtype=np.float32)
            C2 = np.concatenate([C2, pad], axis=0)
        cent2[c * k2:(c + 1) * k2] = C2
        c2 = _assign_chunked(E, C2[:kk], rows_c)
        leaf_parts.append((c * k2 + c2, rows_c.astype(np.int32)))
        if verbose and (c % 128 == 0 or c == k1 - 1):
            print(f"  kmeans2 {c+1}/{k1} ({time.time()-t0:.0f}s)", flush=True)

    leaf_ids = np.concatenate([p[0] for p in leaf_parts])
    leaf_rows = np.concatenate([p[1] for p in leaf_parts])
    order = np.argsort(leaf_ids, kind="stable")
    leaf_rows = leaf_rows[order]
    leaf_ids = leaf_ids[order]
    leaf_ptr = np.zeros(k1 * k2 + 1, dtype=np.int64)
    cnt = np.bincount(leaf_ids, minlength=k1 * k2)
    leaf_ptr[1:] = np.cumsum(cnt)
    np.savez(out_path, cent1=cent1, cent2=cent2, leaf_rows=leaf_rows,
             leaf_ptr=leaf_ptr, k1=np.int64(k1), k2=np.int64(k2))
    if verbose:
        sz = cnt[cnt > 0]
        print(f"ivf built: {len(sz)} leaves, mean leaf size {sz.mean():.0f} "
              f"p90 {np.percentile(sz, 90):.0f} ({time.time()-t0:.0f}s) "
              f"-> {out_path}", flush=True)


# ------------------------------------------------------------------ query
def query_ivf(E, ivf, Q, n1=2, n2=4, topk=50, cutoff=None, self_rows=None,
              chunk=1024, verbose=True):
    """Top-`topk` indexed records per query row of Q (cosine).

    Returns (qs, rows, cos) concatenated across chunks; per query results are
    sorted by descending cosine.  self_rows[i] (optional) is excluded from
    query i's results.  cutoff drops candidates with cos < cutoff.
    """
    cent1 = ivf["cent1"]
    cent2 = ivf["cent2"]
    leaf_rows = ivf["leaf_rows"]
    leaf_ptr = ivf["leaf_ptr"]
    k2 = int(ivf["k2"])
    nq = len(Q)
    Qn = l2n(Q.astype(np.float32))
    out_q, out_r, out_s = [], [], []
    t0 = time.time()
    for lo in range(0, nq, chunk):
        hi = min(lo + chunk, nq)
        nl = hi - lo
        Qc = Qn[lo:hi]
        # --- level 1: top n1 centroids per query
        s1 = Qc @ cent1.T
        n1_i = min(n1, cent1.shape[0])
        sel1 = np.argpartition(-s1, n1_i - 1, axis=1)[:, :n1_i]
        # --- level 2: within each selected L1 cluster, top n2 leaves
        leaf_sel = np.empty((nl, n1_i * n2), dtype=np.int64)
        C2r = cent2.reshape(-1, k2, cent2.shape[1])   # [k1, k2, D]
        for j in range(n1_i):
            c1s = sel1[:, j]
            s2 = np.einsum("qd,qcd->qc", Qc, C2r[c1s])
            n2_i = min(n2, k2)
            sel2 = np.argpartition(-s2, n2_i - 1, axis=1)[:, :n2_i]
            leaf_sel[:, j * n2:(j + 1) * n2] = c1s[:, None] * k2 + sel2
        leaf_sel = np.unique(leaf_sel, axis=1)  # dedupe overlapping leaves
        # --- gather leaf members into per-query contiguous segments
        sizes = (leaf_ptr[leaf_sel + 1] - leaf_ptr[leaf_sel])  # [nl, L]
        L = leaf_sel.shape[1]
        flat_sizes = sizes.ravel()
        keep_leaf = flat_sizes > 0
        nc = sizes.sum(axis=1)
        total = int(nc.sum())
        if total == 0:
            continue
        # leaf-major order: q0-leaf0 members, then q0-leaf1, ...
        rep_leaf = leaf_sel.ravel()[keep_leaf]
        starts = leaf_ptr[rep_leaf]
        lens = flat_sizes[keep_leaf]
        cum = np.zeros(len(lens) + 1, dtype=np.int64)
        cum[1:] = np.cumsum(lens)
        gpos = np.repeat(starts - cum[:-1], lens) + np.arange(int(cum[-1]),
                                                              dtype=np.int64)
        rows = leaf_rows[gpos]
        # query id per gathered slot: leaf slot q = rep pattern of leaf_sel
        slot_q = np.repeat(np.arange(nl, dtype=np.int64), L)[keep_leaf]
        qid = slot_q[np.repeat(np.arange(len(lens), dtype=np.int64), lens)]
        # --- refine: exact cosine (E rows pre-normalized fp16)
        qrep = Qn[lo + qid]
        cos = np.einsum("ij,ij->i", E[rows].astype(np.float32), qrep)
        if self_rows is not None:
            m = rows != self_rows[lo + qid]
            rows, cos, qid = rows[m], cos[m], qid[m]
        if cutoff is not None:
            m = cos >= cutoff
            rows, cos, qid = rows[m], cos[m], qid[m]
        # --- top-k per query: sort by (qid, -cos), keep first topk per group
        order = np.lexsort((-cos, qid))
        rows, cos, qid = rows[order], cos[order], qid[order]
        newq = np.concatenate(([True], qid[1:] != qid[:-1]))
        grp = np.cumsum(newq) - 1
        gstart = np.nonzero(newq)[0]  # start offset of each group
        rank = np.arange(len(qid), dtype=np.int64) - gstart[grp]
        keep = rank < topk
        out_q.append(qid[keep] + lo)
        out_r.append(rows[keep])
        out_s.append(cos[keep])
        if verbose and (lo // chunk) % 50 == 0:
            print(f"  query {hi}/{nq} ({time.time()-t0:.0f}s)", flush=True)
    if not out_q:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0, dtype=np.float32)
    return (np.concatenate(out_q), np.concatenate(out_r),
            np.concatenate(out_s))


# ------------------------------------------------------------------ eval util
def macro_recall_at_k(E, val_anchors, gold_map, ks=(10, 25, 50, 100),
                      n_eval=3000, seed=7, verbose=True):
    """Exact-search recall@k for sampled anchors (upper bound of the ANN).

    gold_map: dict anchor_row -> np.array of gold record rows.
    Returns dict k -> (macro fraction of gold in top-k, any-gold@k).
    """
    rng = np.random.default_rng(seed)
    anc = np.array(sorted(val_anchors), dtype=np.int64)
    if len(anc) > n_eval:
        anc = np.sort(rng.choice(anc, size=n_eval, replace=False))
    N = E.shape[0]
    kmax = max(ks)
    hits = {k: [0.0, 0, 0] for k in ks}  # [sum frac, n_gold_total, n_any]
    n_scored = 0
    t0 = time.time()
    B = 512
    for lo in range(0, len(anc), B):
        A = anc[lo:lo + B]
        Q = l2n(E[A].astype(np.float32))
        sims_top = np.zeros((len(A), kmax), dtype=np.float32)
        sims_idx = np.zeros((len(A), kmax), dtype=np.int64)
        for rlo in range(0, N, 1000000):
            X = E[rlo:rlo + 1000000].astype(np.float32)
            S = Q @ X.T
            part = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
            vals = np.take_along_axis(S, part, axis=1)
            idx = part + rlo
            if rlo == 0:
                sims_top, sims_idx = vals, idx
            else:
                merged_v = np.concatenate([sims_top, vals], axis=1)
                merged_i = np.concatenate([sims_idx, idx], axis=1)
                sel = np.argsort(-merged_v, axis=1)[:, :kmax]
                sims_top = np.take_along_axis(merged_v, sel, axis=1)
                sims_idx = np.take_along_axis(merged_i, sel, axis=1)
        for r, a in enumerate(A):
            g = gold_map.get(int(a))
            if g is None or len(g) == 0:
                continue
            gset = set(g.tolist())
            n_scored += 1
            for k in ks:
                got = sum(1 for x in sims_idx[r, :k] if int(x) in gset)
                hits[k][0] += got / len(g)
                hits[k][1] += len(g)
                hits[k][2] += got > 0
    out = {}
    for k in ks:
        s, tot, anyc = hits[k]
        out[k] = (s / max(tot, 1), anyc / max(n_scored, 1))
    if verbose:
        for k in ks:
            print(f"  emb exact recall@{k}: macro={out[k][0]:.4f} "
                  f"any={out[k][1]:.4f}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return out


# ------------------------------------------------------------------ shared
def gold_map_from_gt(eid2idx):
    import csv
    gm = {}
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8",
              newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = eid2idx.get(row["source1_entity_id"])
            if i is None:
                continue
            g = [eid2idx[m] for m in row["matched_entity_ids"].split(",")
                 if m and m in eid2idx]
            if g:
                gm[i] = np.array(g, dtype=np.int64)
    return gm


def src_from_eid(d):
    if "src" in d:
        return d["src"]
    eid = d["eid"]
    return np.fromiter((0 if e.startswith("S1-") else 1 for e in eid),
                       dtype=np.int8, count=len(eid))


def pair_recall(pk, gk):
    """Fraction of gold pair keys gk present in sorted-unique candidates pk."""
    if len(gk) == 0 or len(pk) == 0:
        return 0.0
    hit = np.searchsorted(pk, gk)
    hit[hit >= len(pk)] = len(pk) - 1
    return float((pk[hit] == gk).mean())


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build IVF index")
    ap.add_argument("--run", action="store_true", help="generate candidates")
    ap.add_argument("--tag", default="train")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n1", type=int, default=2)
    ap.add_argument("--n2", type=int, default=4)
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--merge", action="store_true",
                    help="also write {tag}_pairs_all.npz = union with keys")
    ap.add_argument("--k1", type=int, default=1024)
    ap.add_argument("--k2", type=int, default=64)
    args = ap.parse_args()

    d, meta = load_data(args.tag)
    eid = d["eid"]
    src = src_from_eid(d)
    N = len(eid)
    E = np.load(f"{ART}/emb_E_{args.tag}.npy", mmap_mode="r")

    if args.build:
        s23 = np.nonzero(src == 1)[0]
        print(f"indexing {len(s23)} S2/S3 records", flush=True)
        build_ivf(E, s23, f"{ART}/emb_ivf_{args.tag}.npz", k1=args.k1,
                  k2=args.k2)
        return

    ivf = dict(np.load(f"{ART}/emb_ivf_{args.tag}.npz"))
    s1_rows = np.nonzero(src == 0)[0]
    print(f"{args.tag}: {len(s1_rows)} S1 queries", flush=True)

    # fresh S1 embeddings with the current encoder weights
    import emb_model as em
    enc = em.BiEncoder.load(f"{ART}/emb.npz")
    blob, offs = em.build_text_blob(blocking.RecData(d, meta), meta, args.tag)
    Q, _, _, _ = enc.embed_batches(blob, offs, s1_rows)

    qs, rows, cos = query_ivf(E, ivf, Q, n1=args.n1, n2=args.n2,
                              topk=args.topk, cutoff=args.cutoff,
                              self_rows=s1_rows)
    print(f"emb candidates: {len(qs)} pairs "
          f"({len(qs)/max(len(s1_rows),1):.1f} per S1)", flush=True)
    np.savez(f"{ART}/{args.tag}_pairs_emb.npz", s1=s1_rows[qs], s2=rows)

    N64 = np.int64(N)
    if args.merge:
        kp = np.load(f"{ART}/{args.tag}_pairs.npz")
        k1_, k2_ = kp["s1"].astype(np.int64), kp["s2"].astype(np.int64)
        allk = np.unique(np.concatenate([k1_ * N64 + k2_,
                                         s1_rows[qs].astype(np.int64) * N64
                                         + rows.astype(np.int64)]))
        ua, ub = (allk // N64).astype(np.int64), (allk % N64).astype(np.int64)
        np.savez(f"{ART}/{args.tag}_pairs_all.npz", s1=ua, s2=ub)
        print(f"union with key pairs: {len(k1_)} + {len(qs)} -> {len(ua)} "
              f"unique", flush=True)

    if args.tag == "train":
        gm = gold_map_from_gt(meta["eid2idx"])
        gk = (np.concatenate([g * N64 + v for g, v in gm.items()])
              if gm else np.zeros(0, dtype=np.int64))
        gk.sort()
        pk = np.unique(s1_rows[qs].astype(np.int64) * N64
                       + rows.astype(np.int64))
        print(f"emb pair recall: {pair_recall(pk, gk):.4%} of gold pairs",
              flush=True)
        if args.merge:
            pa = np.unique(ua * N64 + ub)
            print(f"union pair recall: {pair_recall(pa, gk):.4%} of gold "
                  f"pairs", flush=True)
        val = macro_recall_at_k(E, list(gm.keys()), gm, ks=(10, 25, 50, 100))
        print("RECALL@K exact (val anchors):",
              {k: tuple(round(x, 4) for x in v) for k, v in val.items()},
              flush=True)


if __name__ == "__main__":
    main()
