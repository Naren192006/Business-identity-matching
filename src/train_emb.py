"""Train the bi-encoder for embedding-based blocking (numpy-only, ADEN-style).

What it does
------------
1. Cache per-record text blobs (artifacts/train_emb_text.npz) once.
2. Build the encoder (char/word n-gram hashing -> linear -> normalize).
3. Compute idf per hash bucket (document frequency over all records).
4. Training loop (sampled-softmax InfoNCE, a.k.a. ADEN-style contrastive):
     anchors   = S1 records with >=1 gold match (train half of the split)
     positives = one uniformly-sampled gold match per anchor
     negatives = stored hard negatives (IVF-mined) + in-batch (via softmax
                 denominator over all anchors' positives/negatives in batch)
     loss      = -log softmax(anchor . [pos, negs] / T)[0]
   Backward: d loss/d W = sum over involved records of d e_r/d W, chained
   through L2-normalization, scattered to hash buckets (see train_step).
5. Every --neg-rounds epochs the IVF is rebuilt with the current encoder and
   fresh hard negatives are mined (hard-negative curriculum).
6. Eval on the val split: IVF -> query -> per-entity macro recall (the same
   metric family used by cand_tradeoff.py), so the embedding blocking can be
   compared directly with the key blocking (70.14% ceiling baseline).

Run:  python3 src/train_emb.py [--epochs N] [--dim D] [--limit N (smoke)]
Artifacts: artifacts/emb.npz (W, idf, config), artifacts/emb_E_train.npy
(fp16 memmap of ALL train embeddings), artifacts/emb_ivf_train.npz.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocking
import emb_candidates as ec
import emb_model as em
from emb_model import (ART, BiEncoder, build_text_blob, compute_idf)

TAU = 0.05
LR = 0.01
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


# ------------------------------------------------------------------ helpers
def split_path_fexists(ART):
    return os.path.exists(f"{ART}/split.pkl")


def load_gt_sets(meta):
    """S1 row -> sorted np.array of gold match rows (train tag only)."""
    return ec.gold_map_from_gt(meta["eid2idx"])


def gold_hit(g, r):
    """Is record r in sorted gold array g?"""
    if g is None or len(g) == 0:
        return False
    p = np.searchsorted(g, r)
    return p < len(g) and g[min(p, len(g) - 1)] == r


def build_negatives(E, ivf, blob, offs, anchors, enc, per_anchor=8,
                    topk=50, verbose=True):
    """Mine hard negatives: top IVF hits per anchor that are NOT gold.

    anchors must be sorted ascending.  Returns (neg_anchor, neg_rows) sorted
    by anchor then descending cosine (top-ranked first), capped at per_anchor
    negatives per anchor.
    """
    Q, _, _, _ = enc.embed_batches(blob, offs, anchors)
    qs, rows, cos = ec.query_ivf(E, ivf, Q, n1=2, n2=4, topk=topk,
                                 verbose=verbose)
    q_anchor = anchors[qs]
    is_gold = np.array([gold_hit(gold_cache.get(int(a)), int(r))
                        for a, r in zip(q_anchor, rows)])
    m = ~is_gold
    order = np.lexsort((-cos[m], q_anchor[m]))
    neg_rows = rows[m][order]
    neg_anchor = q_anchor[m][order]
    # cap per anchor: keep first per_anchor of each run of equal anchors
    newq = np.concatenate(([True], neg_anchor[1:] != neg_anchor[:-1]))
    grp = np.cumsum(newq) - 1
    gstart = np.nonzero(newq)[0]  # start offset of each group
    rank = np.arange(len(neg_anchor), dtype=np.int64) - gstart[grp]
    keep = rank < per_anchor
    return neg_anchor[keep], neg_rows[keep]


def train_step(enc, blob, offs, rows_a, rows_p, rows_n, adam, T=TAU):
    """One InfoNCE step; rows_* arrays aligned per anchor (same length B).

    Adam state dict: m, v, t.  Applies the update to enc.W in place.
    Returns the step loss.
    """
    allr, inv = np.unique(np.concatenate([rows_a, rows_p, rows_n]),
                          return_inverse=True)
    B = len(rows_a)
    K = len(rows_n) // B
    ia = inv[:B]
    ip = inv[B:2 * B]
    inz = inv[2 * B:]
    E, VN, ids, grows = enc.embed_batches(blob, offs, allr)
    D = enc.dim
    n = len(allr)

    # ---- forward: per-anchor logits [B, 1 + K]
    Ga = E[ia]                                   # [B, D]
    Gp = E[ip]                                   # [B, D]
    Gn = E[inz].reshape(B, K, D)                 # [B, K, D]
    logits = np.concatenate(
        [np.einsum("bd,bd->b", Ga, Gp)[:, None],
         np.einsum("bkd,bd->bk", Gn, Ga)], axis=1) / T
    mmax = logits.max(axis=1, keepdims=True)
    loss = float(np.mean(-logits[:, 0] + mmax[:, 0]
                         + np.log(np.exp(logits - mmax).sum(axis=1))))

    # ---- backward
    p = np.exp(logits - mmax)
    p = p / p.sum(axis=1, keepdims=True)
    dL = p
    dL[:, 0] -= 1.0
    dL /= B

    gE = np.zeros((n, D), dtype=np.float32)
    # anchor gradient: pos term + neg terms
    d_a = (dL[:, [0]] * Gp + np.einsum("bk,bkd->bd", dL[:, 1:], Gn))
    np.add.at(gE, ia, d_a)
    # positive gradient
    np.add.at(gE, ip, dL[:, [0]] * Ga)
    # negative gradients
    gn = (dL[:, 1:].reshape(-1)[:, None] *
          np.repeat(Ga, K, axis=0))
    np.add.at(gE, inz, gn)

    # chain through L2 normalization: de = (g - e*(g.e)) / ||v||
    g_v = gE / np.maximum(VN, 1e-12)[:, None]
    dot = np.einsum("nd,nd->n", g_v, E)
    dpre = (g_v - E * dot[:, None]) / np.maximum(VN, 1e-12)[:, None]

    # ---- scatter to hash buckets, chunked (gather+scatter of all grams at
    # once was materializing ~1GB per step)
    gW = np.zeros_like(enc.W)
    SC = 400_000
    for lo in range(0, len(ids), SC):
        idc = ids[lo:lo + SC]
        dgc = dpre[grows[lo:lo + SC]]
        order = np.argsort(idc, kind="stable")
        sids = idc[order]
        uniq, first = np.unique(sids, return_index=True)
        gW[uniq] += np.add.reduceat(dgc[order], first, axis=0)

    # ---- Adam update on enc.W
    adam["t"] += 1
    t = adam["t"]
    adam["m"] = BETA1 * adam["m"] + (1 - BETA1) * gW
    adam["v"] = BETA2 * adam["v"] + (1 - BETA2) * gW * gW
    mhat = adam["m"] / (1 - BETA1 ** t)
    vhat = adam["v"] / (1 - BETA2 ** t)
    enc.W -= (LR_STEP * mhat / (np.sqrt(vhat) + EPS)).astype(np.float32)
    return loss


# ------------------------------------------------------------------ eval
def eval_val(enc, E, ivf, blob, offs, val_anchors, gold_map, n_eval=4000,
             topk=50, seed=7):
    """Val macro recall for the embedding candidates (sampled anchors)."""
    rng = np.random.default_rng(seed)
    anc = np.array(sorted(val_anchors), dtype=np.int64)
    if len(anc) > n_eval:
        anc = np.sort(rng.choice(anc, size=n_eval, replace=False))
    Q, _, _, _ = enc.embed_batches(blob, offs, anc)
    qs, rows, cos = ec.query_ivf(E, ivf, Q, n1=2, n2=4, topk=topk,
                                 verbose=False)
    hit = np.array([gold_hit(gold_map.get(int(anc[q])), int(r))
                    for q, r in zip(qs, rows)])
    per_a = np.zeros(len(anc), dtype=np.int64)
    np.add.at(per_a, qs, hit)
    tot = np.array([len(gold_map[int(a)]) if int(a) in gold_map else 1
                    for a in anc])
    # macro recall over entities that HAVE gold (singletons excluded)
    mask = np.array([int(a) in gold_map and len(gold_map[int(a)]) > 0
                     for a in anc])
    if not mask.any():
        return 0.0
    return float(np.mean((per_a / np.maximum(tot, 1))[mask]))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="train")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--buckets", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--negs", type=int, default=8)
    ap.add_argument("--neg-rounds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N records (smoke test)")
    ap.add_argument("--prep-only", action="store_true",
                    help="build text blob + idf + embeddings + IVF index, "
                         "then exit (no training).  Run this first at full "
                         "scale; training resumes from the saved encoder.")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    tag = args.tag
    t0 = time.time()
    global LR_STEP
    LR_STEP = args.lr

    # ---------------- data + text blobs
    d, meta = blocking.load_data(tag)
    src = ec.src_from_eid(d)
    if args.limit:
        # self-contained subset: all train-half S1 anchors that have gold +
        # their gold partners + filler records up to the limit
        keep_idx = f"{ART}/emb_limit_idx.npy"
        if os.path.exists(keep_idx):
            keep = np.load(keep_idx)
        else:
            rng0 = np.random.default_rng(5)
            gold_all = ec.gold_map_from_gt(meta["eid2idx"])
            anchors = np.array(sorted(gold_all.keys()), dtype=np.int64)
            tr_mask = np.ones(len(anchors), dtype=bool)
            if split_path_fexists(ART):
                split = pickle.load(open(f"{ART}/split.pkl", "rb"))
                val_set = np.isin(anchors, np.asarray(split["val_s1_idx"]))
                tr_mask = ~val_set
            anchors_tr = anchors[tr_mask]
            anchor_set = set(anchors_tr.tolist())  # build ONCE (was O(N^2))
            parts = [g for a, g in gold_all.items() if a in anchor_set]
            partners = (np.unique(np.concatenate(parts)) if parts
                        else np.zeros(0, dtype=np.int64))
            # respect the cap: subsample anchors so anchors+partners fit
            budget = max(args.limit - len(partners), 1000)
            if len(anchors_tr) > budget:
                anchors_tr = np.sort(rng0.choice(anchors_tr, size=budget,
                                                 replace=False))
                parts = [gold_all[int(a)] for a in anchors_tr
                         if int(a) in gold_all]
                partners = (np.unique(np.concatenate(parts)) if parts
                            else np.zeros(0, dtype=np.int64))
            else:
                # few anchors vs huge partner set (realistic for big limits):
                # partner mass dominates; that's fine, they're needed for
                # positive supervision -- only cap FILLER
                pass
            filler_pool = np.nonzero(src == 1)[0]
            filler = filler_pool[~np.isin(filler_pool, partners)]
            n_extra = max(0, args.limit - len(anchors_tr) - len(partners))
            keep = np.unique(np.concatenate([
                anchors_tr, partners,
                filler[rng0.choice(len(filler), size=min(n_extra, len(filler)),
                                   replace=False)] if n_extra > 0
                else np.zeros(0, dtype=np.int64)]))
            np.save(keep_idx, keep)
        d, meta = em.subset_records(d, meta, keep)
        src = src[keep]
        meta["_subset_note"] = f"subset of {len(keep)} records"
    rd = blocking.RecData(d, meta)
    blob, offs = build_text_blob(rd, meta, tag,
                                 suffix=(f"_lim{args.limit}" if args.limit
                                         else None))
    N = rd.n
    print(f"records={N} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- gold pairs (train tag only)
    global gold_cache, gold_map
    gold_map, gold_cache = {}, {}
    if tag == "train":
        gold_map = load_gt_sets(meta)
        gold_cache = gold_map

    # ---------------- split
    split_path = f"{ART}/split.pkl"
    split = pickle.load(open(split_path, "rb")) if os.path.exists(split_path) \
        else None
    if split is not None and not args.limit:
        val_s1 = np.sort(np.asarray(split["val_s1_idx"], dtype=np.int64))
        tr_s1 = np.sort(np.asarray(split["train_s1_idx"], dtype=np.int64))
    else:
        val_s1 = np.nonzero(src == 0)[0][:0]  # empty
        tr_s1 = np.nonzero(src == 0)[0]
    anchors_all = np.array([a for a in tr_s1
                            if int(a) in gold_map and len(gold_map[int(a)])],
                           dtype=np.int64) if gold_map else np.zeros(0,
                                                                     dtype=np.int64)
    print(f"train anchors: {len(anchors_all)}", flush=True)

    # ---------------- encoder + idf
    dim = args.dim
    if os.path.exists(f"{ART}/emb.npz") and dim is None:
        enc = BiEncoder.load(f"{ART}/emb.npz")
        print(f"resuming encoder from {ART}/emb.npz "
              f"(dim={enc.dim}, buckets={enc.buckets})", flush=True)
    else:
        enc = BiEncoder(dim=dim or 128, buckets=args.buckets or (1 << 17))
    if enc.idf is None:
        enc.idf = compute_idf(blob, offs, enc, verbose=True)
        print(f"idf computed ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- embed everything (fp16 memmap, cached)
    E_path = f"{ART}/emb_E_{tag}{'_lim' + str(args.limit) if args.limit else ''}.npy"
    ivf_path = f"{ART}/emb_ivf_{tag}{'_lim' + str(args.limit) if args.limit else ''}.npz"
    E = enc.embed_all(blob, offs, E_path)
    ivf = None

    adam = dict(m=np.zeros_like(enc.W), v=np.zeros_like(enc.W), t=0)
    if args.prep_only:
        ec.build_ivf(E, np.nonzero(src == 1)[0], ivf_path, verbose=True)
        enc.save(f"{ART}/emb.npz", extra=dict(trained_steps=0,
                                               prep_only=True))
        print(f"prep-only done: encoder/idf/E/IVF cached "
              f"({time.time()-t0:.0f}s total)", flush=True)
        return
    rand_pool = np.nonzero(src == 1)[0]  # random-negative pool (S2/S3 rows)
    rng = np.random.default_rng(11)
    steps_per_epoch = max(1, len(anchors_all) // args.batch)
    print(f"training: {args.epochs} epochs x {steps_per_epoch} steps "
          f"(batch={args.batch}, negs={args.negs})", flush=True)

    neg_anchor = neg_rows = None
    for ep in range(args.epochs):
        te = time.time()
        perm = rng.permutation(len(anchors_all))
        ep_loss, nb = 0.0, 0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            if len(idx) < 8:
                continue
            A = anchors_all[idx]
            P = np.array([gold_map[int(a)][rng.integers(len(gold_map[int(a)]))]
                          for a in A], dtype=np.int64)
            # negatives: stored hard negatives first, then random pool rows
            # (NEVER pad with the anchor's own positive -- that would push
            # the anchor away from a true match)
            K = args.negs
            rows_n = rng.choice(rand_pool, size=(len(A), K), replace=True) \
                if rand_pool.size else np.tile(P[:, None], (1, K))
            if neg_rows is not None:
                starts = np.searchsorted(neg_anchor, A)
                ends = np.searchsorted(neg_anchor, A, side="right")
                for i in range(len(A)):
                    take = min(K, int(ends[i] - starts[i]))
                    if take:
                        rows_n[i, :take] = neg_rows[starts[i]:starts[i] + take]
            ep_loss += train_step(enc, blob, offs, A, P, rows_n.ravel(), adam)
            nb += 1
        print(f"epoch {ep+1}/{args.epochs}: loss={ep_loss/max(nb,1):.4f} "
              f"({time.time()-te:.0f}s)", flush=True)
        # checkpoint every epoch: long runs survive kills (resume = rerun)
        enc.save(f"{ART}/emb.npz", extra=dict(trained_steps=adam["t"],
                                               epoch=ep + 1))

        # ---- periodic hard-negative refresh (re-embed + rebuild IVF)
        if (ep + 1) % args.neg_rounds == 0 or ep == args.epochs - 1:
            import gc
            del E  # release the old memmap (Windows locks open files)
            gc.collect()
            E = enc.embed_all(blob, offs, E_path, force=True)
            ec.build_ivf(E, np.nonzero(src == 1)[0], ivf_path, verbose=True)
            ivf = dict(np.load(ivf_path))
            neg_anchor, neg_rows = build_negatives(
                E, ivf, blob, offs, anchors_all, enc, per_anchor=args.negs)
            np.savez(f"{ART}/emb_negs.npz", a=neg_anchor, n=neg_rows)
            print(f"hard negatives refreshed: {len(neg_anchor)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        # ---- eval
        if (not args.no_eval and gold_map and split is not None
                and len(val_s1)):
            if ivf is None and os.path.exists(ivf_path):
                ivf = dict(np.load(ivf_path))
            if ivf is not None:
                mac = eval_val(enc, E, ivf, blob, offs, val_s1, gold_map)
                print(f"  val macro recall@50: {mac:.4f} "
                      f"(key blocking ceiling: 0.7014)", flush=True)

    enc.save(f"{ART}/emb.npz", extra=dict(trained_steps=adam["t"]))
    print(f"saved {ART}/emb.npz ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
