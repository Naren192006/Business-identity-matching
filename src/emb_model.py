"""Lightweight bi-encoder for embedding-based blocking (pure numpy).

Sits next to src/blocking.py (13 key families, 70.14% candidate-recall
ceiling).  Key hashing misses pairs whose name/address differ by typos,
transliteration drift, reordering or abbreviation -- character n-gram
embeddings recover much of that mass.

Model
-----
record text (normalized name + address + alt romanization + country token)
  -> hashed char 3/4-grams + word 1-grams  ->  idf-weighted multi-hot x
  -> embedding e = normalize(x @ W)        W: BUCKETS x DIM (trained here)
similarity = cosine(e_a, e_b) == dot product (rows are L2-normalized).

Training lives in src/train_emb.py (InfoNCE on gold pairs from the PROVIDED
ground truth only -- competition legal: no external data/weights, W is a few
MB, far below the 8B-param cap).  Retrieval lives in src/emb_candidates.py.

Everything is numpy-only (no sklearn / torch / faiss on this machine).
"""
import os
import time

import numpy as np

ART = "artifacts"

CHAR_GRAMS = (3, 4)            # char n-gram lengths
DEFAULT_BUCKETS = 1 << 17      # 131072 hash buckets for n-grams
DEFAULT_DIM = 128              # embedding dimension
DEFAULT_MAX_NNZ = 320          # max n-grams kept per record (name first)

# hash constants (odd uint64; any fixed odd values work)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_FNV = np.uint64(0x100000001B3)
_SALT = {3: np.uint64(0x243F6A8885A308D3), 4: np.uint64(0x13198A2E03707344),
         11: np.uint64(0xA4093822299F31D0), 12: np.uint64(0xD82EFA9831C4E6C9)}
_PW = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5, 0x9E3779B185EBCA87,
                0xC2B2AE3D27D4EB4F, 0x3C6EF372FE94F82B, 0xA24BAED4963EE407,
                0x9FB21C651E98DF25, 0x68E31DA436C2F6C4, 0x76DC9E7D2FCB0B5B,
                0xD6E8FEB86659FD93, 0x165667B19E3779F9, 0xB5C7F3B0A5A5A2B1,
                0x846CA68BD1B3B5BD], dtype=np.uint64)


# ---------------------------------------------------------------- text blob
def build_text_blob(rd, meta, tag, limit=None, verbose=True, suffix=None):
    """Per-record normalized text as one flat uint8 blob + record offsets.

    text_i = name_tokens + " | " + addr_tokens + " | q<country>" and, when the
    record has an alt (romanized) view, " | " + alt_name [+ " | " + alt_addr].
    A trailing space is appended to every record so words never span records.
    Cached to {ART}/{tag}_emb_text{suffix}.npz (suffix = _lim{N} for --limit
    runs, which build their own subset blob).
    """
    suf = suffix if suffix is not None else (f"_lim{limit}" if limit else "")
    path = f"{ART}/{tag}_emb_text{suf}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return z["blob"], z["offs"]
    V = len(meta["tok_vocab"])
    inv = [None] * V
    for s, i in meta["tok_vocab"].items():
        inv[i] = s
    # country token: name when country_ids is available (older meta pickles
    # lack it), else the numeric id -- either way a consistent per-country
    # token that the encoder can condition on
    cname = {i: "".join(ch for ch in s.lower() if ch.isalnum())
             for s, i in (meta.get("country_ids") or {}).items()}
    t0 = time.time()
    n = rd.n if limit is None else min(rd.n, limit)
    buf = bytearray()
    offs = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        c = int(rd.country[i])
        ckey = cname.get(c)
        ckey = ckey if ckey else str(c)
        parts = [" ".join([inv[t] for t in rd.name_toks(i).tolist()]),
                 " ".join([inv[t] for t in rd.addr_toks(i).tolist()]),
                 "q" + ckey]
        alt = rd.alt_name_toks(i)
        if alt is not None and len(alt):
            parts.append(" ".join([inv[t] for t in alt.tolist()]))
            aa = rd.alt_addr_toks(i)
            if aa is not None and len(aa):
                parts.append(" ".join([inv[t] for t in aa.tolist()]))
        s = " | ".join(p for p in parts if p)
        buf += s.encode("utf-8", "ignore")
        buf += b" "
        offs[i + 1] = len(buf)
        if verbose and i % 2000000 == 0 and i:
            print(f"  text blob {i} ({time.time()-t0:.0f}s)", flush=True)
    blob = np.frombuffer(bytes(buf), dtype=np.uint8)
    np.savez(path, blob=blob, offs=offs)
    if verbose:
        print(f"text blob: {n} records, {len(blob)/1e6:.0f} MB "
              f"({time.time()-t0:.0f}s) -> {path}", flush=True)
    return blob, offs


# ------------------------------------------------------------- n-gram hashing
def _gather_records(blob, offs, rec_idx):
    """Contiguous copy of records rec_idx (must be sorted ascending)."""
    lo = offs[rec_idx]
    hi = offs[rec_idx + 1]
    lens = hi - lo
    total = int(lens.sum())
    starts_cum = np.zeros(len(rec_idx) + 1, dtype=np.int64)
    starts_cum[1:] = np.cumsum(lens)
    gpos = np.repeat(lo - starts_cum[:-1], lens) + np.arange(total, dtype=np.int64)
    return blob[gpos], starts_cum


def _char_hashes(gb, L, buckets):
    """FNV-style rolling hash of all length-L windows of gb."""
    m = len(gb)
    if m < L:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    win = np.lib.stride_tricks.sliding_window_view(gb, L)
    h = np.zeros(m - L + 1, dtype=np.uint64)
    for j in range(L):
        h = h * _FNV + win[:, j].astype(np.uint64)
    h = ((h ^ _SALT[L]) & np.uint64(buckets - 1)).astype(np.int64)
    pos = np.arange(m - L + 1, dtype=np.int64)
    return h, pos


def _word_hashes(gb, buckets):
    """Hash every whitespace-delimited word (order-sensitive, mod-16 power)."""
    m = len(gb)
    nsp = np.nonzero(gb != 32)[0]
    if len(nsp) == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    brk = np.nonzero(np.diff(nsp) > 1)[0] + 1
    wstart = np.concatenate(([nsp[0]], nsp[brk])).astype(np.int64)
    wid = np.zeros(m, dtype=np.int64)
    wid[wstart] = 1
    wid = np.cumsum(wid) - 1
    pos_in_word = np.arange(m, dtype=np.int64) - wstart[wid]
    h = gb.astype(np.uint64) * _PW[pos_in_word & 15]
    hw = np.add.reduceat(h, wstart)
    hw = ((hw ^ _SALT[11]) & np.uint64(buckets - 1)).astype(np.int64)
    return hw, wstart, wid


def ngrams_for_records(blob, offs, rec_idx, buckets, max_nnz):
    """Hashed n-grams for records rec_idx (sorted, unique).

    Returns (ids int64 in [0,buckets), rows int64 aligned to rec_idx order).
    Granularities are emitted in order char3, char4, word -- so the per-record
    cap keeps name-leading char n-grams first (name is the more discriminative
    field; long landmark-style addresses get truncated).
    """
    rec_idx = np.asarray(rec_idx, dtype=np.int64)
    gblob, loffs = _gather_records(blob, offs, rec_idx)
    m = len(gblob)
    pos_all = np.arange(m, dtype=np.int64)
    rec_of_pos = np.searchsorted(loffs, pos_all, side="right") - 1
    out_ids, out_rows = [], []

    for L in CHAR_GRAMS:
        h, pos = _char_hashes(gblob, L, buckets)
        if len(h):
            ok = rec_of_pos[pos] == rec_of_pos[pos + (L - 1)]
            out_ids.append(h[ok])
            out_rows.append(rec_of_pos[pos[ok]])

    hw, wstart, _ = _word_hashes(gblob, buckets)
    if len(hw):
        wrow = rec_of_pos[wstart]
        out_ids.append(hw)
        out_rows.append(wrow)
        if len(hw) > 1:  # adjacent-word bigrams within the same record
            adj = wrow[1:] == wrow[:-1]
            a = hw[:-1][adj].astype(np.uint64) * np.uint64(0x9E3779B1)
            b = hw[1:][adj].astype(np.uint64)
            hb = ((a ^ b ^ _SALT[12]) & np.uint64(buckets - 1)).astype(np.int64)
            out_ids.append(hb)
            out_rows.append(wrow[:-1][adj])

    if not out_ids:
        z = np.zeros(0, dtype=np.int64)
        return z, z
    ids = np.concatenate(out_ids)
    rows = np.concatenate(out_rows)
    # per-record cap: keep the FIRST max_nnz grams of each record
    rstart = np.searchsorted(rows, np.arange(len(rec_idx) + 1))[:-1]
    rank = np.arange(len(rows), dtype=np.int64) - rstart[rows]
    keep = rank < max_nnz
    return ids[keep], rows[keep]


def compute_idf(blob, offs, enc, limit=None, chunk=200000, verbose=True):
    """Document frequency per hash bucket over all records -> idf vector."""
    n = len(offs) - 1 if limit is None else min(len(offs) - 1, limit)
    df = np.zeros(enc.buckets, dtype=np.int64)
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        ids, _ = ngrams_for_records(blob, offs, np.arange(lo, hi, dtype=np.int64),
                                    enc.buckets, enc.max_nnz)
        if len(ids):
            u = np.unique(ids)
            df += np.bincount(u, minlength=enc.buckets)
        if verbose and (lo // chunk) % 20 == 0:
            print(f"  idf {hi}/{n} ({time.time()-t0:.0f}s)", flush=True)
    idf = np.log1p(n / (1.0 + df)).astype(np.float32)
    return idf


# ------------------------------------------------------------- subsetting
def subset_records(d, meta, keep):
    """Subset the packed record arrays to `keep` (sorted unique record ids).

    Returns (d2, meta2) with every ragged array re-offset, alt_rows remapped,
    eid2idx rebuilt, and states sliced.  Used by train_emb.py --limit to build
    a self-contained mini-dataset that still contains gold partners.
    """
    keep = np.sort(np.asarray(keep, dtype=np.int64))
    n = len(keep)
    out = {}
    for k in ("pin", "dom", "country", "eid"):
        out[k] = d[k][keep]
    for flat_k, off_k in (("name_flat", "name_off"), ("addr_flat", "addr_off"),
                          ("dig_flat", "dig_off")):
        off = d[off_k]
        starts = off[keep]
        lens = off[keep + 1] - starts
        total = int(lens.sum())
        cum = np.zeros(n + 1, dtype=np.int64)
        cum[1:] = np.cumsum(lens)
        gpos = np.repeat(starts - cum[:-1], lens) + np.arange(total,
                                                              dtype=np.int64)
        out[flat_k] = d[flat_k][gpos]
        out[off_k] = cum
    # alt (romanized) views: alt arrays are indexed by POSITION in alt_rows
    ar = d["alt_rows"]
    inkeep = np.isin(ar, keep)
    old_pos = np.nonzero(inkeep)[0]
    out["alt_rows"] = np.searchsorted(keep, ar[inkeep])
    for flat_k, off_k in (("alt_name_flat", "alt_name_off"),
                          ("alt_addr_flat", "alt_addr_off")):
        off = d[off_k]
        starts = off[old_pos]
        lens = off[old_pos + 1] - starts
        total = int(lens.sum())
        cum = np.zeros(len(old_pos) + 1, dtype=np.int64)
        cum[1:] = np.cumsum(lens)
        gpos = np.repeat(starts - cum[:-1], lens) + np.arange(total,
                                                              dtype=np.int64)
        out[flat_k] = d[flat_k][gpos]
        out[off_k] = cum
    # raw text blob
    roff = d["raw_off"]
    starts = roff[keep]
    lens = roff[keep + 1] - starts
    total = int(lens.sum())
    cum = np.zeros(n + 1, dtype=np.int64)
    cum[1:] = np.cumsum(lens)
    gpos = np.repeat(starts - cum[:-1], lens) + np.arange(total, dtype=np.int64)
    out["raw_blob"] = d["raw_blob"][gpos]
    out["raw_off"] = cum
    # meta: shallow-copy EVERYTHING (vocab dicts etc.), then replace the
    # record-aligned entries
    meta2 = dict(meta)
    meta2["states"] = [meta["states"][i] for i in keep]
    meta2["eid2idx"] = {e: i for i, e in enumerate(out["eid"])}
    return out, meta2


# ------------------------------------------------------------------ encoder
class BiEncoder:
    """e = normalize( (idf-weighted hashed n-gram bag) @ W )."""

    def __init__(self, dim=DEFAULT_DIM, buckets=DEFAULT_BUCKETS,
                 max_nnz=DEFAULT_MAX_NNZ, seed=13):
        self.dim = int(dim)
        self.buckets = int(buckets)
        self.max_nnz = int(max_nnz)
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((self.buckets, self.dim))
                  * (1.0 / np.sqrt(self.dim))).astype(np.float32)
        self.idf = None

    # ---- forward ----
    def _project(self, ids, rows, n, chunk=400_000):
        """Sparse bag -> dense embeddings for n records (rows sorted).

        Chunked over n-grams so the W-row gather never materializes more
        than `chunk` x dim floats (~200 MB) at once -- unchunked gathers of
        ~1M grams x 128 dims were the training bottleneck.
        """
        v = np.zeros((n, self.dim), dtype=np.float32)
        for lo in range(0, len(ids), chunk):
            idc = ids[lo:lo + chunk]
            rowc = rows[lo:lo + chunk]
            w = self.W[idc]
            if self.idf is not None:
                w *= self.idf[idc][:, None]
            uq, first = np.unique(rowc, return_index=True)
            v[uq] += np.add.reduceat(w, first, axis=0)
        vn = np.linalg.norm(v, axis=1)
        e = v / np.maximum(vn, 1e-12)[:, None]
        return e, vn.astype(np.float32), v

    def embed_batches(self, blob, offs, rec_idx, nnz_cap=4_000_000):
        """Embed arbitrary (sorted, unique) records; caps RAM per sub-chunk.

        Returns (E float32 [n,dim], vnorm float32 [n], ids, rows) where
        ids/rows are the n-gram triplets of ALL input records (rows index the
        input order) -- reused for backprop by the trainer.
        """
        rec_idx = np.asarray(rec_idx, dtype=np.int64)
        n = len(rec_idx)
        # split by records so nnz stays bounded (offs has len n_all + 1)
        if n:
            last = int(rec_idx[-1])
            approx = int(offs[min(last + 1, len(offs) - 1)] - offs[int(rec_idx[0])])
        else:
            approx = 0
        step = max(1, int(nnz_cap / max(approx, 1)))
        E = np.zeros((n, self.dim), dtype=np.float32)
        VN = np.zeros(n, dtype=np.float32)
        id_l, row_l = [], []
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            idx = rec_idx[lo:hi]
            ids, rows = ngrams_for_records(blob, offs, idx, self.buckets,
                                           self.max_nnz)
            e, vn, _ = self._project(ids, rows, len(idx))
            E[lo:hi] = e
            VN[lo:hi] = vn
            id_l.append(ids)
            row_l.append(rows + lo)
        if id_l:
            return E, VN, np.concatenate(id_l), np.concatenate(row_l)
        z = np.zeros(0, dtype=np.int64)
        return E, VN, z, z

    def embed_all(self, blob, offs, out_path, chunk=8192, verbose=True,
                  force=False):
        """Embed every record -> float16 memmap (N, dim), cached on disk."""
        n = len(offs) - 1
        if os.path.exists(out_path) and not force:
            mm = np.load(out_path, mmap_mode="r")
            if mm.shape == (n, self.dim):
                return mm
        if os.path.exists(out_path):
            # Windows: an open memmap locks the file; copy to a temp path,
            # swap in the new array, then unlink the old handle's target.
            mm_old = np.load(out_path, mmap_mode="r")
            del mm_old
            tmp = out_path + ".new"
        else:
            tmp = out_path
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                       shape=(n, self.dim))
        t0 = time.time()
        ar = np.arange(n, dtype=np.int64)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            e, _, _, _ = self.embed_batches(blob, offs, ar[lo:hi])
            mm[lo:hi] = e.astype(np.float16)
            if verbose and (lo // chunk) % 200 == 0:
                print(f"  embed {hi}/{n} ({time.time()-t0:.0f}s)", flush=True)
        mm.flush()
        del mm
        if tmp != out_path:
            os.replace(tmp, out_path)
        if verbose:
            print(f"embedded {n} records -> {out_path} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return np.load(out_path, mmap_mode="r")

    # ---- io ----
    def save(self, path, extra=None):
        d = dict(W=self.W, dim=np.int64(self.dim),
                 buckets=np.int64(self.buckets),
                 max_nnz=np.int64(self.max_nnz))
        if self.idf is not None:
            d["idf"] = self.idf
        if extra:
            d.update(extra)
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        enc = cls(dim=int(z["dim"]), buckets=int(z["buckets"]),
                  max_nnz=int(z["max_nnz"]))
        enc.W = z["W"]
        if "idf" in z.files:
            enc.idf = z["idf"]
        return enc


if __name__ == "__main__":
    # self-test on synthetic data: featurize -> embed -> cosine sanity
    rng = np.random.default_rng(0)
    texts = [b"state bank of india main branch mg road bangalore",
             b"state bank of india main brach mg road banglore",   # typo pair
             b"walmart supercenter store 1234"]
    offs = np.zeros(4, dtype=np.int64)
    blob = bytearray()
    for t in texts:
        blob += t + b" "
        offs[1:] = np.cumsum([len(t) + 1 for t in texts])
    blob = np.frombuffer(bytes(blob), dtype=np.uint8)
    enc = BiEncoder(dim=32, buckets=4096)
    enc.idf = np.ones(enc.buckets, dtype=np.float32)
    idx = np.arange(3, dtype=np.int64)
    ids, rows = ngrams_for_records(blob, offs, idx, enc.buckets, enc.max_nnz)
    assert len(ids) > 100 and rows.max() == 2, (len(ids), rows.max())
    E, VN, _, _ = enc.embed_batches(blob, offs, idx)
    sims = E @ E.T
    assert sims[0, 1] > sims[0, 2] + 0.15, sims  # typo pair must beat random
    p = f"{ART}/emb_selftest.npz"
    os.makedirs(ART, exist_ok=True)
    enc.save(p)
    enc2 = BiEncoder.load(p)
    E2, _, _, _ = enc2.embed_batches(blob, offs, idx)
    assert np.allclose(E, E2, atol=1e-5)
    os.remove(p)
    print("emb_model self-test OK  "
          f"sims[0,1]={sims[0,1]:.3f} sims[0,2]={sims[0,2]:.3f}")
