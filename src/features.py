"""Pairwise feature extraction (vectorized).

Exact set overlaps (name tokens, core tokens, phones, addr tokens, addr
digits) are computed per chunk by exploding both sides into (pair, token)
keys and intersecting sorted unique key arrays -- set semantics without
per-pair python sets.  String similarities use rapidfuzz process.cpdist.
Per-record strings are predecoded once; per-record set sizes precomputed.
"""
import numpy as np
import rapidfuzz
from rapidfuzz.distance import JaroWinkler, Indel

from blocking import RecData, load_data

FEATURE_NAMES = [
    "name_tok_jac", "name_tok_dice", "name_tok_ovr_co", "name_tok_cont",
    "name_ph_jac", "name_ph_dice", "name_ph_ovr_co",
    "name_exact_norm", "name_exact_core",
    "name_fuzz_norm", "name_fuzz_core", "name_fuzz_alt", "name_fuzz_raw",
    "name_jw", "name_subseq",
    "name_len_diff", "name_shorter_in_longer",
    "name_token_prec", "name_token_rec",
    "addr_tok_jac", "addr_tok_dice", "addr_tok_ovr_co",
    "addr_exact_norm", "addr_fuzz", "addr_jw",
    "addr_digit_jac", "addr_digit_eq", "addr_digit_sub",
    "pin_eq", "state_eq", "state_missing",
    "domain_eq", "dom_any",
    "a_name_df", "b_name_df", "tok_rarity",
    "country_eq", "b_addr_empty", "a_addr_empty",
    "core_subset", "ph_subset", "nums_eq", "num_tokens_eq",
]
NF = len(FEATURE_NAMES)


def _ragged_gather(flat, off, rec_idx):
    """Gather variable-length rows for rec_idx; return (values, pair_ids)."""
    starts = off[rec_idx]
    lengths = off[rec_idx + 1] - starts
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=flat.dtype), np.zeros(0, dtype=np.int64)
    cum = np.zeros(len(lengths) + 1, dtype=np.int64)
    cum[1:] = np.cumsum(lengths)
    out_idx = np.repeat(starts - cum[:-1], lengths) + np.arange(total)
    pair_ids = np.repeat(np.arange(len(rec_idx), dtype=np.int64), lengths)
    return flat[out_idx], pair_ids


def _overlap_counts(A_vals, A_pid, B_vals, B_pid, n_pairs, vocab_size):
    """Per-pair distinct-token intersection counts between two ragged sides.

    Returns (inter_counts int64[n_pairs], a_sizes, b_sizes).
    """
    # dedupe (pair, token) within each side
    A_keys = A_pid.astype(np.int64) * vocab_size + A_vals.astype(np.int64)
    B_keys = B_pid.astype(np.int64) * vocab_size + B_vals.astype(np.int64)
    A_keys = np.unique(A_keys)
    B_keys = np.unique(B_keys)
    # sizes per pair = distinct tokens per (pair) on each side
    a_sizes = np.bincount(A_keys // vocab_size, minlength=n_pairs)
    b_sizes = np.bincount(B_keys // vocab_size, minlength=n_pairs)
    inter_keys, _ = _sorted_intersect_count(A_keys, B_keys)
    if len(inter_keys) == 0:
        return np.zeros(n_pairs, dtype=np.int64), a_sizes, b_sizes
    inter = np.bincount(inter_keys // vocab_size, minlength=n_pairs)
    return inter, a_sizes, b_sizes


def _sorted_intersect_count(A, B):
    """Intersection of two sorted unique int64 arrays."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    idx = np.searchsorted(B, A)
    idx_ok = idx < len(B)
    idx_c = np.minimum(idx, len(B) - 1)
    hit = B[idx_c] == A
    mask = idx_ok & hit
    return A[mask], np.nonzero(mask)[0]


def _ratio_stats(inter, sa, sb):
    """Compute jac/dice/ovr/cont/prec/rec/sub from counts and set sizes."""
    n = len(sa)
    la = sa.astype(np.float64)
    lb = sb.astype(np.float64)
    it = inter.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        union = la + lb - it
        jac = np.where(union > 0, it / union, 0.0)
        dice = np.where((la + lb) > 0, 2 * it / (la + lb), 0.0)
        mn = np.minimum(la, lb)
        mx = np.maximum(la, lb)
        ovr = np.where(mn > 0, it / mn, 0.0)
        cont = np.where(mx > 0, it / mx, 0.0)
        prec = np.where(lb > 0, it / lb, 0.0)
        rec = np.where(la > 0, it / la, 0.0)
        sub = np.where(mn > 0, it / mn, 0.0)  # == ovr; subset fraction
    return (jac.astype(np.float32), dice.astype(np.float32),
            ovr.astype(np.float32), cont.astype(np.float32),
            prec.astype(np.float32), rec.astype(np.float32),
            sub.astype(np.float32))


def _ew_sim(sa, sb, scorer):
    if len(sa) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(sa) < 2000:
        return np.array([scorer(x, y) for x, y in zip(sa, sb)],
                        dtype=np.float32) / 100.0
    res = rapidfuzz.process.cpdist(sa, sb, scorer=scorer, workers=-1)
    return np.asarray(res, dtype=np.float32) / 100.0


def _is_subseq(a, b):
    la, lb = len(a), len(b)
    if la > lb:
        return False
    it = iter(b)
    return all(x in it for x in a)


class FeatureBuilder:
    def __init__(self, tag):
        d, meta = load_data(tag)  # load_data already materializes the dict
        self.d = d
        self.rd = RecData(d, meta)
        self.meta = meta
        self.tag = tag
        tv = meta["tok_vocab"]
        import textnorm
        self.legal_ids = frozenset(tv[w] for w in textnorm.LEGAL_SET if w in tv)
        self.tok_df = np.bincount(d["name_flat"],
                                  minlength=len(tv)).astype(np.float32)
        self.num_ids = frozenset(ti for tok, ti in tv.items() if tok.isdigit())
        self.inv_vocab = {v: k for k, v in tv.items()}
        self.V_tok = len(tv)
        self.V_ph = len(meta["phone_vocab"])

        # ---- per-record precomputed stats (one pass, sets discarded)
        n = self.rd.n
        self.name_nuniq = np.zeros(n, dtype=np.int32)
        self.core_nuniq = np.zeros(n, dtype=np.int32)
        self.addr_nuniq = np.zeros(n, dtype=np.int32)
        self.ph_nuniq = np.zeros(n, dtype=np.int32)
        self.dig_nuniq = np.zeros(n, dtype=np.int32)
        t2p = np.asarray(self.rd.tok2phone)
        # string blobs: normalized / core / alt name strings, "\x00"-separated
        norm_blob = bytearray()
        core_blob = bytearray()
        alt_blob = bytearray()
        str_off = np.zeros(n + 1, dtype=np.int64)
        core_off = np.zeros(n + 1, dtype=np.int64)
        alt_off = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            nt = self.rd.name_toks(i).tolist()
            toks = [self.inv_vocab[int(t)] for t in nt]
            ns = " ".join(toks)
            cs = " ".join(t for t in toks if t not in self.legal_ids)
            self.name_nuniq[i] = len(set(nt))
            self.core_nuniq[i] = len(cs.split()) if cs else 0
            self.ph_nuniq[i] = len({int(t2p[t]) for t in nt})
            at = self.rd.addr_toks(i).tolist()
            self.addr_nuniq[i] = len(set(at))
            self.dig_nuniq[i] = len(set(self.rd.addr_digits(i).tolist()))
            # alt romanization (only when it differs from the norm view)
            alt = self.rd.alt_name_toks(i)
            if alt is not None and len(alt):
                als = " ".join(self.inv_vocab[int(t)] for t in alt)
            else:
                als = ns
            norm_blob += ns.encode("utf-8") + b"\x00"
            core_blob += cs.encode("utf-8") + b"\x00"
            alt_blob += als.encode("utf-8") + b"\x00"
            str_off[i + 1] = len(norm_blob)
            core_off[i + 1] = len(core_blob)
            alt_off[i + 1] = len(alt_blob)
            if i % 2000000 == 0 and i:
                print(f"  rec stats {i} ({len(norm_blob)//1000000}MB)",
                      flush=True)
        self.norm_blob = np.frombuffer(bytes(norm_blob), dtype=np.uint8)
        self.core_blob = np.frombuffer(bytes(core_blob), dtype=np.uint8)
        self.alt_blob = np.frombuffer(bytes(alt_blob), dtype=np.uint8)
        self.str_off = str_off
        self.core_off = core_off
        self.alt_off = alt_off
        del norm_blob, core_blob, alt_blob
        self._num_id_set = frozenset(self.num_ids)
        # materialize raw text blobs once (npz access otherwise re-reads
        # the whole array from disk on every __getitem__)
        self._raw_blob = np.asarray(d["raw_blob"])
        self._raw_off = np.asarray(d["raw_off"])
        import gc
        gc.collect()

    def _blob_str(self, blob, off, i):
        return bytes(blob[off[i]:off[i + 1] - 1]).decode("utf-8")

    def raw(self, i):
        blob = self._raw_blob
        off = self._raw_off
        chunk = blob[off[i]:off[i + 1] - 1].tobytes()
        nm, ad = chunk.split(b"\x00", 1)
        return nm.decode("utf-8"), ad.decode("utf-8")

    def build(self, s1_idx, s2_idx, out_path, chunk=2000000, log=True):
        rd = self.rd
        n = len(s1_idx)
        F = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                      shape=(n, NF))
        col = {x: j for j, x in enumerate(FEATURE_NAMES)}
        states = self.meta["states"]

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            A = s1_idx[start:end]
            B = s2_idx[start:end]
            na = end - start
            sl = slice(start, end)
            npr = na

            # ---------- name token exact overlaps (explode)
            Av, Ap = _ragged_gather(rd.name_flat, rd.name_off, A)
            Bv, Bp = _ragged_gather(rd.name_flat, rd.name_off, B)
            inter, sa, sb = _overlap_counts(Av, Ap, Bv, Bp, npr, self.V_tok)
            jac, dice, ovr, cont, prec, rec, sub = _ratio_stats(inter, sa, sb)
            F[sl, col["name_tok_jac"]] = jac
            F[sl, col["name_tok_dice"]] = dice
            F[sl, col["name_tok_ovr_co"]] = ovr
            F[sl, col["name_tok_cont"]] = cont
            F[sl, col["name_token_prec"]] = prec
            F[sl, col["name_token_rec"]] = rec
            F[sl, col["name_exact_norm"]] = (
                (jac >= 0.999) & (cont >= 0.999)).astype(np.float32)
            del Av, Ap, Bv, Bp, inter

            # ---------- core token overlaps (filter legal-suffix tokens)
            Av, Ap = _ragged_gather(rd.name_flat, rd.name_off, A)
            Bv, Bp = _ragged_gather(rd.name_flat, rd.name_off, B)
            leg_arr = np.zeros(self.V_tok, dtype=bool)
            for t in self.legal_ids:
                leg_arr[t] = True
            mA = ~leg_arr[Av]
            mB = ~leg_arr[Bv]
            # reindex pair ids after filtering
            Av_c, Ap_c = Av[mA], Ap[mA]
            Bv_c, Bp_c = Bv[mB], Bp[mB]
            inter_c, sa_c, sb_c = _overlap_counts(Av_c, Ap_c, Bv_c, Bp_c,
                                                  npr, self.V_tok)
            _, _, ovr_c, cont_c, _, _, sub_c = _ratio_stats(inter_c, sa_c, sb_c)
            F[sl, col["core_subset"]] = sub_c
            F[sl, col["name_exact_core"]] = (
                (ovr_c >= 0.999) & (cont_c >= 0.999)).astype(np.float32)
            del Av, Ap, Bv, Bp, Av_c, Ap_c, Bv_c, Bp_c, inter_c

            # ---------- phone overlaps
            t2p = np.asarray(rd.tok2phone)
            Av, Ap = _ragged_gather(rd.name_flat, rd.name_off, A)
            Bv, Bp = _ragged_gather(rd.name_flat, rd.name_off, B)
            Av_p = t2p[Av]
            Bv_p = t2p[Bv]
            inter_p, sa_p, sb_p = _overlap_counts(Av_p, Ap, Bv_p, Bp, npr,
                                                  self.V_ph)
            pj, pd, po, pc, _, _, ps = _ratio_stats(inter_p, sa_p, sb_p)
            F[sl, col["name_ph_jac"]] = pj
            F[sl, col["name_ph_dice"]] = pd
            F[sl, col["name_ph_ovr_co"]] = po
            F[sl, col["ph_subset"]] = ps
            del Av, Ap, Bv, Bp, Av_p, Bv_p, inter_p

            # ---------- addr token overlaps
            Av, Ap = _ragged_gather(rd.addr_flat, rd.addr_off, A)
            Bv, Bp = _ragged_gather(rd.addr_flat, rd.addr_off, B)
            inter_a, sa_a, sb_a = _overlap_counts(Av, Ap, Bv, Bp, npr,
                                                  self.V_tok)
            aj, ad, ao, ac, _, _, _ = _ratio_stats(inter_a, sa_a, sb_a)
            F[sl, col["addr_tok_jac"]] = aj
            F[sl, col["addr_tok_dice"]] = ad
            F[sl, col["addr_tok_ovr_co"]] = ao
            F[sl, col["addr_exact_norm"]] = (
                (aj >= 0.999) & (ac >= 0.999)).astype(np.float32)
            del Av, Ap, Bv, Bp, inter_a

            # ---------- addr digit overlaps
            Av, Ap = _ragged_gather(rd.dig_flat, rd.dig_off, A)
            Bv, Bp = _ragged_gather(rd.dig_flat, rd.dig_off, B)
            inter_d, sa_d, sb_d = _overlap_counts(Av, Ap, Bv, Bp, npr,
                                                  1 << 31)
            dj, dd, do_, dc, _, _, ds = _ratio_stats(inter_d, sa_d, sb_d)
            F[sl, col["addr_digit_jac"]] = dj
            F[sl, col["addr_digit_eq"]] = (
                (do_ >= 0.999) & (dj > 0)).astype(np.float32)
            F[sl, col["addr_digit_sub"]] = ds
            F[sl, col["num_tokens_eq"]] = (do_ >= 0.999).astype(np.float32)
            del Av, Ap, Bv, Bp, inter_d

            # ---------- pin / state / domain / country / empties
            F[sl, col["pin_eq"]] = (rd.pin[A] == rd.pin[B])
            F[sl, col["state_eq"]] = [1.0 if (states[i] and states[j]
                                              and states[i] == states[j])
                                      else 0.0 for i, j in zip(A, B)]
            F[sl, col["state_missing"]] = [0.0 if states[i] and states[j]
                                           else 1.0 for i, j in zip(A, B)]
            F[sl, col["domain_eq"]] = (rd.dom[A] == rd.dom[B])
            F[sl, col["dom_any"]] = (rd.dom[A] > 0)
            F[sl, col["country_eq"]] = (rd.country[A] == rd.country[B])
            F[sl, col["b_addr_empty"]] = (self.addr_nuniq[B] == 0)
            F[sl, col["a_addr_empty"]] = (self.addr_nuniq[A] == 0)

            # ---------- string similarities (decoded once per side)
            raw_a = [self.raw(int(i)) for i in A]
            raw_b = [self.raw(int(i)) for i in B]
            an = [x[0] for x in raw_a]
            bn = [x[0] for x in raw_b]
            aa = [x[1] for x in raw_a]
            ba = [x[1] for x in raw_b]
            F[sl, col["name_fuzz_raw"]] = _ew_sim(an, bn,
                                                  Indel.normalized_similarity)
            F[sl, col["name_jw"]] = _ew_sim(
                [x.lower() for x in an], [x.lower() for x in bn],
                JaroWinkler.normalized_similarity)
            F[sl, col["name_fuzz_norm"]] = _ew_sim(
                [self._blob_str(self.norm_blob, self.str_off, i) for i in A],
                [self._blob_str(self.norm_blob, self.str_off, i) for i in B],
                Indel.normalized_similarity)
            F[sl, col["name_fuzz_core"]] = _ew_sim(
                [self._blob_str(self.core_blob, self.core_off, i) for i in A],
                [self._blob_str(self.core_blob, self.core_off, i) for i in B],
                Indel.normalized_similarity)
            F[sl, col["name_fuzz_alt"]] = np.maximum(
                _ew_sim([self._blob_str(self.norm_blob, self.str_off, i) for i in A],
                        [self._blob_str(self.alt_blob, self.alt_off, i) for i in B],
                        Indel.normalized_similarity),
                _ew_sim([self._blob_str(self.alt_blob, self.alt_off, i) for i in A],
                        [self._blob_str(self.norm_blob, self.str_off, i) for i in B],
                        Indel.normalized_similarity))
            F[sl, col["addr_fuzz"]] = _ew_sim(aa, ba,
                                              Indel.normalized_similarity)
            F[sl, col["addr_jw"]] = _ew_sim(
                [x.lower() for x in aa], [x.lower() for x in ba],
                JaroWinkler.normalized_similarity)

            # ---------- misc features (token lists per chunk)
            nt_a = [self.rd.name_toks(i).tolist() for i in A]
            nt_b = [self.rd.name_toks(i).tolist() for i in B]
            F[sl, col["name_len_diff"]] = np.abs(
                np.array([len(x) for x in nt_a]) -
                np.array([len(x) for x in nt_b]))
            ca = [self._blob_str(self.core_blob, self.core_off, i).split()
                  for i in A]
            cb = [self._blob_str(self.core_blob, self.core_off, i).split()
                  for i in B]
            F[sl, col["name_shorter_in_longer"]] = [
                1.0 if (x and set(x) <= set(y)) or (y and set(y) <= set(x))
                else 0.0 for x, y in zip(ca, cb)]
            F[sl, col["name_subseq"]] = [
                1.0 if _is_subseq(x, y) or _is_subseq(y, x) else 0.0
                for x, y in zip(nt_a, nt_b)]
            F[sl, col["nums_eq"]] = [
                1.0 if sorted(t for t in x if t in self._num_id_set) ==
                      sorted(t for t in y if t in self._num_id_set)
                else 0.0 for x, y in zip(nt_a, nt_b)]
            F[sl, col["a_name_df"]] = np.log1p(
                [float(self.tok_df[min(x)]) if x else 0.0 for x in nt_a])
            F[sl, col["b_name_df"]] = np.log1p(
                [float(self.tok_df[min(x)]) if x else 0.0 for x in nt_b])
            core_b_ids = [[t for t in x if t not in self.legal_ids]
                          for x in nt_b]
            F[sl, col["tok_rarity"]] = [
                float(self.tok_df[min(y)]) if y else 0.0 for y in core_b_ids]

            if log:
                print(f"  features {end}/{n} ({end/max(n,1):.0%})", flush=True)
            F.flush()
        return F
