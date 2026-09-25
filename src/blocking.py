"""Blocking / candidate generation via 64-bit key hashing + numpy grouping.

For each record we compute a set of blocking-key hashes; identical hashes form
groups (found by np.argsort on the hash array).  Groups containing both S1 and
S2/S3 records, with an S2/S3 side no larger than `cap`, emit candidate pairs.

Keys (country folded into every key):
  K1  sorted core-name token set       (name tokens minus legal suffixes)
  K2  (first core phone, sorted rest core phones)
  K3  address pin code
  K4  (first addr digit group, first addr phone token)
  K5  sorted core-phone token set
  K6  domain id
  K7  (first core phone, last core phone)
  K8  (first 3 core phones, first 2 addr tokens)

Both the primary and the alternate (romanization variant) views contribute
keys, so a pair is emitted if ANY view matches on ANY key.
"""
import pickle
import sys
import time

import numpy as np

ART = "artifacts"
MASK = (1 << 62) - 1  # stay within signed int64


def load_data(tag):
    d = np.load(f"{ART}/{tag}_data.npz", allow_pickle=False)
    # materialize into a plain dict: npz __getitem__ re-reads/decompresses
    # the whole array on EVERY access otherwise
    d = {k: d[k] for k in d.files}
    with open(f"{ART}/{tag}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return d, meta


class RecData:
    """Convenient per-record views over the packed arrays."""

    def __init__(self, d, meta):
        self.d = d
        self.meta = meta
        self.eid = d["eid"]
        if "src" in d:
            self.src = d["src"]
        else:
            # derive from entity-id prefixes (0 = S1, 1 = S2/S3)
            self.src = np.fromiter(
                (0 if e.startswith("S1-") else 1 for e in d["eid"]),
                dtype=np.int8, count=len(d["eid"]))
        self.country = d["country"]
        self.pin = d["pin"]
        self.dom = d["dom"]
        self.name_off = d["name_off"]
        self.addr_off = d["addr_off"]
        self.name_flat = d["name_flat"]
        self.addr_flat = d["addr_flat"]
        self.alt_rows = d["alt_rows"]
        self.alt_name_flat = d["alt_name_flat"]
        self.alt_name_off = d["alt_name_off"]
        self.alt_addr_flat = d["alt_addr_flat"]
        self.alt_addr_off = d["alt_addr_off"]
        self.dig_flat = d["dig_flat"]
        self.dig_off = d["dig_off"]
        self.tok2phone = meta["tok2phone"]
        self.tok_vocab = meta["tok_vocab"]
        self.n = len(self.eid)
        self.alt_pos = {int(r): i for i, r in enumerate(self.alt_rows)}

    def name_toks(self, i):
        return self.name_flat[self.name_off[i]:self.name_off[i + 1]]

    def addr_toks(self, i):
        return self.addr_flat[self.addr_off[i]:self.addr_off[i + 1]]

    def alt_name_toks(self, i):
        p = self.alt_pos.get(i)
        if p is None:
            return None
        return self.alt_name_flat[self.alt_name_off[p]:self.alt_name_off[p + 1]]

    def alt_addr_toks(self, i):
        p = self.alt_pos.get(i)
        if p is None:
            return None
        return self.alt_addr_flat[self.alt_addr_off[p]:self.alt_addr_off[p + 1]]

    def addr_digits(self, i):
        return self.dig_flat[self.dig_off[i]:self.dig_off[i + 1]]


def _hash_obj(o):
    return hash(o) & MASK


def build_key_hashes(rd, verbose=True, pin=None, state_arr=None):
    """Return (key_ids int64[N, K], names of keys).  Column k = hash of key k."""
    t0 = time.time()
    import textnorm
    tv = rd.tok_vocab
    legal_ids = frozenset(tv[w] for w in textnorm.LEGAL_SET if w in tv)

    N = rd.n
    tok2phone = rd.tok2phone
    country = rd.country
    pin = rd.pin
    dom = rd.dom

    KH = 13  # key families (10 original + 3 rare-token families)
    keys = np.zeros((N, KH), dtype=np.int64)
    # token/phone document frequencies for rare-pair keys
    tok_df = np.bincount(rd.d["name_flat"], minlength=len(rd.tok_vocab))
    t2p_arr = np.asarray(rd.tok2phone)
    ph_df = np.bincount(t2p_arr[rd.d["name_flat"]],
                        minlength=len(rd.meta["phone_vocab"]))

    for i in range(N):
        c = int(country[i])
        t = rd.name_toks(i)
        core = [int(x) for x in t if x not in legal_ids]
        ph = [int(tok2phone[x]) for x in core]
        at = rd.alt_name_toks(i)
        acore = [int(x) for x in at if x not in legal_ids] if at is not None and len(at) else None
        aph = [int(tok2phone[x]) for x in acore] if acore else None

        h = [0] * KH
        if core:
            h[0] = _hash_obj((c, tuple(sorted(core))))
        if ph:
            h[1] = _hash_obj((c, ph[0], tuple(sorted(ph[1:]))))
            h[4] = _hash_obj((c, tuple(sorted(set(ph)))))
            if len(ph) >= 2:
                h[6] = _hash_obj((c, ph[0], ph[-1]))
            if len(ph) >= 3:
                dt = rd.addr_toks(i)
                d1 = int(dt[0]) if len(dt) else -1
                d2 = int(dt[1]) if len(dt) > 1 else -1
                h[7] = _hash_obj((c, ph[0], ph[1], ph[2], d1, d2))
        p = int(pin[i])
        if p >= 0:
            h[2] = _hash_obj((c, p))
        dg = rd.addr_digits(i)
        dt2 = rd.addr_toks(i)
        ap0 = int(tok2phone[dt2[0]]) if len(dt2) else -1
        if len(dg) and ap0 >= 0:
            h[3] = _hash_obj((c, int(dg[0]), ap0))
        dv = int(dom[i])
        if dv:
            h[5] = _hash_obj((c, dv))
        # alt view gets its own key columns (K1alt -> col 8, K5alt -> col 9)
        if acore:
            h[8] = _hash_obj((c, tuple(sorted(acore))))
        if aph:
            h[9] = _hash_obj((c, tuple(sorted(set(aph)))))
        # rare-token-pair keys (col 10: 2 rarest phones, col 11: rarest phone
        # + first addr digit, col 12: 2 rarest core tokens)
        if ph:
            phs = list(dict.fromkeys(ph))
            ph_sorted = sorted(phs, key=lambda p: ph_df[p])
            if len(ph_sorted) >= 2:
                h[10] = _hash_obj((c, tuple(sorted(ph_sorted[:2]))))
            if len(ph_sorted) >= 1 and len(dg):
                h[11] = _hash_obj((c, ph_sorted[0], int(dg[0])))
        if core:
            toks_sorted = sorted(set(core), key=lambda x: tok_df[x])
            if len(toks_sorted) >= 2:
                h[12] = _hash_obj((c, tuple(sorted(toks_sorted[:2]))))
        keys[i] = h
        if verbose and i % 2000000 == 0 and i:
            print(f"  key hashing {i} ({time.time()-t0:.0f}s)", flush=True)

    print(f"key hashing done ({time.time()-t0:.0f}s)", flush=True)
    return keys


# per-key-family caps on S2/S3-side group size
KEY_CAPS = {
    0: 20,    # exact core-token set (generic names explode; filter bigger groups)
    1: 30,    # phone0 + sorted rest
    2: 15,    # pin code
    3: 30,    # first addr digit + first addr phone
    4: 30,    # exact phone set
    5: 60,    # domain
    6: 30,    # first+last phone
    7: 30,    # first-3-phones + 2 addr tokens
    8: 20,    # alt-view exact core-token set
    9: 30,    # alt-view phone set
    10: 30,   # 2 rarest phones
    11: 30,   # rarest phone + first addr digit
    12: 30,   # 2 rarest tokens
}


def _emit_group(g1, g2, pin, state_arr, k, out1, out2, stats):
    """Emit one group's cross-source pairs.

    Exact-name groups (keys 0/8) larger than their cap keep only pairs
    corroborated by pin or state agreement.  Other oversized groups are
    capped by product size and skipped beyond it.
    """
    kcap = KEY_CAPS.get(k, 60)
    prod = len(g1) * len(g2)
    if len(g2) <= kcap and prod <= 250000:
        out1.append(np.repeat(g1, len(g2)))
        out2.append(np.tile(g2, len(g1)))
        stats["emit"] += prod
        return
    if k in (0, 8) and pin is not None and prod <= 40000000:
        g1_pin = pin[g1]
        g2_pin = pin[g2]
        g1_st = state_arr[g1]
        g2_st = state_arr[g2]
        pin_eq = (g1_pin[:, None] == g2_pin[None, :]) & (g1_pin[:, None] >= 0)
        st_eq = (g1_st[:, None] == g2_st[None, :]) & (g1_st[:, None] != 0)
        ok = pin_eq | st_eq
        ar, bc = np.nonzero(ok)
        if len(ar):
            out1.append(g1[ar])
            out2.append(g2[bc])
            stats["emit"] += len(ar)
        stats["filtered"] += 1
    else:
        stats["skipped"] += 1


def emit_pairs_from_keys(keys, src, cap=60, cap_exact=400, verbose=True,
                         pin=None, state_arr=None):
    """Group identical key hashes; emit cross-source pairs with dedup.

    Oversized groups: exact-name groups (keys 0/8) fall back to emitting only
    pairs that also agree on pin or state; other groups are skipped.

    Returns (s1_idx, s23_idx) int64 arrays (unique).
    """
    t0 = time.time()
    N = keys.shape[0]
    is1 = (src == 0)
    seen_keys = np.zeros(0, dtype=np.int64)
    s1_all = np.zeros(0, dtype=np.int64)
    s2_all = np.zeros(0, dtype=np.int64)
    for k in range(keys.shape[1]):
        col = keys[:, k]
        nz = np.nonzero(col)[0]
        if len(nz) < 2:
            continue
        order = np.argsort(col[nz], kind="stable")
        idx = nz[order]
        vals = col[idx]
        starts = np.nonzero(np.diff(vals) != 0)[0] + 1
        bounds = np.concatenate([[0], starts, [len(idx)]])
        kcap = KEY_CAPS.get(k, cap)
        p1 = []
        p2 = []
        stats = dict(emit=0, filtered=0, skipped=0)
        for b in range(len(bounds) - 1):
            lo, hi = bounds[b], bounds[b + 1]
            g = idx[lo:hi]
            if hi - lo < 2:
                continue
            g1 = g[is1[g]]
            g2 = g[~is1[g]]
            if len(g1) == 0 or len(g2) == 0:
                continue
            _emit_group(g1, g2, pin, state_arr, k, p1, p2, stats)
        if p1:
            k1 = np.concatenate(p1)
            k2 = np.concatenate(p2)
            s1_all = np.concatenate([s1_all, k1])
            s2_all = np.concatenate([s2_all, k2])
            del k1, k2
        if verbose:
            print(f"  key {k}: emitted {stats['emit']} (filtered "
                  f"{stats['filtered']}, skipped {stats['skipped']}) "
                  f"-> cum raw {len(s1_all)} ({time.time()-t0:.0f}s)",
                  flush=True)
    if len(s1_all) == 0:
        return s1_all.astype(np.int64), s2_all.astype(np.int64)
    combo = s1_all.astype(np.int64) * N + s2_all.astype(np.int64)
    del s1_all, s2_all
    combo = np.unique(combo)
    s1_u = (combo // N).astype(np.int64)
    s2_u = (combo % N).astype(np.int64)
    del combo
    print(f"pairs: {len(s1_u)} unique ({time.time()-t0:.0f}s)", flush=True)
    return s1_u, s2_u


def build_candidates(tag, cap=60, cap_exact=400):
    d, meta = load_data(tag)
    rd = RecData(d, meta)
    print(f"loaded {rd.n} records")
    # state abbreviations as a small int-coded array for the group filter
    # ("" -> code 0 so "known state" checks are != 0)
    states = meta["states"]
    st_codes = {"": 0}
    state_arr = np.array([st_codes.setdefault(s, len(st_codes)) for s in states],
                         dtype=np.int32)
    keys = build_key_hashes(rd, pin=rd.pin, state_arr=state_arr)
    s1, s2 = emit_pairs_from_keys(keys, rd.src, cap=cap, cap_exact=cap_exact,
                                  pin=rd.pin, state_arr=state_arr)
    np.savez(f"{ART}/{tag}_pairs.npz", s1=s1, s2=s2)
    print(f"saved {ART}/{tag}_pairs.npz: {len(s1)} pairs")


def eval_candidate_recall(tag):
    d, meta = load_data(tag)
    rd = RecData(d, meta)
    pairs = np.load(f"{ART}/{tag}_pairs.npz")
    s1, s2 = pairs["s1"], pairs["s2"]
    eid2idx = meta["eid2idx"]
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8", newline="") as f:
        import csv
        rdr = csv.DictReader(f, delimiter="\t")
        gt_hits = 0
        gt_tot = 0
        full_recall_entities = 0
        n_entities = 0
        cand_per_s1 = {}
        for a, b in zip(s1, s2):
            cand_per_s1.setdefault(int(a), set()).add(int(b))
        for row in rdr:
            i = eid2idx.get(row["source1_entity_id"])
            if i is None:
                continue
            mids = [m for m in row["matched_entity_ids"].split(",") if m]
            gt_idx = {eid2idx[m] for m in mids if m in eid2idx}
            n_entities += 1
            if not gt_idx:
                full_recall_entities += 1  # singleton: trivially fully recalled
                continue
            got = cand_per_s1.get(i, set()) & gt_idx
            gt_tot += len(gt_idx)
            gt_hits += len(got)
            if len(got) == len(gt_idx):
                full_recall_entities += 1
    print(f"candidate pair recall: {gt_hits}/{gt_tot} = {gt_hits/max(gt_tot,1):.4f}")
    print(f"entities with full gold recall: {full_recall_entities}/{n_entities} "
          f"= {full_recall_entities/max(n_entities,1):.4f} (F0.5 ceiling)")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "train"
    build_candidates(tag)
    if tag == "train":
        eval_candidate_recall(tag)
