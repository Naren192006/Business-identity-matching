"""Diagnose blocking quality on a gold-pair sample.

For sampled S1 entities and their gold S2/S3 matches, compute every blocking
key on both sides and report which keys match (ignoring caps).  Also report
token-DF stats of missed pairs and name-similarity distributions.
"""
import csv
import pickle
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")
from blocking import load_data, RecData, _hash_obj, MASK  # noqa: E402
import textnorm  # noqa: E402

ART = "artifacts"


def main():
    t0 = time.time()
    d, meta = load_data("train")
    rd = RecData(d, meta)
    tv = rd.tok_vocab
    legal_ids = frozenset(tv[w] for w in textnorm.LEGAL_SET if w in tv)
    tok2phone = rd.tok2phone

    # gold pairs for a sample of S1 entities
    gt = {}
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8",
              newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gt[row["source1_entity_id"]] = [m for m in
                                            row["matched_entity_ids"].split(",")
                                            if m]
    s1_ids_all = [e for e in d["eid"] if e.startswith("S1-")]
    random.seed(7)
    sample = random.sample(s1_ids_all, 60000)
    eid2idx = meta["eid2idx"]
    pairs = []
    for s1e in sample:
        i = eid2idx[s1e]
        for m in gt.get(s1e, []):
            j = eid2idx.get(m)
            if j is not None:
                pairs.append((i, j))
    print(f"{len(pairs)} gold pairs from 60k S1 sample", flush=True)

    # token DF for name tokens
    tok_df = np.bincount(d["name_flat"], minlength=len(tv)).astype(np.int64)

    KEY_NAMES = ["K1 tokset", "K2 ph0+rest", "K3 pin", "K4 dig+aph",
                 "K5 phset", "K6 dom", "K7 ph0+phlast", "K8 ph3+addr2",
                 "K9 alt-tokset", "K10 alt-phset"]

    def keyvec(i):
        c = int(rd.country[i])
        t = [int(x) for x in rd.name_toks(i)]
        core = [x for x in t if x not in legal_ids]
        ph = [int(tok2phone[x]) for x in core]
        at = rd.alt_name_toks(i)
        acore = [int(x) for x in at if int(x) not in legal_ids] if at is not None and len(at) else None
        aph = [int(tok2phone[x]) for x in acore] if acore else None
        h = [0] * 10
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
        p = int(rd.pin[i])
        if p >= 0:
            h[2] = _hash_obj((c, p))
        dg = rd.addr_digits(i)
        dt2 = rd.addr_toks(i)
        ap0 = int(tok2phone[dt2[0]]) if len(dt2) else -1
        if len(dg) and ap0 >= 0:
            h[3] = _hash_obj((c, int(dg[0]), ap0))
        dv = int(rd.dom[i])
        if dv:
            h[5] = _hash_obj((c, dv))
        if acore:
            h[8] = _hash_obj((c, tuple(sorted(acore))))
        if aph:
            h[9] = _hash_obj((c, tuple(sorted(set(aph)))))
        return h

    hits = Counter()
    anyhit = 0
    nph = 0
    # shared-token stats for pairs missing all keys
    df_shared = []
    ph_shared = []
    both_empty_addr = 0
    name_df_bucket = Counter()
    for (i, j) in pairs[::3]:
        ki = keyvec(i)
        kj = keyvec(j)
        matched = [k for k in range(10) if ki[k] and ki[k] == kj[k]]
        if matched:
            anyhit += 1
            for k in matched:
                hits[KEY_NAMES[k]] += 1
        else:
            nph += 1
            si = {int(x) for x in rd.name_toks(i) if int(x) not in legal_ids}
            sj = {int(x) for x in rd.name_toks(j) if int(x) not in legal_ids}
            sh = si & sj
            if sh:
                df_shared.append(int(tok_df[min(sh)]))
            ph_i = {int(tok2phone[x]) for x in si}
            ph_j = {int(tok2phone[x]) for x in sj}
            psh = ph_i & ph_j
            if psh:
                ph_shared.append(len(psh))
            if len(rd.addr_toks(i)) == 0 or len(rd.addr_toks(j)) == 0:
                both_empty_addr += 1
    tot = len(pairs[::3])
    print(f"ideal recall (any key, no caps): {anyhit}/{tot} = {anyhit/tot:.4f}")
    print(f"missed pairs: {nph} ({nph/tot:.1%}); of which one side empty addr: {both_empty_addr}")
    print("key hit counts (can overlap):", dict(hits))
    if df_shared:
        arr = np.array(df_shared)
        print(f"missed-with-shared-token: {len(arr)}; DF of rarest shared token "
              f"p25={np.percentile(arr,25):.0f} p50={np.percentile(arr,50):.0f} "
              f"p75={np.percentile(arr,75):.0f} p90={np.percentile(arr,90):.0f}")
        print(f"  share DF<=50: {(arr<=50).mean():.3f}  DF<=200: {(arr<=200).mean():.3f}")
    if ph_shared:
        print(f"missed-with-shared-phone: {len(ph_shared)}")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
