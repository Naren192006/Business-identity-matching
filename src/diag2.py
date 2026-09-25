"""Diag 2: would rarest-token/phone-pair keys catch the missed gold pairs?

For gold pairs missed by all current keys, check agreement on keys built from
the 2 rarest phones / tokens, and measure group sizes for cap planning.
"""
import csv
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")
from blocking import load_data, RecData, _hash_obj  # noqa: E402
import textnorm  # noqa: E402


def main():
    d, meta = load_data("train")
    rd = RecData(d, meta)
    tv = rd.tok_vocab
    legal_ids = frozenset(tv[w] for w in textnorm.LEGAL_SET if w in tv)
    tok2phone = np.asarray(rd.tok2phone)

    # phone DF
    ph_df = np.bincount(tok2phone[d["name_flat"]], minlength=len(meta["phone_vocab"]))
    tok_df = np.bincount(d["name_flat"], minlength=len(tv))

    gt = {}
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8",
              newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gt[row["source1_entity_id"]] = [m for m in
                                            row["matched_entity_ids"].split(",")
                                            if m]
    eid2idx = meta["eid2idx"]
    random.seed(7)
    sample = random.sample([e for e in d["eid"] if e.startswith("S1-")], 60000)
    pairs = []
    for s1e in sample:
        i = eid2idx[s1e]
        for m in gt.get(s1e, []):
            j = eid2idx.get(m)
            if j is not None:
                pairs.append((i, j))

    def rec_keys(i):
        c = int(rd.country[i])
        t = [int(x) for x in rd.name_toks(i)]
        core = [x for x in t if x not in legal_ids]
        phs = list(dict.fromkeys(int(tok2phone[x]) for x in core))
        # rarest 2 phones
        ph_sorted = sorted(phs, key=lambda p: ph_df[p]) if phs else []
        r2ph = tuple(sorted(ph_sorted[:2])) if len(ph_sorted) >= 2 else None
        r1ph = ph_sorted[0] if ph_sorted else None
        toks_sorted = sorted(set(core), key=lambda x: tok_df[x])
        r2tok = tuple(sorted(toks_sorted[:2])) if len(toks_sorted) >= 2 else None
        r1tok = toks_sorted[0] if toks_sorted else None
        # rare phone + first digit
        dg = rd.addr_digits(i)
        d0 = int(dg[0]) if len(dg) else None
        k_rph_d = (c, r1ph, d0) if (r1ph is not None and d0 is not None) else None
        return c, phs, core, r2ph, r1ph, r2tok, r1tok, k_rph_d

    catch = Counter()
    missed = 0
    group_r2ph = []
    for (i, j) in pairs[::3]:
        ci, phsi, corei, r2phi, r1phi, r2toki, r1toki, krphdi = rec_keys(i)
        cj, phsj, corej, r2phj, r1phj, r2tokj, r1tokj, krphdj = rec_keys(j)
        hit = False
        if r2phi and r2phi == r2phj and ci == cj:
            catch["r2ph"] += 1
            hit = True
        if r1phi and r1phi == r1phj and ci == cj:
            catch["r1ph"] += 1
            hit = True
            group_r2ph.append(ph_df[r1phi])
        if r2toki and r2toki == r2tokj and ci == cj:
            catch["r2tok"] += 1
            hit = True
        if r1toki and r1toki == r1tokj and ci == cj:
            catch["r1tok"] += 1
            hit = True
        if krphdi and krphdi == krphdj:
            catch["rph+d0"] += 1
            hit = True
        # any shared phone (upper bound)
        if not hit:
            sh = set(phsi) & set(phsj)
            if sh and ci == cj:
                catch["any_shared_ph"] += 1
            sh2 = set(corei) & set(corej)
            if sh2 and ci == cj:
                catch["any_shared_tok"] += 1
            missed += 1
    tot = len(pairs[::3])
    print(f"sampled {tot} gold pairs; still-missed by new keys: {missed} "
          f"({missed/tot:.1%})")
    print("catch counts:", dict(catch))
    if group_r2ph:
        arr = np.array(group_r2ph)
        print(f"r1ph DF at hits: p50={np.percentile(arr,50):.0f} "
              f"p90={np.percentile(arr,90):.0f} max={arr.max()}")


if __name__ == "__main__":
    main()
