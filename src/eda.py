"""EDA for the Business Entity Resolution challenge.

Loads the three train sources + ground truth, prints sizes, match-rate
distributions, country/name noise patterns and samples of matched vs
non-matched record pairs so we can design normalization, blocking and features.
"""
import csv
import random
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

BASE = "dataset/train"


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    random.seed(0)
    s1 = read_tsv(f"{BASE}/train_source1.tsv")
    s2 = read_tsv(f"{BASE}/train_source2.tsv")
    s3 = read_tsv(f"{BASE}/train_source3.tsv")
    gt = read_tsv(f"{BASE}/train_ground_truth.tsv")

    print("sizes:", len(s1), len(s2), len(s3), "gt rows:", len(gt))

    for name, rows in (("s1", s1), ("s2", s2), ("s3", s3)):
        c = Counter(r["country"] for r in rows)
        print(name, "countries:", c)

    # ground truth stats
    n_matches = []
    gt_map = {}
    for row in gt:
        m = [x for x in row["matched_entity_ids"].split(",") if x]
        gt_map[row["source1_entity_id"]] = m
        n_matches.append(len(m))
    dist = Counter(n_matches)
    print("matches-per-S1 distribution (top):", sorted(dist.items())[:15])
    print("singletons (0 matches):", dist.get(0, 0), "/", len(n_matches),
          f"({dist.get(0, 0)/len(n_matches):.1%})")
    print("total positive pairs:", sum(n_matches))

    # how many S1 ids in gt actually exist in s1 file
    s1_ids = {r["entity_id"] for r in s1}
    print("gt ids in s1 file:", len(gt_map), "of", len(s1_ids), "s1 ids",
          "| gt ids missing from s1:", len(set(gt_map) - s1_ids))

    # positive pair composition by source
    comp = Counter()
    for m in gt_map.values():
        comp[sum(1 for x in m if x.startswith("S2-"))] += 0  # placeholder
    comp2 = Counter()
    for m in gt_map.values():
        n2 = sum(1 for x in m if x.startswith("S2-"))
        n3 = sum(1 for x in m if x.startswith("S3-"))
        comp2[(n2 > 0, n3 > 0)] += 1
    print("pairs with S2-only/S3-only/both:",
          comp2[(True, False)], comp2[(False, True)], comp2[(True, True)])

    # duplicate ids?
    for name, rows in (("s1", s1), ("s2", s2), ("s3", s3)):
        ids = [r["entity_id"] for r in rows]
        print(name, "unique ids:", len(set(ids)))

    # field emptiness
    for name, rows in (("s1", s1), ("s2", s2), ("s3", s3)):
        empty_name = sum(1 for r in rows if not r["business_name"].strip())
        empty_addr = sum(1 for r in rows if not r["business_address"].strip())
        print(name, "empty names:", empty_name, "empty addresses:", empty_addr)

    # name token counts, digits presence, non-latin script presence
    def is_devanagari(s):
        return bool(re.search(r"[\u0900-\u097F]", s))

    for name, rows in (("s1", s1), ("s2", s2), ("s3", s3)):
        dev = sum(1 for r in rows if is_devanagari(r["business_name"]))
        dig = sum(1 for r in rows if re.search(r"\d", r["business_name"]))
        print(name, "devanagari names:", dev, "names with digits:", dig)

    # examples of matched pairs
    by_id = {}
    for rows in (s1, s2, s3):
        for r in rows:
            by_id[r["entity_id"]] = r

    matched_examples = [k for k, v in gt_map.items() if v]
    print("\n--- sample matched groups ---")
    for k in random.sample(matched_examples, 8):
        recs = [by_id[k]] + [by_id[m] for m in gt_map[k]]
        for r in recs:
            print(f"  {r['entity_id']}\t[{r['country']}]\t{r['business_name']!r}\t{r['business_address']!r}")
        print()

    # examples of S1 entities whose address is empty in some sources
    print("--- sample singletons ---")
    single = [k for k, v in gt_map.items() if not v]
    for k in random.sample(single, 5):
        r = by_id[k]
        print(f"  {r['entity_id']}\t[{r['country']}]\t{r['business_name']!r}\t{r['business_address']!r}")


if __name__ == "__main__":
    main()
