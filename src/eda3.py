"""EDA 3: quantify Indic-script names vs mixed vs Latin, transliteration difficulty,
and check S1 name/domain patterns."""
import csv
import re
import sys
import random
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

BASE = "dataset/train"

INDIC = re.compile(r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F"
                   r"\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]")


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def script_class(s):
    has_indic = bool(INDIC.search(s))
    has_latin = bool(re.search(r"[A-Za-z]", s))
    if has_indic and has_latin:
        return "mixed"
    if has_indic:
        return "indic"
    return "latin"


def main():
    random.seed(2)
    gt = read_tsv(f"{BASE}/train_ground_truth.tsv")
    s1 = {r["entity_id"]: r for r in read_tsv(f"{BASE}/train_source1.tsv")}
    s23 = {}
    for src in ("2", "3"):
        for r in read_tsv(f"{BASE}/train_source{src}.tsv"):
            s23[r["entity_id"]] = r

    # classes of matched S2/S3 names vs S1 names
    cnt = Counter()
    gt_sample = [row for row in gt if row["matched_entity_ids"]]
    for row in random.sample(gt_sample, 300000):
        c1 = script_class(s1[row["source1_entity_id"]]["business_name"])
        cnt[("s1", c1)] += 1
        for m in row["matched_entity_ids"].split(","):
            cnt[("m", script_class(s23[m]["business_name"]))] += 1
            cnt[("maddr", script_class(s23[m]["business_address"]))] += 1
    print(cnt)

    # S2/S3 distribution overall
    cnt2 = Counter()
    for eid, r in list(s23.items())[::17]:
        cnt2[(eid[:2], script_class(r["business_name"]))] += 1
    print("by source:", dict(cnt2))

    # domain names
    dom = sum(1 for eid, r in list(s23.items())[::17] if re.search(r"\w+\.(com|net|org|in|co)\b", r["business_name"], re.I))
    print("domain-like names (1/17 sample):", dom)

    # name length stats
    lens = [len(r["business_name"]) for r in list(s23.values())[::31]]
    lens.sort()
    print("name len p50/p90/p99/max:", lens[len(lens)//2], lens[int(len(lens)*0.9)], lens[int(len(lens)*0.99)], lens[-1])


if __name__ == "__main__":
    main()
