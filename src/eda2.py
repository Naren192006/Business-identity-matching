"""EDA pass 2: structural properties of the matching problem.

Checks:
1. Does each S2/S3 record match at most one S1 entity? (determines clean
   validation splits and whether pair-wise model needs many-to-one logic)
2. Cross-country matches?
3. Non-latin script inventory (names + addresses) per source.
4. France records in test: what do they look like?
5. Vocabulary sizes for token hashing.
"""
import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

BASE = "dataset/train"
TEST = "dataset/test"


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


SCRIPT_RANGES = [
    ("deva", (0x0900, 0x097F)),
    ("beng", (0x0980, 0x09FF)),
    ("guru", (0x0A00, 0x0A7F)),
    ("gujr", (0x0A80, 0x0AFF)),
    ("orya", (0x0B00, 0x0B7F)),
    ("taml", (0x0B80, 0x0BFF)),
    ("telu", (0x0C00, 0x0C7F)),
    ("knda", (0x0C80, 0x0CFF)),
    ("mlym", (0x0D00, 0x0D7F)),
    ("cjk", (0x4E00, 0x9FFF)),
    ("arab", (0x0600, 0x06FF)),
    ("cyrl", (0x0400, 0x04FF)),
    ("grek", (0x0370, 0x03FF)),
]


def script_profile(s):
    found = set()
    for ch in s:
        o = ord(ch)
        if o < 128:
            continue
        for name, lo, hi in [(n, r[0], r[1]) for n, r in SCRIPT_RANGES]:
            if lo <= o <= hi:
                found.add(name)
                break
        else:
            found.add("other")
    return found


def main():
    s2 = read_tsv(f"{BASE}/train_source2.tsv")
    s3 = read_tsv(f"{BASE}/train_source3.tsv")
    gt = read_tsv(f"{BASE}/train_ground_truth.tsv")

    # 1. S2/S3 -> S1 uniqueness
    rev = defaultdict(list)
    for row in gt:
        for m in row["matched_entity_ids"].split(","):
            if m:
                rev[m].append(row["source1_entity_id"])
    multi = {k: v for k, v in rev.items() if len(v) > 1}
    print("S2/S3 records with >1 S1 match:", len(multi))
    for k, v in list(multi.items())[:5]:
        print("  ", k, v)

    # 2. country consistency of matches
    s2c = {r["entity_id"]: r["country"] for r in s2}
    s3c = {r["entity_id"]: r["country"] for r in s3}
    s1c = {r["entity_id"]: r["country"] for r in read_tsv(f"{BASE}/train_source1.tsv")}
    bad = 0
    checked = 0
    for row in gt:
        c1 = s1c[row["source1_entity_id"]]
        for m in row["matched_entity_ids"].split(","):
            if not m:
                continue
            checked += 1
            cm = s2c.get(m) or s3c.get(m)
            if cm != c1:
                bad += 1
    print(f"cross-country matched pairs: {bad} / {checked}")

    # 3. scripts per source field
    for label, rows in (("s2", s2), ("s3", s3)):
        nscripts = Counter()
        ascripts = Counter()
        for r in rows:
            nscripts.update(script_profile(r["business_name"]))
            ascripts.update(script_profile(r["business_address"]))
        print(label, "name scripts:", dict(nscripts), "addr scripts:", dict(ascripts))

    # 4. France peek
    for src in ("1", "2", "3"):
        with open(f"{TEST}/test_source{src}.tsv", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            fr = []
            countries = Counter()
            for r in rdr:
                countries[r["country"]] += 1
                if r["country"] == "France" and len(fr) < 6:
                    fr.append(r)
            print(f"test_source{src} countries:", dict(countries))
            for r in fr:
                print(f"  {r['entity_id']}\t{r['business_name']!r}\t{r['business_address']!r}")

    # 5. vocab sizes on a subsample
    import random
    random.seed(1)
    toks = set()
    for rows in (s2, s3):
        for r in random.sample(rows, 300000):
            toks.update(re.findall(r"\w+", r["business_name"].lower()))
            toks.update(re.findall(r"\w+", r["business_address"].lower()))
    print("sampled S2/S3 token vocab:", len(toks))

    # 6. duplicate normalized records inside S2/S3?
    def norm(s):
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"\W+", " ", s.lower()).strip()

    seen = Counter()
    for r in random.sample(s2, 300000):
        seen[(norm(r["business_name"]), norm(r["business_address"]))] += 1
    dups = sum(v - 1 for v in seen.values() if v > 1)
    print("duplicate norm (name,addr) within sampled S2:", dups)

    # 7. how many matched pairs have identical normalized name?
    same_name = same_addr = total = 0
    by_id = {}
    for rows in (s2, s3):
        for r in rows:
            by_id[r["entity_id"]] = r
    for row in gt[:400000]:
        if not row["matched_entity_ids"]:
            continue
        r1 = by_id.get(row["source1_entity_id"])
        if r1 is None:
            continue
        n1, a1 = norm(r1["business_name"]), norm(r1["business_address"])
        for m in row["matched_entity_ids"].split(","):
            r2 = by_id.get(m)
            if r2 is None:
                continue
            total += 1
            if norm(r2["business_name"]) == n1:
                same_name += 1
            if norm(r2["business_address"]) == a1:
                same_addr += 1
    print(f"exact-norm-name pairs: {same_name}/{total}, exact-norm-addr: {same_addr}/{total}")


if __name__ == "__main__":
    main()
