"""Quick smoke test of textnorm on real data rows."""
import csv
import random
import sys

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from textnorm import build_record

BASE = "dataset/train"


def read_tsv(path, n=None):
    with open(path, encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        rows = []
        for i, r in enumerate(rdr):
            if n and i >= n:
                break
            rows.append(r)
    return rows


random.seed(3)
s1 = read_tsv(f"{BASE}/train_source1.tsv", 200000)
s2 = read_tsv(f"{BASE}/train_source2.tsv", 200000)

for r in random.sample(s1, 4):
    rec = build_record(r["entity_id"], "S1", r["country"], r["business_name"], r["business_address"])
    print(f"[{rec.country}] {rec.name!r} | core={rec.name_core!r} | dom={rec.domain!r} pin={rec.addr_pin!r} st={rec.addr_state!r}")
    print(f"    addr={rec.addr!r}")
for r in random.sample(s2, 6):
    rec = build_record(r["entity_id"], "S2", r["country"], r["business_name"], r["business_address"])
    print(f"[{rec.country}] {rec.name!r} | core={rec.name_core!r} | dom={rec.domain!r} pin={rec.addr_pin!r} st={rec.addr_state!r}")
    print(f"    addr={rec.addr!r}")
