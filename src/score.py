"""Scoring: macro-averaged F0.5 over S1 entities (challenge formula).

pred: dict s1_eid -> set(matched eids)
gold: dict s1_eid -> set(matched eids)
"""
import csv


def f05_macro(pred, gold, verbose=False):
    """pred/gold: dict s1_eid -> iterable of matched S2/S3 eids."""
    total = 0.0
    n = 0
    worst = []
    for s1, gold_set in gold.items():
        gold_set = set(gold_set)
        pred_set = set(pred.get(s1, ())) & set(
            m for m in pred.get(s1, ()) if m.startswith(("S2-", "S3-")))
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        if prec + rec == 0:
            f = 0.0
        else:
            f = (1.25 * prec * rec) / (0.25 * prec + rec)
        total += f
        n += 1
        if verbose and f < 0.5:
            worst.append((f, s1, sorted(pred_set), sorted(gold_set)))
    if verbose:
        worst.sort()
        for f, s1, p, g in worst[:15]:
            print(f"  F={f:.3f} {s1}: pred={p[:6]} gold={g[:6]}")
    return total / max(n, 1)


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            mids = [m for m in row["matched_entity_ids"].split(",") if m]
            gold[row["source1_entity_id"]] = set(mids)
    return gold


if __name__ == "__main__":
    import sys
    gold = load_gold(sys.argv[1])
    pred = load_gold(sys.argv[2])
    print(f"F0.5 macro = {f05_macro(pred, gold):.5f}")
