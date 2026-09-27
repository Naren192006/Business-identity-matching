"""End-to-end driver for the embedding-blocking stage.

Stages (each cached; delete artifacts to rerun):
  python3 src/run_emb_stage.py              # full train pipeline + report
  python3 src/run_emb_stage.py --smoke      # tiny subset, sanity only
  python3 src/run_emb_stage.py --test       # embed test tag + generate test candidates

Pipeline:
  1. text blobs (train, test)        -> artifacts/{tag}_emb_text.npz
  2. encoder artifacts/emb.npz       (train_emb.py: InfoNCE + hard negs)
  3. embeddings emb_E_{tag}.npy      (fp16 memmap, all records)
  4. IVF index emb_ivf_{tag}.npz     (two-level k-means)
  5. candidates {tag}_pairs_emb.npz  (ANN top-k per S1)
  6. union {tag}_pairs_all.npz       (keys + emb candidates)
  7. metrics report                  (pair recall / ceiling vs key blocking)
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import load_data

ART = "artifacts"


def sh(cmd):
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd)
    print(f"=== rc={r.returncode} ({time.time()-t0:.0f}s) ===", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="generate test-tag candidates (union too)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n1", type=int, default=2)
    ap.add_argument("--n2", type=int, default=4)
    args = ap.parse_args()

    lim = ["--limit", "20000"] if args.smoke else []
    lims = f"_lim{20000}" if args.smoke else ""

    # ---------------- train side
    if not args.test:
        sh(["src/train_emb.py", "--tag", "train", "--epochs",
            str(args.epochs)] + lim)
        if args.smoke:
            print("--smoke: trained on a 20K subset only; candidate "
                  "generation requires the full run (rerun without "
                  "--smoke)", flush=True)
            return

        # candidates + union + metrics (train tag)
        sh(["src/emb_candidates.py", "--run", "--tag", "train",
            "--topk", str(args.topk), "--n1", str(args.n1),
            "--n2", str(args.n2), "--merge"])

        # pairdata for the union (labels + val split) + blocking report
        sh(["src/pairs_data_emb.py", "--tag", "train",
            "--in", "train_pairs_all", "--suffix", "_all"])
        return

    # ---------------- test side (requires trained emb.npz + emb_E_train)
    sh(["src/emb_candidates.py", "--run", "--tag", "test",
        "--topk", str(args.topk), "--n1", str(args.n1), "--n2", str(args.n2),
        "--merge"])
    sh(["src/pairs_data_emb.py", "--tag", "test",
        "--in", "test_pairs_all", "--suffix", "_all"])


if __name__ == "__main__":
    main()
