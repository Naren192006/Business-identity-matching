"""Build feature memmaps for train or test pairs.

Usage: python3 src/build_features.py <tag>
Requires artifacts/{tag}_pairs.npz; writes artifacts/{tag}_feats.npy.
Supports resume: completed chunks are detected by a marker file.
"""
import os
import sys
import time

import numpy as np

from features import FeatureBuilder, NF
from blocking import load_data

ART = "artifacts"


def main(tag):
    t0 = time.time()
    pairs = np.load(f"{ART}/{tag}_pairs.npz")
    s1 = pairs["s1"].astype(np.int64)
    s2 = pairs["s2"].astype(np.int64)
    n = len(s1)
    out = f"{ART}/{tag}_feats.npy"
    print(f"{tag}: {n} pairs -> {out}", flush=True)

    fb = FeatureBuilder(tag)
    fb.build(s1, s2, out, chunk=200000)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
