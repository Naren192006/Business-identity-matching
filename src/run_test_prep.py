"""Driver: run all test preprocessing steps sequentially."""
import subprocess
import sys
import time

STEPS = [
    ["-X", "utf8", "src/preprocess.py", "test", "S2",
     "dataset/test/test_source2.tsv"],
    ["-X", "utf8", "src/preprocess.py", "test", "S3",
     "dataset/test/test_source3.tsv"],
    ["-X", "utf8", "src/preprocess.py", "test", "assemble"],
]

if __name__ == "__main__":
    for i, step in enumerate(STEPS):
        t0 = time.time()
        print(f"=== step {i}: {' '.join(step)} ===", flush=True)
        r = subprocess.run([sys.executable] + step)
        print(f"=== step {i} rc={r.returncode} ({time.time()-t0:.0f}s) ===",
              flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("ALL DONE", flush=True)
