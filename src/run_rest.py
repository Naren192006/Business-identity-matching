"""Driver: train model, block test, features test, predict, validate."""
import subprocess
import sys
import time

STEPS = [
    ["-X", "utf8", "src/train_match.py"],
    ["-X", "utf8", "src/blocking.py", "test"],
    ["-X", "utf8", "src/pairs_data.py", "test"],
    ["-X", "utf8", "src/build_features.py", "test"],
    ["-X", "utf8", "src/predict_test.py", "test"],
    ["-X", "utf8", "utils/validate_submission.py",
     "--matching", "output/matching_results.tsv",
     "--candidate", "output/candidate_pairs.tsv",
     "--test-dir", "dataset/test"],
]

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for i, step in enumerate(STEPS):
        if i < start:
            continue
        t0 = time.time()
        print(f"=== step {i}: {' '.join(step)} ===", flush=True)
        r = subprocess.run([sys.executable] + step)
        print(f"=== step {i} rc={r.returncode} ({time.time()-t0:.0f}s) ===",
              flush=True)
        if r.returncode != 0:
            print("HALTING on failure", flush=True)
            sys.exit(r.returncode)
    print("ALL DONE", flush=True)
