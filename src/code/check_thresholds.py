"""
Compares the latest eval run's average scores against minimum acceptable
thresholds. Exits with a non-zero code (failing the CI build) if any
metric falls below its threshold.

Thresholds are set from the validated v9_0 + dual-query-fusion baseline
(Context Recall 0.92, Context Precision 0.88, Faithfulness 0.54), with a
small safety margin below each, so normal run-to-run judge variance
doesn't fail the build on its own -- only a real regression should.
"""

import os
import sys

import pandas as pd

THRESHOLDS = {
    "context_recall": 0.80,
    "context_precision": 0.75,
    "faithfulness": 0.40,
}

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval_results.csv")

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: {RESULTS_PATH} not found. Did run_eval.py run successfully?")
        sys.exit(1)

    df = pd.read_csv(RESULTS_PATH)

    missing = df[["context_recall", "context_precision", "faithfulness"]].isna().sum()
    if missing.sum() > 0:
        print("WARNING: some rows have missing (NaN) scores:")
        print(missing)

    averages = df[["context_recall", "context_precision", "faithfulness"]].mean()

    print("=== Evaluation Averages ===")
    failed = False
    for metric, threshold in THRESHOLDS.items():
        score = averages[metric]
        status = "PASS" if score >= threshold else "FAIL"
        if score < threshold:
            failed = True
        print(f"{metric:20s} {score:.4f}  (threshold: {threshold:.2f})  [{status}]")

    if failed:
        print("\nOne or more metrics fell below threshold. Failing build.")
        sys.exit(1)

    print("\nAll metrics passed threshold.")
    sys.exit(0)


if __name__ == "__main__":
    main()