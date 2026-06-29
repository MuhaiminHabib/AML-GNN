import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd


def mean_std(series):
    return f"{series.mean():.4f} ± {series.std():.4f}"


def main():
    results_path = (
        PROJECT_ROOT
        / "outputs"
        / "results"
        / "multiseed"
        / "elliptic_multiseed_results.csv"
    )

    df = pd.read_csv(results_path)

    summary_rows = []

    for (dataset, model), group in df.groupby(["dataset", "model"]):
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "accuracy": mean_std(group["accuracy"]),
                "illicit_precision": mean_std(group["illicit_precision"]),
                "illicit_recall": mean_std(group["illicit_recall"]),
                "illicit_f1": mean_std(group["illicit_f1"]),
                "roc_auc": mean_std(group["roc_auc"]),
                "pr_auc": mean_std(group["pr_auc"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "results"
        / "multiseed"
        / "elliptic_multiseed_summary.csv"
    )

    summary_df.to_csv(output_path, index=False)

    print("\nMulti-seed summary:")
    print(summary_df.to_string(index=False))

    print("\nSaved summary to:")
    print(output_path)


if __name__ == "__main__":
    main()