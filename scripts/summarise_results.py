import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd


def mean_std(series):
    return f"{series.mean():.4f} ± {series.std():.4f}"


def extract_mean(value):
    """
    Extract the mean value from a string like '0.5969 ± 0.0160'
    so the summary can be sorted numerically.
    """
    return value.str.extract(r"([0-9.]+)").astype(float)[0]


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

    group_cols = [
        "experiment_name",
        "dataset",
        "model",
        "normalize",
        "undirected",
    ]

    for keys, group in df.groupby(group_cols):
        experiment_name, dataset, model, normalize, undirected = keys

        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "dataset": dataset,
                "model": model,
                "normalize": normalize,
                "undirected": undirected,
                "accuracy": mean_std(group["accuracy"]),
                "illicit_precision": mean_std(group["illicit_precision"]),
                "illicit_recall": mean_std(group["illicit_recall"]),
                "illicit_f1": mean_std(group["illicit_f1"]),
                "roc_auc": mean_std(group["roc_auc"]),
                "pr_auc": mean_std(group["pr_auc"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by="illicit_f1",
        ascending=False,
        key=extract_mean,
    )

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