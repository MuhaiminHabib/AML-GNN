from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "faithfulness"
    / "summary"
    / "combined_explainer_faithfulness_summary.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "faithfulness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "faithfulness" / "summary"


FINAL_EXPERIMENTS = [
    "gcn_norm_undirected",
    "graphsage_raw_undirected_cw05",
    "gatv2_dir_heads8_hidden32_cw035",
]

MODEL_NAME_MAP = {
    "gcn_norm_undirected": "GCN",
    "graphsage_raw_undirected_cw05": "GraphSAGE",
    "gatv2_dir_heads8_hidden32_cw035": "GATv2-dir",
}

EXPLAINER_NAME_MAP = {
    "gnnexplainer": "GNNExplainer",
    "pgexplainer": "PGExplainer",
    "subgraphx_mcts": "SubgraphX-MCTS",
}

MODEL_ORDER = ["GCN", "GraphSAGE", "GATv2-dir"]
EXPLAINER_ORDER = ["GNNExplainer", "PGExplainer", "SubgraphX-MCTS"]


def prepare_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}\n\n"
            "Run this first:\n"
            "python scripts\\combine_explainer_summaries.py"
        )

    df = pd.read_csv(INPUT_PATH)

    df = df[df["experiment_name"].isin(FINAL_EXPERIMENTS)].copy()

    available_experiments = set(df["experiment_name"].unique())
    missing_experiments = [
        exp for exp in FINAL_EXPERIMENTS if exp not in available_experiments
    ]

    if missing_experiments:
        raise ValueError(
            "Some final experiments are missing from the combined faithfulness file:\n"
            + "\n".join(missing_experiments)
            + "\n\nThis usually means you have not rerun explainers for the final model yet.\n"
            "For GraphSAGE, rerun explainers using:\n"
            "graphsage_raw_undirected_cw05\n\n"
            "Then run:\n"
            "python scripts\\combine_explainer_summaries.py"
        )

    df["model"] = df["experiment_name"].map(MODEL_NAME_MAP)
    df["explainer_name"] = df["explainer"].map(EXPLAINER_NAME_MAP)

    missing_models = df[df["model"].isna()]
    missing_explainers = df[df["explainer_name"].isna()]

    if len(missing_models) > 0:
        raise ValueError(
            "Some experiment names are missing from MODEL_NAME_MAP:\n"
            + str(missing_models["experiment_name"].unique())
        )

    if len(missing_explainers) > 0:
        raise ValueError(
            "Some explainers are missing from EXPLAINER_NAME_MAP:\n"
            + str(missing_explainers["explainer"].unique())
        )

    duplicate_rows = df.duplicated(
        subset=["model", "explainer_name"],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Duplicate model/explainer rows found after filtering:\n"
            + str(
                df.loc[
                    duplicate_rows,
                    ["experiment_name", "model", "explainer", "explainer_name"],
                ]
            )
        )

    return df


def save_clean_report_table(df):
    table = df[
        [
            "explainer_name",
            "model",
            "num_nodes",
            "mean_deletion_drop",
            "mean_insertion_prob",
            "mean_sparsity",
            "deletion_label_flip_rate",
            "insertion_label_preservation_rate",
        ]
    ].copy()

    table["explainer_name"] = pd.Categorical(
        table["explainer_name"],
        categories=EXPLAINER_ORDER,
        ordered=True,
    )

    table["model"] = pd.Categorical(
        table["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    table = table.sort_values(["explainer_name", "model"])

    output_path = SUMMARY_OUTPUT_DIR / "faithfulness_report_table.csv"
    table.to_csv(output_path, index=False)

    print("Saved clean report table:")
    print(output_path)


def plot_metric(df, metric_column, ylabel, output_filename):
    pivot_df = df.pivot(
        index="model",
        columns="explainer_name",
        values=metric_column,
    )

    pivot_df = pivot_df.reindex(index=MODEL_ORDER, columns=EXPLAINER_ORDER)

    ax = pivot_df.plot(kind="bar", figsize=(9, 5))

    ax.set_xlabel("GNN Model")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by Explainer and Model")
    ax.legend(title="Explainer")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    df = prepare_data()

    save_clean_report_table(df)

    plot_metric(
        df,
        metric_column="mean_deletion_drop",
        ylabel="Mean Deletion Drop",
        output_filename="mean_deletion_drop.png",
    )

    plot_metric(
        df,
        metric_column="mean_insertion_prob",
        ylabel="Mean Insertion Probability",
        output_filename="mean_insertion_probability.png",
    )

    plot_metric(
        df,
        metric_column="mean_sparsity",
        ylabel="Mean Sparsity",
        output_filename="mean_sparsity.png",
    )

    plot_metric(
        df,
        metric_column="deletion_label_flip_rate",
        ylabel="Deletion Label Flip Rate",
        output_filename="deletion_label_flip_rate.png",
    )

    plot_metric(
        df,
        metric_column="insertion_label_preservation_rate",
        ylabel="Insertion Label Preservation Rate",
        output_filename="insertion_label_preservation_rate.png",
    )


if __name__ == "__main__":
    main()