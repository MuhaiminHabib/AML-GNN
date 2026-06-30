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


MODEL_NAME_MAP = {
    "graphsage_raw_undirected": "GraphSAGE",
    "gatv2_dir_heads8_hidden32_cw035": "GATv2-dir",
    "gcn_norm_undirected": "GCN",
}

EXPLAINER_NAME_MAP = {
    "gnnexplainer": "GNNExplainer",
    "pgexplainer": "PGExplainer",
    "subgraphx_mcts": "SubgraphX-MCTS",
}


def prepare_data():
    df = pd.read_csv(INPUT_PATH)

    df["model"] = df["experiment_name"].map(MODEL_NAME_MAP)
    df["explainer_name"] = df["explainer"].map(EXPLAINER_NAME_MAP)

    return df


def plot_metric(df, metric_column, ylabel, output_filename):
    pivot_df = df.pivot(
        index="model",
        columns="explainer_name",
        values=metric_column,
    )

    pivot_df = pivot_df.loc[["GCN", "GraphSAGE", "GATv2-dir"]]

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

    report_table = df[
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

    report_table = report_table.sort_values(
        by=["explainer_name", "model"]
    )

    report_table_path = (
        PROJECT_ROOT
        / "outputs"
        / "faithfulness"
        / "summary"
        / "faithfulness_report_table.csv"
    )

    report_table.to_csv(report_table_path, index=False)

    print("Saved clean report table:")
    print(report_table_path)

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