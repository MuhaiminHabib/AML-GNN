import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = [
    "graphsage_raw_undirected",
    "gatv2_dir_heads8_hidden32_cw035",
    "gcn_norm_undirected",
]

SEED = 42


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


def summarise_experiment(explainer_name, experiment_name, seed):
    input_path = (
        PROJECT_ROOT
        / "outputs"
        / "faithfulness"
        / explainer_name
        / experiment_name
        / f"seed_{seed}_faithfulness.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Faithfulness file not found: {input_path}")

    rows = read_csv(input_path)

    summary = {
        "explainer": explainer_name,
        "experiment_name": experiment_name,
        "seed": seed,
        "num_nodes": len(rows),
        "mean_original_prob": mean(row["original_prob"] for row in rows),
        "mean_deletion_prob": mean(row["deletion_prob"] for row in rows),
        "mean_deletion_drop": mean(row["deletion_drop"] for row in rows),
        "deletion_label_flip_rate": mean(row["deletion_label_flip"] for row in rows),
        "mean_insertion_prob": mean(row["insertion_prob"] for row in rows),
        "mean_insertion_gap": mean(row["insertion_gap"] for row in rows),
        "mean_sufficiency_ratio": mean(row["sufficiency_ratio"] for row in rows),
        "insertion_label_preservation_rate": mean(
            row["insertion_label_preserved"] for row in rows
        ),
        "mean_sparsity": mean(row["sparsity"] for row in rows),
        "mean_subgraph_num_nodes": mean(row["subgraph_num_nodes"] for row in rows),
        "mean_subgraph_num_edges": mean(row["subgraph_num_edges"] for row in rows),
        "mean_selected_edges": mean(row["selected_edges"] for row in rows),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explainer",
        type=str,
        required=True,
        choices=["gnnexplainer", "pgexplainer", "subgraphx", "subgraphx_mcts"],
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    summaries = []

    for experiment_name in EXPERIMENTS:
        print(f"Summarising: {args.explainer} | {experiment_name}")
        summary = summarise_experiment(
            explainer_name=args.explainer,
            experiment_name=experiment_name,
            seed=args.seed,
        )
        summaries.append(summary)

    output_dir = PROJECT_ROOT / "outputs" / "faithfulness" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.explainer}_faithfulness_summary.csv"

    fieldnames = list(summaries[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print("\nFaithfulness summary:")
    for row in summaries:
        print(
            f"{row['explainer']} | "
            f"{row['experiment_name']} | "
            f"nodes={row['num_nodes']} | "
            f"deletion_drop={row['mean_deletion_drop']:.4f} | "
            f"insertion_prob={row['mean_insertion_prob']:.4f} | "
            f"sparsity={row['mean_sparsity']:.4f} | "
            f"flip_rate={row['deletion_label_flip_rate']:.4f} | "
            f"preservation={row['insertion_label_preservation_rate']:.4f}"
        )

    print("\nSaved summary to:")
    print(output_path)


if __name__ == "__main__":
    main()