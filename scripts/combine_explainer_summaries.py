import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPLAINERS = ["gnnexplainer", "pgexplainer", "subgraphx_mcts"]

INPUT_DIR = PROJECT_ROOT / "outputs" / "faithfulness" / "summary"
OUTPUT_PATH = INPUT_DIR / "combined_explainer_faithfulness_summary.csv"


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    all_rows = []

    for explainer in EXPLAINERS:
        input_path = INPUT_DIR / f"{explainer}_faithfulness_summary.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Summary file not found: {input_path}")

        rows = read_csv(input_path)
        all_rows.extend(rows)

    fieldnames = list(all_rows[0].keys())

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("Combined explainer faithfulness summary:")
    for row in all_rows:
        print(
            f"{row['explainer']} | "
            f"{row['experiment_name']} | "
            f"nodes={row['num_nodes']} | "
            f"deletion_drop={float(row['mean_deletion_drop']):.4f} | "
            f"insertion_prob={float(row['mean_insertion_prob']):.4f} | "
            f"sparsity={float(row['mean_sparsity']):.4f} | "
            f"flip_rate={float(row['deletion_label_flip_rate']):.4f} | "
            f"preservation={float(row['insertion_label_preservation_rate']):.4f}"
        )

    print("\nSaved combined summary to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()