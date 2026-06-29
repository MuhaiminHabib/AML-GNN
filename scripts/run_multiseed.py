import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch

from datasets.registry import load_dataset
from models.registry import build_model
from training.trainer import TrainConfig, set_seed, train_model


SEEDS = [42, 7, 21, 123, 2025]

EXPERIMENTS = [
    {
        "experiment_name": "gcn_raw",
        "model": "gcn",
        "dataset": "elliptic",
        "normalize": False,
        "epochs": 100,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "hidden_channels": 128,
        "dropout": 0.5,
        "heads": 4,
        "class_weight_strength": 1.0,
        "threshold_tuning": False,
    },
    {
        "experiment_name": "gcn_norm",
        "model": "gcn",
        "dataset": "elliptic",
        "normalize": True,
        "epochs": 100,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "hidden_channels": 128,
        "dropout": 0.5,
        "heads": 4,
        "class_weight_strength": 1.0,
        "threshold_tuning": False,
    },
    {
        "experiment_name": "graphsage_raw",
        "model": "graphsage",
        "dataset": "elliptic",
        "normalize": False,
        "epochs": 100,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "hidden_channels": 128,
        "dropout": 0.5,
        "heads": 4,
        "class_weight_strength": 1.0,
        "threshold_tuning": False,
    },
    {
        "experiment_name": "gatv2_raw_tuned",
        "model": "gatv2",
        "dataset": "elliptic",
        "normalize": False,
        "epochs": 200,
        "lr": 0.001,
        "weight_decay": 5e-4,
        "hidden_channels": 64,
        "dropout": 0.2,
        "heads": 2,
        "class_weight_strength": 0.25,
        "threshold_tuning": False,
    },
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_rows = []
    data_root = PROJECT_ROOT / "data"

    for experiment in EXPERIMENTS:
        experiment_name = experiment["experiment_name"]
        dataset_name = experiment["dataset"]
        model_name = experiment["model"]

        print("\n" + "=" * 80)
        print(f"Experiment: {experiment_name}")
        print(f"Dataset: {dataset_name} | Model: {model_name}")
        print(f"Normalize: {experiment['normalize']}")
        print("=" * 80)

        for seed in SEEDS:
            print("\n" + "-" * 80)
            print(f"Running seed: {seed}")
            print("-" * 80)

            set_seed(seed)

            data, summary_fn = load_dataset(
                dataset_name,
                data_root,
                normalize=experiment["normalize"],
            )

            model = build_model(
                model_name=model_name,
                in_channels=data.num_node_features,
                hidden_channels=experiment["hidden_channels"],
                out_channels=2,
                dropout=experiment["dropout"],
                heads=experiment["heads"],
            )

            config = TrainConfig(
                seed=seed,
                epochs=experiment["epochs"],
                lr=experiment["lr"],
                weight_decay=experiment["weight_decay"],
                hidden_channels=experiment["hidden_channels"],
                dropout=experiment["dropout"],
                threshold_tuning=experiment["threshold_tuning"],
                heads=experiment["heads"],
                class_weight_strength=experiment["class_weight_strength"],
            )

            checkpoint_path = (
                PROJECT_ROOT
                / "outputs"
                / "checkpoints"
                / dataset_name
                / experiment_name
                / f"seed_{seed}.pt"
            )

            result = train_model(
                model=model,
                data=data,
                config=config,
                device=device,
                checkpoint_path=checkpoint_path,
            )

            metrics = result["test_metrics"]

            row = {
                "experiment_name": experiment_name,
                "dataset": dataset_name,
                "model": model_name,
                "seed": seed,
                "normalize": experiment["normalize"],
                "best_epoch": result["best_epoch"],
                "best_val_f1": result["best_val_f1"],
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "illicit_precision": metrics["illicit_precision"],
                "illicit_recall": metrics["illicit_recall"],
                "illicit_f1": metrics["illicit_f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "epochs": experiment["epochs"],
                "lr": experiment["lr"],
                "weight_decay": experiment["weight_decay"],
                "hidden_channels": experiment["hidden_channels"],
                "dropout": experiment["dropout"],
                "heads": experiment["heads"],
                "class_weight_strength": experiment["class_weight_strength"],
                "threshold_tuning": experiment["threshold_tuning"],
            }

            all_rows.append(row)

            print("\nSeed result:")
            print(f"experiment_name: {experiment_name}")
            print(f"model: {model_name}")
            print(f"normalize: {experiment['normalize']}")

            for key in [
                "accuracy",
                "illicit_precision",
                "illicit_recall",
                "illicit_f1",
                "roc_auc",
                "pr_auc",
                "threshold",
            ]:
                print(f"{key}: {row[key]:.4f}")

    output_dir = PROJECT_ROOT / "outputs" / "results" / "multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "elliptic_multiseed_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nSaved multi-seed results to:")
    print(output_path)


if __name__ == "__main__":
    main()