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
        "experiment_name": "graphsage_raw_undirected",
        "model": "graphsage",
        "dataset": "elliptic",
        "normalize": False,
        "undirected": True,
        "direction_aware": False,
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
        "experiment_name": "gatv2_dir_heads8_hidden32_cw035",
        "model": "gatv2_dir",
        "dataset": "elliptic",
        "normalize": False,
        "undirected": False,
        "direction_aware": True,
        "epochs": 250,
        "lr": 0.0005,
        "weight_decay": 5e-4,
        "hidden_channels": 32,
        "dropout": 0.2,
        "heads": 8,
        "class_weight_strength": 0.35,
        "threshold_tuning": False,
    },
    {
        "experiment_name": "gcn_norm_undirected",
        "model": "gcn",
        "dataset": "elliptic",
        "normalize": True,
        "undirected": True,
        "direction_aware": False,
        "epochs": 100,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "hidden_channels": 128,
        "dropout": 0.5,
        "heads": 4,
        "class_weight_strength": 1.0,
        "threshold_tuning": False,
    },
]


def validate_experiment(experiment: dict) -> None:
    model_name = experiment["model"].lower()
    undirected = experiment.get("undirected", False)
    direction_aware = experiment.get("direction_aware", False)

    if undirected and direction_aware:
        raise ValueError(
            f"{experiment['experiment_name']} is invalid: "
            "use either undirected=True or direction_aware=True, not both."
        )

    if model_name == "gatv2_dir" and not direction_aware:
        raise ValueError(
            f"{experiment['experiment_name']} is invalid: "
            "gatv2_dir requires direction_aware=True."
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_rows = []
    data_root = PROJECT_ROOT / "data"

    for experiment in EXPERIMENTS:
        validate_experiment(experiment)

        experiment_name = experiment["experiment_name"]
        dataset_name = experiment["dataset"]
        model_name = experiment["model"]

        normalize = experiment["normalize"]
        undirected = experiment.get("undirected", False)
        direction_aware = experiment.get("direction_aware", False)

        print("\n" + "=" * 80)
        print(f"Experiment: {experiment_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Model: {model_name}")
        print(f"Normalize: {normalize}")
        print(f"Undirected: {undirected}")
        print(f"Direction-aware: {direction_aware}")
        print("=" * 80)

        for seed in SEEDS:
            print("\n" + "-" * 80)
            print(f"Running seed: {seed}")
            print("-" * 80)

            set_seed(seed)

            data, _ = load_dataset(
                dataset_name,
                data_root,
                normalize=normalize,
                make_undirected=undirected,
                direction_aware=direction_aware,
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
                "normalize": normalize,
                "undirected": undirected,
                "direction_aware": direction_aware,
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
            print(f"normalize: {normalize}")
            print(f"undirected: {undirected}")
            print(f"direction_aware: {direction_aware}")

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