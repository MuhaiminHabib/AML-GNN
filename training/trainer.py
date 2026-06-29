import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import (
    evaluate,
    get_probs_labels,
    find_best_threshold,
    compute_metrics_from_probs,
)


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 100
    lr: float = 0.005
    weight_decay: float = 5e-4
    hidden_channels: int = 128
    dropout: float = 0.5
    threshold_tuning: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    class_counts = torch.bincount(labels, minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * class_counts)
    return class_weights


def train_model(
    model: torch.nn.Module,
    data,
    config: TrainConfig,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
):
    model = model.to(device)
    data = data.to(device)

    train_labels = data.y[data.train_mask]
    class_weights = compute_class_weights(train_labels).to(device)

    print("Class weights:", class_weights.detach().cpu().numpy())

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    best_threshold = 0.5
    best_val_metrics = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)

        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights,
        )

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            if config.threshold_tuning:
                val_probs, val_labels = get_probs_labels(
                    model,
                    data,
                    data.val_mask,
                    device,
                )
                threshold, val_metrics = find_best_threshold(
                    val_probs,
                    val_labels,
                    metric="f1",
                )
            else:
                threshold = 0.5
                val_metrics = evaluate(
                    model,
                    data,
                    data.val_mask,
                    device,
                    threshold=threshold,
                )

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Val illicit F1: {val_metrics['illicit_f1']:.4f} | "
                f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
                f"Threshold: {threshold:.3f}"
            )

            if val_metrics["illicit_f1"] > best_val_f1:
                best_val_f1 = val_metrics["illicit_f1"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_threshold = threshold
                best_val_metrics = val_metrics

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)

    test_probs, test_labels = get_probs_labels(
        model,
        data,
        data.test_mask,
        device,
    )

    test_metrics = compute_metrics_from_probs(
        test_probs,
        test_labels,
        threshold=best_threshold,
    )

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "best_threshold": best_threshold,
                "best_val_metrics": best_val_metrics,
                "test_metrics": test_metrics,
                "config": config.__dict__,
            },
            checkpoint_path,
        )

    return {
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_threshold": float(best_threshold),
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }