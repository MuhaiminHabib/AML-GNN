import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


@torch.no_grad()
def get_probs_labels(model, data, mask, device):
    model.eval()

    logits = model(data.x.to(device), data.edge_index.to(device))
    probs = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
    labels = data.y[mask].detach().cpu().numpy()

    return probs, labels


def compute_metrics_from_probs(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = float("nan")

    return {
        "accuracy": float(acc),
        "illicit_precision": float(precision),
        "illicit_recall": float(recall),
        "illicit_f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
    }


def find_best_threshold(probs, labels, metric="f1"):
    thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for threshold in thresholds:
        metrics = compute_metrics_from_probs(probs, labels, threshold)

        if metric == "f1":
            score = metrics["illicit_f1"]
        elif metric == "precision":
            score = metrics["illicit_precision"]
        elif metric == "recall":
            score = metrics["illicit_recall"]
        else:
            raise ValueError(f"Unknown threshold selection metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return float(best_threshold), best_metrics


@torch.no_grad()
def evaluate(model, data, mask, device, threshold=0.5):
    probs, labels = get_probs_labels(model, data, mask, device)
    return compute_metrics_from_probs(probs, labels, threshold=threshold)


@torch.no_grad()
def evaluate_with_best_threshold(model, data, val_mask, test_mask, device):
    val_probs, val_labels = get_probs_labels(model, data, val_mask, device)
    best_threshold, val_metrics = find_best_threshold(val_probs, val_labels, metric="f1")

    test_probs, test_labels = get_probs_labels(model, data, test_mask, device)
    test_metrics = compute_metrics_from_probs(
        test_probs,
        test_labels,
        threshold=best_threshold,
    )

    return best_threshold, val_metrics, test_metrics