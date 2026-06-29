import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score


@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()

    logits = model(data.x.to(device), data.edge_index.to(device))
    probs = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
    preds = logits[mask].argmax(dim=1).detach().cpu().numpy()
    labels = data.y[mask].detach().cpu().numpy()

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )

    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    return {
        "accuracy": acc,
        "illicit_precision": precision,
        "illicit_recall": recall,
        "illicit_f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }