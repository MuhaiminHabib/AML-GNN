import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from datasets.registry import load_dataset
from models.registry import build_model


FINAL_EXPERIMENTS = {
    "graphsage_raw_undirected": {
        "model": "graphsage",
        "dataset": "elliptic",
        "normalize": False,
        "undirected": True,
        "direction_aware": False,
        "hidden_channels": 128,
        "dropout": 0.5,
        "heads": 4,
    },
    "gatv2_dir_heads8_hidden32_cw035": {
        "model": "gatv2_dir",
        "dataset": "elliptic",
        "normalize": False,
        "undirected": False,
        "direction_aware": True,
        "hidden_channels": 32,
        "dropout": 0.2,
        "heads": 8,
    },
    "gcn_norm_undirected": {
        "model": "gcn",
        "dataset": "elliptic",
        "normalize": True,
        "undirected": True,
        "direction_aware": False,
        "hidden_channels": 128,
        "dropout": 0.5,
        "heads": 4,
    },
}


class ExplainModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None):
        if getattr(self.model, "uses_edge_attr", False):
            return self.model(x, edge_index, edge_attr)
        return self.model(x, edge_index)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


@torch.no_grad()
def predict_target(wrapper, x, edge_index, edge_attr, target_idx):
    logits = wrapper(x, edge_index, edge_attr)
    probs = F.softmax(logits, dim=1)

    illicit_prob = float(probs[target_idx, 1].item())
    pred = int(logits[target_idx].argmax().item())

    return illicit_prob, pred


def build_top_edge_mask(subset, sub_edge_index, top_edges):
    """
    Creates a boolean mask over subgraph edges.
    True = edge is selected by the explainer.
    """

    selected_pairs = set()

    for edge in top_edges:
        selected_pairs.add((int(edge["global_src"]), int(edge["global_dst"])))

    mask = torch.zeros(sub_edge_index.size(1), dtype=torch.bool)

    for edge_pos in range(sub_edge_index.size(1)):
        local_src = int(sub_edge_index[0, edge_pos].item())
        local_dst = int(sub_edge_index[1, edge_pos].item())

        global_src = int(subset[local_src].item())
        global_dst = int(subset[local_dst].item())

        if (global_src, global_dst) in selected_pairs:
            mask[edge_pos] = True

    return mask


def evaluate_one_explanation(
    model,
    data,
    explanation,
    device,
    num_hops=2,
):
    node_idx = int(explanation["target_node"])
    top_edges = explanation["top_edges"]

    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )

    x_sub = data.x[subset].to(device)
    sub_edge_index = sub_edge_index.to(device)

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr_sub = data.edge_attr[edge_mask].to(device)
    else:
        edge_attr_sub = None

    local_target_idx = int(mapping.item())

    wrapper = ExplainModelWrapper(model).to(device)
    wrapper.eval()

    selected_mask = build_top_edge_mask(
        subset=subset,
        sub_edge_index=sub_edge_index.cpu(),
        top_edges=top_edges,
    ).to(device)

    total_edges = int(sub_edge_index.size(1))
    selected_edges = int(selected_mask.sum().item())

    if selected_edges == 0:
        raise RuntimeError(f"No selected edges matched for node {node_idx}")

    deletion_mask = ~selected_mask
    insertion_mask = selected_mask

    deletion_edge_index = sub_edge_index[:, deletion_mask]
    insertion_edge_index = sub_edge_index[:, insertion_mask]

    if edge_attr_sub is not None:
        deletion_edge_attr = edge_attr_sub[deletion_mask]
        insertion_edge_attr = edge_attr_sub[insertion_mask]
    else:
        deletion_edge_attr = None
        insertion_edge_attr = None

    with torch.no_grad():
        original_prob, original_pred = predict_target(
            wrapper,
            x_sub,
            sub_edge_index,
            edge_attr_sub,
            local_target_idx,
        )

        deletion_prob, deletion_pred = predict_target(
            wrapper,
            x_sub,
            deletion_edge_index,
            deletion_edge_attr,
            local_target_idx,
        )

        insertion_prob, insertion_pred = predict_target(
            wrapper,
            x_sub,
            insertion_edge_index,
            insertion_edge_attr,
            local_target_idx,
        )

    deletion_drop = original_prob - deletion_prob
    insertion_gap = original_prob - insertion_prob

    sufficiency_ratio = insertion_prob / original_prob if original_prob > 0 else 0.0
    sparsity = selected_edges / total_edges if total_edges > 0 else 0.0

    deletion_label_flip = int(original_pred != deletion_pred)
    insertion_label_preserved = int(original_pred == insertion_pred)

    return {
        "target_node": node_idx,
        "original_prob": original_prob,
        "original_pred": original_pred,
        "deletion_prob": deletion_prob,
        "deletion_pred": deletion_pred,
        "deletion_drop": deletion_drop,
        "deletion_label_flip": deletion_label_flip,
        "insertion_prob": insertion_prob,
        "insertion_pred": insertion_pred,
        "insertion_gap": insertion_gap,
        "sufficiency_ratio": sufficiency_ratio,
        "insertion_label_preserved": insertion_label_preserved,
        "subgraph_num_nodes": int(x_sub.size(0)),
        "subgraph_num_edges": total_edges,
        "selected_edges": selected_edges,
        "sparsity": sparsity,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(FINAL_EXPERIMENTS.keys()),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument(
        "--explainer",
        type=str,
        default="gnnexplainer",
        choices=["gnnexplainer", "pgexplainer", "subgraphx"],
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Explainer:", args.explainer)

    exp = FINAL_EXPERIMENTS[args.experiment]

    dataset_name = exp["dataset"]
    model_name = exp["model"]

    data_root = PROJECT_ROOT / "data"

    data, _ = load_dataset(
        dataset_name,
        data_root,
        normalize=exp["normalize"],
        make_undirected=exp["undirected"],
        direction_aware=exp["direction_aware"],
    )

    model = build_model(
        model_name=model_name,
        in_channels=data.num_node_features,
        hidden_channels=exp["hidden_channels"],
        out_channels=2,
        dropout=exp["dropout"],
        heads=exp["heads"],
    ).to(device)

    checkpoint_path = (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / dataset_name
        / args.experiment
        / f"seed_{args.seed}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    explanation_path = (
        PROJECT_ROOT
        / "outputs"
        / "explanations"
        / args.explainer
        / args.experiment
        / f"seed_{args.seed}_explanations.json"
    )

    if not explanation_path.exists():
        raise FileNotFoundError(f"Explanation file not found: {explanation_path}")

    with open(explanation_path, "r", encoding="utf-8") as f:
        explanation_data = json.load(f)

    explanations = explanation_data["explanations"]

    if len(explanations) == 0:
        raise RuntimeError(f"No explanations found in: {explanation_path}")

    rows = []

    for explanation in explanations:
        node_idx = explanation["target_node"]
        print(f"Evaluating node: {node_idx}")

        result = evaluate_one_explanation(
            model=model,
            data=data,
            explanation=explanation,
            device=device,
            num_hops=args.num_hops,
        )

        rows.append(result)

        print(
            f"original={result['original_prob']:.4f} | "
            f"deletion={result['deletion_prob']:.4f} | "
            f"insertion={result['insertion_prob']:.4f} | "
            f"drop={result['deletion_drop']:.4f} | "
            f"sparsity={result['sparsity']:.4f}"
        )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "faithfulness"
        / args.explainer
        / args.experiment
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"seed_{args.seed}_faithfulness.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved faithfulness results to:")
    print(output_path)

    avg_deletion_drop = sum(r["deletion_drop"] for r in rows) / len(rows)
    avg_insertion_prob = sum(r["insertion_prob"] for r in rows) / len(rows)
    avg_sufficiency_ratio = sum(r["sufficiency_ratio"] for r in rows) / len(rows)
    avg_sparsity = sum(r["sparsity"] for r in rows) / len(rows)
    deletion_flip_rate = sum(r["deletion_label_flip"] for r in rows) / len(rows)
    insertion_preservation_rate = sum(r["insertion_label_preserved"] for r in rows) / len(rows)

    print("\nSummary:")
    print(f"Mean deletion drop: {avg_deletion_drop:.4f}")
    print(f"Mean insertion probability: {avg_insertion_prob:.4f}")
    print(f"Mean sufficiency ratio: {avg_sufficiency_ratio:.4f}")
    print(f"Mean sparsity: {avg_sparsity:.4f}")
    print(f"Deletion label flip rate: {deletion_flip_rate:.4f}")
    print(f"Insertion label preservation rate: {insertion_preservation_rate:.4f}")


if __name__ == "__main__":
    main()