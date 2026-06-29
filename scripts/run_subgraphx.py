import argparse
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


@torch.no_grad()
def get_candidate_nodes(
    model,
    data,
    device,
    max_candidates=100,
    min_prob=0.60,
    max_prob=0.95,
    min_subgraph_edges=10,
    num_hops=2,
):
    model.eval()
    wrapper = ExplainModelWrapper(model).to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = (
        data.edge_attr.to(device)
        if hasattr(data, "edge_attr") and data.edge_attr is not None
        else None
    )

    logits = wrapper(x, edge_index, edge_attr)
    probs = F.softmax(logits, dim=1)[:, 1]
    preds = logits.argmax(dim=1)

    mask = (
        data.test_mask.to(device)
        & (data.y.to(device) == 1)
        & (preds == 1)
        & (probs >= min_prob)
        & (probs <= max_prob)
    )

    candidate_nodes = mask.nonzero(as_tuple=False).view(-1).cpu().tolist()

    filtered_nodes = []

    for node_idx in candidate_nodes:
        _, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

        if sub_edge_index.size(1) >= min_subgraph_edges:
            filtered_nodes.append(node_idx)

    filtered_nodes = sorted(
        filtered_nodes,
        key=lambda n: abs(float(probs[n].item()) - 0.80),
    )

    return filtered_nodes[:max_candidates]


def edge_positions_to_edge_index(edge_index, edge_attr, kept_positions):
    kept_tensor = torch.tensor(
        kept_positions,
        dtype=torch.long,
        device=edge_index.device,
    )

    new_edge_index = edge_index[:, kept_tensor]

    if edge_attr is not None:
        new_edge_attr = edge_attr[kept_tensor]
    else:
        new_edge_attr = None

    return new_edge_index, new_edge_attr


@torch.no_grad()
def score_kept_edges(
    wrapper,
    x,
    edge_index,
    edge_attr,
    kept_positions,
    target_idx,
):
    if len(kept_positions) == 0:
        empty_edge_index = edge_index[:, :0]
        empty_edge_attr = edge_attr[:0] if edge_attr is not None else None
        prob, _ = predict_target(
            wrapper,
            x,
            empty_edge_index,
            empty_edge_attr,
            target_idx,
        )
        return prob

    new_edge_index, new_edge_attr = edge_positions_to_edge_index(
        edge_index=edge_index,
        edge_attr=edge_attr,
        kept_positions=kept_positions,
    )

    prob, _ = predict_target(
        wrapper,
        x,
        new_edge_index,
        new_edge_attr,
        target_idx,
    )

    return prob


@torch.no_grad()
def compute_single_edge_removal_scores(
    wrapper,
    x,
    edge_index,
    edge_attr,
    target_idx,
):
    total_edges = edge_index.size(1)
    all_edges = list(range(total_edges))

    original_prob, _ = predict_target(
        wrapper,
        x,
        edge_index,
        edge_attr,
        target_idx,
    )

    removal_scores = {}

    for edge_pos in all_edges:
        kept_positions = [e for e in all_edges if e != edge_pos]

        prob_without_edge = score_kept_edges(
            wrapper=wrapper,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            kept_positions=kept_positions,
            target_idx=target_idx,
        )

        drop = original_prob - prob_without_edge
        removal_scores[edge_pos] = drop

    return original_prob, removal_scores


@torch.no_grad()
def greedy_subgraphx_search(
    wrapper,
    x,
    edge_index,
    edge_attr,
    target_idx,
    top_k_edges=5,
    max_candidates_per_step=25,
):
    """
    Lightweight SubgraphX-style search.

    It starts from the full local subgraph and removes edges one by one.
    At each step, it removes the edge whose removal preserves the target
    illicit probability the most.

    The final remaining edges are used as the explanation subgraph.
    """

    total_edges = edge_index.size(1)

    if total_edges <= top_k_edges:
        original_prob, _ = predict_target(
            wrapper,
            x,
            edge_index,
            edge_attr,
            target_idx,
        )
        return list(range(total_edges)), original_prob

    original_prob, removal_scores = compute_single_edge_removal_scores(
        wrapper=wrapper,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target_idx=target_idx,
    )

    kept_positions = list(range(total_edges))

    while len(kept_positions) > top_k_edges:
        removable_edges = sorted(
            kept_positions,
            key=lambda e: removal_scores.get(e, 0.0),
        )

        candidate_edges = removable_edges[:max_candidates_per_step]

        best_edge_to_remove = None
        best_prob_after_removal = -1.0

        for edge_pos in candidate_edges:
            trial_kept = [e for e in kept_positions if e != edge_pos]

            prob_after_removal = score_kept_edges(
                wrapper=wrapper,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                kept_positions=trial_kept,
                target_idx=target_idx,
            )

            if prob_after_removal > best_prob_after_removal:
                best_prob_after_removal = prob_after_removal
                best_edge_to_remove = edge_pos

        kept_positions = [e for e in kept_positions if e != best_edge_to_remove]

    final_prob = score_kept_edges(
        wrapper=wrapper,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        kept_positions=kept_positions,
        target_idx=target_idx,
    )

    return kept_positions, final_prob


def build_explanation_for_node(
    model,
    data,
    node_idx,
    device,
    num_hops=2,
    top_k_edges=5,
    max_candidates_per_step=25,
):
    model.eval()
    wrapper = ExplainModelWrapper(model).to(device)
    wrapper.eval()

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

    original_prob, original_pred = predict_target(
        wrapper,
        x_sub,
        sub_edge_index,
        edge_attr_sub,
        local_target_idx,
    )

    if original_pred != 1:
        print(
            f"Skipped node {node_idx} because subgraph prediction is not illicit "
            f"(pred={original_pred}, prob={original_prob:.4f})"
        )
        return None

    if not (0.60 <= original_prob <= 0.95):
        print(
            f"Skipped node {node_idx} because subgraph probability is outside target range "
            f"(prob={original_prob:.4f})"
        )
        return None

    kept_edges, final_prob = greedy_subgraphx_search(
        wrapper=wrapper,
        x=x_sub,
        edge_index=sub_edge_index,
        edge_attr=edge_attr_sub,
        target_idx=local_target_idx,
        top_k_edges=top_k_edges,
        max_candidates_per_step=max_candidates_per_step,
    )

    top_edges = []

    for edge_pos in kept_edges:
        local_src = int(sub_edge_index[0, edge_pos].item())
        local_dst = int(sub_edge_index[1, edge_pos].item())

        global_src = int(subset[local_src].item())
        global_dst = int(subset[local_dst].item())

        top_edges.append(
            {
                "global_src": global_src,
                "global_dst": global_dst,
                "score": float(final_prob),
            }
        )

    return {
        "target_node": int(node_idx),
        "local_target_idx": local_target_idx,
        "full_graph_illicit_prob": original_prob,
        "subgraph_illicit_prob": original_prob,
        "subgraph_prediction": original_pred,
        "subgraph_num_nodes": int(x_sub.size(0)),
        "subgraph_num_edges": int(sub_edge_index.size(1)),
        "selected_subgraph_prob": float(final_prob),
        "top_edges": top_edges,
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
    parser.add_argument("--num_nodes", type=int, default=5)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--top_k_edges", type=int, default=5)
    parser.add_argument("--max_candidates_per_step", type=int, default=25)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    candidate_nodes = get_candidate_nodes(
        model=model,
        data=data,
        device=device,
        max_candidates=args.num_nodes * 20,
        num_hops=args.num_hops,
    )

    print(f"Candidate illicit test nodes found: {len(candidate_nodes)}")

    explanations = []

    for node_idx in candidate_nodes:
        if len(explanations) >= args.num_nodes:
            break

        print(f"\nExplaining node: {node_idx}")

        explanation = build_explanation_for_node(
            model=model,
            data=data,
            node_idx=node_idx,
            device=device,
            num_hops=args.num_hops,
            top_k_edges=args.top_k_edges,
            max_candidates_per_step=args.max_candidates_per_step,
        )

        if explanation is None:
            continue

        explanations.append(explanation)

        print(
            f"Done | subgraph nodes: {explanation['subgraph_num_nodes']} | "
            f"edges: {explanation['subgraph_num_edges']} | "
            f"selected edges: {len(explanation['top_edges'])} | "
            f"original prob: {explanation['subgraph_illicit_prob']:.4f} | "
            f"selected prob: {explanation['selected_subgraph_prob']:.4f}"
        )

    if len(explanations) == 0:
        raise RuntimeError("No valid SubgraphX explanations were generated.")

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "explanations"
        / "subgraphx"
        / args.experiment
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"seed_{args.seed}_explanations.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": args.experiment,
                "dataset": dataset_name,
                "model": model_name,
                "seed": args.seed,
                "num_explanations": len(explanations),
                "num_hops": args.num_hops,
                "top_k_edges": args.top_k_edges,
                "method_note": "Lightweight SubgraphX-style greedy subgraph search",
                "explanations": explanations,
            },
            f,
            indent=2,
        )

    print("\nSaved SubgraphX explanations to:")
    print(output_path)


if __name__ == "__main__":
    main()