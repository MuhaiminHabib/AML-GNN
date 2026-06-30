import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from datasets.registry import load_dataset
from models.registry import build_model


FINAL_EXPERIMENTS = {
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
    "graphsage_raw_undirected_cw05": {
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
}


class ExplainModelWrapper(nn.Module):
    """
    Wrapper so GNNExplainer can call all models using the same signature.

    Normal models:
        model(x, edge_index)

    Direction-aware GATv2:
        model(x, edge_index, edge_attr)
    """

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
def get_correct_illicit_test_nodes(
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
    edge_attr = data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None

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
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

        if sub_edge_index.size(1) >= min_subgraph_edges:
            filtered_nodes.append(node_idx)

    if len(filtered_nodes) == 0:
        raise RuntimeError(
            "No suitable correctly classified illicit test nodes found. "
            "Try lowering min_subgraph_edges or widening the probability range."
        )

    filtered_nodes = sorted(
        filtered_nodes,
        key=lambda n: abs(float(probs[n].item()) - 0.80)
    )

    return filtered_nodes[:max_candidates], probs.cpu()

def explain_one_node(
    model,
    data,
    node_idx,
    full_probs,
    device,
    num_hops=2,
    explainer_epochs=100,
    top_k_edges=20,
):
    model.eval()

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

    with torch.no_grad():
        sub_logits = wrapper(x_sub, sub_edge_index, edge_attr_sub)
        sub_probs = F.softmax(sub_logits, dim=1)[:, 1]
        sub_pred = int(sub_logits[local_target_idx].argmax().item())
        sub_illicit_prob = float(sub_probs[local_target_idx].item())

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=explainer_epochs, lr=0.01),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )

    if edge_attr_sub is not None:
        explanation = explainer(
            x_sub,
            sub_edge_index,
            edge_attr=edge_attr_sub,
            index=local_target_idx,
        )
    else:
        explanation = explainer(
            x_sub,
            sub_edge_index,
            index=local_target_idx,
        )

    edge_scores = explanation.edge_mask.detach().cpu()

    top_k = min(top_k_edges, edge_scores.numel())
    top_scores, top_indices = torch.topk(edge_scores, k=top_k)

    top_edges = []
    for score, edge_pos in zip(top_scores.tolist(), top_indices.tolist()):
        local_src = int(sub_edge_index[0, edge_pos].detach().cpu().item())
        local_dst = int(sub_edge_index[1, edge_pos].detach().cpu().item())

        global_src = int(subset[local_src].item())
        global_dst = int(subset[local_dst].item())

        top_edges.append(
            {
                "global_src": global_src,
                "global_dst": global_dst,
                "score": float(score),
            }
        )

    result = {
        "target_node": int(node_idx),
        "local_target_idx": local_target_idx,
        "full_graph_illicit_prob": float(full_probs[node_idx].item()),
        "subgraph_illicit_prob": sub_illicit_prob,
        "subgraph_prediction": sub_pred,
        "subgraph_num_nodes": int(x_sub.size(0)),
        "subgraph_num_edges": int(sub_edge_index.size(1)),
        "top_edges": top_edges,
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(FINAL_EXPERIMENTS.keys()),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_nodes", type=int, default=10)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--explainer_epochs", type=int, default=100)
    parser.add_argument("--top_k_edges", type=int, default=20)

    args = parser.parse_args()

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

    candidate_nodes, full_probs = get_correct_illicit_test_nodes(
        model=model,
        data=data,
        device=device,
        max_candidates=args.num_nodes * 20,
        min_prob=0.60,
        max_prob=0.95,
        min_subgraph_edges=10,
        num_hops=args.num_hops,
    )

    print(f"Found candidate illicit test nodes: {len(candidate_nodes)}")

    explanations = []

    for node_idx in candidate_nodes:
        if len(explanations) >= args.num_nodes:
            break

        print(f"\nExplaining node: {node_idx}")

        try:
            explanation_result = explain_one_node(
                model=model,
                data=data,
                node_idx=node_idx,
                full_probs=full_probs,
                device=device,
                num_hops=args.num_hops,
                explainer_epochs=args.explainer_epochs,
                top_k_edges=args.top_k_edges,
            )

            subgraph_prob = explanation_result["subgraph_illicit_prob"]
            subgraph_pred = explanation_result["subgraph_prediction"]

            if subgraph_pred != 1:
                print(
                    f"Skipped node {node_idx} because subgraph prediction is not illicit "
                    f"(pred={subgraph_pred}, prob={subgraph_prob:.4f})"
                )
                continue

            if not (0.60 <= subgraph_prob <= 0.95):
                print(
                    f"Skipped node {node_idx} because subgraph probability is outside target range "
                    f"(prob={subgraph_prob:.4f})"
                )
                continue

            explanations.append(explanation_result)

            print(
                f"Done | subgraph nodes: {explanation_result['subgraph_num_nodes']} | "
                f"edges: {explanation_result['subgraph_num_edges']} | "
                f"subgraph prob: {explanation_result['subgraph_illicit_prob']:.4f}"
            )

        except RuntimeError as e:
            print(f"Skipped node {node_idx} because of runtime error:")
            print(e)

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "explanations"
        / "gnnexplainer"
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
                "explanations": explanations,
            },
            f,
            indent=2,
        )

    print("\nSaved explanations to:")
    print(output_path)


if __name__ == "__main__":
    main()