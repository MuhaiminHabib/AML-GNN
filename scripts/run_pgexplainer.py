import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain import Explainer, PGExplainer
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
def get_candidate_nodes(
    model,
    data,
    device,
    split_name,
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

    if split_name == "train":
        split_mask = data.train_mask.to(device)
    elif split_name == "val":
        split_mask = data.val_mask.to(device)
    elif split_name == "test":
        split_mask = data.test_mask.to(device)
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    mask = (
        split_mask
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
        key=lambda n: abs(float(probs[n].item()) - 0.80)
    )

    return filtered_nodes[:max_candidates]


@torch.no_grad()
def prepare_sample(model, data, node_idx, device, num_hops=2):
    model.eval()
    wrapper = ExplainModelWrapper(model).to(device)

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

    logits = wrapper(x_sub, sub_edge_index, edge_attr_sub)
    probs = F.softmax(logits, dim=1)[:, 1]
    preds = logits.argmax(dim=1)

    subgraph_prob = float(probs[local_target_idx].item())
    subgraph_pred = int(preds[local_target_idx].item())

    if subgraph_pred != 1:
        return None

    if not (0.60 <= subgraph_prob <= 0.95):
        return None

    target = preds.detach()

    return {
        "target_node": int(node_idx),
        "subset": subset.cpu(),
        "x": x_sub,
        "edge_index": sub_edge_index,
        "edge_attr": edge_attr_sub,
        "local_target_idx": local_target_idx,
        "target": target,
        "subgraph_prob": subgraph_prob,
        "subgraph_pred": subgraph_pred,
        "subgraph_num_nodes": int(x_sub.size(0)),
        "subgraph_num_edges": int(sub_edge_index.size(1)),
    }


def top_edges_from_explanation(sample, explanation, top_k_edges):
    edge_scores = explanation.edge_mask.detach().cpu()

    top_k = min(top_k_edges, edge_scores.numel())
    top_scores, top_indices = torch.topk(edge_scores, k=top_k)

    subset = sample["subset"]
    sub_edge_index_cpu = sample["edge_index"].detach().cpu()

    top_edges = []

    for score, edge_pos in zip(top_scores.tolist(), top_indices.tolist()):
        local_src = int(sub_edge_index_cpu[0, edge_pos].item())
        local_dst = int(sub_edge_index_cpu[1, edge_pos].item())

        global_src = int(subset[local_src].item())
        global_dst = int(subset[local_dst].item())

        top_edges.append(
            {
                "global_src": global_src,
                "global_dst": global_dst,
                "score": float(score),
            }
        )

    return top_edges


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
    parser.add_argument("--train_nodes", type=int, default=30)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--explainer_epochs", type=int, default=30)
    parser.add_argument("--top_k_edges", type=int, default=5)
    parser.add_argument("--pg_lr", type=float, default=0.003)

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

    train_candidates = get_candidate_nodes(
        model=model,
        data=data,
        device=device,
        split_name="train",
        max_candidates=args.train_nodes * 10,
        num_hops=args.num_hops,
    )

    explain_candidates = get_candidate_nodes(
        model=model,
        data=data,
        device=device,
        split_name="test",
        max_candidates=args.num_nodes * 10,
        num_hops=args.num_hops,
    )

    print(f"Train candidates found: {len(train_candidates)}")
    print(f"Explain candidates found: {len(explain_candidates)}")

    train_samples = []

    for node_idx in train_candidates:
        if len(train_samples) >= args.train_nodes:
            break

        sample = prepare_sample(
            model=model,
            data=data,
            node_idx=node_idx,
            device=device,
            num_hops=args.num_hops,
        )

        if sample is not None:
            train_samples.append(sample)

    explain_samples = []

    for node_idx in explain_candidates:
        if len(explain_samples) >= args.num_nodes:
            break

        sample = prepare_sample(
            model=model,
            data=data,
            node_idx=node_idx,
            device=device,
            num_hops=args.num_hops,
        )

        if sample is not None:
            explain_samples.append(sample)

    if len(train_samples) == 0:
        raise RuntimeError("No valid PGExplainer training samples found.")

    if len(explain_samples) == 0:
        raise RuntimeError("No valid PGExplainer explanation samples found.")

    print(f"PGExplainer training samples: {len(train_samples)}")
    print(f"PGExplainer explanation samples: {len(explain_samples)}")

    wrapper = ExplainModelWrapper(model).to(device)
    wrapper.eval()

    algorithm = PGExplainer(
        epochs=args.explainer_epochs,
        lr=args.pg_lr,
    ).to(device)

    explainer = Explainer(
        model=wrapper,
        algorithm=algorithm,
        explanation_type="phenomenon",
        node_mask_type=None,
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )

    print("\nTraining PGExplainer...")

    for epoch in range(args.explainer_epochs):
        total_loss = 0.0

        for sample in train_samples:
            if sample["edge_attr"] is not None:
                loss = explainer.algorithm.train(
                    epoch,
                    wrapper,
                    sample["x"],
                    sample["edge_index"],
                    target=sample["target"],
                    index=sample["local_target_idx"],
                    edge_attr=sample["edge_attr"],
                )
            else:
                loss = explainer.algorithm.train(
                    epoch,
                    wrapper,
                    sample["x"],
                    sample["edge_index"],
                    target=sample["target"],
                    index=sample["local_target_idx"],
                )

            total_loss += float(loss)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_samples)
            print(f"Epoch {epoch + 1:03d} | PGExplainer loss: {avg_loss:.4f}")

    print("\nGenerating PGExplainer explanations...")

    explanations = []

    for sample in explain_samples:
        node_idx = sample["target_node"]
        print(f"Explaining node: {node_idx}")

        if sample["edge_attr"] is not None:
            explanation = explainer(
                sample["x"],
                sample["edge_index"],
                target=sample["target"],
                index=sample["local_target_idx"],
                edge_attr=sample["edge_attr"],
            )
        else:
            explanation = explainer(
                sample["x"],
                sample["edge_index"],
                target=sample["target"],
                index=sample["local_target_idx"],
            )

        top_edges = top_edges_from_explanation(
            sample=sample,
            explanation=explanation,
            top_k_edges=args.top_k_edges,
        )

        explanations.append(
            {
                "target_node": sample["target_node"],
                "local_target_idx": sample["local_target_idx"],
                "full_graph_illicit_prob": sample["subgraph_prob"],
                "subgraph_illicit_prob": sample["subgraph_prob"],
                "subgraph_prediction": sample["subgraph_pred"],
                "subgraph_num_nodes": sample["subgraph_num_nodes"],
                "subgraph_num_edges": sample["subgraph_num_edges"],
                "top_edges": top_edges,
            }
        )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "explanations"
        / "pgexplainer"
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
                "train_samples": len(train_samples),
                "explainer_epochs": args.explainer_epochs,
                "top_k_edges": args.top_k_edges,
                "explanations": explanations,
            },
            f,
            indent=2,
        )

    print("\nSaved PGExplainer explanations to:")
    print(output_path)


if __name__ == "__main__":
    main()