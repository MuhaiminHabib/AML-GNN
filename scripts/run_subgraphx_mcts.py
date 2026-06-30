import argparse
import json
import math
import random
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
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None):
        if getattr(self.model, "uses_edge_attr", False):
            return self.model(x, edge_index, edge_attr)
        return self.model(x, edge_index)


class MCTSNode:
    def __init__(self, coalition, parent=None):
        self.coalition = frozenset(coalition)
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_reward = 0.0
        self.reward = None

    @property
    def mean_reward(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count


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
    wrapper.eval()

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


def induced_edge_mask_from_nodes(edge_index, node_set):
    node_set = set(int(n) for n in node_set)
    mask = []

    for edge_pos in range(edge_index.size(1)):
        src = int(edge_index[0, edge_pos].item())
        dst = int(edge_index[1, edge_pos].item())
        mask.append(src in node_set and dst in node_set)

    return torch.tensor(mask, dtype=torch.bool, device=edge_index.device)


@torch.no_grad()
def score_node_coalition(
    wrapper,
    x,
    edge_index,
    edge_attr,
    target_idx,
    coalition,
):
    if len(coalition) == 0:
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

    edge_mask = induced_edge_mask_from_nodes(edge_index, coalition)

    selected_edge_index = edge_index[:, edge_mask]

    if edge_attr is not None:
        selected_edge_attr = edge_attr[edge_mask]
    else:
        selected_edge_attr = None

    prob, _ = predict_target(
        wrapper,
        x,
        selected_edge_index,
        selected_edge_attr,
        target_idx,
    )

    return prob


@torch.no_grad()
def approximate_shapley_value(
    wrapper,
    x,
    edge_index,
    edge_attr,
    target_idx,
    coalition,
    all_nodes,
    num_samples=40,
):
    coalition = set(int(n) for n in coalition)
    all_nodes = set(int(n) for n in all_nodes)

    outside_nodes = list(all_nodes - coalition)

    if len(coalition) == 0:
        return 0.0

    marginal_scores = []

    for _ in range(num_samples):
        sampled_context = set()

        for node in outside_nodes:
            if random.random() < 0.5:
                sampled_context.add(node)

        context_score = score_node_coalition(
            wrapper=wrapper,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            target_idx=target_idx,
            coalition=sampled_context,
        )

        context_plus_coalition = sampled_context | coalition

        coalition_score = score_node_coalition(
            wrapper=wrapper,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            target_idx=target_idx,
            coalition=context_plus_coalition,
        )

        marginal_scores.append(coalition_score - context_score)

    return float(sum(marginal_scores) / len(marginal_scores))


def get_removable_nodes(coalition, edge_index, target_idx):
    coalition = set(int(n) for n in coalition)

    removable = []

    for node in coalition:
        if node == target_idx:
            continue

        remaining = coalition - {node}

        if len(remaining) == 0:
            continue

        edge_mask = induced_edge_mask_from_nodes(edge_index, remaining)

        if int(edge_mask.sum().item()) > 0:
            removable.append(node)

    return removable


def uct_score(parent, child, exploration_weight):
    if child.visit_count == 0:
        return float("inf")

    exploitation = child.mean_reward
    exploration = exploration_weight * math.sqrt(
        math.log(parent.visit_count + 1) / child.visit_count
    )

    return exploitation + exploration


def expand_node(
    node,
    edge_index,
    target_idx,
    max_children,
):
    if len(node.children) > 0:
        return node.children

    removable_nodes = get_removable_nodes(
        coalition=node.coalition,
        edge_index=edge_index,
        target_idx=target_idx,
    )

    if len(removable_nodes) == 0:
        return []

    removable_nodes = removable_nodes[:]
    random.shuffle(removable_nodes)
    removable_nodes = removable_nodes[:max_children]

    for remove_node in removable_nodes:
        child_coalition = set(node.coalition) - {remove_node}
        child = MCTSNode(coalition=child_coalition, parent=node)
        node.children.append(child)

    return node.children


def select_child(node, exploration_weight):
    scores = [
        uct_score(
            parent=node,
            child=child,
            exploration_weight=exploration_weight,
        )
        for child in node.children
    ]

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return node.children[best_idx]


def backpropagate(node, reward):
    current = node

    while current is not None:
        current.visit_count += 1
        current.total_reward += reward
        current = current.parent


def collect_tree_nodes(root):
    nodes = []
    stack = [root]

    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)

    return nodes


def run_mcts_subgraphx(
    wrapper,
    x,
    edge_index,
    edge_attr,
    target_idx,
    min_nodes,
    max_nodes,
    num_mcts_rollouts,
    shapley_samples,
    max_children,
    exploration_weight,
):
    all_nodes = list(range(x.size(0)))

    root = MCTSNode(coalition=all_nodes)

    score_cache = {}

    def get_reward(node):
        key = tuple(sorted(node.coalition))

        if key not in score_cache:
            score_cache[key] = approximate_shapley_value(
                wrapper=wrapper,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                target_idx=target_idx,
                coalition=node.coalition,
                all_nodes=all_nodes,
                num_samples=shapley_samples,
            )

        node.reward = score_cache[key]
        return score_cache[key]

    for rollout in range(num_mcts_rollouts):
        node = root

        # Important fix:
        # Keep moving down the tree until coalition size reaches max_nodes.
        # This prevents MCTS from stopping too early with very large coalitions.
        while len(node.coalition) > max_nodes:
            children = expand_node(
                node=node,
                edge_index=edge_index,
                target_idx=target_idx,
                max_children=max_children,
            )

            if len(children) == 0:
                break

            unvisited = [child for child in children if child.visit_count == 0]

            if len(unvisited) > 0:
                node = random.choice(unvisited)
            else:
                node = select_child(
                    node=node,
                    exploration_weight=exploration_weight,
                )

        # Once the coalition is within the target range,
        # sometimes continue deeper toward min_nodes.
        while len(node.coalition) > min_nodes and random.random() < 0.5:
            children = expand_node(
                node=node,
                edge_index=edge_index,
                target_idx=target_idx,
                max_children=max_children,
            )

            if len(children) == 0:
                break

            unvisited = [child for child in children if child.visit_count == 0]

            if len(unvisited) > 0:
                node = random.choice(unvisited)
            else:
                node = select_child(
                    node=node,
                    exploration_weight=exploration_weight,
                )

        reward = get_reward(node)
        backpropagate(node, reward)

        if rollout == 0 or (rollout + 1) % 20 == 0:
            print(
                f"  MCTS rollout {rollout + 1:03d}/{num_mcts_rollouts} | "
                f"coalition_size={len(node.coalition)} | reward={reward:.4f}"
            )

    tree_nodes = collect_tree_nodes(root)

    valid_nodes = [
        node
        for node in tree_nodes
        if min_nodes <= len(node.coalition) <= max_nodes
    ]

    if len(valid_nodes) == 0:
        raise RuntimeError(
            "MCTS did not find any valid coalition within the requested node range. "
            "Try increasing --num_mcts_rollouts or --max_children."
        )

    for node in valid_nodes:
        if node.reward is None:
            get_reward(node)

    best_node = max(valid_nodes, key=lambda n: n.reward)

    selected_prob = score_node_coalition(
        wrapper=wrapper,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target_idx=target_idx,
        coalition=best_node.coalition,
    )

    return best_node.coalition, best_node.reward, selected_prob


@torch.no_grad()
def rank_edges_inside_coalition(
    wrapper,
    x,
    edge_index,
    edge_attr,
    target_idx,
    coalition,
    max_edges,
):
    edge_mask = induced_edge_mask_from_nodes(edge_index, coalition)
    candidate_positions = edge_mask.nonzero(as_tuple=False).view(-1).tolist()

    if len(candidate_positions) <= max_edges:
        return candidate_positions

    scored_edges = []

    for edge_pos in candidate_positions:
        single_edge_index = edge_index[:, [edge_pos]]

        if edge_attr is not None:
            single_edge_attr = edge_attr[[edge_pos]]
        else:
            single_edge_attr = None

        prob, _ = predict_target(
            wrapper,
            x,
            single_edge_index,
            single_edge_attr,
            target_idx,
        )

        scored_edges.append((edge_pos, prob))

    scored_edges = sorted(scored_edges, key=lambda item: item[1], reverse=True)

    return [edge_pos for edge_pos, _ in scored_edges[:max_edges]]


def build_explanation_for_node(
    model,
    data,
    node_idx,
    device,
    num_hops=2,
    min_nodes=3,
    max_nodes=8,
    num_mcts_rollouts=80,
    shapley_samples=30,
    max_children=12,
    exploration_weight=10.0,
    max_edges=5,
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

    local_num_nodes = int(x_sub.size(0))

    safe_min_nodes = min(min_nodes, local_num_nodes)
    safe_max_nodes = min(max_nodes, local_num_nodes)

    if safe_min_nodes < 1:
        safe_min_nodes = 1

    if safe_max_nodes < safe_min_nodes:
        safe_max_nodes = safe_min_nodes

    selected_nodes, shapley_reward, selected_prob = run_mcts_subgraphx(
        wrapper=wrapper,
        x=x_sub,
        edge_index=sub_edge_index,
        edge_attr=edge_attr_sub,
        target_idx=local_target_idx,
        min_nodes=safe_min_nodes,
        max_nodes=safe_max_nodes,
        num_mcts_rollouts=num_mcts_rollouts,
        shapley_samples=shapley_samples,
        max_children=max_children,
        exploration_weight=exploration_weight,
    )

    selected_edge_positions = rank_edges_inside_coalition(
        wrapper=wrapper,
        x=x_sub,
        edge_index=sub_edge_index,
        edge_attr=edge_attr_sub,
        target_idx=local_target_idx,
        coalition=selected_nodes,
        max_edges=max_edges,
    )

    top_edges = []

    for edge_pos in selected_edge_positions:
        local_src = int(sub_edge_index[0, edge_pos].item())
        local_dst = int(sub_edge_index[1, edge_pos].item())

        global_src = int(subset[local_src].item())
        global_dst = int(subset[local_dst].item())

        top_edges.append(
            {
                "global_src": global_src,
                "global_dst": global_dst,
                "score": float(shapley_reward),
            }
        )

    if len(top_edges) == 0:
        print(f"Skipped node {node_idx} because MCTS selected no edges.")
        return None

    selected_global_nodes = [
        int(subset[local_idx].item())
        for local_idx in sorted(selected_nodes)
    ]

    return {
        "target_node": int(node_idx),
        "local_target_idx": local_target_idx,
        "full_graph_illicit_prob": original_prob,
        "subgraph_illicit_prob": original_prob,
        "subgraph_prediction": original_pred,
        "subgraph_num_nodes": int(x_sub.size(0)),
        "subgraph_num_edges": int(sub_edge_index.size(1)),
        "selected_node_count": int(len(selected_nodes)),
        "selected_global_nodes": selected_global_nodes,
        "selected_subgraph_prob": float(selected_prob),
        "shapley_reward": float(shapley_reward),
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

    parser.add_argument("--min_nodes", type=int, default=3)
    parser.add_argument("--max_nodes", type=int, default=8)
    parser.add_argument("--max_edges", type=int, default=5)

    parser.add_argument("--num_mcts_rollouts", type=int, default=80)
    parser.add_argument("--shapley_samples", type=int, default=30)
    parser.add_argument("--max_children", type=int, default=12)
    parser.add_argument("--exploration_weight", type=float, default=10.0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

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
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            num_mcts_rollouts=args.num_mcts_rollouts,
            shapley_samples=args.shapley_samples,
            max_children=args.max_children,
            exploration_weight=args.exploration_weight,
            max_edges=args.max_edges,
        )

        if explanation is None:
            continue

        explanations.append(explanation)

        print(
            f"Done | subgraph nodes: {explanation['subgraph_num_nodes']} | "
            f"edges: {explanation['subgraph_num_edges']} | "
            f"selected nodes: {explanation['selected_node_count']} | "
            f"selected edges: {len(explanation['top_edges'])} | "
            f"original prob: {explanation['subgraph_illicit_prob']:.4f} | "
            f"selected prob: {explanation['selected_subgraph_prob']:.4f} | "
            f"shapley reward: {explanation['shapley_reward']:.4f}"
        )

    if len(explanations) == 0:
        raise RuntimeError("No valid SubgraphX-MCTS explanations were generated.")

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "explanations"
        / "subgraphx_mcts"
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
                "min_nodes": args.min_nodes,
                "max_nodes": args.max_nodes,
                "max_edges": args.max_edges,
                "num_mcts_rollouts": args.num_mcts_rollouts,
                "shapley_samples": args.shapley_samples,
                "max_children": args.max_children,
                "exploration_weight": args.exploration_weight,
                "method_note": "SubgraphX-MCTS with Shapley-value-style subgraph scoring, adapted for node classification.",
                "explanations": explanations,
            },
            f,
            indent=2,
        )

    print("\nSaved SubgraphX-MCTS explanations to:")
    print(output_path)


if __name__ == "__main__":
    main()