from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def normalize_features_by_train_time(
    x: torch.Tensor,
    time_step: torch.Tensor,
    train_end_time: int = 34,
) -> torch.Tensor:
    """
    Normalize node features using only train-time nodes.
    This avoids leaking validation/test-period feature statistics.
    """
    train_time_mask = time_step <= train_end_time

    mean = x[train_time_mask].mean(dim=0, keepdim=True)
    std = x[train_time_mask].std(dim=0, keepdim=True)

    std = torch.where(std == 0, torch.ones_like(std), std)

    return (x - mean) / std


def load_elliptic(
    data_dir: str | Path,
    normalize: bool = True,
    make_undirected: bool = False,
) -> Data:
    data_dir = Path(data_dir)

    features_path = data_dir / "elliptic_txs_features.csv"
    classes_path = data_dir / "elliptic_txs_classes.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"

    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edges_path)

    tx_ids = features.iloc[:, 0].astype(str).to_numpy()
    tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    time_step = torch.tensor(features.iloc[:, 1].to_numpy(), dtype=torch.long)
    x = torch.tensor(features.iloc[:, 2:].to_numpy(), dtype=torch.float)

    if normalize:
        x = normalize_features_by_train_time(
            x=x,
            time_step=time_step,
            train_end_time=34,
        )

    y = torch.full((len(tx_ids),), -1, dtype=torch.long)

    class_map = {
        "2": 0,          # licit
        "1": 1,          # illicit
        "licit": 0,
        "illicit": 1,
    }

    classes_tx_ids = classes["txId"].astype(str)
    classes_labels = classes["class"].astype(str)

    for tx_id, label in zip(classes_tx_ids, classes_labels):
        if tx_id in tx_id_to_idx and label in class_map:
            y[tx_id_to_idx[tx_id]] = class_map[label]

    edge_src_ids = edges["txId1"].astype(str)
    edge_dst_ids = edges["txId2"].astype(str)

    src = edge_src_ids.map(tx_id_to_idx)
    dst = edge_dst_ids.map(tx_id_to_idx)

    valid_edges = src.notna() & dst.notna()

    src = src[valid_edges].astype(np.int64).to_numpy()
    dst = dst[valid_edges].astype(np.int64).to_numpy()

    edge_index = torch.from_numpy(np.vstack([src, dst])).long()
    if make_undirected:
        edge_index = to_undirected(edge_index)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        time_step=time_step,
    )

    labelled_mask = y != -1

    data.train_mask = labelled_mask & (time_step <= 34)
    data.val_mask = labelled_mask & (time_step >= 35) & (time_step <= 38)
    data.test_mask = labelled_mask & (time_step >= 39)

    return data


def print_elliptic_summary(data: Data) -> None:
    print("Total nodes:", data.num_nodes)
    print("Total edges:", data.num_edges)
    print("Node features:", data.num_node_features)

    print("Minimum time step:", int(data.time_step.min()))
    print("Maximum time step:", int(data.time_step.max()))

    print("\nFeature statistics:")
    print("Feature mean approx:", float(data.x.mean()))
    print("Feature std approx:", float(data.x.std()))

    print("\nLabelled nodes:", int((data.y != -1).sum()))
    print("Unknown nodes:", int((data.y == -1).sum()))

    for name, mask in [
        ("Train", data.train_mask),
        ("Validation", data.val_mask),
        ("Test", data.test_mask),
    ]:
        labels = data.y[mask]
        counts = torch.bincount(labels, minlength=2)

        print(f"\n{name} labelled nodes:", int(mask.sum()))
        print(f"{name} class distribution [licit, illicit]:", counts.tolist())