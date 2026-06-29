from pathlib import Path

from utils.elliptic_loader import load_elliptic, print_elliptic_summary


def load_dataset(
    dataset_name: str,
    data_root: str | Path,
    normalize: bool = True,
    make_undirected: bool = False,
):
    data_root = Path(data_root)

    if dataset_name.lower() == "elliptic":
        data = load_elliptic(
            data_root / "elliptic",
            normalize=normalize,
            make_undirected=make_undirected,
        )
        return data, print_elliptic_summary

    if dataset_name.lower() == "amlsim":
        raise NotImplementedError(
            "AMLSim loader is not implemented yet. It will be added after Elliptic framework is stable."
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")