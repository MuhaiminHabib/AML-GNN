from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.gatv2 import GATv2


def build_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float,
):
    model_name = model_name.lower()

    if model_name == "gcn":
        return GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    if model_name == "graphsage":
        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    if model_name == "gatv2":
        return GATv2(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=4,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model: {model_name}")