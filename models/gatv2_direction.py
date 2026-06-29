import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv


class DirectionAwareGATv2(torch.nn.Module):
    """
    Residual GATv2 with edge direction features.

    edge_attr format:
    [1, 0] = original transaction direction
    [0, 1] = reverse message-passing direction
    """

    uses_edge_attr = True

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
        edge_dim: int = 2,
    ):
        super().__init__()

        self.dropout = dropout

        self.input_proj = nn.Linear(in_channels, hidden_channels * heads)

        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        self.norm1 = nn.LayerNorm(hidden_channels * heads)

        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        self.norm2 = nn.LayerNorm(hidden_channels * heads)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        residual = self.input_proj(x)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x + residual)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        residual = x

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x + residual)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)