import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv


class GATv2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.classifier = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)