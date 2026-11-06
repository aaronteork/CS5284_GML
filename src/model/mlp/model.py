import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    """No-graph encoder: node features pass through unchanged (the paper's MLP baseline)."""

    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x, edge_index, edge_attr):
        return x


