"""Plain GCN (Kipf & Welling, 2017) over the directed road graph, in native PyTorch.

    out_i = b + sum over j in N(i) + {i} of  W h_j / sqrt(deg_j * deg_i)
deg counts incoming edges plus the self-loop (same as PyG's GCNConv on a
directed edge_index). The GCN layer does not use road features; the shared
per-road predictor still receives them. Layers are stacked like GraphSAGE:
conv, BatchNorm, ReLU (not after the last layer), dropout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index):
        n = x.size(0)
        h = self.lin(x)
        src, dst = edge_index
        deg = torch.bincount(dst, minlength=n).to(h.dtype) + 1   # +1 for the self-loop
        inv_sqrt = deg.rsqrt()
        weight = (inv_sqrt[src] * inv_sqrt[dst]).unsqueeze(1)
        out = (h * (inv_sqrt * inv_sqrt).unsqueeze(1)).index_add(0, dst, weight * h[src])
        return out + self.bias


class GCN(nn.Module):
    """Node encoder compatible with the shared per-road AccidentModel head."""

    def __init__(self, in_channels_node, in_channels_edge, hidden_channels,
                 num_layers=2, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GCN requires at least one layer")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_channels = in_channels_node if layer == 0 else hidden_channels
            self.convs.append(GCNLayer(in_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = dropout
        self.out_channels = hidden_channels

    def forward(self, x, edge_index, edge_attr=None):
        h = x
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = norm(conv(h, edge_index))
            if layer < len(self.convs) - 1:
                h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        return h
