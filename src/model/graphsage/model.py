import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeSAGEConv(nn.Module):
    """The paper's GraphSAGE layer.

    Each message is the transformed neighbour plus the transformed connecting edge's
    features; messages are mean-aggregated with a self-loop (zero edge features) and
    the result is L2-normalised. Unlike standard GraphSAGE there is no separate root weight.
    """

    def __init__(self, in_channels_node, in_channels_edge, out_channels):
        super().__init__()
        self.lin_node = nn.Linear(in_channels_node, out_channels)
        self.lin_edge = nn.Linear(in_channels_edge, out_channels) if in_channels_edge else None

    def forward(self, x, edge_index, edge_attr):
        h = self.lin_node(x)
        src, dst = edge_index

        messages = h[src]
        self_loop = h
        if self.lin_edge is not None:
            messages = messages + self.lin_edge(edge_attr)
            self_loop = self_loop + self.lin_edge.bias

        out = self_loop.index_add(0, dst, messages)
        degree = torch.bincount(dst, minlength=x.size(0)).to(h.dtype) + 1
        return F.normalize(out / degree.unsqueeze(1), p=2, dim=-1)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_channels = in_channels_node if layer == 0 else hidden_channels
            self.convs.append(EdgeSAGEConv(in_channels, in_channels_edge, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = dropout
        self.out_channels = hidden_channels

    def forward(self, x, edge_index, edge_attr):
        h = x
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = norm(conv(h, edge_index, edge_attr))
            if layer < len(self.convs) - 1:
                h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        return h


