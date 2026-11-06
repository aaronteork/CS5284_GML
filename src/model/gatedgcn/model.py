"""Residual GatedGCN with directed, feature-wise normalised edge gates.

Uses the node/edge update formulation from the Benchmarking GNNs GatedGCN:
https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/layers/gated_gcn_layer.py
Implemented with native PyTorch; no PyG or DGL dependency is required.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedGCNLayer(nn.Module):
    """Jointly update hidden node and edge states, with residual connections.

    For each directed edge j -> i, compute
        z_ji = C e_ji + D h_j + E h_i
        g_ji = sigmoid(z_ji)
        m_i = sum_j(g_ji * B h_j) / (sum_j(g_ji) + eps)
    Node updates use A h_i + m_i. Node and edge updates then receive batch
    normalisation, ReLU, residual addition and dropout. No self-loops are added:
    the A transform and residual already preserve each node's own information.
    """

    def __init__(self, hidden_channels, dropout=0.0, eps=1e-6):
        super().__init__()
        self.lin_self = nn.Linear(hidden_channels, hidden_channels)
        self.lin_message = nn.Linear(hidden_channels, hidden_channels)
        self.lin_edge = nn.Linear(hidden_channels, hidden_channels)
        self.lin_src = nn.Linear(hidden_channels, hidden_channels)
        self.lin_dst = nn.Linear(hidden_channels, hidden_channels)
        self.node_norm = nn.BatchNorm1d(hidden_channels)
        self.edge_norm = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout
        self.eps = eps

    @staticmethod
    def _normalise(norm, values):
        # BatchNorm cannot estimate training variance from a single row. Use its
        # running statistics for singleton graphs/edges; empty edges stay empty.
        if values.size(0) == 0:
            return values
        if norm.training and values.size(0) == 1:
            return F.batch_norm(values, norm.running_mean, norm.running_var,
                                norm.weight, norm.bias, training=False, eps=norm.eps)
        return norm(values)

    def forward(self, h, edge_index, e):
        src, dst = edge_index
        edge_update = self.lin_edge(e) + self.lin_src(h)[src] + self.lin_dst(h)[dst]
        gates = torch.sigmoid(edge_update)
        messages = gates * self.lin_message(h)[src]
        numerator = torch.zeros_like(h).index_add(0, dst, messages)
        denominator = torch.zeros_like(h).index_add(0, dst, gates)
        node_update = self.lin_self(h) + numerator / (denominator + self.eps)
        node_update = F.relu(self._normalise(self.node_norm, node_update))
        edge_update = F.relu(self._normalise(self.edge_norm, edge_update))
        h = F.dropout(h + node_update, self.dropout, training=self.training)
        e = F.dropout(e + edge_update, self.dropout, training=self.training)
        return h, e


class GatedGCN(nn.Module):
    """Project road/node inputs and apply residual gated message passing.

    Returns node embeddings for the shared endpoint/road predictor. Hidden edge
    states evolve between layers; the predictor still receives the original road
    attributes, keeping its inputs consistent with the GraphSAGE comparison.
    """

    def __init__(self, in_channels_node, in_channels_edge, hidden_channels,
                 num_layers=2, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GatedGCN requires at least one layer")
        self.node_projection = nn.Linear(in_channels_node, hidden_channels)
        self.edge_projection = nn.Linear(in_channels_edge, hidden_channels)
        self.convs = nn.ModuleList(
            GatedGCNLayer(hidden_channels, dropout) for _ in range(num_layers)
        )
        self.out_channels = hidden_channels

    def forward(self, x, edge_index, edge_attr):
        h = self.node_projection(x)
        e = self.edge_projection(edge_attr)
        for conv in self.convs:
            h, e = conv(h, edge_index, e)
        return h
