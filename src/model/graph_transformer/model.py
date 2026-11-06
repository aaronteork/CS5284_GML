"""Sparse edge-aware dot-product attention over the directed road graph.

The edge-modulated score follows Week 09 Lab 03. Blocks use pre-LayerNorm
attention and feed-forward residual updates. Raw road attributes are projected
independently in each block; this variant does not evolve hidden edge states.
Existing shared structural inputs (including optional Node2Vec) are retained;
no new Laplacian positional encodings or dense node-pair tensors are introduced.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseEdgeAttention(nn.Module):
    """Multi-head attention, normalised over incoming edges per destination.

    For j -> i and head width d:
        score_ij = sum(Q(h_i) * E(road_ji) * K(h_j)) / sqrt(d)
        message_ij = softmax_j(score_ij) * V(h_j)
    Heads are concatenated and projected to the hidden width. No self-loops are
    added; isolated nodes receive zero attention output and retain their state
    through the block's residual path. Memory scales with edges, not node pairs.
    """

    def __init__(self, hidden_channels, in_channels_edge, num_heads=4, dropout=0.0):
        super().__init__()
        if num_heads < 1 or hidden_channels < 1 or hidden_channels % num_heads:
            raise ValueError("hidden_channels must be positive and divisible by positive num_heads")
        self.num_heads = num_heads
        self.head_channels = hidden_channels // num_heads
        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.key = nn.Linear(hidden_channels, hidden_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        self.edge_projection = nn.Linear(in_channels_edge, hidden_channels)
        self.output = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.dropout = dropout

    def forward(self, h, edge_index, edge_attr):
        num_nodes = h.size(0)
        src, dst = edge_index
        shape = (num_nodes, self.num_heads, self.head_channels)
        q, k, v = (projection(h).reshape(shape)
                   for projection in (self.query, self.key, self.value))
        edge = self.edge_projection(edge_attr).reshape(
            edge_attr.size(0), self.num_heads, self.head_channels)
        logits = (q[dst] * edge * k[src]).sum(-1) / math.sqrt(self.head_channels)

        # Numerically stable segmented softmax. Missing edges never enter the
        # softmax; maxima are detached because they only shift each segment.
        destinations = dst[:, None].expand(-1, self.num_heads)
        maxima = logits.new_full((num_nodes, self.num_heads), -torch.inf)
        maxima.scatter_reduce_(0, destinations, logits.detach(), reduce="amax",
                               include_self=True)
        weights = torch.exp(logits - maxima[dst])
        totals = logits.new_zeros((num_nodes, self.num_heads)).index_add(0, dst, weights)
        alpha = weights / totals[dst].clamp_min(torch.finfo(weights.dtype).tiny)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        messages = alpha.unsqueeze(-1) * v[src]
        aggregated = h.new_zeros(shape).index_add(0, dst, messages)
        return self.output(aggregated.reshape(num_nodes, -1))


class GraphTransformerLayer(nn.Module):
    """Pre-normalised attention and 2x-width feed-forward residual updates."""

    def __init__(self, hidden_channels, in_channels_edge, num_heads=4, dropout=0.0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_channels)
        self.attention = SparseEdgeAttention(hidden_channels, in_channels_edge,
                                             num_heads, dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_channels, hidden_channels),
        )
        self.dropout = dropout

    def forward(self, h, edge_index, edge_attr):
        update = self.attention(self.attention_norm(h), edge_index, edge_attr)
        h = h + F.dropout(update, self.dropout, training=self.training)
        update = self.feedforward(self.feedforward_norm(h))
        return h + F.dropout(update, self.dropout, training=self.training)


class GraphTransformer(nn.Module):
    """Node encoder compatible with the shared per-road AccidentModel head."""

    def __init__(self, in_channels_node, in_channels_edge, hidden_channels,
                 num_layers=2, dropout=0.0, num_heads=4):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GraphTransformer requires at least one layer")
        self.node_projection = nn.Linear(in_channels_node, hidden_channels)
        self.layers = nn.ModuleList(
            GraphTransformerLayer(hidden_channels, in_channels_edge, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.out_channels = hidden_channels

    def forward(self, x, edge_index, edge_attr):
        h = self.node_projection(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return h
