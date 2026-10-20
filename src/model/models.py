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


class RoadPredictor(nn.Module):
    """MLP on [h_u, h_v, road features] for each road (u, v). Returns raw scores."""

    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [1]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, h_u, h_v, road_attr):
        z = torch.cat([h_u, h_v, road_attr], dim=1)
        for lin in self.lins[:-1]:
            z = F.dropout(F.relu(lin(z)), self.dropout, training=self.training)
        return self.lins[-1](z).squeeze(-1)


class AccidentModel(nn.Module):
    """Encoder over the road graph followed by a per-road predictor.

    `forward` returns logits for classification and accident counts for regression.
    """

    def __init__(self, encoder, in_channels_edge, hidden_channels, num_predictor_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.predictor = RoadPredictor(
            2 * encoder.out_channels + in_channels_edge, hidden_channels, num_predictor_layers, dropout,
        )

    def forward(self, data, road_ids=None):
        h = self.encoder(data.x, data.edge_index, data.edge_attr)
        road_index, road_attr = data.road_index, data.road_attr
        if road_ids is not None:
            road_index, road_attr = road_index[:, road_ids], road_attr[road_ids]
        return self.predictor(h[road_index[0]], h[road_index[1]], road_attr)


def build_model(name, in_channels_node, in_channels_edge, hidden_channels=256, num_layers=2, dropout=0.0):
    if name == "mlp":
        encoder = Identity(in_channels_node)
    elif name == "graphsage":
        encoder = GraphSAGE(in_channels_node, in_channels_edge, hidden_channels, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model {name!r}; choose 'mlp' or 'graphsage'")
    return AccidentModel(encoder, in_channels_edge, hidden_channels, num_layers, dropout)
