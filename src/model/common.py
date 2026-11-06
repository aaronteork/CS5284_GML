import torch
import torch.nn as nn
import torch.nn.functional as F


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


