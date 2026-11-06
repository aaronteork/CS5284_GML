"""Model factory and backward-compatible imports for the training pipeline."""
from .common import AccidentModel, RoadPredictor
from .mlp import Identity
from .graphsage import EdgeSAGEConv, GraphSAGE
from .gatedgcn import GatedGCN, GatedGCNLayer
from .graph_transformer import GraphTransformer

MODEL_NAMES = ("mlp", "graphsage", "gatedgcn", "graph_transformer")


def build_model(name, in_channels_node, in_channels_edge, hidden_channels=256, num_layers=2,
                dropout=0.0, num_heads=4):
    if name == "mlp":
        encoder = Identity(in_channels_node)
    elif name == "graphsage":
        encoder = GraphSAGE(in_channels_node, in_channels_edge, hidden_channels, num_layers, dropout)
    elif name == "gatedgcn":
        encoder = GatedGCN(in_channels_node, in_channels_edge, hidden_channels, num_layers, dropout)
    elif name == "graph_transformer":
        encoder = GraphTransformer(in_channels_node, in_channels_edge, hidden_channels,
                                   num_layers, dropout, num_heads)
    else:
        raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")
    return AccidentModel(encoder, in_channels_edge, hidden_channels, num_layers, dropout)
