"""Graph Attention Network (Velickovic et al., 2018), in native PyTorch.

For each head k and each edge j -> i (self-loop i -> i included):
    e_ij  = LeakyReLU(a_k . [W_k h_i || W_k h_j])
    alpha = softmax of e_ij over the neighbours j of i
    out_i = || over heads k of  sum_j alpha_ij W_k h_j
The GAT layer does not use road features; the shared per-road predictor still
receives them. Layers are stacked like GraphSAGE: conv, BatchNorm, ReLU (not
after the last layer), dropout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_softmax(scores, index, num_nodes):
    """Softmax of [E, H] scores over the edges that share the same destination."""
    heads = scores.size(1)
    maxima = scores.new_full((num_nodes, heads), -torch.inf)
    maxima.scatter_reduce_(0, index[:, None].expand(-1, heads), scores.detach(),
                           reduce="amax", include_self=True)
    weights = torch.exp(scores - maxima[index])          # subtract max for stability
    totals = scores.new_zeros((num_nodes, heads)).index_add(0, index, weights)
    return weights / totals[index]


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.0,
                 negative_slope=0.2):
        super().__init__()
        if out_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        self.lin = nn.Linear(in_channels, out_channels, bias=False)       # W (all heads)
        # a = [a_dst || a_src], so a . [W h_i || W h_j] = a_dst . W h_i + a_src . W h_j
        self.att_dst = nn.Parameter(torch.empty(num_heads, self.head_channels))
        self.att_src = nn.Parameter(torch.empty(num_heads, self.head_channels))
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_src)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.dropout = dropout
        self.negative_slope = negative_slope

    def forward(self, x, edge_index):
        n = x.size(0)
        h = self.lin(x).view(n, self.num_heads, self.head_channels)      # W h, per head

        loops = torch.arange(n, device=x.device)                          # self-loops
        src = torch.cat([edge_index[0], loops])
        dst = torch.cat([edge_index[1], loops])

        score_dst = (h * self.att_dst).sum(-1)                            # [N, H]
        score_src = (h * self.att_src).sum(-1)
        e = F.leaky_relu(score_dst[dst] + score_src[src], self.negative_slope)   # [E+N, H]

        alpha = segment_softmax(e, dst, n)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        out = h.new_zeros(n, self.num_heads, self.head_channels)
        out = out.index_add(0, dst, alpha.unsqueeze(-1) * h[src])
        return out.reshape(n, -1) + self.bias                             # concat heads


class GAT(nn.Module):
    """Node encoder compatible with the shared per-road AccidentModel head."""

    def __init__(self, in_channels_node, in_channels_edge, hidden_channels,
                 num_layers=2, dropout=0.0, num_heads=4):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GAT requires at least one layer")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_channels = in_channels_node if layer == 0 else hidden_channels
            self.convs.append(GATLayer(in_channels, hidden_channels, num_heads, dropout))
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
