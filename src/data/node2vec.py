"""Generate Node2Vec embeddings for a state's road graph.

    python -m src.data.node2vec --state DE MD

Uses the paper's walk settings (karateclub's defaults: 10 walks of length 80 per
node, p = q = 1, window 5, 128 dimensions) on the undirected road graph. The paper
trains these walks with gensim's CBOW word2vec; here they're trained as skip-gram
with negative sampling in PyTorch, because gensim's training silently corrupts
the embeddings on large corpora on some platforms.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import DATA_DIR, download_state


def node2vec_path(state, data_dir=DATA_DIR, dimensions=128):
    return Path(data_dir) / "embeddings" / "node2vec" / f"{state}_{dimensions}.npy"


def undirected_csr(edge_index, num_nodes):
    """Row pointers and column indices of the undirected, deduplicated graph."""
    src, dst = edge_index
    both = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    key = torch.unique(both[0] * num_nodes + both[1])
    rows, cols = key // num_nodes, key % num_nodes
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.bincount(rows, minlength=num_nodes).cumsum(0)
    return rowptr, cols


def uniform_random_walks(start, rowptr, cols, walk_length):
    """[len(start), walk_length] uniform random walks; isolated nodes stay put."""
    walks = torch.empty(len(start), walk_length, dtype=torch.long, device=start.device)
    walks[:, 0] = start
    degree = rowptr[1:] - rowptr[:-1]
    for step in range(1, walk_length):
        current = walks[:, step - 1]
        deg = degree[current]
        offset = (torch.rand(len(current), device=start.device) * deg).long()
        neighbour = cols[(rowptr[current] + offset).clamp(max=len(cols) - 1)]
        walks[:, step] = torch.where(deg > 0, neighbour, current)
    return walks


def context_pairs(walks, window_size):
    """(centre, context) pairs for every position within `window_size` steps, both directions."""
    centres, contexts = [], []
    for k in range(1, window_size + 1):
        a, b = walks[:, :-k].reshape(-1), walks[:, k:].reshape(-1)
        centres += [a, b]
        contexts += [b, a]
    return torch.cat(centres), torch.cat(contexts)


class SkipGram(nn.Module):
    def __init__(self, num_nodes, dimensions):
        super().__init__()
        self.input = nn.Embedding(num_nodes, dimensions, sparse=True)
        self.output = nn.Embedding(num_nodes, dimensions, sparse=True)
        nn.init.uniform_(self.input.weight, -0.5 / dimensions, 0.5 / dimensions)
        nn.init.zeros_(self.output.weight)

    def loss(self, centres, contexts, negatives):
        u = self.input(centres)                               # [P, D]
        pos = (u * self.output(contexts)).sum(-1)             # [P]
        neg = torch.bmm(self.output(negatives), u.unsqueeze(-1)).squeeze(-1)  # [P, K]
        return -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).sum(-1).mean())


def generate_node2vec(state, data_dir=DATA_DIR, dimensions=128, walk_length=80, walks_per_node=10,
                      window_size=5, num_negative=5, epochs=1, lr=0.01, batch_size=256, seed=42,
                      device=None):
    out_path = node2vec_path(state, data_dir, dimensions)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    state_dir = download_state(state, data_dir)
    adj = torch.load(state_dir / "adj_matrix.pt", weights_only=False).coalesce()
    num_nodes = adj.shape[0]
    rowptr, cols = (t.to(device) for t in undirected_csr(adj.indices(), num_nodes))

    # word2vec's negative-sampling distribution: token frequency^0.75, and walk frequency ~ degree.
    degree = (rowptr[1:] - rowptr[:-1]).float().clamp(min=1)
    negative_probs = degree.pow(0.75)

    model = SkipGram(num_nodes, dimensions).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)

    starts = torch.arange(num_nodes, device=device).repeat(walks_per_node)
    num_batches = (len(starts) + batch_size - 1) // batch_size
    print(f"{state}: {len(starts):,} walks of length {walk_length} over {num_nodes:,} nodes on {device}")
    for epoch in range(1, epochs + 1):
        start_time, total = time.time(), 0.0
        for b, batch in enumerate(starts[torch.randperm(len(starts), device=device)].split(batch_size)):
            walks = uniform_random_walks(batch, rowptr, cols, walk_length)
            centres, contexts = context_pairs(walks, window_size)
            negatives = torch.multinomial(negative_probs, len(centres) * num_negative, replacement=True)
            loss = model.loss(centres, contexts, negatives.view(-1, num_negative))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            if (b + 1) % 500 == 0 or b + 1 == num_batches:
                print(f"  epoch {epoch} batch {b + 1}/{num_batches} | loss {total / (b + 1):.4f} | "
                      f"{time.time() - start_time:.0f}s")

    embeddings = model.input.weight.detach().cpu().numpy().astype(np.float32)
    np.save(out_path, embeddings)
    print(f"{state}: saved {embeddings.shape} to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", nargs="+", default=["DE"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    for state in args.state:
        generate_node2vec(state, epochs=args.epochs, device=args.device)
