"""Fast Node2Vec-style GPU embeddings adapted from baseline-fw.

Custom skip-gram/negative sampling, p=q=1, 20-step walks. This is a documented
change from the paper's karateclub/gensim implementation, not strict replication.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .dataset import DATA_DIR, download_state

BACKEND = "baseline-fw.torch-skipgram-v2"


def node2vec_path(state, data_dir=DATA_DIR, dimensions=128):
    return Path(data_dir) / "embeddings" / "node2vec_merged" / f"{state}_{dimensions}.npy"


def train_embeddings(edge_index, num_nodes, dimensions=128, walk_length=20,
                     walks_per_node=10, window_size=5, epochs=1, batch_size=1024,
                     lr=0.01, seed=0, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if num_nodes < 1 or min(dimensions, walks_per_node, epochs, batch_size) < 1:
        raise ValueError("Node count, dimensions, walks, epochs and batch size must be positive")
    if walk_length < 2 or not 1 <= window_size < walk_length:
        raise ValueError("Require 1 <= window_size < walk_length")
    # Deduplicate reverse edges before sampling: bidirectional roads must not
    # receive extra probability simply because both directions were recorded.
    ei = edge_index.to(device)
    keys = torch.unique(torch.cat([ei[0] * num_nodes + ei[1], ei[1] * num_nodes + ei[0]]))
    src, col = keys // num_nodes, keys % num_nodes
    degree = torch.bincount(src, minlength=num_nodes)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    rowptr[1:] = degree.cumsum(0)
    rng = torch.Generator(device=device).manual_seed(seed)

    def walks(start):
        path, current = [start], start
        for _ in range(walk_length - 1):
            d = degree[current]
            offset = (torch.rand(current.shape, generator=rng, device=device) * d.clamp(min=1)).long()
            if col.numel():
                nxt = col[(rowptr[current] + offset).clamp(max=col.numel() - 1)]
                current = torch.where(d > 0, nxt, current)
            path.append(current)
        return torch.stack(path, dim=1)

    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        embedding = torch.nn.Embedding(num_nodes, dimensions, device=device)
        context = torch.nn.Embedding(num_nodes, dimensions, device=device)
        torch.nn.init.normal_(embedding.weight, std=0.1)
        torch.nn.init.zeros_(context.weight)
        optimizer = torch.optim.Adam(list(embedding.parameters()) + list(context.parameters()), lr=lr)
        for epoch in range(epochs):
            starts = torch.arange(num_nodes, device=device).repeat(walks_per_node)
            starts = starts[torch.randperm(starts.numel(), generator=rng, device=device)]
            progress = tqdm(starts.split(batch_size), desc=f"Node2Vec epoch {epoch+1}/{epochs}", unit="batch")
            for start in progress:
                w = walks(start)
                centres = torch.cat([w[:, :-offset] for offset in range(1, window_size + 1)], 1).reshape(-1)
                contexts = torch.cat([w[:, offset:] for offset in range(1, window_size + 1)], 1).reshape(-1)
                negative = torch.randint(num_nodes, (centres.numel(),), generator=rng, device=device)
                z = embedding(centres)
                loss = -(F.logsigmoid((z * context(contexts)).sum(-1))
                         + F.logsigmoid(-(z * context(negative)).sum(-1))).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
        return embedding.weight.detach().cpu().numpy().copy()


def generate_node2vec(state, data_dir=DATA_DIR, dimensions=128, walk_length=20,
                      walks_per_node=10, window_size=5, epochs=1, batch_size=1024,
                      lr=0.01, seed=0, device=None, force=False):
    params = dict(dimensions=dimensions, walk_length=walk_length, walks_per_node=walks_per_node,
                  window_size=window_size, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
                  p=1, q=1, negatives_per_pair=1, context_direction="forward")
    expected = dict(backend=BACKEND, state=state, parameters=params)
    out = node2vec_path(state, data_dir, dimensions)
    meta = out.with_suffix(".json")
    state_dir = download_state(state, data_dir)
    adj = torch.load(state_dir / "adj_matrix.pt", map_location="cpu", weights_only=False).coalesce()
    num_nodes = adj.shape[0]
    if out.exists() and not force:
        metadata = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        if any(metadata.get(k) != v for k, v in expected.items()) or metadata.get("num_nodes") != num_nodes:
            raise ValueError(f"{out} belongs to another embedding configuration. Use --force to regenerate.")
        _validate_embeddings(np.load(out), num_nodes, dimensions)
        print(f"{state}: reusing {out}", flush=True)
        return out
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{state}: generating {BACKEND} on {device} ({num_nodes:,} nodes)", flush=True)
    emb = train_embeddings(adj.indices(), num_nodes, dimensions, walk_length, walks_per_node,
                           window_size, epochs, batch_size, lr, seed, device)
    _validate_embeddings(emb, num_nodes, dimensions)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp.npy")
    np.save(temp, emb)
    temp.replace(out)
    expected.update(num_nodes=num_nodes, torch_version=str(torch.__version__), device=device)
    meta.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {emb.shape} to {out}", flush=True)
    return out


def _validate_embeddings(embeddings, num_nodes, dimensions):
    if embeddings.shape != (num_nodes, dimensions) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding shape or non-finite values")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Invalid embedding norms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", nargs="+", default=["DE"], choices=["DE", "MD"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for state in args.state:
        generate_node2vec(state, epochs=args.epochs, seed=args.seed, device=args.device, force=args.force)
