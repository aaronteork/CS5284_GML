"""Generate embeddings with karateclub.Node2Vec, without an alternative backend.

Install requirements.txt in Python 3.10 for generation and training. Existing embeddings
without matching provenance are rejected. Use --force to regenerate them.
"""
import argparse
import json
import os
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .dataset import DATA_DIR, download_state

BACKEND = "karateclub.Node2Vec"
KARATECLUB_VERSION = "1.3.3"


def available_walk_workers():
    """Respect Linux CPU affinity/quotas and cap automatic graph copies at eight."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, (int(quota) + int(period) - 1) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, min(count, 8))


def node2vec_path(state, data_dir=DATA_DIR, dimensions=128):
    return Path(data_dir) / "embeddings" / "node2vec" / f"{state}_{dimensions}.npy"


def generate_node2vec(state, data_dir=DATA_DIR, dimensions=128, walk_length=80,
                      walks_per_node=10, window_size=5, epochs=1, seed=42,
                      workers=4, force=False, walk_workers=0):
    if walk_workers < 0 or workers < 1:
        raise ValueError("walk_workers must be >= 0 and workers must be >= 1")
    walk_workers = walk_workers or available_walk_workers()
    params = dict(dimensions=dimensions, walk_length=walk_length,
                  walk_number=walks_per_node, window_size=window_size,
                  p=1.0, q=1.0, epochs=epochs, learning_rate=0.05,
                  min_count=1, seed=seed, workers=workers)
    provenance = {"backend": BACKEND, "karateclub_version": KARATECLUB_VERSION,
                  "state": state, "parameters": params}
    execution = {"mode": "parallel" if walk_workers > 1 else "serial",
                 "walk_workers": walk_workers,
                 "chunk_nodes": 128 if walk_workers > 1 else None,
                 "seed_policy": "SeedSequence([seed, chunk_id])" if walk_workers > 1 else "original"}
    out_path = node2vec_path(state, data_dir, dimensions)
    metadata_path = out_path.with_suffix(".json")
    if out_path.exists() and not force:
        if not metadata_path.exists():
            raise ValueError(f"{out_path} has no Node2Vec provenance; regenerate with --force")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(k) != v for k, v in provenance.items()):
            raise ValueError(f"{out_path} uses a different backend or settings; regenerate with --force")
        if "execution" in metadata and metadata["execution"] != execution:
            raise ValueError(f"{out_path} uses different walk execution settings; regenerate with --force")
        _validate_embeddings(np.load(out_path), metadata["num_nodes"], dimensions)
        print(f"{state}: reusing validated Node2Vec embeddings from {out_path}", flush=True)
        return out_path

    try:
        import networkx as nx
        from karateclub import Node2Vec
    except ImportError as exc:
        raise RuntimeError("Node2Vec requires the original karateclub backend. Install "
                           "requirements.txt in a Python 3.10 environment. "
                           "No replacement backend is used.") from exc
    if version("karateclub") != KARATECLUB_VERSION:
        raise RuntimeError(f"Use karateclub=={KARATECLUB_VERSION} for reproducible embeddings")

    state_dir = download_state(state, data_dir)
    adj = torch.load(state_dir / "adj_matrix.pt", weights_only=False).coalesce()
    num_nodes = adj.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(adj.indices().t().tolist())

    print(f"{state}: fitting {BACKEND} on {num_nodes:,} nodes (CPU)", flush=True)
    model = Node2Vec(**params)
    # Preserve karateclub's Node2Vec.fit/Word2Vec training and original do_walk.
    # Only dispatch walks across processes; --walk-workers 1 is fully serial.
    walk_context = nullcontext()
    if walk_workers > 1:
        from .parallel_walks import parallel_node2vec_walks
        walk_context = parallel_node2vec_walks(walk_workers, seed)
    with walk_context:
        model.fit(graph)
    embeddings = np.asarray(model.get_embedding(), dtype=np.float32)
    _validate_embeddings(embeddings, num_nodes, dimensions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    provenance.update(num_nodes=num_nodes, gensim_version=version("gensim"), execution=execution)
    metadata_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"{state}: saved {embeddings.shape} to {out_path}")
    return out_path


def _validate_embeddings(embeddings, num_nodes, dimensions):
    if embeddings.shape != (num_nodes, dimensions):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all() or (np.linalg.norm(embeddings, axis=1) == 0).any():
        raise ValueError("Node2Vec produced non-finite or zero embeddings; generation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", nargs="+", default=["DE"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--walk-workers", type=int, default=0,
                        help="Walk processes: 0=automatic (up to 8), 1=original serial execution")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for state in args.state:
        generate_node2vec(state, epochs=args.epochs, workers=args.workers, force=args.force,
                         walk_workers=args.walk_workers)
