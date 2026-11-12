# Pinned comparison baseline; extracted from osm_baseline_comparison.ipynb, cell 7.
"""Persistent monthly and structural feature caches adapted from baseline-fw."""
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def load_node_cache(state_dir, columns, num_nodes):
    state_dir = Path(state_dir)
    files = sorted((state_dir / "Nodes").glob("node_features_*_*.csv"),
                   key=lambda p: tuple(map(int, p.stem.split("_")[-2:])))
    if not files:
        raise ValueError(f"No monthly node features in {state_dir}")
    signature = {"version": 2, "columns": columns, "num_nodes": num_nodes,
                 "sources": [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files]}
    cache_dir = state_dir / "cache"
    metadata = cache_dir / "monthly_inputs.json"
    paths = {key: cache_dir / f"{key}.npy" for key in ("weather", "coords")}
    if metadata.exists() and all(p.exists() for p in paths.values()):
        if json.loads(metadata.read_text(encoding="utf-8")) == signature:
            return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}, files
    cache_dir.mkdir(exist_ok=True)
    weather = np.lib.format.open_memmap(paths["weather"].with_suffix(".tmp.npy"),
                mode="w+", dtype=np.float32, shape=(len(files), num_nodes, len(columns)))
    coords = None
    for i, path in enumerate(tqdm(files, desc=f"Cache {state_dir.name} weather", unit="month")):
        df = pd.read_csv(path)
        if len(df) != num_nodes:
            raise ValueError(f"{path} has {len(df)} nodes; expected {num_nodes}")
        if i == 0:
            coords = df[["lon", "lat"]].to_numpy(dtype=np.float32)
        weather[i] = df[columns].to_numpy(dtype=np.float32)
    weather.flush()
    del weather
    np.save(paths["coords"].with_suffix(".tmp.npy"), coords)
    for path in paths.values():
        path.with_suffix(".tmp.npy").replace(path)
    metadata.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}, files


def load_static_cache(state_dir, edge_index, num_nodes, k=500, seed=0):
    state_dir = Path(state_dir)
    cache_dir = state_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    graph_file = state_dir / "adj_matrix.pt"
    signature = dict(version=1, num_nodes=num_nodes, k=min(k, num_nodes), seed=seed,
                     graph_size=graph_file.stat().st_size, graph_mtime=graph_file.stat().st_mtime_ns)
    meta, path = cache_dir / "static_inputs.json", cache_dir / "static_inputs.npz"
    if meta.exists() and path.exists() and json.loads(meta.read_text(encoding="utf-8")) == signature:
        with np.load(path) as cache:
            return {key: cache[key].copy() for key in cache.files}
    src, dst = edge_index.numpy()
    in_degree = np.bincount(dst, minlength=num_nodes)
    out_degree = np.bincount(src, minlength=num_nodes)
    eye = np.eye(max(in_degree.max(), out_degree.max()) + 1, dtype=np.float32)
    degree = np.concatenate([eye[in_degree], eye[out_degree]], axis=1)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.t().tolist())
    print(f"{state_dir.name}: computing approximate betweenness (k={signature['k']}, once per graph)", flush=True)
    started = time.perf_counter()
    centrality = nx.betweenness_centrality(graph, k=signature["k"], normalized=True, seed=seed)
    bc = np.array([centrality[i] for i in range(num_nodes)], dtype=np.float32)[:, None]
    arrays = dict(degree=degree, centrality=bc)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)
    meta.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    print(f"{state_dir.name}: structural cache ready in {time.perf_counter()-started:.1f}s", flush=True)
    return arrays
