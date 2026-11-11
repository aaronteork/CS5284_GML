"""Past-only PHYSICAL-road history; directed graph edges are never collapsed.

Original NPZ labels are summed by unordered endpoints, then windows are rebuilt.
The archived directed X cannot be sliced at representative edges or blindly summed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.history.features import (
    FEATURE_NAMES,
    NeighbourAggregator,
    build_features,
)

GROUPS = {"none": [], "own": [0, 1, 2, 3, 4, 5, 9], "all": list(range(10))}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PhysicalNeighbours(NeighbourAggregator):
    """Use the shared history aggregator for already-merged physical roads."""

    def __init__(self, pairs, num_nodes):
        super().__init__(pairs, num_nodes)
        self.degree = self.edge_degree


build_physical_features = build_features


class HistoryScaler:
    """Streaming training-only statistics; mean imputation -> zero after scaling."""
    def fit(self, matrices):
        total, square, number = (np.zeros(10, np.float64) for _ in range(3))
        for matrix in matrices:
            values = np.asarray(matrix, dtype=np.float64)
            finite = np.isfinite(values)
            clean = np.where(finite, values, 0)
            total += clean.sum(0)
            square += (clean * clean).sum(0)
            number += finite.sum(0)
        self.mean = total / np.maximum(number, 1)
        variance = np.maximum(square / np.maximum(number, 1) - self.mean**2, 0)
        self.std = np.sqrt(variance)
        self.std[self.std < 1e-8] = 1
        return self

    def transform(self, matrix, mode):
        ids = GROUPS[mode]
        values = np.asarray(matrix[:, ids], dtype=np.float32)
        missing = ~np.isfinite(values)
        scaled = np.where(missing, 0, (values - self.mean[ids]) / self.std[ids]).astype(np.float32)
        masks = [i for i, column in enumerate(ids) if column < 8]
        return np.concatenate([scaled, missing[:, masks].astype(np.float32)], axis=1)


class PhysicalHistory:
    def __init__(self, history_dir, state_dir, edge_index, road_keys, accidents,
                 weather_months, train_years, mode):
        if mode not in GROUPS:
            raise ValueError(f"Unknown history mode: {mode}")
        self.mode = mode
        self.directory = Path(history_dir)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        pairs = edge_index.cpu().numpy().T
        mapping = pd.read_csv(self.directory / "edge_mapping.csv")
        if not np.array_equal(mapping.edge_id, np.arange(len(pairs))) or not np.array_equal(
                mapping[["node_1_idx", "node_2_idx"]].to_numpy(), pairs):
            raise ValueError("History mapping does not match the directed adjacency order")
        source_hashes = {}
        for filename, expected in manifest["source_sha256"].items():
            actual = sha256(Path(state_dir) / filename)
            if actual != expected:
                raise ValueError(f"History was built from different source data: {filename}")
            source_hashes[filename] = actual
        # Canonical keys have exactly the same ordering as the existing target loader.
        num_nodes = int(edge_index.max()) + 1
        # Use adjacency node count from the graph itself (isolated trailing nodes may exist).
        import torch
        num_nodes = torch.load(Path(state_dir) / "adj_matrix.pt", map_location="cpu",
                               weights_only=False).shape[0]
        keys = np.minimum(pairs[:, 0], pairs[:, 1]) * num_nodes + np.maximum(pairs[:, 0], pairs[:, 1])
        self.edge_to_road = np.searchsorted(road_keys, keys)
        if not np.array_equal(road_keys[self.edge_to_road], keys):
            raise ValueError("Physical history road mapping differs from target mapping")
        coverage = pd.read_csv(self.directory / "monthly_coverage_audit.csv")
        use = coverage["usable_under_current_assumption"].astype(str).str.lower().eq("true")
        allowed = {pd.Period(m, freq="M") for m in coverage.loc[use, "month"]}
        if not allowed:
            raise ValueError("No usable history months")
        self.periods = pd.period_range(min(allowed), max(allowed), freq="M")
        usable = np.asarray([p in allowed for p in self.periods])
        self.months = [(p.year, p.month) for p in self.periods if p in allowed and (p.year, p.month) in weather_months]
        if not self.months:
            raise ValueError("No overlap between history and weather months")
        self.positions = {(p.year, p.month): i for i, p in enumerate(self.periods)}
        self.counts = np.zeros((len(self.periods), len(road_keys)), dtype=np.float32)
        archives = {}
        for t, period in enumerate(self.periods):
            if not usable[t]:
                continue
            path = self.directory / "monthly_features" / f"history_{period}.npz"
            archives[path.name] = sha256(path)
            with np.load(path, allow_pickle=False) as archive:
                if (str(archive["target_month"].item()) != str(period)
                    or str(archive["history_cutoff"].item()) != str(period - 1)
                    or not np.array_equal(archive["edge_id"], np.arange(len(pairs)))):
                    raise ValueError(f"Invalid archive alignment/temporal metadata: {path.name}")
                y = archive["y"]
                if y.shape != (len(pairs),) or not np.isfinite(y).all() or (y < 0).any() or not np.equal(y, np.floor(y)).all():
                    raise ValueError(f"Invalid directed counts: {path.name}")
                self.counts[t] = np.bincount(self.edge_to_road, weights=y, minlength=len(road_keys))
            expected = np.zeros(len(road_keys), np.float32)
            ids, values = accidents.get((period.year, period.month), ([], []))
            expected[np.asarray(ids, dtype=np.int64)] = values
            if not np.array_equal(expected, self.counts[t]):
                raise ValueError(f"Archive physical counts disagree with CSV labels: {period}")
        signature = {"version": 1, "implementation": sha256(__file__), "source": source_hashes,
                     "archives": archives, "periods": list(map(str, self.periods)),
                     "usable": usable.tolist(), "num_roads": len(road_keys), "features": FEATURE_NAMES}
        self.fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        cache = Path(state_dir) / "cache" / "physical_history"
        cache.mkdir(parents=True, exist_ok=True)
        meta, path = cache / "metadata.json", cache / "features.npy"
        shape = (len(self.periods), len(road_keys), 10)
        valid_cache = meta.exists() and path.exists() and json.loads(meta.read_text()) == signature
        if not valid_cache:
            print("Building past-only physical-road history cache...", flush=True)
            canonical = np.column_stack([road_keys // num_nodes, road_keys % num_nodes])
            neighbours = PhysicalNeighbours(canonical, num_nodes)
            temporary = path.with_suffix(".tmp.npy")
            features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
            for t in range(len(self.periods)):
                features[t] = build_physical_features(self.counts, usable, t, neighbours)
            features.flush()
            del features
            temporary.replace(path)
            meta.write_text(json.dumps(signature, indent=2) + "\n")
        self.raw = np.load(path, mmap_mode="r")
        if self.raw.shape != shape or self.raw.dtype != np.float32:
            raise ValueError("Invalid history cache; remove data/DE/cache/physical_history and rebuild")
        training = [ym for ym in self.months if ym[0] in train_years]
        if not training:
            raise ValueError("No training months for history preprocessing")
        self.scaler = HistoryScaler().fit(self.raw[self.positions[ym]] for ym in training)
        ids = GROUPS[mode]
        self.names = [FEATURE_NAMES[i] for i in ids] + [FEATURE_NAMES[i] + "_missing" for i in ids if i < 8]
        self.metadata = {"mode": mode, "target_unit": "physical", "fingerprint": self.fingerprint,
                         "features": self.names, "training_months": [f"{y:04d}-{m:02d}" for y, m in training],
                         "mean": self.scaler.mean.tolist(), "std": self.scaler.std.tolist(),
                         "history_policy": "strictly-before-target; rolling one-month-ahead",
                         "missing_policy": "training-mean imputation and explicit indicators"}

    def for_edges(self, year, month):
        if (year, month) not in self.months:
            raise ValueError(f"History not usable for {year}-{month:02d}")
        physical = self.scaler.transform(self.raw[self.positions[(year, month)]], self.mode)
        return physical[self.edge_to_road]
