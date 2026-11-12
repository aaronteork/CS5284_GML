"""Past-only features for the supplied Delaware ML4RoadSafety files.

Road identity is the ORDERED node-index pair in adj_matrix.pt, in coalesced
edge order. A separate unordered-pair mapping prevents counting the two
orientations of one physical connection twice when defining neighbours.
No target-month labels enter build_features().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
import torch


FEATURE_NAMES = [
    "acc_lag_1m", "acc_mean_3m", "acc_mean_6m", "acc_mean_12m",
    "acc_lag_12m", "acc_nonzero_fraction_12m",
    "neighbor_acc_lag_1m_mean", "neighbor_acc_mean_3m",
    "neighbor_count", "history_months_available_12m",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class HistoryData:
    source_dir: Path
    raw: pd.DataFrame
    nodes: pd.DataFrame
    edges: pd.DataFrame
    months: pd.PeriodIndex
    counts: np.ndarray
    observed_month: np.ndarray
    usable_month: np.ndarray
    monthly_audit: pd.DataFrame
    audit: dict


def load_data(source_dir, last_usable_month="2022-09", excluded_months=()):
    """Zero-fill omitted pairs only in explicitly usable observed months.

    Presence of positive records is not proof of complete reporting. The
    supplied cutoff is a documented working coverage assumption, not a fact
    established by the CSV. Unobserved months are never marked usable.
    """
    source_dir = Path(source_dir).resolve()
    raw = pd.read_csv(source_dir / "accidents_monthly.csv")
    required = ["year", "month", "node_1", "node_2", "node_1_idx", "node_2_idx", "acc_count"]
    if not set(required).issubset(raw.columns):
        raise ValueError(f"Missing columns: {set(required) - set(raw.columns)}")
    if raw[required].isna().any().any():
        raise ValueError("Missing identifiers, dates or accident counts")
    if not raw.month.between(1, 12).all() or not (raw.acc_count >= 0).all():
        raise ValueError("Invalid month/count")
    if not np.equal(raw.acc_count, np.floor(raw.acc_count)).all():
        raise ValueError("Accident counts must be integers")
    raw["period"] = pd.PeriodIndex(pd.to_datetime(dict(year=raw.year, month=raw.month, day=1)), freq="M")
    adjacency = torch.load(source_dir / "adj_matrix.pt", map_location="cpu", weights_only=True).coalesce()
    edge_pairs = adjacency.indices().numpy().T.copy()
    edges = pd.DataFrame(edge_pairs, columns=["node_1_idx", "node_2_idx"])
    edges.insert(0, "edge_id", np.arange(len(edges), dtype=np.int64))
    first = raw.period.min()
    nodes = pd.read_csv(source_dir / "Nodes" / f"node_features_{first.year}_{first.month}.csv")
    if len(nodes) != adjacency.shape[0] or nodes.node_id.duplicated().any():
        raise ValueError("Node table does not match graph")
    for endpoint in (1, 2):
        idx = raw[f"node_{endpoint}_idx"].to_numpy()
        if idx.min() < 0 or idx.max() >= len(nodes):
            raise ValueError("Accident endpoint outside graph")
        if not np.array_equal(nodes.node_id.to_numpy()[idx], raw[f"node_{endpoint}"].to_numpy()):
            raise ValueError("OSM node IDs and graph indices disagree")
        edges[f"node_{endpoint}"] = nodes.node_id.to_numpy()[edges[f"node_{endpoint}_idx"]]
    lookup = pd.MultiIndex.from_frame(edges[["node_1_idx", "node_2_idx"]])
    raw["edge_id"] = lookup.get_indexer(pd.MultiIndex.from_frame(raw[["node_1_idx", "node_2_idx"]]))
    if (raw.edge_id < 0).any():
        raise ValueError("Accident records do not align with graph edges")
    if raw.duplicated(["period", "edge_id"]).any():
        raise ValueError("Duplicate road-month rows: inspect before summing")
    # Include the rest of the final calendar year so missing terminal months
    # are visible in the audit, without labeling them as zero-accident months.
    months = pd.period_range(first, f"{raw.year.max()}-12", freq="M")
    counts = np.zeros((len(months), len(edges)), dtype=np.int32)
    ti = months.get_indexer(raw.period)
    counts[ti, raw.edge_id.to_numpy()] = raw.acc_count.to_numpy(dtype=np.int32)
    observed = np.asarray(months.isin(raw.period.unique()))
    cutoff = pd.Period(last_usable_month, freq="M")
    excluded = {pd.Period(m, freq="M") for m in excluded_months}
    usable = observed & np.asarray(months <= cutoff) & np.asarray([m not in excluded for m in months])
    monthly = pd.DataFrame({"month": months.astype(str), "has_accident_rows": observed,
                            "usable_under_current_assumption": usable})
    monthly["recorded_accidents"] = np.where(observed, counts.sum(axis=1), np.nan)
    monthly["positive_road_month_rows"] = np.where(observed, (counts > 0).sum(axis=1), np.nan)
    monthly["implicit_zero_fraction"] = np.where(usable, (counts == 0).mean(axis=1), np.nan)
    unique_physical = np.unique(np.sort(edge_pairs, axis=1), axis=0)
    audit = {
        "accident_rows": len(raw), "recorded_accidents": int(raw.acc_count.sum()),
        "nodes": len(nodes), "ordered_graph_edges": len(edges),
        "unordered_endpoint_groups": len(unique_physical),
        "self_loop_edges": int((edge_pairs[:, 0] == edge_pairs[:, 1]).sum()),
        "self_loop_accident_rows": int(raw.node_1_idx.eq(raw.node_2_idx).sum()),
        "months_with_accident_rows": int(observed.sum()), "usable_months": int(usable.sum()),
        "missing_months": months[~observed].astype(str).tolist(),
        "last_usable_month": str(cutoff), "excluded_months": sorted(map(str, excluded)),
        "first_record_month": str(first), "last_record_month": str(raw.period.max()),
        "all_positive_records_match_graph": True, "node_index_mapping_verified": True,
        "coverage_assumption": "Omitted ordered edges are zero only in usable months; reporting completeness and historical road availability are not independently verified.",
        "road_identity": "Ordered endpoint pair in the supplied static graph; edge_id is coalesced adjacency order, not an OSM way ID.",
        "source_sha256": {name: sha256_file(source_dir / name) for name in ["accidents_monthly.csv", "adj_matrix.pt"]},
    }
    return HistoryData(source_dir, raw, nodes, edges, months, counts, observed, usable, monthly, audit)


class NeighbourAggregator:
    """Distinct unordered connections sharing an endpoint; excludes self/reverse.

    Own-road labels/features retain the benchmark's ordered pairs. For neighbour
    features only, counts across the orientations of each connection are summed.
    This is topological proximity, not directed traffic reachability. Parallel
    OSM ways cannot be recovered from a graph that only stores endpoint pairs.
    """
    def __init__(self, edge_pairs, n_nodes):
        edge_pairs = np.asarray(edge_pairs, dtype=np.int64)
        self.pairs, self.group_for_edge = np.unique(np.sort(edge_pairs, axis=1), axis=0, return_inverse=True)
        groups = np.arange(len(self.pairs))
        distinct = self.pairs[:, 0] != self.pairs[:, 1]
        rows = np.r_[self.pairs[:, 0], self.pairs[distinct, 1]]
        cols = np.r_[groups, groups[distinct]]
        self.incidence = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_nodes, len(groups)))
        self.group_sum = sparse.csr_matrix((np.ones(len(edge_pairs), np.float32), (self.group_for_edge, np.arange(len(edge_pairs)))), shape=(len(groups), len(edge_pairs)))
        self.endpoint_count = np.asarray(self.incidence.sum(axis=0)).ravel()
        node_degree = np.asarray(self.incidence.sum(axis=1)).ravel()
        self.group_degree = np.asarray(self.incidence.T @ node_degree).ravel() - self.endpoint_count
        self.edge_degree = self.group_degree[self.group_for_edge].astype(np.int32)

    def mean(self, edge_values):
        edge_values = np.asarray(edge_values, dtype=np.float32)
        if np.isnan(edge_values).all():
            return np.full(len(edge_values), np.nan, np.float32)
        if not np.isfinite(edge_values).all():
            raise ValueError("Neighbour input must be finite or entirely missing")
        group_values = self.group_sum @ edge_values
        totals = self.incidence.T @ (self.incidence @ group_values) - self.endpoint_count * group_values
        means = np.divide(totals, self.group_degree, out=np.zeros_like(totals), where=self.group_degree > 0)
        return means[self.group_for_edge]


def build_features(counts, usable_month, target_idx, neighbours):
    """Features for month t access only counts[:t]. Unavailable history is NaN."""
    counts = np.asarray(counts)
    usable_month = np.asarray(usable_month, dtype=bool)
    if not 0 <= target_idx <= len(counts):
        raise ValueError("Target index outside timeline")
    n_edges = counts.shape[1]
    missing = lambda: np.full(n_edges, np.nan, np.float32)

    def lag(k):
        idx = target_idx - k
        return counts[idx].astype(np.float32) if idx >= 0 and usable_month[idx] else missing()

    def rolling(k, nonzero=False):
        if target_idx < k or not usable_month[target_idx-k:target_idx].all():
            return missing()
        values = counts[target_idx-k:target_idx]
        return (values > 0).mean(axis=0, dtype=np.float32) if nonzero else values.mean(axis=0, dtype=np.float32)

    lag1, mean3 = lag(1), rolling(3)
    available = int(usable_month[max(0, target_idx-12):target_idx].sum())
    return np.column_stack([
        lag1, mean3, rolling(6), rolling(12), lag(12), rolling(12, nonzero=True),
        neighbours.mean(lag1), neighbours.mean(mean3), neighbours.edge_degree,
        np.full(n_edges, available, np.float32),
    ]).astype(np.float32)


def fixed_history_mean(counts, usable_month, start_idx, stop_idx, target_idx):
    """Frozen baseline over [start_idx, stop_idx), strictly before target."""
    if not 0 <= start_idx < stop_idx <= target_idx <= len(counts):
        raise ValueError("Fixed reference window must precede the target month")
    if not np.asarray(usable_month)[start_idx:stop_idx].all():
        raise ValueError("Fixed reference window contains unavailable months")
    return np.asarray(counts)[start_idx:stop_idx].mean(axis=0, dtype=np.float32)


def walk_forward_indices(months, usable_month, start_month, end_month):
    """Yield (target index, strictly prior usable indices), without model fitting."""
    start, end = pd.Period(start_month, "M"), pd.Period(end_month, "M")
    for t, month in enumerate(months):
        if start <= month <= end and usable_month[t]:
            yield t, np.flatnonzero(np.asarray(usable_month)[:t])


def export_features(data, neighbours, output_dir):
    """One compressed, edge-aligned NPZ per usable target month; labels separate."""
    output_dir = Path(output_dir)
    monthly_dir = output_dir / "monthly_features"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    data.edges.to_csv(output_dir / "edge_mapping.csv", index=False)
    data.monthly_audit.to_csv(output_dir / "monthly_coverage_audit.csv", index=False)
    files = []
    for t in np.flatnonzero(data.usable_month):
        month = str(data.months[t])
        X = build_features(data.counts, data.usable_month, t, neighbours)
        path = monthly_dir / f"history_{month}.npz"
        np.savez_compressed(path, X=X, y=data.counts[t], edge_id=data.edges.edge_id.to_numpy(),
                            feature_names=np.asarray(FEATURE_NAMES), target_month=np.asarray(month),
                            history_cutoff=np.asarray(str(data.months[t]-1)),
                            full_12m_history=np.asarray(t >= 12 and data.usable_month[t-12:t].all()))
        files.append(path.name)
    manifest = dict(data.audit)
    manifest.update({"feature_names": FEATURE_NAMES, "feature_files": len(files),
        "feature_dtype": "float32", "label_dtype": str(data.counts.dtype),
        "missing_history": "NaN; complete windows required. Fit any imputer on training only.",
        "isolated_road_neighbour_mean": "0 for known history, with neighbor_count=0; NaN if history unavailable.",
        "neighbours": "Distinct unordered endpoint pairs sharing an intersection, excluding the target pair including reverse orientation; sum counts across orientations before averaging distinct neighbours.",
        "forecast_protocol": "One month ahead; prior-month accident reports assumed available. Not a whole-year forecast from a fixed January origin.",
        "official_team_split": None, "model_performance_evaluated": False,
        "integration": "Join by node_1_idx,node_2_idx or verified edge_mapping; never assume another loader has identical edge order.",
    })
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_self_checks():
    """Small tests for exact lags, future invariance, and neighbour semantics."""
    pairs = np.array([[0, 1], [1, 0], [1, 2], [2, 3], [4, 5], [1, 1]])
    neighbours = NeighbourAggregator(pairs, 6)
    sample = np.array([2, 3, 7, 11, 13, 17], dtype=np.float32)
    # Canonical 0-1 excludes both itself and its reverse; neighbours: 1-2,1-1.
    np.testing.assert_allclose(neighbours.mean(sample), [12, 12, 11, 7, 0, 6])
    np.testing.assert_array_equal(neighbours.edge_degree, [2, 2, 3, 1, 0, 2])
    counts = np.arange(16 * len(pairs), dtype=np.int32).reshape(16, -1)
    usable = np.ones(16, bool)
    X = build_features(counts, usable, 12, neighbours)
    np.testing.assert_allclose(X[:, 0], counts[11])
    np.testing.assert_allclose(X[:, 1], counts[9:12].mean(axis=0))
    np.testing.assert_allclose(X[:, 3], counts[:12].mean(axis=0))
    np.testing.assert_allclose(X[:, 4], counts[0])
    changed = counts.copy()
    changed[12:] = 999999
    np.testing.assert_allclose(X, build_features(changed, usable, 12, neighbours))
    # A genuinely unavailable month invalidates windows; it is not a zero.
    usable[10] = False
    incomplete = build_features(counts, usable, 12, neighbours)
    assert np.isnan(incomplete[:, 1]).all() and np.isfinite(incomplete[:, 0]).all()
    assert (incomplete[:, -1] == 11).all()
    assert np.isnan(build_features(counts, usable, 0, neighbours)[:, 0]).all()
    np.testing.assert_allclose(fixed_history_mean(counts, np.ones(16, bool), 0, 3, 12), counts[:3].mean(axis=0))
    try:
        fixed_history_mean(counts, np.ones(16, bool), 0, 13, 12)
    except ValueError:
        pass
    else:
        raise AssertionError("Future fixed-window labels were accepted")
    return {"past_only_and_future_mutation": "passed", "exact_lags_and_windows": "passed",
            "missing_month_and_warmup": "passed", "neighbours_reverse_loop_isolated": "passed",
            "fixed_reference_cutoff": "passed"}


if __name__ == "__main__":
    print(json.dumps(run_self_checks(), indent=2))
