"""Attach one target month's accident-history features to ML4RoadSafety data.

The interface mirrors the OSM notebook's ``attach_osm_features`` function.
History is dynamic, so the caller selects a target month before attaching it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


FEATURE_GROUPS = {
    "own_history": [
        "acc_lag_1m",
        "acc_mean_3m",
        "acc_mean_6m",
        "acc_mean_12m",
        "acc_lag_12m",
        "acc_nonzero_fraction_12m",
    ],
    "neighbour_history": [
        "neighbor_acc_lag_1m_mean",
        "neighbor_acc_mean_3m",
        "neighbor_count",
    ],
    "history_coverage": ["history_months_available_12m"],
}


def attach_accident_history(data, history_dir, target_month, states=("DE",)):
    """Attach one month's features to a loaded ML4RoadSafety data dictionary.

    This follows the OSM notebook's ``attach_osm_features`` pattern. For each
    selected state it adds ``data[state]["accident_history"]`` containing a
    road-aligned feature matrix, names, groups and temporal metadata. The
    target ``y`` array is intentionally not attached as an input feature; use
    the benchmark's accident labels for that month.

    Args:
        data: The dictionary returned by the benchmark ``utils.read_data`` loader.
        history_dir: The ``accident_history/outputs`` directory.
        target_month: A month such as ``"2018-07"``.
        states: States with generated history exports. Delaware is currently
            the only exported state.

    Returns:
        The same ``data`` dictionary, with ``data[state]["accident_history"]``
        added for each requested state.
    """
    month = str(target_month)
    try:
        import pandas as pd

        period = pd.Period(month, freq="M")
    except Exception as exc:
        raise ValueError(f"Invalid target month {target_month!r}; use YYYY-MM") from exc
    if str(period) != month:
        raise ValueError(f"Invalid target month {target_month!r}; use YYYY-MM")

    history_dir = Path(history_dir)
    monthly_dir = history_dir / "monthly_features"
    mapping_path = history_dir / "edge_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Road mapping not found: {mapping_path}")

    import pandas as pd

    mapping = pd.read_csv(mapping_path)
    required_mapping = {"edge_id", "node_1_idx", "node_2_idx"}
    if not required_mapping.issubset(mapping.columns):
        raise ValueError(f"Road mapping must contain {sorted(required_mapping)}")
    mapping_ids = mapping["edge_id"].to_numpy()
    if not np.array_equal(mapping_ids, np.arange(len(mapping_ids))):
        raise ValueError("Road mapping edge_id must be contiguous and in adjacency order")
    mapping_pairs = mapping[["node_1_idx", "node_2_idx"]].to_numpy()

    if isinstance(states, str):
        states = (states,)
    for state in states:
        if state not in data:
            raise KeyError(f"State {state!r} is not present in the loaded benchmark data")
        path = monthly_dir / f"history_{month}.npz"
        if not path.exists():
            raise FileNotFoundError(f"History features for {month} not found: {path}")

        adjacency = data[state]["adj"].coalesce()
        edge_pairs = adjacency.indices().cpu().numpy().T
        if not np.array_equal(edge_pairs, mapping_pairs):
            raise ValueError(
                f"{state}: accident-history roads do not match the benchmark "
                "adjacency order; join by endpoint pair or rebuild the exports"
            )

        with np.load(path, allow_pickle=False) as saved:
            if str(saved["target_month"].item()) != month:
                raise ValueError(f"Target month metadata does not match {path.name}")
            edge_id = saved["edge_id"]
            features = saved["X"]
            names = [str(name) for name in saved["feature_names"]]
            history_cutoff = str(saved["history_cutoff"].item())
            full_12m_history = bool(saved["full_12m_history"].item())

        if not np.array_equal(edge_id, mapping_ids):
            raise ValueError(f"{state}: edge_id in {path.name} does not match edge_mapping.csv")
        if features.shape != (len(edge_pairs), len(names)):
            raise ValueError(f"Unexpected feature shape in {path.name}: {features.shape}")
        if np.isinf(features).any():
            raise ValueError(f"Infinite feature value found in {path.name}; missing values should be NaN")
        if history_cutoff != str(period - 1):
            raise ValueError(f"History cutoff in {path.name} is not strictly before the target month")
        if set(names) != {name for group in FEATURE_GROUPS.values() for name in group}:
            raise ValueError("Feature names do not match the documented feature groups")

        attached_features = features.copy()
        try:
            import torch

            attached_features = torch.from_numpy(attached_features)
        except ImportError:
            pass

        data[state]["accident_history"] = {
            "features": attached_features,
            "names": names,
            "groups": FEATURE_GROUPS,
            "edge_id": edge_id.copy(),
            "node_1_idx": edge_pairs[:, 0].copy(),
            "node_2_idx": edge_pairs[:, 1].copy(),
            "target_month": month,
            "history_cutoff": history_cutoff,
            "full_12m_history": full_12m_history,
        }
    return data


def read_with_accident_history(
    data_dir, local_data_dir, history_dir, target_month, states=("DE",)
):
    """Load the original benchmark with one month's history features attached.

    ``utils`` must be importable from the benchmark project environment.
    """
    import utils

    data = utils.read_data(Path(data_dir), Path(local_data_dir))
    return attach_accident_history(data, history_dir, target_month, states=states)
