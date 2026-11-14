"""Original OSM model-input transformations; static scaling uses all graph edges."""
import json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

BASELINE_COMMIT = "299962d"
OSM_GROUPS = ("road_character", "road_character_filled", "intersection_control", "cycle_pedestrian_lighting", "nearby_places", "road_shape")
FEATURE_GROUP_PREFIXES = ("road_character", "intersection_control", "cycle_pedestrian_lighting", "nearby_places", "road_shape")
TAG_BASED_GROUPS = ("road_character", "road_character_filled", "intersection_control", "cycle_pedestrian_lighting")
SURFACE_COLUMNS = ("road_character_surface", "road_character_surface_filled")
OLD_GROUP_NAMES = {"g1_filled": "road_character_filled", "g1": "road_character", "g2": "intersection_control",
                   "g3": "cycle_pedestrian_lighting", "g4": "nearby_places", "g5": "road_shape"}
MIN_CATEGORY_ROADS = 50
MISSING_FLAG_THRESHOLD = 0.005
SINUOSITY_RANGE = (1.0, 5.0)
LANES_MAX = 8
STANDARDISED_CLIP = 10.0

def osm_table_path(state, data_dir=None):
    """The model-ready feature table for a state, written by notebooks/osm_features_and_eda.ipynb (Stage 13)."""
    return Path(data_dir or Path(os.environ.get("CS5284_DATA_DIR", "data"))) / f"{state}_osm_features.parquet"

def osm_group_columns(table, state, data_dir=None):
    """{group: [columns]}: which feature columns belong to which group.

    Taken from the meta file written next to the table (`feature_groups_for_ablation`) when there is one, otherwise worked out
    from the column prefixes (`_filled` / `_is_est` columns form road_character_filled; `_est_source` columns are diagnostics).
    """
    data_dir = Path(data_dir or Path(os.environ.get("CS5284_DATA_DIR", "data")))
    meta = data_dir / f"{state}_osm_features_meta.json"
    if meta.exists():
        groups = json.loads(meta.read_text(encoding="utf-8")).get("feature_groups_for_ablation", {})
        if groups:
            return {group: [c for c in columns if c in table.columns] for group, columns in groups.items()}
    groups = {}
    for column in table.columns:
        prefix = next((p for p in FEATURE_GROUP_PREFIXES if column.startswith(p + "_")), None)
        if prefix is None or column.endswith("_est_source"):
            continue
        group = "road_character_filled" if prefix == "road_character" and column.endswith(("_filled", "_is_est")) else prefix
        groups.setdefault(group, []).append(column)
    return groups

def build_osm_features(table, group_columns, groups, src, dst):
    """Turn the saved OSM feature table into model inputs, one row per directed edge of the graph.

    Why: the saved table holds raw values (missing stays NaN, counts and distances are heavy-tailed). The model needs finite,
    comparably scaled numbers, handled the way the table's preprocessing recipe says and the way the baseline handles its own
    road features.

    Takes: the table, the {group: columns} map from `osm_group_columns`, the groups to use, and the graph's edge endpoints
    `src`, `dst` (from `adj.coalesce().indices()`). The table must be in that same order; this is checked.

    How each column is treated:
      * counts (places within a radius, mid-block counts) and distances: log(1 + value). A distance that is missing because the
        state has no place of that kind at all is filled with the largest known distance, with a `__missing` indicator;
      * sinuosity: clipped to SINUOSITY_RANGE; lane counts: clipped to 1 to LANES_MAX;
      * surface (raw or filled): one-hot for types with at least MIN_CATEGORY_ROADS roads, the rest grouped as "other";
      * 0/1 flags stay as they are; every other column is standardised over all edges (static, like the baseline's
        coordinates and road length), and standardised values are clipped at +/- STANDARDISED_CLIP;
      * other missing values become 0 (the standardised mean, or "absent" for flags). A column gets a `__missing` indicator
        when more than MISSING_FLAG_THRESHOLD of matched roads lack it (the sparse tags). Three-state tag columns (cycle_pedestrian_lighting)
        end up as value plus indicator: yes = (1, 0), tagged "no" = (0, 0), not tagged = (0, 1);
      * if any tag-based group (TAG_BASED_GROUPS) is used, an `osm_unmatched` indicator marks roads with no OSM match;
      * columns with no variation are dropped.

    road_character (OSM values only) and road_character_filled (values plus estimates) must not be used together: an estimate is not a measurement.

    Returns: (float32 array [num_edges, num_features], feature names, the group each feature belongs to). With no groups, zero columns.
    """
    groups = tuple(groups)
    missing_groups = [g for g in groups if g not in group_columns]
    if missing_groups:
        raise ValueError(f"OSM groups {missing_groups} are not in the feature table (it has {sorted(group_columns)})")
    if {"road_character", "road_character_filled"} <= set(groups):
        raise ValueError("Use either road_character (OSM values only) or road_character_filled (values plus estimates), not both")
    if not groups:
        return np.zeros((len(src), 0), dtype=np.float32), [], []
    if len(table) != len(src) or not (table["node_1_idx"].to_numpy() == src).all() \
            or not (table["node_2_idx"].to_numpy() == dst).all():
        raise ValueError("OSM feature rows are not in the same order as the graph's edges; rebuild the feature table")

    matched = (table["has_osm_match"].to_numpy() > 0) if "has_osm_match" in table.columns else (table["match_type"] != "unmatched").to_numpy()
    parts, names, owners = [], [], []

    def add(values, name, owner):
        values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if values.max() == values.min():
            return  # no variation: nothing to learn
        parts.append(values.astype(np.float32))
        names.append(name)
        owners.append(owner)

    def missing_flag(is_missing, name, owner, always=False):
        if is_missing.any() and (always or (matched.any() and is_missing[matched].mean() > MISSING_FLAG_THRESHOLD)):
            add(is_missing.astype(float), f"{name}__missing", owner)

    for group in groups:
        for column in group_columns[group]:
            values = table[column]
            if column in SURFACE_COLUMNS:
                counts = values.value_counts()
                grouped = values.where(values.isin(counts[counts >= MIN_CATEGORY_ROADS].index) | values.isna(), "other")
                for category in sorted(grouped.dropna().unique()):
                    add((grouped == category).to_numpy(dtype=float), f"{column}={category}", group)
                missing_flag(grouped.isna().to_numpy(), column, group)
                continue
            v = values.to_numpy(dtype=float)
            always_flag = False
            if column.endswith("_nearest_m") or re.search(r"_\d+m$", column) or "street_count" in column or column.endswith("_midblock"):
                v = np.log1p(v)  # counts and distances are heavy-tailed
                if column.endswith("_nearest_m") and (~np.isfinite(v)).any() and np.isfinite(v).any():
                    always_flag = True
                    v = np.where(np.isfinite(v), v, v[np.isfinite(v)].max())
                    missing_here = ~np.isfinite(values.to_numpy(dtype=float))
            elif column == "road_shape_sinuosity":
                v = np.clip(v, *SINUOSITY_RANGE)
            elif column in ("road_character_lanes", "road_character_lanes_filled"):
                v = np.clip(v, 1, LANES_MAX)
            missing = missing_here if always_flag else ~np.isfinite(v)
            finite = v[np.isfinite(v)]
            if len(finite) and not set(np.unique(finite)) <= {0.0, 1.0}:
                std = finite.std()
                v = np.clip((v - finite.mean()) / (std if std > 0 else 1.0), -STANDARDISED_CLIP, STANDARDISED_CLIP)
            add(v, column, group)
            missing_flag(missing, column, group, always=always_flag)

    if set(groups) & set(TAG_BASED_GROUPS):
        add((~matched).astype(float), "osm_unmatched", "flag")
    if not parts:
        return np.zeros((len(src), 0), dtype=np.float32), [], []
    features = np.stack(parts, axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite OSM features after preprocessing")
    return features, names, owners
