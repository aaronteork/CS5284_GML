"""Run the merged baseline with OpenStreetMap edge features added.

Nothing in src/ is changed. This script swaps in a dataset class that appends OSM columns to the baseline's static
edge features, then calls the baseline's own training and evaluation code. Run it from the repository root:

    python -m src.osm.train --state DE --model graphsage --task classification --osm-groups road_character intersection_control \
        cycle_pedestrian_lighting nearby_places road_shape --run-name DE_graphsage_classification_n2v_osm_all
    python -m src.osm.train --state DE --check-only --osm-groups ...            # feature checks, no training
    python -m src.osm.train ... --permutation-importance                       # needs the trained run
    python -m src.osm.train --state DE --check-reuse RUN [RUN ...] --osm-groups  # are saved runs still valid for the renamed table?
    python -m src.osm.train --state DE --traffic-check --osm-groups              # does the traffic (AADT) input vary between roads?
    python -m src.osm.train --state DE --overlap-check --osm-groups ...          # how much of Node2Vec do these features describe?

The features come from `{STATE}_osm_features.parquet` (built by notebooks/osm_features_and_eda.ipynb, Stage 13) in the data folder.
Groups: road_character (speed limit, lanes, surface, bridge, tunnel as OSM has them), road_character_filled (the same three
tags with validated estimates for missing values; use it INSTEAD of road_character), intersection_control, cycle_pedestrian_lighting,
nearby_places and road_shape (curviness).

`--osm-groups` is required. With no value (`--osm-groups` alone) it is the control run: the inputs are exactly the
baseline's. All other options are the baseline's (see `python -m src.model.train --help`).
"""
import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.osm._baseline import RoadAccidentDataset
from src.osm._baseline import dataset as baseline_dataset
from src.osm._baseline.node2vec import generate_node2vec
from src.osm._baseline import train as baseline_train

from src.osm.preprocessing import osm_table_path, osm_group_columns, build_osm_features

BASELINE_COMMIT = "299962d"  # the baseline code (aaronteork/CS5284_GML, main) this notebook was built and tested against
OSM_GROUPS = ("road_character", "road_character_filled", "intersection_control", "cycle_pedestrian_lighting", "nearby_places", "road_shape")
FEATURE_GROUP_PREFIXES = ("road_character", "intersection_control", "cycle_pedestrian_lighting", "nearby_places", "road_shape")
TAG_BASED_GROUPS = ("road_character", "road_character_filled", "intersection_control", "cycle_pedestrian_lighting")  # unknown for unmatched roads
SURFACE_COLUMNS = ("road_character_surface", "road_character_surface_filled")
# Group names before the rename (osm_pipeline_v7). Used only to check that runs trained on the old table can be reused.
OLD_GROUP_NAMES = {"g1_filled": "road_character_filled", "g1": "road_character", "g2": "intersection_control",
                   "g3": "cycle_pedestrian_lighting", "g4": "nearby_places", "g5": "road_shape"}
MIN_CATEGORY_ROADS = 50          # surface types with fewer roads than this are grouped as "other"
MISSING_FLAG_THRESHOLD = 0.005   # add a "missing" indicator when more than this share of matched roads lack the value
SINUOSITY_RANGE = (1.0, 5.0)     # sinuosity below 1 is impossible and a few roads reach 180; clip both ends
LANES_MAX = 8                    # a few roads carry lane counts near 20 (mostly tagging errors); clip at 8
STANDARDISED_CLIP = 10.0         # sparse counts give a handful of roads values of 20 to 45 standard deviations; clip standardised values at +/- this








class OsmRoadAccidentDataset(RoadAccidentDataset):
    """The baseline dataset with OpenStreetMap columns appended to the static edge features.

    Everything else (node features, traffic, labels, splits, caches) is the baseline's own code. The OSM columns come
    right after the baseline's road flags and log length, before the traffic columns. With `osm_groups=()` the result
    is identical to `RoadAccidentDataset`.
    """

    def __init__(self, *args, osm_groups=(), osm_path=None, **kwargs):
        state, data_dir = args[0], kwargs.get("data_dir", baseline_dataset.DATA_DIR)
        if kwargs.get("download") is False and not (Path(data_dir) / state / "adj_matrix.pt").exists():
            raise FileNotFoundError(f"{Path(data_dir) / state} is not extracted. Run the setup cell that extracts the dataset first.")
        self.osm_groups = tuple(osm_groups)
        self._osm_path = osm_path
        self.osm_feature_names, self.osm_feature_groups = [], []
        super().__init__(*args, **kwargs)

    def _load_static_edge_attr(self):
        base = super()._load_static_edge_attr()
        if not self.osm_groups:
            return base
        path = Path(self._osm_path) if self._osm_path else osm_table_path(self.state, self.dir.parent)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Build it with notebooks/osm_features_and_eda.ipynb (Stage 13); it is saved in the Drive data folder.")
        table = pd.read_parquet(path)
        src, dst = self.edge_index.numpy()
        features, self.osm_feature_names, self.osm_feature_groups = build_osm_features(
            table, osm_group_columns(table, self.state, path.parent), self.osm_groups, src, dst)
        return torch.cat([base, torch.from_numpy(features)], dim=1)


def osm_provenance(state, groups, dataset):
    """What the run recorded about its OSM inputs (goes into run_metadata.json and metrics.json)."""
    table_path = osm_table_path(state)
    record = {"groups": list(groups), "num_features": len(dataset.osm_feature_names), "feature_names": dataset.osm_feature_names,
              "table_file": table_path.name, "baseline_commit": BASELINE_COMMIT}
    for key, meta_name in (("snapshot_date", f"{state}_osm_matching_meta.json"), ("pipeline_version", f"{state}_osm_features_meta.json")):
        meta_path = table_path.with_name(meta_name)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            record[key] = meta.get("osm_snapshot_date" if key == "snapshot_date" else "pipeline_version")
    return record


def install(groups):
    """Point the baseline's training code at the OSM-aware dataset (module-level swap; no files are edited)."""
    holder = {}

    def load_dataset(cfg):
        embedding_path = generate_node2vec(cfg.state, device=cfg.device) if cfg.use_node2vec else None
        holder["dataset"] = OsmRoadAccidentDataset(
            cfg.state, train_years=baseline_train.SPLITS[cfg.state][0],
            node_embedding_path=embedding_path, osm_groups=groups, download=False)  # never run the baseline's downloader (it can delete zips)
        return holder["dataset"]

    original_provenance = baseline_train.run_provenance

    def run_provenance(cfg):
        record = original_provenance(cfg)
        record["osm"] = osm_provenance(cfg.state, groups, holder["dataset"])
        return record

    baseline_train.load_dataset = load_dataset
    baseline_train.run_provenance = run_provenance


def check(state, groups):
    """Feature checks without training: row alignment, finite values, and a summary of every OSM column."""
    adj = torch.load(Path(baseline_dataset.DATA_DIR) / state / "adj_matrix.pt", map_location="cpu", weights_only=False).coalesce()
    src, dst = adj.indices().numpy()
    path = osm_table_path(state)
    table = pd.read_parquet(path)
    group_columns = osm_group_columns(table, state)
    features, names, owners = build_osm_features(table, group_columns, groups, src, dst)
    print(f"{state}: {len(src):,} directed edges; OSM table {path.name} has {len(table):,} rows (row order matches the graph)")
    print(f"groups available in the table: { {g: len(c) for g, c in group_columns.items()} }")
    print(f"groups used: {list(groups) or 'none (control)'}: {features.shape[1]} OSM columns added to the baseline's "
          f"{len(baseline_dataset.ROAD_FLAG_KEYS) + 1} static road features")
    if not names:
        return
    print("columns added per group:", {g: owners.count(g) for g in dict.fromkeys(owners)})
    summary = pd.DataFrame({"group": owners, "mean": features.mean(0), "std": features.std(0), "min": features.min(0),
                            "max": features.max(0), "share_nonzero": (features != 0).mean(0)}, index=names)
    pd.set_option("display.width", 200, "display.max_rows", 300)
    print(summary.round(3).to_string())


class _PermutedEdgeAttr:
    """A view of the dataset whose edge features are shuffled across roads in the chosen columns (same shuffle every month)."""

    def __init__(self, dataset, columns, permutation):
        self._dataset, self._columns, self._permutation = dataset, columns, permutation

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def monthly(self, year, month):
        data = self._dataset.monthly(year, month)
        if not self._columns:
            return data
        edge_attr = data.edge_attr.clone()
        edge_attr[:, self._columns] = data.edge_attr[self._permutation][:, self._columns]
        return dataclasses.replace(data, edge_attr=edge_attr)


def permutation_importance(cfg, groups, repeats=3):
    """How much does a trained model rely on each feature group? Shuffle the group across roads and re-score the test months.

    Why: it shows which inputs the model actually uses, without training new models. A group the model ignores can be
    shuffled with no change in the metric; a group it relies on makes the metric worse (AUROC lower, MAE higher).

    Takes: the run's config (the run must already be trained: its checkpoints are read from weights/<run_name>/), and the
    OSM groups it was trained with. Each group is shuffled `repeats` times per seed with different shuffles; the results
    are averaged. Scored with the baseline's own `evaluate` on the test months, "all roads" protocol.

    Sets shuffled: the baseline's road flags and length, traffic (AADT), each OSM group, and all OSM columns together.
    Saves results/<run_name>/permutation_importance.json and returns the same dictionary.
    """
    from src.osm._baseline import evaluate
    from src.osm._baseline.models import build_model

    embedding_path = generate_node2vec(cfg.state, device=cfg.device) if cfg.use_node2vec else None
    train_years, _, test_years = baseline_train.SPLITS[cfg.state]
    dataset = OsmRoadAccidentDataset(cfg.state, train_years=train_years, node_embedding_path=embedding_path, osm_groups=groups, download=False)
    test_months = dataset.months(test_years)
    sample = dataset.monthly(*dataset.months(train_years)[0])
    width, num_edges = sample.edge_attr.shape[1], sample.edge_attr.shape[0]
    base_columns = len(baseline_dataset.ROAD_FLAG_KEYS) + 1
    owners = dataset.osm_feature_groups
    column_sets = {"baseline road flags and length": list(range(base_columns)), "traffic (AADT)": [width - 2, width - 1]}
    for group in groups:
        column_sets[f"OSM {group}"] = [base_columns + i for i, owner in enumerate(owners) if owner == group]
    if owners:
        column_sets["all OSM columns"] = list(range(base_columns, base_columns + len(owners)))
    column_sets = {name: columns for name, columns in column_sets.items() if columns}

    results = {"none": {}, **{name: {} for name in column_sets}}
    for seed in cfg.seeds:
        checkpoint = torch.load(baseline_train.WEIGHTS_DIR / cfg.run_name / f"seed{seed}.pt", map_location=cfg.device, weights_only=False)
        model = build_model(cfg.model, sample.x.size(1), width, cfg.hidden_channels, cfg.num_layers, cfg.dropout).to(cfg.device)
        model.load_state_dict(checkpoint["state_dict"])
        results["none"][str(seed)] = evaluate(model, dataset, test_months, cfg.task, cfg.device)["all"]
        for name, columns in column_sets.items():
            scored = []
            for repeat in range(repeats):
                permutation = torch.from_numpy(np.random.default_rng(1000 * seed + repeat).permutation(num_edges)).long()
                scored.append(evaluate(model, _PermutedEdgeAttr(dataset, columns, permutation), test_months, cfg.task, cfg.device)["all"])
            results[name][str(seed)] = {metric: float(np.mean([s[metric] for s in scored])) for metric in scored[0]}
        print(f"seed {seed}: " + ", ".join(f"{name} {results[name][str(seed)]['AUROC' if cfg.task == 'classification' else 'MAE']:.4f}"
                                           for name in ["none", *column_sets]), flush=True)
    output = {"run": cfg.run_name, "task": cfg.task, "groups": list(groups), "repeats": repeats,
              "pipeline_version": osm_provenance(cfg.state, groups, dataset).get("pipeline_version"),
              "column_sets": {name: len(columns) for name, columns in column_sets.items()}, "scores": results}
    path = baseline_train.RESULTS_DIR / cfg.run_name / "permutation_importance.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def same_settings(saved, current):
    """True if a saved seed was trained with the same settings as the current run (seed list and device may differ)."""
    ignore = {"seeds", "device"}
    return {k: v for k, v in saved.items() if k not in ignore} == {k: v for k, v in current.items() if k not in ignore}


def run_with_resume(cfg, fresh=False):
    """The baseline's `run`, except that a seed already saved with the same settings and the same OSM inputs is loaded instead of retrained.

    Why: a free Colab session can stop mid-run. The baseline saves each seed as it finishes, but its loop always restarts at
    the first seed. Here a seed counts as finished when its metrics file, weights and test predictions all exist, its saved
    settings match, and it was trained on the same OSM table, version and groups. With `fresh=True` every seed is retrained.
    Everything else (data, training, evaluation, the files written) is the baseline's own code and format.
    Prints one line per seed. Returns the summary over seeds.
    """
    from dataclasses import asdict
    dataset = baseline_train.load_dataset(cfg)
    data_summary = {"num_nodes": dataset.num_nodes, "num_roads": dataset.num_roads, "target_unit": dataset.target_unit,
                    "months": {split: len(dataset.months(years)) for split, years in zip(("train", "valid", "test"), baseline_train.SPLITS[cfg.state])}}
    print(f"Data coverage: {data_summary}", flush=True)
    results_dir = baseline_train.RESULTS_DIR / cfg.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    provenance = baseline_train.run_provenance(cfg)
    with open(results_dir / "run_metadata.json", "w", encoding="utf-8") as stream:
        json.dump({"config": asdict(cfg), "data": data_summary, "provenance": provenance}, stream, indent=2)
    key = "AUROC" if cfg.task == "classification" else "MAE"
    runs = []
    for seed in cfg.seeds:
        saved_path = results_dir / f"seed{seed}_metrics.json"
        finished = (saved_path.exists() and (baseline_train.WEIGHTS_DIR / cfg.run_name / f"seed{seed}.pt").exists()
                    and (results_dir / f"seed{seed}_test_predictions.pt").exists())
        saved = json.loads(saved_path.read_text(encoding="utf-8")) if finished else None
        same_inputs = saved is not None and {k: (saved.get("provenance") or {}).get("osm", {}).get(k) for k in ("table_file", "pipeline_version", "groups")} \
            == {k: provenance["osm"].get(k) for k in ("table_file", "pipeline_version", "groups")}
        if not fresh and saved is not None and same_settings(saved["config"], asdict(cfg)) and same_inputs:
            result = {k: saved[k] for k in ("seed", "best_epoch", "valid", "test")}
            status = "reused (already trained)"
        else:
            result = baseline_train.train_seed(cfg, seed, dataset)
            with open(saved_path, "w") as stream:
                json.dump({"config": asdict(cfg), "data": data_summary, "provenance": provenance, **result}, stream, indent=2)
            status = "trained"
        runs.append(result)
        with open(results_dir / "metrics.json", "w") as stream:
            json.dump({"config": asdict(cfg), "data": data_summary, "summary": baseline_train.summarise(runs), "runs": runs,
                       "provenance": provenance, "complete": len(runs) == len(cfg.seeds)}, stream, indent=2)
        print(f"[seed {seed}] {status}: best epoch {result['best_epoch']}, valid {key} {result['valid'][cfg.select_protocol][key]:.4f}, "
              f"test {key} {result['test']['all'][key]:.4f}", flush=True)
    summary = baseline_train.summarise(runs)
    print(f"\n{cfg.run_name}: test results over seeds {cfg.seeds}")
    for protocol, metrics in summary["test"].items():
        print(f"  {protocol}: " + ", ".join(f"{k} {v['mean']:.4f} ± {v['std']:.4f}" for k, v in metrics.items()))
    return summary


def renamed_from_old(name):
    """A column, feature or group name from before the rename (g1_..., g4_...) in today's names."""
    for old_name in ("g1_filled", "g1", "g2", "g3", "g4", "g5"):
        if name == old_name or name.startswith(old_name + "_") or name.startswith(old_name + "="):
            return OLD_GROUP_NAMES[old_name] + name[len(old_name):]
    return name


def check_reuse(state, run_names):
    """Can runs trained on the old feature table (old group names) be reused with the renamed table?

    Why: the three Node2Vec runs were trained before the rename. Their saved weights expect the same inputs in the same order.
    Checks: (1) after renaming, the old and new tables have the same columns in the same order; (2) the same values, where a
    text column counts as equal if its values agree once missing markers (None / NaN) and storage type are ignored; (3) the same
    feature groups; (4) for each run, the model inputs built from the new table equal, element for element, the ones built from
    the old table, and their columns equal the ones recorded when the run was trained. Check 4 decides whether the saved weights
    are valid. Writes results/reuse_check_<state>.json. Returns True only if all pass.
    """
    data_dir = Path(baseline_dataset.DATA_DIR)
    old_path, old_meta = data_dir / f"{state}_osm_model_features.parquet", data_dir / f"{state}_osm_model_features_meta.json"
    new_path, new_meta = osm_table_path(state), data_dir / f"{state}_osm_features_meta.json"
    report = {"old_table": old_path.name, "new_table": new_path.name, "runs": {}}
    if not old_path.exists():
        report["result"] = "old table not found: cannot verify, so the runs must be retrained"
        print(report["result"])
        Path("results", f"reuse_check_{state}.json").write_text(json.dumps(report, indent=2))
        return False
    old, new = pd.read_parquet(old_path), pd.read_parquet(new_path)
    old.columns = [renamed_from_old(c) for c in old.columns]
    storage = {c: f"{old[c].dtype} vs {new[c].dtype}" for c in new.columns if c in old.columns and old[c].dtype != new[c].dtype}

    def plain(table):
        """Text columns as plain Python strings with None for missing, whatever pandas type they were saved with."""
        table = table.copy()
        for column in table.columns:
            if not (pd.api.types.is_bool_dtype(table[column]) or pd.api.types.is_numeric_dtype(table[column])):
                table[column] = pd.Series([None if pd.isna(v) else str(v) for v in table[column].astype(object)], index=table.index, dtype=object)
        return table

    old, new = plain(old), plain(new)
    report["same_columns_in_order"] = list(old.columns) == list(new.columns)
    def as_values(series):
        """Values for comparison: numbers as floats, text as plain strings, every missing marker as None."""
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            return series.astype(float).to_numpy()
        return np.array([None if pd.isna(value) else str(value) for value in series.astype(object)], dtype=object)

    def same_values(a, b):
        if len(a) != len(b):
            return False
        if a.dtype == object or b.dtype == object:
            return all(x == y for x, y in zip(a, b))
        return bool(np.array_equal(a, b, equal_nan=True))

    differing, storage_only = [], []
    if report["same_columns_in_order"]:
        for column in new.columns:
            if not same_values(as_values(old[column]), as_values(new[column])):
                differing.append(column)
            elif column in storage:
                storage_only.append(f"{column} ({storage[column]})")
    report["columns_with_different_values"] = differing
    report["columns_differing_only_in_storage"] = storage_only
    old_groups = {renamed_from_old(g): [renamed_from_old(c) for c in cols] for g, cols in json.loads(old_meta.read_text())["feature_groups_for_ablation"].items()}
    new_groups = json.loads(new_meta.read_text())["feature_groups_for_ablation"]
    report["same_groups"] = old_groups == new_groups
    adj = torch.load(data_dir / state / "adj_matrix.pt", map_location="cpu", weights_only=False).coalesce()
    src, dst = adj.indices().numpy()
    all_ok = report["same_columns_in_order"] and not differing and report["same_groups"]
    for name in run_names:
        metrics_path = Path("results") / name / "metrics.json"
        if not metrics_path.exists():
            report["runs"][name] = "not trained yet"
            continue
        osm = json.loads(metrics_path.read_text())["provenance"].get("osm") or {}
        groups = [renamed_from_old(g) for g in osm.get("groups", [])]
        new_inputs, names, _ = build_osm_features(new, osm_group_columns(new, state), groups, src, dst)
        old_inputs, old_names, _ = build_osm_features(old, {g: [c for c in cols if c in old.columns] for g, cols in old_groups.items()}, groups, src, dst)
        recorded = [renamed_from_old(n) for n in osm.get("feature_names", [])]
        same_inputs = new_inputs.shape == old_inputs.shape and bool(np.array_equal(new_inputs, old_inputs))
        report["runs"][name] = {"groups": groups, "columns": len(names), "same_input_columns_in_order": names == recorded == old_names,
                                "same_model_inputs": same_inputs}
        all_ok = all_ok and names == recorded == old_names and same_inputs
    report["result"] = "PASS: the saved runs see exactly the same inputs" if all_ok else "FAIL: retrain these runs (set FORCE_RERUN = True)"
    Path("results", f"reuse_check_{state}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return all_ok


def traffic_check(state):
    """Does the traffic (AADT) input the model sees ever differ between roads? Explains a zero in the permutation test.

    For the first available month of every year: the share of roads with a reported AADT value, and the spread (standard
    deviation) and number of distinct values of the two traffic columns of the edge features (the last two, as the baseline builds them).
    If a column is the same for every road in the test years, shuffling it across roads cannot change a prediction.
    Writes results/traffic_check_<state>.json and prints the table.
    """
    train_years, valid_years, test_years = baseline_train.SPLITS[state]
    dataset = OsmRoadAccidentDataset(state, train_years=train_years, node_embedding_path=None, osm_groups=(), download=False)
    rows = []
    for split, years in (("train", train_years), ("valid", valid_years), ("test", test_years)):
        for year in years:
            months = dataset.months([year])
            if not months:
                continue
            edge_attr = dataset.monthly(*months[0]).edge_attr
            traffic, missing = edge_attr[:, -2].numpy(), edge_attr[:, -1].numpy()
            rows.append({"split": split, "year": int(year), "share_of_roads_with_AADT": float(1 - missing.mean()),
                         "traffic_column_std": float(traffic.std()), "traffic_distinct_values": int(len(np.unique(traffic))),
                         "missing_flag_distinct_values": int(len(np.unique(missing)))})
    table = pd.DataFrame(rows)
    Path("results", f"traffic_check_{state}.json").write_text(table.to_json(orient="records", indent=2))
    print(table.round(4).to_string(index=False))
    return table


def overlap_check(state, groups, components=10, folds=5, alpha=1.0, seed=0):
    """How much of what Node2Vec knows about a road can be predicted from its road features?

    Definition. A road's embedding is the average of its two end intersections' Node2Vec vectors (128 numbers), one row per
    physical road. Its top `components` principal directions are the target. Three predictor sets are compared: (a) the baseline's
    road flags and log length, (b) the OSM features (as the model sees them), (c) both. Model: ridge regression. Score: R^2 on
    held-out roads, with folds made of spatial blocks (a 10 x 10 grid over the state), because neighbouring roads share intersections
    and random folds would inflate the score. The overall R^2 weights each component by the variance it explains.
    Reading: a high R^2 for (b), and a clear rise from (a) to (c), means the OSM features describe much of what Node2Vec encodes.
    Writes results/overlap_check_<state>.json and returns the same dictionary.
    """
    embedding_path = generate_node2vec(state)
    train_years = baseline_train.SPLITS[state][0]
    dataset = OsmRoadAccidentDataset(state, train_years=train_years, node_embedding_path=embedding_path, osm_groups=groups, download=False)
    src, dst = dataset.edge_index.numpy()
    base_columns = len(baseline_dataset.ROAD_FLAG_KEYS) + 1
    edge = dataset.static_edge_attr.numpy().astype(np.float64)
    nodes = int(max(src.max(), dst.max())) + 1
    _, first = np.unique(np.minimum(src, dst).astype(np.int64) * nodes + np.maximum(src, dst), return_index=True)
    embeddings = dataset.node_embeddings.numpy().astype(np.float64)
    target = ((embeddings[src] + embeddings[dst]) / 2)[first]
    target = target - target.mean(axis=0)
    _, singular, vt = np.linalg.svd(target, full_matrices=False)
    share = singular ** 2 / (singular ** 2).sum()
    target = target @ vt[:components].T
    weights = share[:components] / share[:components].sum()

    coords = dataset.static_node_features.numpy()[:, :2]
    middle = ((coords[src] + coords[dst]) / 2)[first]
    cells = [np.clip(np.searchsorted(np.quantile(middle[:, k], np.linspace(0, 1, 11)[1:-1]), middle[:, k]), 0, 9) for k in (0, 1)]
    block = cells[0] * 10 + cells[1]
    fold_of_block = np.random.default_rng(seed).permutation(100) % folds
    fold = fold_of_block[block]

    def held_out_r2(x):
        error, total = np.zeros(components), np.zeros(components)
        for k in range(folds):
            train, test = fold != k, fold == k
            mean, std = x[train].mean(axis=0), x[train].std(axis=0)
            std[std == 0] = 1.0
            design = lambda rows: np.column_stack([np.ones(rows.sum()), (x[rows] - mean) / std])
            a, b = design(train), design(test)
            penalty = alpha * np.eye(a.shape[1]); penalty[0, 0] = 0.0
            beta = np.linalg.solve(a.T @ a + penalty, a.T @ target[train])
            error += ((target[test] - b @ beta) ** 2).sum(axis=0)
            total += ((target[test] - target[train].mean(axis=0)) ** 2).sum(axis=0)
        by_component = 1 - error / total
        return {"weighted": float((weights * by_component).sum()), "by_component": [float(v) for v in by_component]}

    sets = {"baseline road features": edge[first][:, :base_columns], "OSM features": edge[first][:, base_columns:],
            "both": edge[first]}
    result = {"state": state, "groups": list(groups), "physical_roads": int(len(first)), "components": components,
              "variance_explained_by_components": float(share[:components].sum()),
              "r2": {name: held_out_r2(x) for name, x in sets.items() if x.shape[1] > 0}}
    Path("results", f"overlap_check_{state}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "r2"}, indent=2))
    for name, values in result["r2"].items():
        print(f"{name}: weighted held-out R^2 {values['weighted']:.3f}")
    return result


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--osm-groups", nargs="*", default=None)
    pre.add_argument("--check-only", action="store_true")
    pre.add_argument("--permutation-importance", action="store_true")
    pre.add_argument("--check-reuse", nargs="+", default=None)
    pre.add_argument("--traffic-check", action="store_true")
    pre.add_argument("--overlap-check", action="store_true")
    pre.add_argument("--fresh", action="store_true")
    known, rest = pre.parse_known_args()
    if known.osm_groups is None:
        raise SystemExit("--osm-groups is required (give it no value for the control run, which uses only the baseline's features)")
    groups = tuple(known.osm_groups)
    sys.argv = [sys.argv[0], *rest]
    cfg = baseline_train.parse_args()
    if known.check_only:
        check(cfg.state, groups)
        return
    if known.check_reuse:
        if not check_reuse(cfg.state, known.check_reuse):
            raise SystemExit(1)
        return
    if known.traffic_check:
        traffic_check(cfg.state)
        return
    if known.overlap_check:
        overlap_check(cfg.state, groups)
        return
    if known.permutation_importance:
        permutation_importance(cfg, groups)
        return
    install(groups)
    run_with_resume(cfg, fresh=known.fresh)


if __name__ == "__main__":
    main()
