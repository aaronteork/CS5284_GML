"""Saved-run comparisons extracted from the OSM notebook; no training on import."""
from src.reporting import apply_report_style as apply_chart_style

from src.osm.validation import validate_prediction_pairs

import json

import re

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

from sklearn.metrics import average_precision_score, roc_auc_score

PRIMARY_METRIC = {"classification": "AUROC", "regression": "MAE"}

SECONDARY_METRICS = {"classification": ["AUROC (pooled)", "PR-AUC", "precision@top1%", "recall@top1%"],
                     "regression": ["MSE", "MAE (accident roads)", "MAE (no-accident roads)"]}

HIGHER_IS_BETTER = {"AUROC": True, "AUROC (pooled)": True, "PR-AUC": True, "precision@top1%": True, "recall@top1%": True,
                    "MAE": False, "MSE": False, "MAE (accident roads)": False, "MAE (no-accident roads)": False}

BASELINE_LABELS = ("committed baseline", "baseline code")

SURFACE, INK, INK_SECONDARY, GRID_COLOR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

VARIANT_COLORS = {"none": "#2a78d6", "all": "#eb6834", "all_filled": "#1baf7a", "road_character_filled": "#1baf7a", "road_character": "#eda100",
                  "intersection_control": "#e87ba4", "cycle_pedestrian_lighting": "#008300", "nearby_places": "#6250d6", "road_shape": "#e34948"}

PAPER_AUROC = {("DE", "graphsage"): 0.876, ("DE", "mlp"): 0.755}

OLD_GROUP_NAMES = {"g1_filled": "road_character_filled", "g1": "road_character", "g2": "intersection_control", "g3": "cycle_pedestrian_lighting",
                   "g4": "nearby_places", "g5": "road_shape"}

NEUTRAL = "#8a8984"

def load_runs(results_dir, committed_dir=None):
    """Read every finished run's metrics into one table, one row per run.

    Why: runs made by this notebook, runs of the team's own training code, and the baseline results committed to the
    repository all end up as `metrics.json` files. This puts them side by side with a label saying which is which.

    Labels: runs made by osm_train.py are labelled by their OSM group set ("none" is the control run, "all" is all five groups,
    otherwise the groups joined by "-"); runs in `results_dir` made by the baseline's own script are "baseline code";
    runs read from `committed_dir` are "committed baseline".

    Takes: the results folder, and optionally a folder holding a copy of the committed baseline results.
    Returns: a DataFrame with the config, and `test_by_seed` / `valid_by_seed` (seed -> {"all": metrics, "balanced": metrics}).
    """
    rows = []

    def add(path, committed):
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        config = run["config"]
        if not run.get("complete") or "use_node2vec" not in config:
            return  # unfinished, or from an older version of the baseline
        osm = (run.get("provenance") or {}).get("osm")
        if committed:
            label = "committed baseline"
        elif osm is None:
            label = "baseline code"
        else:
            suffix = re.search(r"_osm_(.+)$", config["run_name"])
            label = suffix.group(1) if suffix else ("none" if not osm["groups"] else "-".join(osm["groups"]))
        rows.append({"run": config["run_name"], "label": label, "state": config["state"], "model": config["model"],
                     "task": config["task"], "n2v": config["use_node2vec"], "epochs": config["epochs"],
                     "num_osm_features": osm["num_features"] if osm else 0,
                     "test_by_seed": {r["seed"]: r["test"] for r in run["runs"]},
                     "valid_by_seed": {r["seed"]: r["valid"] for r in run["runs"]},
                     "best_epoch": {r["seed"]: r["best_epoch"] for r in run["runs"]}})

    for path in sorted(Path(results_dir).glob("*/metrics.json")):
        add(path, committed=False)
    if committed_dir is not None and Path(committed_dir).exists():
        for path in sorted(Path(committed_dir).glob("*/metrics.json")):
            add(path, committed=True)
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs

    # Two runs with the same setting and label would make a comparison ambiguous: keep the one with the most seeds.
    runs["n_seeds"] = runs["test_by_seed"].map(len)
    key = ["state", "model", "task", "n2v", "label"]
    kept = runs.sort_values("n_seeds", ascending=False, kind="stable").drop_duplicates(key)
    for _, dropped in runs.drop(kept.index).iterrows():
        print(f"note: ignoring {dropped['run']} (another {dropped['label']} run of the same setting has more or equal seeds)")
    return kept.sort_index().drop(columns="n_seeds").reset_index(drop=True)

def metric_by_seed(run, metric, protocol="all", split="test"):
    """One run's metric for each seed, as {seed: value}."""
    source = run["test_by_seed"] if split == "test" else run["valid_by_seed"]
    return {seed: values[protocol][metric] for seed, values in source.items() if metric in values[protocol]}

def summarise_runs(runs, protocol="all"):
    """One line per run: the primary test metric as mean and standard deviation over seeds (plus its validation score)."""
    rows = []
    for _, run in runs.iterrows():
        metric = PRIMARY_METRIC[run["task"]]
        test = np.array(list(metric_by_seed(run, metric, protocol).values()))
        valid = np.array(list(metric_by_seed(run, metric, protocol, split="valid").values()))
        rows.append({"state": run["state"], "model": run["model"], "task": run["task"], "node2vec": run["n2v"],
                     "variant": run["label"], "osm_columns": run["num_osm_features"], "metric": metric, "seeds": len(test),
                     "test": f"{test.mean():.4f} ± {test.std():.4f}", "validation": f"{valid.mean():.4f}",
                     "run": run["run"]})
    return pd.DataFrame(rows).sort_values(["state", "task", "model", "node2vec", "variant"]).reset_index(drop=True)

def compare_with_control(runs, protocol="all", secondary=True):
    """For every setting that has a control run, the change from adding OSM features, paired by seed.

    Why: the control run uses the same code, splits, seeds and embeddings, and differs only by having no OSM columns, so a
    difference between it and an OSM run is due to the OSM features (plus training noise, which the per-seed pairing shows).

    Returns: one row per (setting, variant, metric) with the control value, the variant value, the change (variant minus
    control, averaged over seeds), the change for each seed, and how many seeds improved. "Improved" means a higher AUROC /
    PR-AUC / precision / recall, or a lower MAE / MSE.
    """
    rows = []
    for (state, model, task, n2v), setting in runs.groupby(["state", "model", "task", "n2v"]):
        control = setting[setting["label"] == "none"]
        if control.empty:   # no control trained here: use the team's committed run of the same setting
            control = setting[setting["label"] == "committed baseline"]
        if control.empty:
            continue
        control = control.iloc[0]
        metrics = [PRIMARY_METRIC[task]] + (SECONDARY_METRICS[task] if secondary else [])
        for _, variant in setting[~setting["label"].isin(["none", *BASELINE_LABELS])].iterrows():
            for metric in metrics:
                a, b = metric_by_seed(control, metric, protocol), metric_by_seed(variant, metric, protocol)
                seeds = sorted(set(a) & set(b))
                if not seeds:
                    continue
                deltas = np.array([b[s] - a[s] for s in seeds])
                better = (deltas > 0) if HIGHER_IS_BETTER[metric] else (deltas < 0)
                rows.append({"state": state, "model": model, "task": task, "node2vec": n2v, "variant": variant["label"],
                             "control_source": "trained here" if control["label"] == "none" else "team's committed run",
                             "metric": metric, "control": np.mean([a[s] for s in seeds]), "variant_value": np.mean([b[s] for s in seeds]),
                             "change": deltas.mean(), "change_by_seed": ", ".join(f"{d:+.4f}" for d in deltas),
                             "seeds_improved": f"{int(better.sum())}/{len(seeds)}"})
    return pd.DataFrame(rows)

def reproduction_check(runs, protocol="all"):
    """Does the control run reproduce the team's committed baseline? (It should, up to training noise.)

    Compares the control run's primary test metric with the committed baseline run of the same setting. The committed runs
    came from an earlier checkout of the code, so small differences are expected; a large gap would mean the inputs differ.
    """
    rows = []
    for (state, model, task, n2v), setting in runs.groupby(["state", "model", "task", "n2v"]):
        control = setting[setting["label"] == "none"]
        committed = setting[setting["label"] == "committed baseline"]
        if control.empty or committed.empty:
            continue
        metric = PRIMARY_METRIC[task]
        a = np.array(list(metric_by_seed(control.iloc[0], metric, protocol).values()))
        b = np.array(list(metric_by_seed(committed.iloc[0], metric, protocol).values()))
        rows.append({"state": state, "model": model, "task": task, "node2vec": n2v, "metric": metric,
                     "control_here": f"{a.mean():.4f} ± {a.std():.4f}", "committed": f"{b.mean():.4f} ± {b.std():.4f}",
                     "difference": a.mean() - b.mean()})
    return pd.DataFrame(rows)

def plot_metric_comparison(runs, state, task, protocol="all"):
    """Primary test metric for each variant: a dot for the mean over seeds with a bar of one standard deviation, hollow dots for single seeds.

    One panel per model setting. The dashed line is the control run (no OSM features). Dots are used instead of bars because the
    differences are small, and a bar chart would need a zoomed axis that exaggerates them.
    """
    apply_chart_style()
    metric = PRIMARY_METRIC[task]
    subset = runs[(runs["state"] == state) & (runs["task"] == task) & ~runs["label"].isin(BASELINE_LABELS)]
    settings = sorted(subset.groupby(["model", "n2v"]).groups)
    if not settings:
        print(f"no finished runs for {state} {task}")
        return
    fig, axes = plt.subplots(1, len(settings), figsize=(5.0 * len(settings), 4.4), layout="constrained", squeeze=False)
    for ax, (model, n2v) in zip(axes[0], settings):
        group = subset[(subset["model"] == model) & (subset["n2v"] == n2v)]
        order = [l for l in ["none"] + sorted(l for l in group["label"].unique() if l != "none") if l in set(group["label"])]
        values = {l: np.array(list(metric_by_seed(group[group["label"] == l].iloc[0], metric, protocol).values())) for l in order}
        low, high = min(v.min() for v in values.values()), max(v.max() for v in values.values())
        pad = (high - low) * 0.25 + 1e-3
        for position, label in enumerate(order):
            v = values[label]
            ax.errorbar(position, v.mean(), yerr=v.std(), fmt="o", ms=10, color=VARIANT_COLORS.get(label, NEUTRAL), ecolor=INK,
                        elinewidth=1.4, capsize=4, zorder=3)
            ax.plot([position + 0.18] * len(v), v, "o", ms=5, mfc=SURFACE, mec=INK, mew=1.1, zorder=3)
            ax.text(position - 0.12, v.mean(), f"{v.mean():.3f}", ha="right", va="center", fontsize=10)
        if "none" in values:
            ax.axhline(values["none"].mean(), color=INK_SECONDARY, lw=1, ls="--", zorder=1)
        ax.set_xlim(-0.8, len(order) - 0.3)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["no OSM (control)" if l == "none" else l for l in order], rotation=20, ha="right")
        ax.set_ylabel(f"test {metric}" + (" (higher is better)" if HIGHER_IS_BETTER[metric] else " (lower is better)"))
        ax.set_title(f"{state} {model}, {'with' if n2v else 'without'} Node2Vec")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    return fig

def load_permutation_importance(results_dir, run):
    """Read results/<run>/permutation_importance.json, or None if it was not run."""
    path = Path(results_dir) / run / "permutation_importance.json"
    if not path.exists():
        return None
    importance = json.loads(path.read_text(encoding="utf-8"))
    # runs from before the group rename use "OSM g4" and so on; show today's names
    rename = lambda name: f"OSM {OLD_GROUP_NAMES[name[4:]]}" if name.startswith("OSM ") and name[4:] in OLD_GROUP_NAMES else name
    importance["scores"] = {rename(k): v for k, v in importance["scores"].items()}
    importance["column_sets"] = {rename(k): v for k, v in importance["column_sets"].items()}
    return importance

def permutation_table(importance):
    """Per feature set: how much worse the test metric gets when that set is shuffled across roads (averaged over seeds).

    For AUROC the drop is "unshuffled minus shuffled"; for MAE it is "shuffled minus unshuffled". Larger means the model
    relies on that set more; about zero means the model ignores it.
    """
    metric = PRIMARY_METRIC[importance["task"]]
    sign = 1 if HIGHER_IS_BETTER[metric] else -1
    base = {seed: v[metric] for seed, v in importance["scores"]["none"].items()}
    rows = []
    for name, per_seed in importance["scores"].items():
        if name == "none":
            continue
        drops = np.array([sign * (base[seed] - per_seed[seed][metric]) for seed in per_seed])
        rows.append({"feature set": name, "columns": importance["column_sets"][name], "worsening": drops.mean(), "std over seeds": drops.std(),
                     "unshuffled": np.mean(list(base.values())), "shuffled": np.mean([per_seed[s][metric] for s in per_seed])})
    return pd.DataFrame(rows).sort_values("worsening", ascending=False).reset_index(drop=True)

def plot_permutation_importance(importance, title):
    """Horizontal bars of how much the metric worsens when each feature set is shuffled (dots are single seeds)."""
    apply_chart_style()
    metric = PRIMARY_METRIC[importance["task"]]
    sign = 1 if HIGHER_IS_BETTER[metric] else -1
    base = {seed: v[metric] for seed, v in importance["scores"]["none"].items()}
    names = [n for n in importance["scores"] if n != "none"]
    per_seed = {n: np.array([sign * (base[s] - importance["scores"][n][s][metric]) for s in base]) for n in names}
    names = sorted(names, key=lambda n: per_seed[n].mean())
    colour = lambda n: VARIANT_COLORS.get(n.split()[-1], NEUTRAL) if n.startswith("OSM ") else (VARIANT_COLORS["all"] if n == "all OSM columns" else NEUTRAL)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(names) + 1.6), layout="constrained")
    for row, name in enumerate(names):
        ax.barh(row, per_seed[name].mean(), height=0.6, color=colour(name))
        ax.plot(per_seed[name], [row] * len(per_seed[name]), "o", ms=5, mfc=SURFACE, mec=INK, mew=1.2, zorder=3)
    ax.axvline(0, color=INK_SECONDARY, lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}  ({importance['column_sets'][n]} column{'s' if importance['column_sets'][n] != 1 else ''})" for n in names])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"worsening of test {metric} when the set is shuffled")
    ax.set_title(title)
    return fig

def load_predictions(results_dir, run, seeds):
    """The saved test predictions of a run, one dictionary per seed (months, pred [months, roads], y, road_index, ...)."""
    return [torch.load(Path(results_dir) / run / f"seed{s}_test_predictions.pt", map_location="cpu", weights_only=False) for s in seeds]

def coarse_road_type(names):
    """Group the dataset's 14 road types (motorway_link counts as motorway, and so on) into 7."""
    base = pd.Series(names).str.replace("_link", "", regex=False)
    return base.where(base.isin(["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]), "other").to_numpy()

def road_attributes(state, prediction, data_dir):
    """Per physical road (in prediction order): road type, length, OSM match status and a few OSM features.

    Each road is represented by one directed edge (`road_index`); its row in the saved OSM feature table gives the rest.
    """
    data_dir = Path(data_dir)
    src, dst = prediction["road_index"].numpy()
    n_nodes = torch.load(data_dir / state / "adj_matrix.pt", map_location="cpu", weights_only=False).shape[0]
    keys = src.astype(np.int64) * n_nodes + dst
    label = np.full(len(keys), "other", dtype=object)
    for name, matrix in torch.load(data_dir / state / "Edges" / "edge_features.pt", map_location="cpu", weights_only=False).items():
        if str(name) in ("oneway", "length"):
            continue
        matrix = matrix.coalesce()
        index, value = matrix.indices().numpy(), matrix.values().numpy()
        label[np.isin(keys, index[0].astype(np.int64)[value > 0] * n_nodes + index[1][value > 0])] = str(name)
    path = data_dir / f"{state}_osm_features.parquet"
    table = pd.read_parquet(path, columns=["node_1_idx", "node_2_idx", "length_m", "match_type", "road_character_maxspeed_mph", "nearby_places_signal_250m"])
    attributes = pd.DataFrame({"node_1_idx": src, "node_2_idx": dst}).merge(table, how="left", on=["node_1_idx", "node_2_idx"])
    if len(attributes) != len(src):
        raise ValueError("road table did not line up with the predictions")
    attributes["road_type"] = coarse_road_type(label)
    return attributes

def subgroup_masks(attributes):
    """Named groups of roads to compare models on (True/False arrays over the roads)."""
    masks = {f"road type: {t}": (attributes["road_type"] == t).to_numpy()
             for t in ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "other"]}
    length = attributes["length_m"]
    for name, low, high in [("under 100 m", 0, 100), ("100 to 300 m", 100, 300), ("300 m to 1 km", 300, 1000), ("over 1 km", 1000, np.inf)]:
        masks[f"length: {name}"] = ((length >= low) & (length < high)).to_numpy()
    matched = (attributes["match_type"] != "unmatched").to_numpy()
    masks["OSM match: matched"], masks["OSM match: not matched"] = matched, ~matched
    tagged = attributes["road_character_maxspeed_mph"].notna().to_numpy()
    masks["speed limit tagged in OSM"], masks["speed limit not tagged (matched roads)"] = tagged, matched & ~tagged
    signals = (attributes["nearby_places_signal_250m"] > 0).to_numpy()
    masks["signal within 250 m"], masks["no signal within 250 m"] = signals, ~signals
    return {name: mask for name, mask in masks.items() if mask.sum() > 0}

def subgroup_scores(predictions, mask, task):
    """Pooled test scores over all months for the roads in `mask`, averaged over seeds: AUROC and average precision, or MAE."""
    scores = []
    for p in predictions:
        pred, y = p["pred"][:, mask].numpy().ravel(), p["y"][:, mask].numpy().ravel()
        if task == "classification":
            positive = y > 0
            scores.append((roc_auc_score(positive, pred), average_precision_score(positive, pred)) if 0 < positive.sum() < len(positive) else (np.nan, np.nan))
        else:
            scores.append((np.abs(pred - y).mean(),))
    return np.nanmean(np.array(scores, dtype=float), axis=0)

def compare_subgroups(state, task, control_predictions, variant_predictions, data_dir):
    """Control and OSM-feature model scored on the same groups of roads (road type, length, OSM match, ...).

    Returns one row per group: how many roads, the share of road-months with an accident, the control's score, the variant's score
    and the change (variant minus control). For classification the score is the pooled AUROC and the average precision.
    """
    validate_prediction_pairs(control_predictions, variant_predictions)
    attributes = road_attributes(state, control_predictions[0], data_dir)
    y = control_predictions[0]["y"].numpy()
    rows = []
    for name, mask in subgroup_masks(attributes).items():
        a, b = subgroup_scores(control_predictions, mask, task), subgroup_scores(variant_predictions, mask, task)
        row = {"group": name, "roads": int(mask.sum()), "accident rate": float((y[:, mask] > 0).mean())}
        for k, label in enumerate(["AUROC", "AP"] if task == "classification" else ["MAE"]):
            row.update({f"{label} control": a[k], f"{label} OSM": b[k], f"{label} change": b[k] - a[k]})
        rows.append(row)
    return pd.DataFrame(rows)

def monthly_scores(predictions, task):
    """Per test month, the primary metric (AUROC, or MAE) averaged over seeds, plus the month's number of accident roads."""
    months = predictions[0]["months"]
    values = np.full((len(predictions), len(months)), np.nan)
    positives = np.array([(predictions[0]["y"][i] > 0).sum().item() for i in range(len(months))])
    for s, p in enumerate(predictions):
        for i in range(len(months)):
            y, pred = p["y"][i].numpy(), p["pred"][i].numpy()
            if task == "classification":
                values[s, i] = roc_auc_score(y > 0, pred) if 0 < (y > 0).sum() < len(y) else np.nan
            else:
                values[s, i] = np.abs(pred - y).mean()
    return pd.DataFrame({"year": [m[0] for m in months], "month": [m[1] for m in months], "positives": positives,
                         "score": np.nanmean(values, axis=0)})

def compare_months(control_predictions, variant_predictions, task, n_boot=2000, seed=0):
    """Is the change consistent across test months? The monthly change, a bootstrap interval, and the share of months improved.

    Overall change: the primary metric averaged over months (AUROC weighted by the month's accident roads, as the baseline does;
    MAE as the plain mean). The interval resamples whole months, so it reflects month-to-month variation (months close in time
    are alike, so it is somewhat too narrow). It does not cover variation between training runs; see the per-seed changes.
    """
    validate_prediction_pairs(control_predictions, variant_predictions)
    a, b = monthly_scores(control_predictions, task), monthly_scores(variant_predictions, task)
    table = a.rename(columns={"score": "control"}).assign(variant=b["score"].to_numpy())
    table["change"] = table["variant"] - table["control"]
    ok = table["change"].notna().to_numpy()
    weights = table["positives"].to_numpy(float) if task == "classification" else np.ones(len(table))
    table = table.assign(weight=weights)
    usable = table[ok]
    overall = np.average(usable["change"], weights=usable["weight"])
    rng = np.random.default_rng(seed)
    draws = [np.average(usable["change"].to_numpy()[i], weights=usable["weight"].to_numpy()[i])
             for i in (rng.integers(0, len(usable), len(usable)) for _ in range(n_boot))]
    improved = (usable["change"] > 0) if task == "classification" else (usable["change"] < 0)
    summary = {"overall change": overall, "interval low": float(np.percentile(draws, 2.5)), "interval high": float(np.percentile(draws, 97.5)),
               "months improved": f"{int(improved.sum())}/{len(usable)}"}
    return table, summary

def plot_monthly_change(table, summary, task, title):
    """Bars of the month-by-month change in the primary metric (variant minus control), with the overall change."""
    apply_chart_style()
    metric = PRIMARY_METRIC[task]
    labels = [f"{y}-{m:02d}" for y, m in zip(table["year"], table["month"])]
    fig, ax = plt.subplots(figsize=(max(7, 0.28 * len(table) + 2), 4), layout="constrained")
    ax.bar(range(len(table)), table["change"], color=VARIANT_COLORS["all"], width=0.7)
    ax.axhline(0, color=INK_SECONDARY, lw=1)
    ax.axhline(summary["overall change"], color=INK, lw=1.2, ls="--")
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel(f"change in {metric}" + (" (above 0 is better)" if HIGHER_IS_BETTER[metric] else " (below 0 is better)"))
    ax.set_title(title, pad=24)
    ax.text(0, 1.03, f"overall {summary['overall change']:+.4f} (95% interval {summary['interval low']:+.4f} to {summary['interval high']:+.4f}); "
                     f"{summary['months improved']} months improved", transform=ax.transAxes, fontsize=10, color=INK_SECONDARY, va="bottom")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    return fig

def per_seed_table(runs, protocol="all"):
    """One row per run and seed: best epoch (out of the run's epochs), validation and test primary metric, and test PR-AUC.

    Replaces the training logs: it shows that every seed trained, which validation checkpoint was selected, and how close the seeds are.
    """
    rows = []
    for _, run in runs[~runs["label"].isin(BASELINE_LABELS)].iterrows():
        metric = PRIMARY_METRIC[run["task"]]
        for seed in sorted(run["test_by_seed"]):
            rows.append({"model": run["model"], "node2vec": run["n2v"], "variant": run["label"], "seed": seed,
                         "best epoch": f"{run['best_epoch'][seed]} of {run['epochs']}", f"valid {metric}": run["valid_by_seed"][seed][protocol][metric],
                         f"test {metric}": run["test_by_seed"][seed][protocol][metric], "test PR-AUC": run["test_by_seed"][seed][protocol].get("PR-AUC", np.nan)})
    return pd.DataFrame(rows).sort_values(["model", "node2vec", "variant", "seed"]).reset_index(drop=True)

def two_by_two(runs, state, model, task, variant="all", protocol="all"):
    """Test scores with Node2Vec off and on, without and with the OSM features (mean and standard deviation over seeds).

    Returns (table, gains). `gains` holds the Node2Vec gain (control with minus control without), the OSM gain without Node2Vec,
    the OSM gain with Node2Vec, and the share of the OSM gain that disappears once Node2Vec is present.
    """
    metric = PRIMARY_METRIC[task]
    subset = runs[(runs["state"] == state) & (runs["model"] == model) & (runs["task"] == task)]

    def scores(n2v, label):
        match = subset[(subset["n2v"] == n2v) & (subset["label"] == label)]
        if not len(match) and label == "none":   # the team's committed run is the control where none was trained here
            match = subset[(subset["n2v"] == n2v) & (subset["label"] == "committed baseline")]
        return np.array(list(metric_by_seed(match.iloc[0], metric, protocol).values())) if len(match) else np.array([])

    rows, means = [], {}
    for n2v in (False, True):
        row = {"Node2Vec": "with" if n2v else "without"}
        for label, column in (("none", "control (no OSM)"), (variant, f"with OSM '{variant}'")):
            values = scores(n2v, label)
            means[(n2v, label)] = values.mean() if len(values) else np.nan
            row[column] = f"{values.mean():.4f} ± {values.std():.4f}" if len(values) else "not run"
        rows.append(row)
    gains = {"Node2Vec gain (control)": means[(True, "none")] - means[(False, "none")],
             "OSM gain without Node2Vec": means[(False, variant)] - means[(False, "none")],
             "OSM gain with Node2Vec": means[(True, variant)] - means[(True, "none")]}
    without, with_ = gains["OSM gain without Node2Vec"], gains["OSM gain with Node2Vec"]
    gains["share of the OSM gain lost once Node2Vec is present"] = (1 - with_ / without) if np.isfinite(without) and without > 0 else np.nan
    return pd.DataFrame(rows).set_index("Node2Vec"), gains

def plot_two_by_two(runs, state, model, task, variant="all", protocol="all"):
    """Dot plot of the 2 x 2: test score without and with Node2Vec, for the control and the OSM run (single seeds as small dots)."""
    apply_chart_style()
    metric = PRIMARY_METRIC[task]
    subset = runs[(runs["state"] == state) & (runs["model"] == model) & (runs["task"] == task)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    for offset, label in ((-0.08, "none"), (0.08, variant)):
        xs, ys = [], []
        for position, n2v in enumerate((False, True)):
            match = subset[(subset["n2v"] == n2v) & (subset["label"] == label)]
            if not len(match) and label == "none":
                match = subset[(subset["n2v"] == n2v) & (subset["label"] == "committed baseline")]
            if not len(match):
                continue
            values = np.array(list(metric_by_seed(match.iloc[0], metric, protocol).values()))
            ax.plot([position + offset] * len(values), values, "o", ms=4, mfc=SURFACE, mec=VARIANT_COLORS.get(label, NEUTRAL), zorder=3)
            xs.append(position + offset); ys.append(values.mean())
        ax.plot(xs, ys, "-o", ms=9, color=VARIANT_COLORS.get(label, NEUTRAL), label="control (no OSM)" if label == "none" else f"with OSM '{label}'")
    if (state, model) in PAPER_AUROC and task == "classification":
        ax.axhline(PAPER_AUROC[(state, model)], color=INK_SECONDARY, lw=1, ls=":", label="paper, with Node2Vec (for scale)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["without Node2Vec", "with Node2Vec"]); ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(f"test {metric}"); ax.yaxis.grid(True); ax.legend(loc="lower right")
    ax.set_title(f"{state} {model}: Node2Vec adds much more than OSM; does OSM add more without it?")
    return fig

def osm_fill_rates(state, data_dir="data"):
    """Share of directed roads with a value for a few key OSM columns, from the feature table's meta file."""
    meta = json.loads((Path(data_dir) / f"{state}_osm_features_meta.json").read_text())
    fill = meta.get("fill_rate", {})
    keys = {"speed limit": "road_character_maxspeed_mph", "lanes": "road_character_lanes", "surface": "road_character_surface",
            "lighting": "cycle_pedestrian_lighting_lit", "signal at the start node": "intersection_control_signal_start",
            "nearby places (signals within 250 m)": "nearby_places_signal_250m", "curviness": "road_shape_sinuosity"}
    return {label: fill.get(column, np.nan) for label, column in keys.items()}
