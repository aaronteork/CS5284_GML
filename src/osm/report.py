"""Report tables/figures from traceable saved OSM notebook outputs.

The exported evidence contains rounded displays, not raw prediction tensors.
Loading it never executes either source notebook or triggers training. Use
src.osm.comparison for fresh analysis when the original result files are present.
"""

import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_evidence(project_dir):
    """Read the bundled evidence and reject stale snapshots of present sources."""
    root = Path(project_dir)
    evidence = json.loads((root / "reports" / "osm_notebook_evidence.json").read_text(encoding="utf-8"))
    for name, digest in evidence["sources"].items():
        source = root / name
        if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name} changed since evidence extraction; review and rerun scripts/extract_osm_modules.py.")
    return evidence


def saved_table(evidence, name):
    """Read a static HTML table; retain source row names as the DataFrame index."""
    rows = evidence["tables"][name]["rows"]
    header = rows[0]
    # pandas adds a second header row for a named index (the 2x2 table).
    body = [row for row in rows[1:] if any(value for value in row[1:])]
    if any(len(row) != len(header) for row in body):
        raise ValueError(f"Unexpected table shape for {name}")
    table = pd.DataFrame([row[1:] for row in body], index=[row[0] for row in body], columns=header[1:])
    for column in table:
        series = table[column]
        if set(series.dropna().unique()) <= {"True", "False"}:
            table[column] = series.map({"True": True, "False": False})
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().all():
                table[column] = numeric
    return table


def _markdown_table(text):
    rows = [line.strip().strip("|").split("|") for line in text.splitlines() if line.strip().startswith("|")]
    rows = [[value.strip() for value in row] for row in rows]
    if len(rows) < 3:
        raise ValueError("No saved Markdown table")
    return pd.DataFrame(rows[2:], columns=rows[0])


def feature_coverage(evidence):
    """Directed-row raw-tag coverage; accident-weighted shares have another denominator."""
    item = next(item for item in evidence["feature_markdown"]
                if item["cell"] == 64 and "weighted by accidents" in item["text"])
    table = _markdown_table(item["text"])
    table = table.rename(columns={"": "Feature", "all roads": "Directed roads (%)",
                                   "weighted by accidents": "Accident-weighted coverage (%)"})
    for column in ("Directed roads (%)", "Accident-weighted coverage (%)"):
        table[column] = 100 * pd.to_numeric(table[column])
    table["Feature"] = table.Feature.replace({"lit": "Lighting"}).str.capitalize()
    return table.set_index("Feature")[["Directed roads (%)", "Accident-weighted coverage (%)"]]


def feature_groups(evidence):
    """Five raw OSM groups and their column counts from the saved catalogue."""
    item = next(item for item in evidence["feature_markdown"]
                if item["cell"] == 64 and "By feature group" in item["text"])
    table = _markdown_table(item["text"]).rename(columns={"": "Group", "what": "Description", "columns": "Raw columns"})
    table["Raw columns"] = pd.to_numeric(table["Raw columns"])
    return table.set_index("Group")[["Description", "Raw columns"]]


def gap_validation(evidence):
    item = next(item for item in evidence["feature_markdown"] if item["cell"] == 71)
    table = _markdown_table(item["text"])
    return table.set_index("state / feature")[["decision", "neighbours + road type", "expected accuracy on the missing roads"]]


def score_summary(evidence):
    """Mean +/- population SD across model seeds, in source precision."""
    table = saved_table(evidence, "two_by_two")
    table.index = pd.Index(["Without", "With"], name="Node2Vec")
    table.columns = ["Baseline AUROC", "With OSM AUROC"]
    changes = saved_table(evidence, "changes")
    primary = changes[changes.metric == "AUROC"].set_index("node2vec")
    table["AUROC change"] = [primary.loc[False, "change"], primary.loc[True, "change"]]
    table["Seeds improved"] = [primary.loc[False, "seeds_improved"], primary.loc[True, "seeds_improved"]]
    return table


def metric_changes(evidence):
    table = saved_table(evidence, "changes").copy()
    table["node2vec"] = table.node2vec.map({False: "Without", True: "With"})
    table["metric"] = table.metric.replace({"PR-AUC": "AP"})
    return table.rename(columns={"node2vec": "Node2Vec", "metric": "Metric", "control": "Baseline",
                                  "variant_value": "With OSM", "change": "Change"})[
        ["Node2Vec", "Metric", "Baseline", "With OSM", "Change"]]


def permutation_summary(evidence):
    tables = []
    for n2v, key in (("With", "importance_n2v"), ("Without", "importance_no_n2v")):
        table = saved_table(evidence, key).copy()
        table["Node2Vec"] = n2v
        tables.append(table)
    return pd.concat(tables, ignore_index=True).rename(columns={"feature set": "Feature group", "worsening": "AUROC drop"})


def plot_osm_comparison(evidence):
    """Compact 2x2 AUROC plot and per-group permutation reliance."""
    scores = saved_table(evidence, "two_by_two")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for i, (column, label) in enumerate(zip(scores.columns, ("Baseline", "With OSM"))):
        pairs = [re.findall(r"[0-9.]+", value) for value in scores[column]]
        means, stds = np.array([[float(p[0]), float(p[1])] for p in pairs]).T
        axes[0].errorbar([0, 1], means, yerr=stds, marker="o", capsize=3, label=label)
    axes[0].set(xticks=[0, 1], xticklabels=["Without Node2Vec", "With Node2Vec"], ylabel="Test AUROC", ylim=(.83, .89))
    axes[0].legend()
    table = permutation_summary(evidence)
    groups = ["nearby_places", "road_character", "intersection_control", "road_shape", "cycle_pedestrian_lighting"]
    labels = ["Nearby places", "Road characteristics", "Intersection controls", "Road shape", "Cycle / pedestrian / lighting"]
    for i, (n2v, label) in enumerate((("Without", "Without Node2Vec"), ("With", "With Node2Vec"))):
        selected = table[table.Node2Vec == n2v].set_index("Feature group")
        values = [selected.loc[f"OSM {g}", "AUROC drop"] for g in groups]
        errors = [selected.loc[f"OSM {g}", "std over seeds"] for g in groups]
        axes[1].barh(np.arange(len(groups)) + (i - .5) * .35, values, .35, xerr=errors, label=label)
    axes[1].set(yticks=range(len(groups)), yticklabels=labels, xlabel="Test AUROC drop after group permutation")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="grey", linewidth=.7)
    axes[1].legend(fontsize=8)
    return fig
