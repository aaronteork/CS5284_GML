"""Tables and figures for the report's OpenStreetMap section (§5.1).

Why: the report must render from small committed files, without the OSM data, the
prediction tensors or a GPU. Everything here reads `metrics.json`,
`permutation_importance.json` and `run_metadata.json` under `results/`, written by
`notebooks/osm_baseline_comparison.ipynb`. Nothing here trains, rebuilds features or reads the
OSM notebooks.

Takes: the repository folder. Returns: pandas tables and matplotlib figures.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# The four runs of the 2x2 comparison, keyed by (Node2Vec setting, OSM setting).
# The no-Node2Vec control is the team's committed baseline run, not retrained here.
COMPARISON_RUNS = {
    ("Without", "Baseline"): "DE_graphsage_classification_no_n2v",
    ("Without", "With OSM"): "osm/DE_graphsage_classification_no_n2v_osm_all",
    ("With", "Baseline"): "osm/DE_graphsage_classification_n2v_osm_none",
    ("With", "With OSM"): "osm/DE_graphsage_classification_n2v_osm_all",
}

# Feature-group permutation was run on the two models that use the OSM features.
PERMUTATION_RUNS = {
    "Without": "osm/DE_graphsage_classification_no_n2v_osm_all",
    "With": "osm/DE_graphsage_classification_n2v_osm_all",
}

ALL_OSM_GROUPS = ["road_character", "intersection_control", "cycle_pedestrian_lighting", "nearby_places", "road_shape"]

# Readable names for the permutation feature sets, in the order they are plotted.
PERMUTATION_LABELS = {
    "OSM nearby_places": "Nearby places",
    "OSM road_character": "Road character",
    "OSM intersection_control": "Intersection controls",
    "OSM road_shape": "Road shape",
    "OSM cycle_pedestrian_lighting": "Cycle / pedestrian / lighting",
    "all OSM columns": "All OSM columns",
    "baseline road flags and length": "Baseline road type and length",
    "traffic (AADT)": "Traffic (AADT)",
}

# Metric keys in metrics.json and the names the report uses for them.
REPORT_METRICS = {
    "AUROC": "AUROC",
    "PR-AUC": "AP",
    "precision@top1%": "Precision@top 1%",
    "recall@top1%": "Recall@top 1%",
}

BASELINE_COLOR, OSM_COLOR = "#9a9a94", "#2a78d6"
NODE2VEC_COLORS = {"Without": "#eb6834", "With": "#6b4fa3"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_osm_results(project_dir):
    """Read the OSM comparison runs and check that they form a fair 2x2.

    Why: a missing seed, a run from another state or task, or an OSM run with fewer
    groups would silently change the comparison. Each check raises with the run name.
    Takes: the repository folder. Returns a dict with `runs` {(node2vec, osm): metrics},
    and `permutation` {node2vec: permutation results}.
    """
    results_dir = Path(project_dir) / "results"
    runs = {}
    for (node2vec, osm), name in COMPARISON_RUNS.items():
        metrics = _read_json(results_dir / name / "metrics.json")
        metadata = _read_json(results_dir / name / "run_metadata.json")
        config = metrics["config"]

        # Same state, model, task and seeds in every cell of the 2x2.
        expected = {"state": "DE", "model": "graphsage", "task": "classification",
                    "use_node2vec": node2vec == "With", "seeds": [0, 1, 2]}
        wrong = {key: config.get(key) for key, value in expected.items() if config.get(key) != value}
        if wrong or not metrics.get("complete") or len(metrics["runs"]) != 3:
            raise ValueError(f"{name}: not a complete 3-seed DE GraphSAGE classification run as expected ({wrong})")

        # OSM runs must use all five groups; control runs must use none.
        osm_record = metadata["provenance"].get("osm") or {}
        groups = osm_record.get("groups", [])
        if (osm == "With OSM") != (groups == ALL_OSM_GROUPS):
            raise ValueError(f"{name}: OSM groups {groups} do not match the '{osm}' cell")
        runs[(node2vec, osm)] = metrics

    permutation = {node2vec: _read_json(results_dir / name / "permutation_importance.json")
                   for node2vec, name in PERMUTATION_RUNS.items()}
    return {"runs": runs, "permutation": permutation}


def _seed_values(metrics, metric):
    """Test-set value of one metric for each seed, in seed order (all roads, not the balanced subset)."""
    ordered = sorted(metrics["runs"], key=lambda run: run["seed"])
    return np.array([run["test"]["all"][metric] for run in ordered])


def _mean_sd(values):
    # Population SD (ddof=0) across the three seeds, as everywhere else in the report.
    return f"{values.mean():.4f} ± {values.std():.4f}"


def osm_comparison_table(results):
    """Table: baseline vs with-OSM test scores for both Node2Vec settings.

    Why: the main result of §5.1. Change is the difference of seed means; 'Seeds
    improved' pairs runs by seed number, which share the training seed but not
    necessarily the same pipeline commit (the no-Node2Vec control is the team's run).
    Takes: `load_osm_results` output. Returns a DataFrame indexed by (Node2Vec, metric).
    """
    rows = []
    for node2vec in ("Without", "With"):
        for key, label in REPORT_METRICS.items():
            baseline = _seed_values(results["runs"][(node2vec, "Baseline")], key)
            with_osm = _seed_values(results["runs"][(node2vec, "With OSM")], key)
            rows.append({
                "Node2Vec": node2vec, "Metric": label,
                "Baseline": _mean_sd(baseline), "With OSM": _mean_sd(with_osm),
                "Change": with_osm.mean() - baseline.mean(),
                "Seeds improved": f"{int((with_osm > baseline).sum())}/3",
            })
    return pd.DataFrame(rows).set_index(["Node2Vec", "Metric"])


def permutation_table(results):
    """Table: test AUROC drop when each feature set is shuffled, per Node2Vec setting.

    Why: shows which inputs the fitted model relies on. A drop is the unshuffled
    AUROC minus the shuffled AUROC, averaged over the three seeds' repeats; it measures
    reliance, not the gain from adding the group.
    Takes: `load_osm_results` output. Returns a DataFrame (feature set x Node2Vec).
    """
    columns = {}
    for node2vec, record in results["permutation"].items():
        scores = record["scores"]
        intact = np.array([scores["none"][seed]["AUROC"] for seed in sorted(scores["none"])])
        drops = {}
        for feature_set, label in PERMUTATION_LABELS.items():
            shuffled = np.array([scores[feature_set][seed]["AUROC"] for seed in sorted(scores[feature_set])])
            drops[label] = (intact - shuffled).mean()
        columns[f"{node2vec} Node2Vec"] = pd.Series(drops)
    table = pd.DataFrame(columns)
    table.index.name = "Shuffled feature set"
    return table


def plot_osm_comparison(results):
    """Figure: (left) the 2x2 test AUROC; (right) AUROC drop per shuffled OSM group.

    Why: one picture of the main result and of what the model leans on.
    Takes: `load_osm_results` output. Returns the matplotlib figure.
    """
    fig, (ax_scores, ax_drops) = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained",
                                              gridspec_kw={"width_ratios": [1, 1.35]})

    # Left: points with seed SD (not bars: the axis is zoomed, and bars would exaggerate the gaps).
    positions = np.arange(2)
    for offset, (osm, color) in zip((-0.1, 0.1), (("Baseline", BASELINE_COLOR), ("With OSM", OSM_COLOR))):
        values = [_seed_values(results["runs"][(node2vec, osm)], "AUROC") for node2vec in ("Without", "With")]
        means, sds = np.array([v.mean() for v in values]), np.array([v.std() for v in values])
        ax_scores.errorbar(positions + offset, means, yerr=sds, fmt="o", markersize=7, capsize=4,
                           color=color, label=osm)
        for x, mean in zip(positions + offset, means):
            ax_scores.annotate(f"{mean:.4f}", (x, mean), xytext=(9, 0), textcoords="offset points",
                               va="center", fontsize=8.5, color=color)
    ax_scores.set(xticks=positions, xticklabels=["Without Node2Vec", "With Node2Vec"], xlim=(-0.5, 1.6),
                  ylabel="Test AUROC (mean ± SD, 3 seeds)")
    ax_scores.legend(loc="lower right")
    ax_scores.grid(axis="y")

    # Right: horizontal bars of the permutation drop for each OSM group.
    groups = ["Nearby places", "Road character", "Intersection controls", "Road shape", "Cycle / pedestrian / lighting"]
    drops = permutation_table(results)
    rows = np.arange(len(groups))
    for offset, node2vec in zip((-0.2, 0.2), ("Without", "With")):
        ax_drops.barh(rows + offset, drops.loc[groups, f"{node2vec} Node2Vec"], 0.38,
                      color=NODE2VEC_COLORS[node2vec], label=f"{node2vec} Node2Vec")
    ax_drops.set(yticks=rows, yticklabels=groups, xlabel="Test AUROC drop when the group is shuffled")
    ax_drops.invert_yaxis()
    ax_drops.axvline(0, color="grey", linewidth=0.7)
    ax_drops.grid(axis="x")
    ax_drops.legend(loc="lower right")
    return fig


def plot_osm_pipeline():
    """Figure: from the OSM snapshot and the benchmark graph to the model inputs.

    Why: the reader sees the whole pipeline, its headline numbers, and where the OSM
    features enter the models (every model's road predictor, and the graph models'
    message passing) before reading the details. Needs no data;
    the numbers are the Delaware values reported in §5.1.
    Returns the matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7.2), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.2)
    ax.axis("off")
    fills = {"source": "#e8f0fb", "step": "#cde2fb", "model": "#e2f3ec"}
    arrow = dict(arrowstyle="-|>", color="#5f5e5a", lw=1.3, shrinkA=1, shrinkB=1)

    def box(x, y, width, title, detail, kind):
        """A rounded box centred on (x, y) with a bold title and a smaller detail line."""
        ax.add_patch(FancyBboxPatch((x - width / 2, y - 0.42), width, 0.84, fc=fills[kind], ec="#c3c2bc",
                                    lw=1.1, boxstyle="round,pad=0.02,rounding_size=0.12"))
        ax.text(x, y + 0.14, title, ha="center", va="center", fontsize=10, weight="bold", color="#1f1f1d")
        ax.text(x, y - 0.18, detail, ha="center", va="center", fontsize=8.6, color="#4a4945")

    def down(x, y_from, y_to):
        ax.annotate("", xy=(x, y_to), xytext=(x, y_from), arrowprops=arrow)

    # Two sources merge into the matching step.
    box(2.6, 9.55, 3.8, "OpenStreetMap snapshot", "Geofabrik extract, Delaware, 2026-10-18", "source")
    box(7.4, 9.55, 3.8, "Benchmark road graph", "49,023 intersections, 116,196 directed roads", "source")
    ax.annotate("", xy=(4.2, 8.52), xytext=(2.6, 9.13), arrowprops=arrow)
    ax.annotate("", xy=(5.8, 8.52), xytext=(7.4, 9.13), arrowprops=arrow)

    # The linear preparation steps.
    steps = [
        ("Match roads to OSM", "89.6% of directed roads matched (85.1% exact, 4.0% nearest)"),
        ("Clean recorded attributes", "free text converted to numbers; missing values = mean + indicator"),
        ("Test gap-filling", "speed limit passes, lanes and surface fail; estimates not used in the model"),
        ("Build and encode features", "58 features in 5 groups  →  70 encoded columns  x_OSM"),
    ]
    y = 8.1
    for i, (title, detail) in enumerate(steps):
        box(5, y, 7.2, title, detail, "step")
        down(5, y - 0.42, y - 0.83)
        y -= 1.25

    # The combined edge vector, then the two places it is used.
    box(5, y, 7.2, "Road input vector", "e_uv,t = [ e_base ‖ x_OSM ]", "model")
    ax.annotate("", xy=(2.8, y - 1.33), xytext=(4.2, y - 0.42), arrowprops=arrow)
    ax.annotate("", xy=(7.2, y - 1.33), xytext=(5.8, y - 0.42), arrowprops=arrow)
    box(2.8, y - 1.75, 4.0, "1. Message passing (graph models)", "GraphSAGE, GatedGCN, Graph Transformer", "model")
    box(7.2, y - 1.75, 4.0, "2. Road predictor (all models)", "s_uv,t = f( [ h_u ‖ h_v ‖ e_uv,t ] )", "model")
    ax.annotate("", xy=(5.2, y - 1.75), xytext=(4.8, y - 1.75), arrowprops=arrow)
    ax.text(5, y - 1.5, "h_u, h_v", ha="center", fontsize=8.6, color="#4a4945")

    ax.text(5, 0.25, "Accident counts are never used to build, match or fill a feature.",
            ha="center", fontsize=9, style="italic", color="#4a4945")
    return fig
