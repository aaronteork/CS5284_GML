"""Shared report chart style. No global changes until explicitly applied."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STATE_COLORS = {"DE": "#2a78d6", "MD": "#eb6834"}


def apply_report_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "grid.color": "#e4e3df", "grid.linewidth": .8,
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=list(STATE_COLORS.values())),
    })


def load_training_results(results_dir):
    """Read completed ordinary training runs; keep OSM results in their own section.

    Every run retains its identity and provenance. No best-test run is selected,
    and repeated configurations are not silently merged. Metrics are recalculated
    from saved seeds using the existing trainer's aggregation function.
    """
    from src.model.train import summarise

    columns = ["Run", "State", "Model", "Task", "Node2Vec", "History", "Seeds", "Epochs",
               "AUROC", "AUROC SD", "AP", "AP SD", "MAE", "MAE SD",
               "Source commit", "Dirty checkout"]
    rows = []
    for path in sorted(Path(results_dir).rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg, runs = payload["config"], payload.get("runs", [])
        provenance = payload.get("provenance") or {}
        if (not payload.get("complete") or payload.get("smoke_only") or not runs
                or "test" not in payload.get("summary", {}) or provenance.get("osm") is not None):
            continue
        metrics = summarise(runs)["test"]["all"]
        row = {"Run": cfg["run_name"], "State": cfg["state"], "Model": cfg["model"],
               "Task": cfg["task"], "Node2Vec": bool(cfg["use_node2vec"]),
               "Seeds": len(runs), "Epochs": cfg["epochs"],
               "Source commit": provenance.get("git_commit"),
               "Dirty checkout": provenance.get("git_dirty")}
        history_mode = cfg.get("history_mode")
        row["History"] = (history_mode if history_mode in ("own", "all") else
                           "+".join(cfg.get("history_groups", [])) or
                           ("Matched warm-up control" if cfg.get("history_min_months", 0) else "None"))
        for source, label in (("AUROC", "AUROC"), ("PR-AUC", "AP"), ("MAE", "MAE")):
            values = metrics.get(source, {})
            row[label] = values.get("mean", float("nan"))
            row[label + " SD"] = values.get("std", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def format_training_results(results, task):
    """Compact report table with task-appropriate primary metrics."""
    if task not in ("classification", "regression"):
        raise ValueError("task must be classification or regression")
    table = results[results.Task == task].copy()
    table["Node2Vec"] = table.Node2Vec.map({False: "Without", True: "With"})
    for metric in (("AUROC", "AP") if task == "classification" else ("MAE", "AUROC")):
        table[metric] = [f"{mean:.4f} ± {std:.4f}" for mean, std in zip(table[metric], table[metric + " SD"])]
    metrics = ["AUROC", "AP"] if task == "classification" else ["MAE", "AUROC"]
    history = ["History"] if table.History.ne("None").any() else []
    return table[["State", "Model", "Node2Vec", *history, "Seeds", *metrics, "Run"]].reset_index(drop=True)


def plot_training_history(results_dir, run_name, seed=0):
    """Plot recorded training and validation curves without test-set tuning."""
    root = Path(results_dir)
    directory = root / run_name
    if not (directory / "metrics.json").is_file():
        matches = [path.parent for path in root.rglob("metrics.json")
                   if json.loads(path.read_text(encoding="utf-8")).get("config", {}).get("run_name") == run_name]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one results directory for run {run_name!r}; found {len(matches)}")
        directory = matches[0]
    payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    history = pd.read_csv(directory / f"seed{seed}_history.csv")
    task = payload["config"]["task"]
    metric, label = ("valid_auroc", "AUROC") if task == "classification" else ("valid_mae", "MAE")
    best = next(row["best_epoch"] for row in payload["runs"] if row["seed"] == seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
    axes[0].plot(history.epoch, history.train_loss)
    axes[0].set_ylabel("Training BCE" if task == "classification" else "Training MAE")
    evaluated = history.dropna(subset=[metric])
    axes[1].plot(evaluated.epoch, evaluated[metric], marker="o", markersize=3)
    axes[1].set_ylabel(f"Validation {label}")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.axvline(best, linestyle="--", color="grey", label=f"Selected epoch {best}")
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle(f"{run_name} / seed {seed}")
    return fig
