"""Summarize completed history runs against committed benchmark references."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.model.train import ROOT


REFERENCE_FIELDS = (
    "state", "model", "task", "use_node2vec", "seeds", "hidden_channels",
    "num_layers", "dropout", "lr", "eval_every", "select_protocol", "num_heads",
)


def _config_key(config):
    defaults = {"num_heads": 4}
    values = []
    for name in REFERENCE_FIELDS:
        value = config.get(name, defaults.get(name))
        if isinstance(value, list):
            value = tuple(value)
        values.append((name, value))
    return tuple(values)


def _completed_runs(directory):
    runs = []
    for path in sorted(Path(directory).glob("*/metrics.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (not saved.get("complete") or saved.get("smoke_only")
                or "test" not in saved.get("summary", {})):
            continue
        runs.append((path, saved))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--baseline-results-dir", default=str(ROOT / "results"),
                        help="Committed original benchmark results used as contextual references")
    parser.add_argument("--output", default="",
                        help="Comparison CSV path; defaults to <results-dir>/comparison.csv")
    args = parser.parse_args()
    directory = Path(args.results_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    baseline_dir = Path(args.baseline_results_dir)
    if not baseline_dir.is_absolute():
        baseline_dir = ROOT / baseline_dir

    references = {}
    for path, saved in _completed_runs(baseline_dir):
        cfg = saved["config"]
        if cfg.get("history_mode", "none") != "none" or cfg.get("history_groups"):
            continue
        references.setdefault(_config_key(cfg), []).append((path, saved))

    rows = []
    for path, saved in _completed_runs(directory):
        cfg = saved["config"]
        metric = "AUROC" if cfg["task"] == "classification" else "MAE"
        values = saved["summary"]["test"]["all"]
        row = {"run": cfg["run_name"], "model": cfg["model"], "task": cfg["task"],
               "history": cfg["history_mode"], "node2vec": cfg["use_node2vec"],
               "seeds": str(cfg["seeds"]), "hidden_channels": cfg["hidden_channels"],
               "epochs": cfg["epochs"],
               "metric": metric, "mean": values[metric]["mean"], "std": values[metric]["std"],
               "AP": values["PR-AUC"]["mean"], "top1_precision": values["precision@top1%"]["mean"]}

        candidates = references.get(_config_key(cfg), [])
        if len(candidates) == 1:
            reference_path, reference = candidates[0]
            reference_cfg = reference["config"]
            reference_values = reference["summary"]["test"]["all"]
            reference_mean = reference_values[metric]["mean"]
            row.update({
                "reference_run": reference_cfg["run_name"],
                "reference_epochs": reference_cfg["epochs"],
                "reference_mean": reference_mean,
                "reference_std": reference_values[metric]["std"],
                "delta_vs_reference": row["mean"] - reference_mean,
                "reference_note": ("Committed original result; epoch budget differs. "
                                   "Use as context, not a paired history ablation."
                                   if cfg["epochs"] != reference_cfg["epochs"] else
                                   "Committed original result; code/data provenance may differ."),
            })
        elif len(candidates) > 1:
            row.update({"reference_run": "", "reference_epochs": "", "reference_mean": "",
                        "reference_std": "", "delta_vs_reference": "",
                        "reference_note": "Multiple committed references match this configuration."})
        else:
            row.update({"reference_run": "", "reference_epochs": "", "reference_mean": "",
                        "reference_std": "", "delta_vs_reference": "",
                        "reference_note": "No committed original result for this model/task configuration."})
        rows.append(row)
    if not rows:
        print("No completed full experiments yet. Smoke tests are excluded.")
        return
    frame = pd.DataFrame(rows)
    output = Path(args.output) if args.output else directory / "comparison.csv"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print("Reference delta = history result minus committed benchmark result; positive favors AUROC, negative favors MAE.")


if __name__ == "__main__":
    main()
