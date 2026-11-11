"""Collect completed experiments; show paired history-minus-baseline differences."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.model.train import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    directory = Path(args.results_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    rows = []
    for path in sorted(directory.glob("*/metrics.json")):
        saved = json.loads(path.read_text())
        if not saved.get("complete") or saved.get("smoke_only") or "test" not in saved["summary"]:
            continue
        cfg = saved["config"]
        metric = "AUROC" if cfg["task"] == "classification" else "MAE"
        values = saved["summary"]["test"]["all"]
        row = {"run": cfg["run_name"], "model": cfg["model"], "task": cfg["task"],
               "history": cfg["history_mode"], "node2vec": cfg["use_node2vec"],
               "seeds": str(cfg["seeds"]), "hidden_channels": cfg["hidden_channels"],
               "metric": metric, "mean": values[metric]["mean"], "std": values[metric]["std"],
               "AP": values["PR-AUC"]["mean"], "top1_precision": values["precision@top1%"]["mean"]}
        # Only compare identical code/data/hyperparameters; history is the intended difference.
        identity = saved["identity"]
        settings = dict(identity["settings"])
        settings.pop("history_mode", None)
        settings.pop("run_name", None)
        row["_match"] = json.dumps([settings, cfg["seeds"], identity["source_sha256"], identity.get("data_fingerprint"),
                                    identity["history_fingerprint"]], sort_keys=True)
        rows.append(row)
    if not rows:
        print("No completed full experiments yet. Smoke tests are excluded.")
        return
    frame = pd.DataFrame(rows)
    baseline = {row["_match"]: row["mean"] for row in rows if row["history"] == "none"}
    frame["delta_vs_matched_baseline"] = [row["mean"] - baseline[row["_match"]]
        if row["history"] != "none" and row["_match"] in baseline else float("nan") for row in rows]
    frame = frame.drop(columns="_match")
    frame.to_csv(directory / "comparison.csv", index=False)
    print(frame.to_string(index=False))
    print("Delta: positive is better for AUROC; negative is better for MAE.")


if __name__ == "__main__":
    main()
