"""Measure graph training/validation time and peak GPU memory without touching test metrics."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.eval import evaluate
from src.model.models import MODEL_NAMES, build_model
from src.model.train import Config, ROOT, load_dataset, train_epoch


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--with-node2vec", action="store_true")
    parser.add_argument("--history-mode", choices=["none", "own", "all"], default="all")
    parser.add_argument("--report", default="reports/hardware_profile.json")
    args = parser.parse_args()
    cfg = Config(device=args.device, hidden_channels=args.hidden_channels,
                 use_node2vec=args.with_node2vec, history_mode=args.history_mode)
    dataset = load_dataset(cfg)
    train_months, valid_months = dataset.months(range(2009, 2013)), dataset.months(range(2013, 2018))
    sample = dataset.monthly(*train_months[-1])
    results = []
    for name in args.models:
        cfg.model = name
        model = optimizer = None
        try:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = build_model(name, sample.x.shape[1], sample.edge_attr.shape[1],
                                args.hidden_channels, 2, 0, num_heads=4).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            rng = np.random.default_rng(0)
            train_epoch(model, optimizer, dataset, train_months[-1:], cfg, rng)  # warm up
            sync(args.device)
            start = time.perf_counter()
            train_epoch(model, optimizer, dataset, train_months[-3:], cfg, rng)
            sync(args.device)
            train_seconds = (time.perf_counter() - start) / 3
            start = time.perf_counter()
            evaluate(model, dataset, valid_months[:3], cfg.task, args.device)
            sync(args.device)
            valid_seconds = (time.perf_counter() - start) / 3
            estimate = (100 * len(train_months) * train_seconds +
                        (20 * len(valid_months) + len(valid_months) + len(dataset.months(range(2018, 2023)))) * valid_seconds) / 3600
            result = {"model": name, "train_seconds_per_month": train_seconds,
                      "validation_seconds_per_month_including_metrics": valid_seconds,
                      "estimated_hours_100_epochs_one_seed": estimate,
                      "planning_hours_with_io_margin": [estimate * 1.2, estimate * 2.0],
                      "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3
                            if args.device.startswith("cuda") else None,
                      "parameters": sum(p.numel() for p in model.parameters())}
        except torch.cuda.OutOfMemoryError:
            result = {"model": name, "error": "CUDA OOM; choose larger GPU or reduce width consistently for every comparison"}
        finally:
            del optimizer, model
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    report = {"device": args.device,
              "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
              "hidden_channels": args.hidden_channels, "history_mode": args.history_mode,
              "node2vec": args.with_node2vec, "results": results,
              "note": "Measured on three train/validation months, no test evaluation; estimates exclude first preprocessing and Node2Vec."}
    path = Path(args.report)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
