"""Train the MLP / GraphSAGE baselines.

    python -m src.model.train --state DE --model mlp --task classification --seeds 0 1 2

Best-on-validation weights go to `weights/<run>/seed<k>.pt`; metrics and test
predictions go to `results/<run>/`.
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import RoadAccidentDataset, download_centrality
from src.eval import evaluate
from src.model.models import build_model

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"

# Train / validation / test years from the paper's Table 7.
SPLITS = {
    "DE": (range(2009, 2013), range(2013, 2018), range(2018, 2023)),
    "MD": (range(2015, 2018), range(2018, 2020), range(2020, 2023)),
}

MIN_ACCIDENT_ROADS = 10


@dataclass
class Config:
    state: str = "DE"
    model: str = "mlp"                      # "mlp" or "graphsage"
    task: str = "classification"            # "classification" or "regression"
    node_features: str = "centrality"       # "centrality", "none", or a path to an .npy file
    use_traffic_volume: bool = False
    epochs: int = 100
    lr: float = 1e-3
    hidden_channels: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    eval_every: int = 5
    select_protocol: str = "balanced"       # protocol whose validation score picks the best epoch
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = ""

    def __post_init__(self):
        if not self.run_name:
            self.run_name = f"{self.state}_{self.model}_{self.task}"


def load_dataset(cfg):
    train_years = SPLITS[cfg.state][0]
    if cfg.node_features == "centrality":
        embedding_path = download_centrality(cfg.state)
    elif cfg.node_features == "none":
        embedding_path = None
    else:
        embedding_path = cfg.node_features
    return RoadAccidentDataset(
        cfg.state, train_years=train_years,
        node_embedding_path=embedding_path, use_traffic_volume=cfg.use_traffic_volume,
    )


def selection_score(results, cfg):
    """Higher is better: validation AUROC for classification, negative MAE for regression."""
    metrics = results[cfg.select_protocol]
    return metrics["AUROC"] if cfg.task == "classification" else -metrics["MAE"]


def train_epoch(model, optimizer, dataset, months, cfg, rng):
    model.train()
    total_loss, total_roads = 0.0, 0
    for year, month in rng.permutation(months).tolist():
        data = dataset.monthly(year, month).to(cfg.device)
        positive = torch.nonzero(data.y > 0).squeeze(1)
        if len(positive) < MIN_ACCIDENT_ROADS:
            continue
        negative_pool = torch.nonzero(data.y == 0).squeeze(1)
        negative = negative_pool[torch.randperm(len(negative_pool), device=cfg.device)[:len(positive)]]
        road_ids = torch.cat([positive, negative])

        out = model(data, road_ids)
        if cfg.task == "classification":
            loss = F.binary_cross_entropy_with_logits(out, (data.y[road_ids] > 0).float())
        else:
            loss = F.l1_loss(out, data.y[road_ids])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(road_ids)
        total_roads += len(road_ids)
    return total_loss / max(total_roads, 1)


def train_seed(cfg, seed, dataset):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_years, valid_years, test_years = SPLITS[cfg.state]
    train_months, valid_months = dataset.months(train_years), dataset.months(valid_years)

    sample = dataset.monthly(*train_months[0])
    model = build_model(
        cfg.model, sample.x.size(1), sample.edge_attr.size(1),
        cfg.hidden_channels, cfg.num_layers, cfg.dropout,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    weights_path = WEIGHTS_DIR / cfg.run_name / f"seed{seed}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch = -float("inf"), 0

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        loss = train_epoch(model, optimizer, dataset, train_months, cfg, rng)
        if epoch % cfg.eval_every and epoch != cfg.epochs:
            continue
        valid = evaluate(model, dataset, valid_months, cfg.task, cfg.device)
        score = selection_score(valid, cfg)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "seed": seed, "epoch": epoch},
                       weights_path)
        key = "AUROC" if cfg.task == "classification" else "MAE"
        print(f"[seed {seed}] epoch {epoch:3d} | loss {loss:.4f} | "
              f"valid {key} ({cfg.select_protocol}) {valid[cfg.select_protocol][key]:.4f} | "
              f"best epoch {best_epoch} | {time.time() - start:.1f}s")

    model.load_state_dict(torch.load(weights_path, map_location=cfg.device)["state_dict"])
    valid = evaluate(model, dataset, valid_months, cfg.task, cfg.device)
    test, predictions = evaluate(model, dataset, dataset.months(test_years), cfg.task, cfg.device,
                                 return_predictions=True)

    results_dir = RESULTS_DIR / cfg.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, results_dir / f"seed{seed}_test_predictions.pt")
    return {"seed": seed, "best_epoch": best_epoch, "valid": valid, "test": test}


def summarise(runs):
    """Mean and standard deviation across seeds for every split / protocol / metric."""
    summary = {}
    for split in ("valid", "test"):
        for protocol, metrics in runs[0][split].items():
            for metric in metrics:
                values = [r[split][protocol][metric] for r in runs]
                summary.setdefault(split, {}).setdefault(protocol, {})[metric] = {
                    "mean": float(np.mean(values)), "std": float(np.std(values)),
                }
    return summary


def run(cfg):
    dataset = load_dataset(cfg)
    runs = [train_seed(cfg, seed, dataset) for seed in cfg.seeds]
    summary = summarise(runs)

    results_dir = RESULTS_DIR / cfg.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({"config": asdict(cfg), "summary": summary, "runs": runs}, f, indent=2)

    print(f"\n{cfg.run_name}: test results over seeds {cfg.seeds}")
    for protocol, metrics in summary["test"].items():
        print(f"  {protocol}: " + ", ".join(f"{k} {v['mean']:.4f} ± {v['std']:.4f}" for k, v in metrics.items()))
    return summary


def parse_args():
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=defaults.state, choices=sorted(SPLITS))
    parser.add_argument("--model", default=defaults.model, choices=["mlp", "graphsage"])
    parser.add_argument("--task", default=defaults.task, choices=["classification", "regression"])
    parser.add_argument("--node-features", default=defaults.node_features)
    parser.add_argument("--use-traffic-volume", action="store_true")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--hidden-channels", type=int, default=defaults.hidden_channels)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--select-protocol", default=defaults.select_protocol, choices=["all", "balanced"])
    parser.add_argument("--seeds", type=int, nargs="+", default=defaults.seeds)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--run-name", default="")
    return Config(**{k.replace("-", "_"): v for k, v in vars(parser.parse_args()).items()})


if __name__ == "__main__":
    run(parse_args())
