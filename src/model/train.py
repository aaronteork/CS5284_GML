"""Train MLP, GraphSAGE, GatedGCN and Graph Transformer road-accident models.

    python -m src.model.train --state DE --model mlp --task classification --seeds 0 1 2

Best-on-validation weights go to `weights/<run>/seed<k>.pt`; metrics and test
predictions go to `results/<run>/`.
"""
import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.data import DATA_DIR, RoadAccidentDataset
from src.data.node2vec import generate_node2vec, node2vec_path
from src.eval import evaluate
from src.model.models import MODEL_NAMES, build_model

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
    model: str = "mlp"                      # See MODEL_NAMES for encoder choices.
    task: str = "classification"            # "classification" or "regression"
    use_node2vec: bool = True              # ablation: all other features stay identical
    epochs: int = 100
    lr: float = 1e-3
    hidden_channels: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    num_heads: int = 4                     # Graph Transformer only; width must be divisible.
    eval_every: int = 5
    select_protocol: str = "all"            # full-network validation picks the best epoch
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = ""
    data_dir: str = ""                    # Empty retains the repository's data folder.
    output_dir: str = ""                  # Root for results/ and weights/; empty retains defaults.
    history_groups: list = field(default_factory=list)
    history_min_months: int = 0            # Set 12 for matched-warm-up history ablations, including control.

    def __post_init__(self):
        if not self.run_name:
            embedding = "n2v" if self.use_node2vec else "no_n2v"
            self.run_name = f"{self.state}_{self.model}_{self.task}_{embedding}"


def load_dataset(cfg):
    train_years = SPLITS[cfg.state][0]
    data_dir = Path(cfg.data_dir) if cfg.data_dir else DATA_DIR
    embedding_path = generate_node2vec(cfg.state, data_dir=data_dir, device=cfg.device) if cfg.use_node2vec else None
    dataset = RoadAccidentDataset(
        cfg.state, train_years=train_years,
        node_embedding_path=embedding_path, data_dir=data_dir,
    )
    if cfg.history_groups or cfg.history_min_months:
        from src.history.integration import HistoryDataset
        dataset = HistoryDataset(dataset, groups=cfg.history_groups, min_history=cfg.history_min_months)
    return dataset


def results_directory(cfg):
    return (Path(cfg.output_dir) / "results" if cfg.output_dir else RESULTS_DIR) / cfg.run_name


def weights_directory(cfg):
    return (Path(cfg.output_dir) / "weights" if cfg.output_dir else WEIGHTS_DIR) / cfg.run_name


def selection_score(results, cfg):
    """Higher is better: validation AUROC for classification, negative MAE for regression."""
    metrics = results[cfg.select_protocol]
    return metrics["AUROC"] if cfg.task == "classification" else -metrics["MAE"]


HISTORY_COLUMNS = ["epoch", "train_loss", "valid_auroc", "valid_mae",
                   "epoch_seconds", "elapsed_seconds", "is_best", "best_epoch"]


def save_curves(history_path, cfg):
    """Plot recorded values; blank validation entries are unevaluated epochs."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with open(history_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    metric = "valid_auroc" if cfg.task == "classification" else "valid_mae"
    evaluated = [row for row in rows if row[metric]]
    # An isolated canvas saves PNGs without switching a notebook's inline backend.
    fig = Figure(figsize=(10, 4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    try:
        axes[0].plot([int(r["epoch"]) for r in rows], [float(r["train_loss"]) for r in rows])
        axes[0].set_ylabel("Training BCE loss" if cfg.task == "classification" else "Training MAE loss")
        if evaluated:
            axes[1].plot([int(r["epoch"]) for r in evaluated],
                         [float(r[metric]) for r in evaluated], marker="o")
        axes[1].set_ylabel(f"Validation {'AUROC' if cfg.task == 'classification' else 'MAE'} ({cfg.select_protocol})")
        best = int(rows[-1]["best_epoch"])
        for ax in axes:
            ax.set_xlabel("Epoch")
            if best:
                ax.axvline(best, linestyle="--", color="tab:red", label=f"Best epoch {best}")
                ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle(f"{cfg.run_name} / {history_path.stem}")
        fig.tight_layout()
        fig.savefig(history_path.with_suffix(".png"), dpi=150)
    finally:
        fig.clear()


def run_provenance(cfg):
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    data_dir = Path(cfg.data_dir) if cfg.data_dir else DATA_DIR
    embedding_meta = node2vec_path(cfg.state, data_dir=data_dir).with_suffix(".json")
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "splits": {name: list(years) for name, years in
                       zip(("train", "valid", "test"), SPLITS[cfg.state])},
            "embedding": json.loads(embedding_meta.read_text(encoding="utf-8"))
                if cfg.use_node2vec and embedding_meta.exists() else None}


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
        num_heads=cfg.num_heads,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    weights_path = weights_directory(cfg) / f"seed{seed}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch = -float("inf"), 0
    results_dir = results_directory(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / f"seed{seed}_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS).writeheader()
    history_path.with_suffix(".png").unlink(missing_ok=True)
    training_start = time.perf_counter()

    progress = tqdm(range(1, cfg.epochs + 1), desc=f"{cfg.state} {cfg.model} {cfg.task} seed {seed}",
                    unit="epoch", dynamic_ncols=True)
    for epoch in progress:
        start = time.time()
        loss = train_epoch(model, optimizer, dataset, train_months, cfg, rng)
        progress.set_postfix(loss=f"{loss:.4f}", refresh=False)
        valid, is_best = None, False
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            valid = evaluate(model, dataset, valid_months, cfg.task, cfg.device)
            score = selection_score(valid, cfg)
            if not np.isfinite(score):
                raise ValueError("Validation selection metric is undefined; check split labels")
            if score > best_score:
                best_score, best_epoch, is_best = score, epoch, True
                torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "seed": seed, "epoch": epoch},
                           weights_path)
            key = "AUROC" if cfg.task == "classification" else "MAE"
            tqdm.write(f"[seed {seed}] epoch {epoch:3d} | loss {loss:.4f} | "
                  f"valid {key} ({cfg.select_protocol}) {valid[cfg.select_protocol][key]:.4f} | "
                  f"best epoch {best_epoch} | {time.time() - start:.1f}s")
        metrics = valid[cfg.select_protocol] if valid is not None else {}
        with open(history_path, "a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS).writerow({
                "epoch": epoch, "train_loss": loss, "valid_auroc": metrics.get("AUROC", ""),
                "valid_mae": metrics.get("MAE", ""), "epoch_seconds": time.time() - start,
                "elapsed_seconds": time.perf_counter() - training_start,
                "is_best": is_best, "best_epoch": best_epoch})
        if valid is not None or epoch == 1:
            save_curves(history_path, cfg)

    model.load_state_dict(torch.load(weights_path, map_location=cfg.device)["state_dict"])
    valid = evaluate(model, dataset, valid_months, cfg.task, cfg.device)
    test, predictions = evaluate(model, dataset, dataset.months(test_years), cfg.task, cfg.device,
                                 return_predictions=True)

    results_dir = results_directory(cfg)
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
    data_summary = {"num_nodes": dataset.num_nodes, "num_roads": dataset.num_roads,
                    "target_unit": dataset.target_unit,
                    "months": {split: len(dataset.months(years)) for split, years in
                               zip(("train", "valid", "test"), SPLITS[cfg.state])}}
    print(f"Data coverage: {data_summary}", flush=True)
    results_dir = results_directory(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    provenance = run_provenance(cfg)
    if hasattr(dataset, "history_provenance"):
        provenance["accident_history"] = dataset.history_provenance()
    with open(results_dir / "run_metadata.json", "w", encoding="utf-8") as stream:
        json.dump({"config": asdict(cfg), "data": data_summary, "provenance": provenance}, stream, indent=2)
    runs = []
    for seed in cfg.seeds:
        result = train_seed(cfg, seed, dataset)
        runs.append(result)
        with open(results_dir / f"seed{seed}_metrics.json", "w") as f:
            json.dump({"config": asdict(cfg), "data": data_summary, "provenance": provenance, **result}, f, indent=2)
        # Persist completed seeds even if a later Colab run is interrupted.
        with open(results_dir / "metrics.json", "w") as f:
            json.dump({"config": asdict(cfg), "data": data_summary, "summary": summarise(runs), "runs": runs,
                       "provenance": provenance, "complete": len(runs) == len(cfg.seeds)}, f, indent=2)
    summary = summarise(runs)

    results_dir = results_directory(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({"config": asdict(cfg), "data": data_summary, "summary": summary,
                   "runs": runs, "provenance": provenance, "complete": True}, f, indent=2)

    print(f"\n{cfg.run_name}: test results over seeds {cfg.seeds}")
    for protocol, metrics in summary["test"].items():
        print(f"  {protocol}: " + ", ".join(f"{k} {v['mean']:.4f} ± {v['std']:.4f}" for k, v in metrics.items()))
    return summary


def parse_args():
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=defaults.state, choices=sorted(SPLITS))
    parser.add_argument("--model", default=defaults.model, choices=MODEL_NAMES)
    parser.add_argument("--task", default=defaults.task, choices=["classification", "regression"])
    parser.add_argument("--without-node2vec", action="store_false", dest="use_node2vec",
                        help="Ablation: omit only Node2Vec embeddings, retain all other features")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--hidden-channels", type=int, default=defaults.hidden_channels)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads,
                        help="Graph Transformer attention heads; must divide hidden channels")
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--select-protocol", default=defaults.select_protocol, choices=["all", "balanced"])
    parser.add_argument("--seeds", type=int, nargs="+", default=defaults.seeds)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--data-dir", default="", help="Extracted benchmark and embedding folder")
    parser.add_argument("--output-dir", default="", help="Root containing results/ and weights/")
    parser.add_argument("--history-groups", nargs="*", default=[],
                        choices=["own_history", "neighbour_history", "history_coverage"])
    parser.add_argument("--history-min-months", type=int, default=0,
                        help="Common warm-up for history and matched controls; use 12 for report ablations")
    return Config(**{k.replace("-", "_"): v for k, v in vars(parser.parse_args()).items()})


if __name__ == "__main__":
    run(parse_args())
