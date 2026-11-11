"""Train MLP, GraphSAGE, GatedGCN and Graph Transformer road-accident models.

    python -m src.model.train --state DE --model mlp --task classification --seeds 0 1 2

Best-on-validation weights go to `weights/<run>/seed<k>.pt`; metrics and test
predictions go to `results/<run>/`.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.data import RoadAccidentDataset
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
    # Keep the original repository notebooks working without a history archive.
    # The history notebook passes `none`/`all` explicitly for matched experiments.
    history_mode: str = "none"
    data_dir: str = "data"
    history_dir: str = "data/accident_history/outputs"
    output_dir: str = ""
    resume: bool = False
    smoke_test: bool = False
    # Backwards-compatible optional feature-group path already present on main.
    history_groups: list = field(default_factory=list)
    history_min_months: int = 0

    def __post_init__(self):
        if self.epochs < 1 or self.eval_every < 1 or not self.seeds:
            raise ValueError("epochs, eval_every and seeds must be nonempty/positive")
        if self.history_mode not in ("none", "own", "all"):
            raise ValueError("history_mode must be none, own or all")
        if self.history_groups and self.history_mode != "none":
            raise ValueError("Choose either history_mode or history_groups, not both")
        if self.history_min_months < 0:
            raise ValueError("history_min_months must be nonnegative")
        if self.smoke_test:
            self.epochs, self.eval_every, self.seeds = 2, 1, [self.seeds[0]]
        if not self.run_name:
            embedding = "n2v" if self.use_node2vec else "no_n2v"
            self.run_name = f"{self.state}_{self.model}_{self.task}_{embedding}"
            if self.history_groups or self.history_min_months:
                groups = "-".join(self.history_groups) or "none"
                self.run_name += f"_groups_{groups}_warmup{self.history_min_months}"
            else:
                self.run_name += f"_hist_{self.history_mode}"
        if self.smoke_test and not self.run_name.endswith("_smoke"):
            self.run_name += "_smoke"
        if Path(self.run_name).name != self.run_name or self.run_name in (".", ".."):
            raise ValueError("run_name must be one directory name")


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def output_paths(cfg):
    if cfg.output_dir:
        root = project_path(cfg.output_dir)
        return root / "weights", root / "results"
    return WEIGHTS_DIR, RESULTS_DIR


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def experiment_identity(cfg, dataset):
    settings = asdict(cfg)
    for key in ("device", "resume", "data_dir", "history_dir", "output_dir", "seeds"):
        settings.pop(key)
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    history = getattr(dataset, "history", None)
    return {"settings": settings, "source_sha256": digest.hexdigest(),
            "data_fingerprint": getattr(dataset, "data_fingerprint", None),
            "history_fingerprint": getattr(history, "fingerprint", None)}


def load_dataset(cfg):
    train_years = SPLITS[cfg.state][0]
    if cfg.state != "DE" and (cfg.history_mode != "none" or cfg.history_groups or cfg.history_min_months):
        raise ValueError("Accident-history features are currently validated for Delaware only")
    data_dir, history_dir = project_path(cfg.data_dir), project_path(cfg.history_dir)
    embedding_path = generate_node2vec(cfg.state, data_dir=data_dir, device=cfg.device) if cfg.use_node2vec else None
    dataset = RoadAccidentDataset(
        cfg.state, train_years=train_years,
        data_dir=data_dir, node_embedding_path=embedding_path,
    )
    if cfg.history_groups or cfg.history_min_months:
        from src.history.integration import HistoryDataset
        dataset = HistoryDataset(dataset, groups=cfg.history_groups, min_history=cfg.history_min_months)
    elif cfg.history_mode != "none" or (history_dir / "manifest.json").is_file():
        from src.history.archive_dataset import ArchiveHistoryDataset
        if not (history_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"History archive not found: {history_dir / 'manifest.json'}")
        dataset = ArchiveHistoryDataset(dataset, history_dir, cfg.history_mode)
    return dataset


def selection_score(results, cfg):
    """Higher is better: validation AUROC for classification, negative MAE for regression."""
    metrics = results[cfg.select_protocol]
    return metrics["AUROC"] if cfg.task == "classification" else -metrics["MAE"]


HISTORY_COLUMNS = ["epoch", "train_loss", "valid_auroc", "valid_mae",
                   "epoch_seconds", "elapsed_seconds", "is_best", "best_epoch"]


def save_curves(history_path, cfg):
    """Plot recorded values; blank validation entries are unevaluated epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    metric = "valid_auroc" if cfg.task == "classification" else "valid_mae"
    evaluated = [row for row in rows if row[metric]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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
        plt.close(fig)


def run_provenance(cfg):
    def git(*args):
        if not (ROOT / ".git").exists():
            return None  # A parent directory's unrelated repository is not this experiment.
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    embedding_meta = node2vec_path(cfg.state, project_path(cfg.data_dir)).with_suffix(".json")
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "device": cfg.device,
            "gpu": torch.cuda.get_device_name() if str(cfg.device).startswith("cuda") else None,
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
        # Dedicated numpy RNG is independent of model width/initialisation/dropout.
        # Paired feature ablations therefore see the same sampled negatives.
        positions = rng.choice(len(negative_pool), size=min(len(positive), len(negative_pool)), replace=False)
        negative = negative_pool[torch.as_tensor(positions, device=cfg.device)]
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
    if cfg.smoke_test:
        train_months, valid_months = train_months[-2:], valid_months[:2]

    sample = dataset.monthly(*train_months[0])
    model = build_model(
        cfg.model, sample.x.size(1), sample.edge_attr.size(1),
        cfg.hidden_channels, cfg.num_layers, cfg.dropout,
        num_heads=cfg.num_heads,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    weights_dir, result_root = output_paths(cfg)
    weights_path = weights_dir / cfg.run_name / f"seed{seed}.pt"
    last_path = weights_path.with_name(f"seed{seed}_last.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch = -float("inf"), 0
    results_dir = result_root / cfg.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / f"seed{seed}_history.csv"
    identity = experiment_identity(cfg, dataset)
    history_rows, start_epoch, elapsed_before, best_state = [], 1, 0.0, None
    if cfg.resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume rejected: code, data or training settings changed; use a new run name")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_epoch = saved["epoch"] + 1
        best_score, best_epoch, best_state = saved["best_score"], saved["best_epoch"], saved["best_state"]
        history_rows, elapsed_before = saved["history"], saved["elapsed_seconds"]
        print(f"Resuming seed {seed} at epoch {start_epoch}", flush=True)
    elif history_path.exists():
        if not cfg.resume:
            raise FileExistsError(f"Existing history without a resumable checkpoint: {history_path}; use --resume or a new run name")
        print(f"Seed {seed} has no completed checkpoint; restarting it from its original seed", flush=True)
    # Reconcile the CSV with the last atomically completed epoch after an interruption.
    with open(history_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(history_rows)
    training_start = time.perf_counter()

    progress = tqdm(range(start_epoch, cfg.epochs + 1), desc=f"{cfg.state} {cfg.model} {cfg.task} seed {seed}",
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
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                atomic_torch_save({"state_dict": best_state, "config": asdict(cfg), "seed": seed,
                                   "epoch": epoch, "identity": identity,
                                   "history_preprocessing": getattr(getattr(dataset, "history", None), "metadata", None)},
                                  weights_path)
            key = "AUROC" if cfg.task == "classification" else "MAE"
            tqdm.write(f"[seed {seed}] epoch {epoch:3d} | loss {loss:.4f} | "
                  f"valid {key} ({cfg.select_protocol}) {valid[cfg.select_protocol][key]:.4f} | "
                  f"best epoch {best_epoch} | {time.time() - start:.1f}s")
        metrics = valid[cfg.select_protocol] if valid is not None else {}
        row = {
                "epoch": epoch, "train_loss": loss, "valid_auroc": metrics.get("AUROC", ""),
                "valid_mae": metrics.get("MAE", ""), "epoch_seconds": time.time() - start,
                "elapsed_seconds": elapsed_before + time.perf_counter() - training_start,
                "is_best": is_best, "best_epoch": best_epoch}
        history_rows.append(row)
        with open(history_path, "a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS).writerow(row)
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch,
                           "best_state": best_state, "history": history_rows, "identity": identity,
                           "elapsed_seconds": row["elapsed_seconds"], "numpy_rng": rng.bit_generator.state,
                           "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, last_path)
        if valid is not None or epoch == 1:
            save_curves(history_path, cfg)

    model.load_state_dict(best_state)
    atomic_torch_save({"state_dict": best_state, "config": asdict(cfg), "seed": seed,
                       "epoch": best_epoch, "identity": identity,
                       "history_preprocessing": getattr(getattr(dataset, "history", None), "metadata", None)}, weights_path)
    valid = evaluate(model, dataset, valid_months, cfg.task, cfg.device)
    if cfg.smoke_test:
        return {"seed": seed, "best_epoch": best_epoch, "valid": valid, "smoke_only": True}
    test, predictions = evaluate(model, dataset, dataset.months(test_years), cfg.task, cfg.device,
                                 return_predictions=True)

    results_dir = result_root / cfg.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(predictions, results_dir / f"seed{seed}_test_predictions.pt")
    return {"seed": seed, "best_epoch": best_epoch, "valid": valid, "test": test}


def summarise(runs):
    """Mean and standard deviation across seeds for every split / protocol / metric."""
    summary = {}
    for split in ("valid", "test"):
        if split not in runs[0]:
            continue
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
                    "history": getattr(getattr(dataset, "history", None), "metadata", None),
                    "months": {split: len(dataset.months(years)) for split, years in
                               zip(("train", "valid", "test"), SPLITS[cfg.state])}}
    print(f"Data coverage: {data_summary}", flush=True)
    results_dir = output_paths(cfg)[1] / cfg.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    provenance = run_provenance(cfg)
    identity = experiment_identity(cfg, dataset)
    metadata_path = results_dir / "run_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if not cfg.resume:
            raise FileExistsError("This run already exists. Use --resume or a new --run-name")
        if old.get("identity") != identity:
            raise ValueError("Run identity changed; use a new run name")
    atomic_json({"config": asdict(cfg), "data": data_summary, "provenance": provenance,
                 "identity": identity}, metadata_path)
    runs = []
    for seed in cfg.seeds:
        seed_path = results_dir / f"seed{seed}_metrics.json"
        if cfg.resume and seed_path.exists():
            saved = json.loads(seed_path.read_text())
            if saved.get("identity") != identity:
                raise ValueError("Completed seed does not match this experiment")
            result = {key: saved[key] for key in ("seed", "best_epoch", "valid", "test", "smoke_only") if key in saved}
            print(f"Seed {seed} already complete; skipping", flush=True)
        else:
            result = train_seed(cfg, seed, dataset)
        runs.append(result)
        atomic_json({"config": asdict(cfg), "data": data_summary, "provenance": provenance,
                     "identity": identity, **result}, seed_path)
        # Persist completed seeds even if a later Colab run is interrupted.
        atomic_json({"config": asdict(cfg), "data": data_summary, "summary": summarise(runs), "runs": runs,
                     "identity": identity, "provenance": provenance, "smoke_only": cfg.smoke_test,
                     "complete": len(runs) == len(cfg.seeds)}, results_dir / "metrics.json")
    summary = summarise(runs)

    split = "valid" if cfg.smoke_test else "test"
    print(f"\n{cfg.run_name}: {split} results over seeds {cfg.seeds}")
    for protocol, metrics in summary[split].items():
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
    parser.add_argument("--history-mode", choices=["none", "own", "all"], default=defaults.history_mode)
    parser.add_argument("--data-dir", default=defaults.data_dir, help="Relative to this project or absolute")
    parser.add_argument("--history-dir", default=defaults.history_dir, help="Original history outputs directory")
    parser.add_argument("--output-dir", default=defaults.output_dir, help="Root containing weights/ and results/")
    parser.add_argument("--resume", action="store_true", help="Resume completed epochs and skip completed seeds")
    parser.add_argument("--smoke-test", action="store_true", help="Two epochs, two train/valid months, no test evaluation")
    parser.add_argument("--history-groups", nargs="*", default=[],
                        choices=["own_history", "neighbour_history", "history_coverage"])
    parser.add_argument("--history-min-months", type=int, default=0,
                        help="Common warm-up for the legacy feature-group integration and its control")
    return Config(**{k.replace("-", "_"): v for k, v in vars(parser.parse_args()).items()})


if __name__ == "__main__":
    run(parse_args())
