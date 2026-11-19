"""Extend a completed run in place. Preserve a snapshot of the previous epoch budget."""
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from src.model import train


def extend(cfg):
    weights_root, results_root = train.output_paths(cfg)
    results = results_root / cfg.run_name
    weights = weights_root / cfg.run_name
    metadata = results / "run_metadata.json"
    if not cfg.resume or not metadata.is_file():
        raise ValueError(f"Extension requires --resume and an existing run: {results}")
    prior = json.loads(metadata.read_text())
    old = prior["identity"]
    previous = old["settings"]["epochs"]
    if cfg.epochs < previous or cfg.seeds != prior["config"]["seeds"]:
        raise ValueError("Keep the original seeds and choose an equal or larger epoch limit")
    dataset = train.load_dataset(cfg)
    current = train.experiment_identity(cfg, dataset)
    expected = dict(current, settings=dict(current["settings"], epochs=previous))
    if expected != old:
        raise ValueError("Extension rejected: code, data or training settings differ")

    for seed in cfg.seeds:
        path = weights / f"seed{seed}_last.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing last checkpoint: {path}")
    if current != old:
        archive = results / "epoch_snapshots" / f"epoch_{previous}"
        if not archive.exists():
            archive.parent.mkdir(exist_ok=True)
            with TemporaryDirectory(dir=archive.parent) as tmp:
                stage = Path(tmp) / "snapshot"
                for seed in cfg.seeds:
                    ckpt = torch.load(weights / f"seed{seed}_last.pt", map_location="cpu", weights_only=False)
                    if ckpt["identity"] != old or ckpt["epoch"] != previous:
                        raise ValueError(f"Finish the previous run before extending seed {seed}")
                    record = json.loads((results / f"seed{seed}_metrics.json").read_text())
                    if record["identity"] != old:
                        raise ValueError(f"Metrics do not match checkpoint for seed {seed}")
                shutil.copytree(weights, stage / "weights")
                shutil.copytree(results, stage / "results", ignore=shutil.ignore_patterns("epoch_snapshots"))
                stage.rename(archive)
        # Replaying this preparation after an interruption uses the immutable snapshot.
        snapshot_meta = json.loads((archive / "results/run_metadata.json").read_text())
        if snapshot_meta["identity"] != old:
            raise ValueError(f"Snapshot belongs to a different experiment: {archive}")
        for seed in cfg.seeds:
            filename = f"seed{seed}_last.pt"
            ckpt = torch.load(archive / "weights" / filename, map_location="cpu", weights_only=False)
            ckpt["identity"] = current
            train.atomic_torch_save(ckpt, weights / filename)
            (results / f"seed{seed}_metrics.json").unlink(missing_ok=True)
        summary_path = results / "metrics.json"
        summary = json.loads((archive / "results/metrics.json").read_text())
        train.atomic_json(dict(summary, complete=False), summary_path)
        train.atomic_json(dict(prior, identity=current, config=train.asdict(cfg)), metadata)
        print(f"Saved previous results in {archive}", flush=True)
    del dataset
    train.run(cfg)


if __name__ == "__main__":
    extend(train.parse_args())
