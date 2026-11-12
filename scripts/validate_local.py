"""Real-data CPU integration checks, width 32; no test-split evaluation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.model.train import Config, load_dataset, train_seed
from src.model.models import MODEL_NAMES


def main():
    torch.set_num_threads(2)
    dataset = load_dataset(Config(use_node2vec=False, device="cpu"))
    results = []
    for model in MODEL_NAMES:
        for task in ("classification", "regression"):
            cfg = Config(model=model, task=task, use_node2vec=False, device="cpu",
                         hidden_channels=32, seeds=[0], smoke_test=True, resume=True,
                         output_dir="reports/local_smoke_final")
            result = train_seed(cfg, 0, dataset)
            results.append({"model": model, "task": task, "best_epoch": result["best_epoch"],
                            "finite_validation_metrics": all(
                                isinstance(v, float) and torch.isfinite(torch.tensor(v)).item()
                                for v in result["valid"]["all"].values())})
    report = {"torch": str(torch.__version__), "device": "cpu", "hidden_channels": 32,
              "full_directed_graph": True, "train_months": ["2012-11", "2012-12"],
              "validation_months": ["2013-01", "2013-02"], "test_evaluated": False,
              "epochs": 2, "history": "all", "results": results,
              "note": "Integration smoke checks, not model performance experiments or GPU memory measurements."}
    (ROOT / "reports/local_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
