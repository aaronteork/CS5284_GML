"""Evaluate prespecified zero/lag/rolling-mean references under the same road protocol."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval import evaluate
from src.model.train import Config, ROOT, SPLITS, load_dataset


class HistoryReference(torch.nn.Module):
    def __init__(self, history, column=None):
        super().__init__()
        self.history, self.column = history, column

    def forward(self, data):
        if self.column is None:
            return torch.zeros_like(data.y)
        t = self.history.positions[(data.year, data.month)]
        values = np.array(self.history.raw[t, :, self.column], copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Reference window unavailable in evaluation month")
        return torch.from_numpy(values).to(data.y.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--output", default="reports/simple_baselines.json")
    args = parser.parse_args()
    dataset = load_dataset(Config(use_node2vec=False, device="cpu"))
    months = dataset.months(SPLITS["DE"][1 if args.split == "valid" else 2])
    results = {}
    for name, column in (("zero", None), ("lag1", 0), ("mean3", 1), ("mean12", 3)):
        model = HistoryReference(dataset.history, column)
        results[name] = evaluate(model, dataset, months, "regression", "cpu")
        print(name, json.dumps(results[name]["all"]), flush=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"split": args.split, "months": months, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
