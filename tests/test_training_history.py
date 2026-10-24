import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.model import train


class TrainingHistoryTests(unittest.TestCase):
    def test_history_records_all_epochs_and_test_uses_best_checkpoint(self):
        for task in ("classification", "regression"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cfg = train.Config(task=task, epochs=4, eval_every=2,
                                   use_node2vec=False, device="cpu")
                dataset = SimpleNamespace(
                    months=lambda years: [(min(years), 1)],
                    monthly=lambda *args: SimpleNamespace(x=torch.zeros(3, 2), edge_attr=torch.zeros(2, 1)))
                model = torch.nn.Linear(1, 1, bias=False)
                epochs = []
                tested_weights = []

                def train_epoch(*args):
                    epoch = len(epochs) + 1
                    epochs.append(epoch)
                    with torch.no_grad():
                        model.weight.fill_(epoch)
                    return 1. / epoch

                def evaluate(model, dataset, months, task, device, return_predictions=False):
                    weight = model.weight.item()
                    # Epoch 2 is better than epoch 4 for both selection metrics.
                    metrics = {"all": {"AUROC": 1. / weight, "MAE": weight}}
                    if return_predictions:
                        tested_weights.append(weight)
                        return metrics, {"pred": torch.ones(1, 2)}
                    return metrics

                with patch.object(train, "WEIGHTS_DIR", root / "weights"), \
                     patch.object(train, "RESULTS_DIR", root / "results"), \
                     patch.object(train, "build_model", return_value=model), \
                     patch.object(train, "train_epoch", side_effect=train_epoch), \
                     patch.object(train, "evaluate", side_effect=evaluate):
                    result = train.train_seed(cfg, 0, dataset)
                directory = root / "results" / cfg.run_name
                history = directory / "seed0_history.csv"
                with open(history, newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(row["epoch"]) for row in rows], [1, 2, 3, 4])
                self.assertEqual(rows[0]["valid_auroc"], "")
                self.assertEqual(rows[2]["valid_auroc"], "")
                self.assertEqual(rows[1]["is_best"], "True")
                self.assertEqual(rows[3]["is_best"], "False")
                self.assertEqual(result["best_epoch"], 2)
                self.assertEqual(tested_weights, [2.])
                elapsed = [float(row["elapsed_seconds"]) for row in rows]
                self.assertEqual(elapsed, sorted(elapsed))
                self.assertEqual(history.with_suffix(".png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertTrue((directory / "seed0_test_predictions.pt").exists())

    def test_history_is_preserved_when_later_training_is_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = train.Config(epochs=3, eval_every=2, device="cpu")
            dataset = SimpleNamespace(
                months=lambda years: [(min(years), 1)],
                monthly=lambda *args: SimpleNamespace(x=torch.zeros(3, 2), edge_attr=torch.zeros(2, 1)))
            root = Path(temporary)
            with patch.object(train, "WEIGHTS_DIR", root / "weights"), \
                 patch.object(train, "RESULTS_DIR", root / "results"), \
                 patch.object(train, "build_model", return_value=torch.nn.Linear(1, 1)), \
                 patch.object(train, "train_epoch", side_effect=[0.5, KeyboardInterrupt]), \
                 patch.object(train, "evaluate") as evaluation:
                with self.assertRaises(KeyboardInterrupt):
                    train.train_seed(cfg, 0, dataset)
                evaluation.assert_not_called()
            history = root / "results" / cfg.run_name / "seed0_history.csv"
            with open(history, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["train_loss"], "0.5")


if __name__ == "__main__":
    unittest.main()
