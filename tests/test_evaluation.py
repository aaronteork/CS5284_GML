import math
import unittest
from types import SimpleNamespace

import torch

from src.eval import evaluate
from src.eval.metrics import classification_metrics


class Month(SimpleNamespace):
    def to(self, device):
        return self


class Dataset:
    edge_index = torch.tensor([[0, 1], [1, 2]])
    road_edge = torch.tensor([0, 1])

    def monthly(self, year, month):
        return Month(y=torch.tensor([1.0, 0.0]) if month == 1 else torch.zeros(2))


class ConstantModel(torch.nn.Module):
    def forward(self, data):
        return torch.ones_like(data.y)


class EvaluationTests(unittest.TestCase):
    def test_auroc_uses_monthly_ranking_and_positive_count_weights(self):
        metrics = classification_metrics([0.2, 0.1, 0.8, 0.7, 0.9], [1, 0, 1, 1, 0],
                                         groups=[0, 0, 1, 1, 1])
        # Monthly AUROCs are 1 and 0, weighted by 1 and 2 accident roads.
        self.assertAlmostEqual(metrics["AUROC"], 1 / 3)
        self.assertAlmostEqual(metrics["AUROC (pooled)"], 0.5)

    def test_accident_free_month_contributes_to_mae_and_exports(self):
        results, predictions = evaluate(ConstantModel(), Dataset(), [(2020, 1), (2020, 2)],
                                        "regression", "cpu", return_predictions=True)
        # Errors are [0, 1] in January and [1, 1] in February.
        self.assertAlmostEqual(results["all"]["MAE"], 0.75)
        self.assertAlmostEqual(results["balanced"]["MAE"], 0.5)
        self.assertEqual(predictions["months"], [(2020, 1), (2020, 2)])
        self.assertEqual(tuple(predictions["pred"].shape), (2, 2))
        self.assertAlmostEqual(results["all"]["precision@top1%"], 0.5)

    def test_entirely_accident_free_split_has_mae_and_undefined_auroc(self):
        results = evaluate(ConstantModel(), Dataset(), [(2020, 2)], "regression", "cpu")
        self.assertEqual(results["all"]["MAE"], 1.0)
        self.assertTrue(math.isnan(results["all"]["AUROC"]))
        self.assertEqual(results["all"]["precision@top1%"], 0.0)
        self.assertTrue(math.isnan(results["all"]["recall@top1%"]))
        self.assertTrue(math.isnan(results["balanced"]["MAE"]))

    def test_classification_keeps_accident_free_months(self):
        results, predictions = evaluate(ConstantModel(), Dataset(), [(2020, 2)],
                                        "classification", "cpu", return_predictions=True)
        self.assertEqual(predictions["months"], [(2020, 2)])
        self.assertEqual(results["all"]["precision@0.5"], 0.0)
        self.assertTrue(math.isnan(results["all"]["AUROC"]))


if __name__ == "__main__":
    unittest.main()
