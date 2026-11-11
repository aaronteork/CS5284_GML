import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data.dataset import RoadAccidentDataset, WEATHER_COLUMNS
from src.data.history import (FEATURE_NAMES, HistoryScaler, PhysicalNeighbours,
                              build_physical_features, sha256)
from src.model import train
from src.model.models import MODEL_NAMES


class HistoryArithmeticTests(unittest.TestCase):
    def test_neighbours_exclude_reverse_and_handle_loop_and_isolated(self):
        # Canonical physical roads: the reverse orientation is already merged.
        pairs = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [1, 1]])
        neighbours = PhysicalNeighbours(pairs, 6)
        np.testing.assert_array_equal(neighbours.degree, [2, 3, 1, 0, 2])
        np.testing.assert_allclose(neighbours.mean(np.array([5, 7, 11, 13, 17], np.float32)),
                                   [12, 11, 7, 0, 6])

    def test_future_invariance_missing_windows_and_exact_arithmetic(self):
        neighbours = PhysicalNeighbours(np.array([[0, 1], [1, 2]]), 3)
        counts = np.arange(32, dtype=np.float32).reshape(16, 2)
        usable = np.ones(16, bool)
        original = build_physical_features(counts, usable, 12, neighbours)
        np.testing.assert_allclose(original[:, 1], counts[9:12].mean(0))
        np.testing.assert_allclose(original[:, 3], counts[:12].mean(0))
        np.testing.assert_allclose(original[:, 5], (counts[:12] > 0).mean(0))
        counts[12:] = 99999
        np.testing.assert_array_equal(original, build_physical_features(counts, usable, 12, neighbours))
        usable[10] = False
        missing = build_physical_features(counts, usable, 12, neighbours)
        self.assertTrue(np.isnan(missing[:, 1]).all())
        self.assertTrue(np.isfinite(missing[:, 0]).all())
        np.testing.assert_array_equal(missing[:, -1], [11, 11])

    def test_scaling_uses_only_passed_training_values_and_retains_missingness(self):
        values = np.tile(np.array([[1.], [3.]], np.float32), (1, 10))
        scaler = HistoryScaler().fit([values])
        test = np.full((2, 10), 101., np.float32)
        test[0, 0] = np.nan
        transformed = scaler.transform(test, "all")
        self.assertEqual(transformed.shape, (2, 18))
        self.assertEqual(transformed[0, 0], 0)
        self.assertEqual(transformed[0, 10], 1)
        self.assertEqual(transformed[1, 0], 99)
        np.testing.assert_array_equal(scaler.mean, np.full(10, 2))


class HistoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state, self.history = self.root / "DE", self.root / "history"
        for directory in (self.state / "Nodes", self.state / "Edges", self.history / "monthly_features"):
            directory.mkdir(parents=True)
        self.edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        def sparse(values):
            return torch.sparse_coo_tensor(self.edges, torch.tensor(values, dtype=torch.float32), (3, 3))
        torch.save(sparse([1, 1, 1, 1]), self.state / "adj_matrix.pt")
        torch.save({"length": sparse([1, 1, 2, 2])}, self.state / "Edges" / "edge_features.pt")
        torch.save({"AADT": sparse([5, 5, 8, 8])}, self.state / "Edges" / "edge_features_traffic_2009.pt")
        # Coalesced order equals the array above; both directions of road 0 are positive.
        records, coverage = [], []
        for period in pd.period_range("2009-01", "2010-03", freq="M"):
            frame = pd.DataFrame({c: [1., 2., 3.] for c in WEATHER_COLUMNS})
            frame["lon"], frame["lat"] = [-75., -74., -73.], [38., 39., 40.]
            frame.to_csv(self.state / "Nodes" / f"node_features_{period.year}_{period.month}.csv", index=False)
            y = np.array([2, 3, 0, 0], np.int32)
            for edge in range(2):
                records.append(dict(year=period.year, month=period.month,
                                    node_1_idx=int(self.edges[0, edge]), node_2_idx=int(self.edges[1, edge]),
                                    acc_count=int(y[edge])))
            np.savez(self.history / "monthly_features" / f"history_{period}.npz",
                     X=np.zeros((4, 10), np.float32), y=y, edge_id=np.arange(4),
                     feature_names=FEATURE_NAMES, target_month=str(period), history_cutoff=str(period-1))
            coverage.append({"month": str(period), "usable_under_current_assumption": True})
        pd.DataFrame(records).to_csv(self.state / "accidents_monthly.csv", index=False)
        pd.DataFrame(coverage).to_csv(self.history / "monthly_coverage_audit.csv", index=False)
        pd.DataFrame({"edge_id": np.arange(4), "node_1_idx": self.edges[0],
                      "node_2_idx": self.edges[1]}).to_csv(self.history / "edge_mapping.csv", index=False)
        self.write_manifest()
        self.datasets = []

    def write_manifest(self):
        (self.history / "manifest.json").write_text(json.dumps({"source_sha256": {
            name: sha256(self.state / name) for name in ("adj_matrix.pt", "accidents_monthly.csv")}}))

    def dataset(self, mode="all"):
        dataset = RoadAccidentDataset("DE", [2009], self.root, download=False,
                                      history_dir=self.history, history_mode=mode)
        self.datasets.append(dataset)
        return dataset

    def tearDown(self):
        RoadAccidentDataset._raw_weather.cache_clear()
        RoadAccidentDataset._raw_traffic.cache_clear()
        for dataset in self.datasets:
            for array in dataset._node_cache.values():
                array._mmap.close()
            dataset.history.raw._mmap.close()
        self.temp.cleanup()

    def test_directional_graph_physical_labels_and_rebuilt_history(self):
        dataset = self.dataset()
        month = dataset.monthly(2010, 1)
        torch.testing.assert_close(month.edge_index, self.edges)
        torch.testing.assert_close(month.y, torch.tensor([5., 0.]))
        raw = dataset.history.raw[12]
        self.assertEqual(raw[0, 0], 5)  # Sum orientations before lagging.
        self.assertEqual(raw[0, 5], 1)  # Nonzero fraction is NOT 1 + 1.
        self.assertEqual(raw[1, 6], 5)
        self.assertEqual(raw[0, 6], 0)  # Own reverse is not a neighbour.
        torch.testing.assert_close(month.edge_attr[0, -18:], month.edge_attr[1, -18:])
        self.assertTrue(torch.isfinite(dataset.monthly(2009, 1).edge_attr).all())
        baseline = self.dataset("none")
        own = self.dataset("own")
        self.assertEqual(dataset.available_months, baseline.available_months)
        self.assertEqual(dataset.available_months, own.available_months)
        self.assertEqual(month.edge_attr.shape[1], 27 + 18)
        self.assertEqual(own.monthly(2010, 1).edge_attr.shape[1], 27 + 13)
        torch.testing.assert_close(month.edge_attr[:, :27], baseline.monthly(2010, 1).edge_attr)

    def test_misaligned_mapping_is_rejected(self):
        mapping = pd.read_csv(self.history / "edge_mapping.csv")
        mapping.loc[0, "node_2_idx"] = 2
        mapping.to_csv(self.history / "edge_mapping.csv", index=False)
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.dataset()

    def test_all_four_models_train_both_tasks(self):
        dataset = self.dataset()
        # Map validation/test years to synthetic months, with both label classes.
        dataset.months = lambda years: [(2009, 12)] if min(years) == 2009 else [(2010, 1)]
        for model in MODEL_NAMES:
            for task in ("classification", "regression"):
                with self.subTest(model=model, task=task):
                    cfg = train.Config(model=model, task=task, epochs=1, eval_every=1,
                                       hidden_channels=8, seeds=[0], device="cpu", use_node2vec=False,
                                       output_dir=str(self.root / "runs"))
                    with patch.object(train, "MIN_ACCIDENT_ROADS", 1):
                        result = train.train_seed(cfg, 0, dataset)
                    self.assertEqual(result["best_epoch"], 1)
                    self.assertTrue(np.isfinite(result["test"]["all"]["AUROC"]))

    def test_resume_restores_optimizer_rng_and_epoch(self):
        dataset = self.dataset()
        dataset.months = lambda years: [(2009, 12)] if min(years) == 2009 else [(2010, 1)]
        cfg = train.Config(model="mlp", epochs=3, eval_every=1, hidden_channels=8,
                           seeds=[0], device="cpu", use_node2vec=False, output_dir=str(self.root / "resume"))
        original_epoch = train.train_epoch
        calls = [0]
        def interrupt(*args):
            calls[0] += 1
            if calls[0] == 2:
                raise KeyboardInterrupt
            return original_epoch(*args)
        with patch.object(train, "MIN_ACCIDENT_ROADS", 1), patch.object(train, "train_epoch", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                train.train_seed(cfg, 0, dataset)
        cfg.resume = True
        with patch.object(train, "MIN_ACCIDENT_ROADS", 1):
            resumed = train.train_seed(cfg, 0, dataset)
        last = self.root / "resume" / "weights" / cfg.run_name / "seed0_last.pt"
        resumed_state = torch.load(last, weights_only=False)
        cfg.resume, cfg.output_dir = False, str(self.root / "continuous")
        with patch.object(train, "MIN_ACCIDENT_ROADS", 1):
            uninterrupted = train.train_seed(cfg, 0, dataset)
        state = torch.load(self.root / "continuous" / "weights" / cfg.run_name / "seed0_last.pt", weights_only=False)
        for name, tensor in state["model"].items():
            torch.testing.assert_close(tensor, resumed_state["model"][name], rtol=0, atol=0)
        self.assertEqual(resumed["best_epoch"], uninterrupted["best_epoch"])
        self.assertEqual([row["epoch"] for row in resumed_state["history"]], [1, 2, 3])

    def test_resume_before_first_completed_epoch_restarts_seed(self):
        dataset = self.dataset()
        dataset.months = lambda years: [(2009, 12)] if min(years) == 2009 else [(2010, 1)]
        cfg = train.Config(model="mlp", epochs=1, eval_every=1, hidden_channels=8,
                           seeds=[0], device="cpu", use_node2vec=False, output_dir=str(self.root / "early"))
        with patch.object(train, "train_epoch", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                train.train_seed(cfg, 0, dataset)
        cfg.resume = True
        with patch.object(train, "MIN_ACCIDENT_ROADS", 1):
            result = train.train_seed(cfg, 0, dataset)
        self.assertEqual(result["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
