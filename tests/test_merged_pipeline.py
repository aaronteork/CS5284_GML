import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.cache import load_node_cache
from src.data.dataset import RoadAccidentDataset, WEATHER_COLUMNS
from src.data.node2vec import generate_node2vec, train_embeddings


class MergedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / "DE"
        (self.state / "Nodes").mkdir(parents=True)
        (self.state / "Edges").mkdir()
        self.datasets = []
        self.edges = torch.tensor([[0, 1, 1], [1, 0, 2]])
        torch.save(self.sparse([1, 1, 1]), self.state / "adj_matrix.pt")
        torch.save({"length": self.sparse([20, 20, 50])},
                   self.state / "Edges" / "edge_features.pt")
        for year, values in [(2009, [1, np.nan, 3]), (2010, [101, np.nan, 103])]:
            frame = pd.DataFrame({column: [1., 2., 3.] for column in WEATHER_COLUMNS})
            frame["tavg"] = values
            frame["lon"], frame["lat"] = [-75., -74., -73.], [38., 39., 40.]
            frame.to_csv(self.state / "Nodes" / f"node_features_{year}_1.csv", index=False)
        for year, values in [(2010, [10, 20, 30]), (2012, [100, 200, 300])]:
            torch.save({"AADT": self.sparse(values)},
                       self.state / "Edges" / f"edge_features_traffic_{year}.pt")
        pd.DataFrame({"year": [2009, 2009, 2010], "month": [1, 1, 1],
                      "node_1_idx": [0, 1, 1], "node_2_idx": [1, 0, 2],
                      "acc_count": [2, 3, 1]}).to_csv(
                          self.state / "accidents_monthly.csv", index=False)

    def sparse(self, values):
        return torch.sparse_coo_tensor(self.edges, torch.tensor(values, dtype=torch.float32), (3, 3))

    def dataset(self, embeddings=None):
        result = RoadAccidentDataset("DE", [2009], self.root,
                                     node_embedding_path=embeddings, download=False)
        self.datasets.append(result)
        return result

    def tearDown(self):
        # Clear cached bound-method references and release Windows memory maps.
        RoadAccidentDataset._raw_weather.cache_clear()
        RoadAccidentDataset._raw_traffic.cache_clear()
        for dataset in self.datasets:
            self.close_cache(dataset._node_cache)
        self.temporary.cleanup()

    @staticmethod
    def close_cache(cache):
        for array in cache.values():
            array._mmap.close()

    def test_weather_uses_training_statistics_and_explicit_missing_flags(self):
        dataset = self.dataset()
        offset = dataset.static_node_features.shape[1]
        train, valid = dataset.monthly(2009, 1), dataset.monthly(2010, 1)
        torch.testing.assert_close(train.x[:, offset], torch.tensor([-1., 0., 1.]))
        torch.testing.assert_close(valid.x[:, offset], torch.tensor([99., 0., 101.]))
        torch.testing.assert_close(valid.x[:, offset + 6], torch.tensor([0., 1., 0.]))
        self.assertTrue(torch.isfinite(valid.x).all())

    def test_node2vec_ablation_changes_only_appended_embedding_columns(self):
        embedding_path = self.root / "embeddings.npy"
        np.save(embedding_path, np.array([[1, 3], [2, 2], [3, 1]], dtype=np.float32))
        base = self.dataset().monthly(2009, 1)
        enhanced = self.dataset(embedding_path).monthly(2009, 1)
        self.assertEqual(enhanced.x.shape[1], base.x.shape[1] + 2)
        torch.testing.assert_close(enhanced.x[:, :-2], base.x, rtol=0, atol=0)
        for field in ("edge_index", "edge_attr", "road_edge", "y"):
            torch.testing.assert_close(getattr(enhanced, field), getattr(base, field), rtol=0, atol=0)

    def test_reverse_direction_accidents_sum_into_one_physical_road(self):
        dataset = self.dataset()
        self.assertEqual(dataset.num_roads, 2)
        torch.testing.assert_close(dataset.monthly(2009, 1).y, torch.tensor([5., 0.]))
        torch.testing.assert_close(dataset.monthly(2010, 1).y, torch.tensor([0., 1.]))

    def test_traffic_carries_forward_without_using_future_reports(self):
        dataset = self.dataset()
        self.assertTrue(torch.isnan(dataset._raw_traffic(2009)).all())
        expected = torch.log1p(torch.tensor([[10.], [20.], [30.]]))
        torch.testing.assert_close(dataset._raw_traffic(2011), expected)
        self.assertFalse(torch.equal(dataset._raw_traffic(2011), dataset._raw_traffic(2012)))
        train = dataset.monthly(2009, 1)
        torch.testing.assert_close(train.edge_attr[:, -2], torch.zeros(3))
        torch.testing.assert_close(train.edge_attr[:, -1], torch.ones(3))

    def test_sparse_duplicates_average_finite_observations(self):
        dataset = self.dataset()
        values = torch.sparse_coo_tensor(torch.tensor([[0, 0, 0], [1, 1, 1]]),
                                         torch.tensor([2., 4., float("nan")]), (3, 3))
        aligned = dataset._edge_values(values)
        self.assertEqual(aligned[0].item(), 3.)
        self.assertTrue(torch.isnan(aligned[1:]).all())

    def test_weather_cache_refreshes_after_source_changes(self):
        cache, _ = load_node_cache(self.state, WEATHER_COLUMNS, 3)
        self.assertEqual(cache["weather"][0, 0, 0], 1.)
        self.close_cache(cache)
        source = self.state / "Nodes" / "node_features_2009_1.csv"
        previous = source.stat()
        frame = pd.read_csv(source)
        frame.loc[0, "tavg"] = 42.
        frame.to_csv(source, index=False)
        os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000))
        cache, _ = load_node_cache(self.state, WEATHER_COLUMNS, 3)
        try:
            self.assertEqual(cache["weather"][0, 0, 0], 42.)
            self.assertTrue(np.isnan(cache["weather"][0, 1, 0]))
        finally:
            self.close_cache(cache)

    def test_embedding_cache_rejects_different_training_parameters(self):
        options = dict(data_dir=self.root, dimensions=4, walk_length=3,
                       window_size=1, walks_per_node=1, batch_size=8, device="cpu")
        path = generate_node2vec("DE", **options)
        original = np.load(path).copy()
        self.assertEqual(generate_node2vec("DE", **options), path)
        np.testing.assert_array_equal(np.load(path), original)
        with self.assertRaisesRegex(ValueError, "another embedding configuration"):
            generate_node2vec("DE", seed=1, **options)


class WalkTests(unittest.TestCase):
    def test_duplicate_reverse_edges_do_not_bias_walks_and_isolates_stay_finite(self):
        options = dict(num_nodes=4, dimensions=4, walk_length=4, walks_per_node=2,
                       window_size=2, batch_size=8, device="cpu", seed=7)
        edges = torch.tensor([[0, 1], [1, 2]])
        duplicated = torch.cat([edges, edges.flip(0), edges], dim=1)
        original = train_embeddings(edges, **options)
        repeated = train_embeddings(duplicated, **options)
        np.testing.assert_array_equal(original, repeated)
        self.assertTrue(np.isfinite(original).all())
        self.assertGreater(np.linalg.norm(original[3]), 0)


if __name__ == "__main__":
    unittest.main()
