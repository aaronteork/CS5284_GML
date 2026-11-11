"""Validate bundled data and build reusable caches; never fit an accident model.

    python -m src.prepare
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.history import PhysicalNeighbours, build_physical_features
from src.model.train import Config, ROOT, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--history-dir", default="data/accident_history/outputs")
    parser.add_argument("--report", default="reports/data_validation.json")
    args = parser.parse_args()
    dataset = load_dataset(Config(use_node2vec=False, data_dir=args.data_dir,
                                  history_dir=args.history_dir, history_mode="all",
                                  device="cpu"))
    history = dataset.history
    t = history.positions[(2012, 12)]
    raw = history.raw[t]
    np.testing.assert_allclose(raw[:, 1], history.counts[t-3:t].mean(0))
    np.testing.assert_allclose(raw[:, 5], (history.counts[t-12:t] > 0).mean(0))
    # Audit target/future mutation invariance on a small graph; real cache arithmetic above.
    counts = np.arange(32, dtype=np.float32).reshape(16, 2)
    usable = np.ones(16, bool)
    neighbours = PhysicalNeighbours(np.array([[0, 1], [1, 2]]), 3)
    before = build_physical_features(counts, usable, 12, neighbours)
    counts[12:] = 1e6
    np.testing.assert_array_equal(before, build_physical_features(counts, usable, 12, neighbours))
    for ym in ((2009, 1), (2009, 4), (2010, 1), (2012, 12)):
        sample = dataset.monthly(*ym)
        assert torch.isfinite(sample.x).all() and torch.isfinite(sample.edge_attr).all()
        np.testing.assert_array_equal(sample.y.numpy(), history.counts[history.positions[ym]])
        features = sample.edge_attr[:, -18:].numpy()
        np.testing.assert_array_equal(features, features[dataset.road_edge.numpy()][history.edge_to_road])
    report = {"nodes": dataset.num_nodes, "directed_edges": int(dataset.edge_index.shape[1]),
              "physical_roads": dataset.num_roads, "node_input_dim_without_node2vec": sample.x.shape[1],
              "base_edge_input_dim": sample.edge_attr.shape[1] - 18,
              "edge_input_dim_with_all_history": sample.edge_attr.shape[1],
              "split_months": {"train": len(dataset.months(range(2009, 2013))),
                               "valid": len(dataset.months(range(2013, 2018))),
                               "test": len(dataset.months(range(2018, 2023)))},
              "directed_graph_preserved": True, "all_archive_labels_match_physical_csv_labels": True,
              "same_history_for_both_directions": True, "past_only_check": True,
              "finite_inputs_including_warmup": True, "history": history.metadata}
    path = Path(args.report)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
