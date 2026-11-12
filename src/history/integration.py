"""Align the branch's past-only history methods with physical-road targets.

The imported branch functions stay unchanged in features.py. Here, directional
counts are summed *before* computing rolling means and nonzero fractions. The
same physical-road attributes are then broadcast to both message-passing edges.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import FEATURE_NAMES, NeighbourAggregator, build_features, load_data
from .loader import FEATURE_GROUPS


class PhysicalHistory:
    def __init__(self, months, counts, usable, pairs, n_nodes, source_audit=None):
        self.months = pd.PeriodIndex(months, freq="M")
        self.counts = np.asarray(counts)
        self.usable = np.asarray(usable, dtype=bool)
        self.pairs = np.asarray(pairs, dtype=np.int64)
        if self.counts.shape != (len(self.months), len(self.pairs)):
            raise ValueError("Physical history counts are not aligned with dates and roads")
        self.neighbours = NeighbourAggregator(self.pairs, n_nodes)
        self.source_audit = source_audit or {}

    @classmethod
    def from_source(cls, source):
        directed = source.edges[["node_1_idx", "node_2_idx"]].to_numpy()
        pairs, group = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
        counts = np.zeros((len(source.months), len(pairs)), dtype=source.counts.dtype)
        np.add.at(counts.T, group, source.counts.T)
        return cls(source.months, counts, source.usable_month, pairs, len(source.nodes), source.audit)

    @classmethod
    def load(cls, state_dir):
        if Path(state_dir).name != "DE":
            raise ValueError("The imported branch is validated for Delaware only")
        return cls.from_source(load_data(state_dir))

    def index(self, year, month):
        return int(self.months.get_loc(pd.Period(year=year, month=month, freq="M")))

    def eligible(self, year, month, min_history=12):
        try:
            t = self.index(year, month)
        except KeyError:
            return False
        return bool(self.usable[t] and t >= min_history
                    and self.usable[t-min_history:t].all())

    def features(self, year, month):
        t = self.index(year, month)
        if not self.usable[t]:
            raise ValueError("Target month is not usable under the declared coverage assumptions")
        return build_features(self.counts, self.usable, t, self.neighbours)


def history_readiness(history):
    """Count original and common-warm-up months under the report's fixed splits."""
    from src.model.train import SPLITS

    rows = []
    for split, years in zip(("Train", "Validation", "Test"), SPLITS["DE"]):
        available = [m for i, m in enumerate(history.months) if m.year in years and history.usable[i]]
        eligible = [m for m in available if history.eligible(m.year, m.month, 12)]
        rows.append({"Split": split, "Usable months": len(available), "After 12-month warm-up": len(eligible)})
    return pd.DataFrame(rows).set_index("Split")


def training_risk_by_history(history):
    """Descriptive training-only association on physical road-months, not model accuracy."""
    from src.model.train import SPLITS

    labels = ["0", "1", "2", "3–5", "6+"]
    number, positive, total = np.zeros(5, np.int64), np.zeros(5, np.int64), np.zeros(5)
    for t, month in enumerate(history.months):
        if month.year not in SPLITS["DE"][0] or not history.eligible(month.year, month.month, 12):
            continue
        group = np.digitize(history.counts[t-3:t].sum(axis=0), [1, 2, 3, 6])
        y = history.counts[t]
        number += np.bincount(group, minlength=5)
        positive += np.bincount(group, weights=y > 0, minlength=5).astype(np.int64)
        total += np.bincount(group, weights=y, minlength=5)
    return pd.DataFrame({"Prior 3-month accidents": labels, "Road-months": number,
        "Next-month accident probability (%)": np.divide(100 * positive, number, out=np.full(5, np.nan), where=number > 0),
        "Mean next-month count": np.divide(total, number, out=np.full(5, np.nan), where=number > 0)})


class HistoryDataset:
    """Wrap an existing dataset; reuse its labels, graph and shared training loop.

    One-month-ahead histories update with observed past labels; model weights do
    not refit at each origin. Imputation/scaling use eligible training months.
    A no-history control can use the same warm-up and retain identical inputs.
    """

    def __init__(self, base, groups=(), min_history=12, history=None):
        if base.state != "DE":
            raise ValueError("History integration is currently validated for DE only")
        if not 0 <= min_history <= 12:
            raise ValueError("min_history must be between zero and twelve")
        unknown = set(groups) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Unknown history groups: {sorted(unknown)}")
        self.base = base
        self.groups = tuple(groups)
        self.min_history = min_history
        self.history = history if history is not None else PhysicalHistory.load(base.dir)
        edges = base.edge_index.cpu().numpy().T
        canonical = np.sort(edges, axis=1)
        keys = self.history.pairs[:, 0] * base.num_nodes + self.history.pairs[:, 1]
        edge_keys = canonical[:, 0] * base.num_nodes + canonical[:, 1]
        self.edge_to_road = np.searchsorted(keys, edge_keys)
        if (self.edge_to_road >= len(keys)).any() or not np.array_equal(keys[self.edge_to_road], edge_keys):
            raise ValueError("History roads and model graph disagree")
        target_keys = edge_keys[base.road_edge.cpu().numpy()]
        if not np.array_equal(target_keys, keys):
            raise ValueError("History physical-road order differs from model targets")
        selected = {name for group in groups for name in FEATURE_GROUPS[group]}
        self.columns = [i for i, name in enumerate(FEATURE_NAMES) if name in selected]
        self.feature_names = [FEATURE_NAMES[i] for i in self.columns]
        self._mean, self._std = self._fit_scaler()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def months(self, years):
        return [(y, m) for y, m in self.base.months(years) if self.history.eligible(y, m, self.min_history)]

    def _raw(self, year, month):
        raw = self.history.features(year, month)[:, self.columns].astype(np.float64)
        # Counts, rates of counts and degrees use log1p; fractions and coverage
        # retain their native scale. This transformation is label-independent.
        for j, column in enumerate(self.columns):
            if column in (0, 1, 2, 3, 4, 6, 7, 8):
                raw[:, j] = np.log1p(raw[:, j])
        return raw

    def _fit_scaler(self):
        width = len(self.columns)
        number, total, squares = np.zeros(width), np.zeros(width), np.zeros(width)
        months = self.months(self.base.train_years)
        if not months:
            raise ValueError("No eligible training months remain after history warm-up")
        for year, month in months if width else []:
            values = self._raw(year, month)
            finite = np.isfinite(values)
            number += finite.sum(axis=0)
            total += np.where(finite, values, 0).sum(axis=0)
            squares += np.where(finite, values ** 2, 0).sum(axis=0)
        mean = total / np.maximum(number, 1)
        std = np.sqrt(np.maximum(squares / np.maximum(number, 1) - mean ** 2, 0))
        return mean, np.where(std > 0, std, 1.)

    def monthly(self, year, month):
        if not self.history.eligible(year, month, self.min_history):
            raise ValueError("Requested month is outside the common history coverage/warm-up population")
        data = self.base.monthly(year, month)
        t = self.history.index(year, month)
        if not np.array_equal(data.y.cpu().numpy(), self.history.counts[t]):
            raise ValueError("History counts and model targets disagree")
        if not self.columns:
            return data
        raw = self._raw(year, month)
        missing = ~np.isfinite(raw)
        values = np.nan_to_num((raw - self._mean) / self._std, nan=0., posinf=0., neginf=0.)
        encoded = np.column_stack([values, missing.astype(float)]).astype(np.float32)
        dynamic = torch.from_numpy(encoded[self.edge_to_road])
        return replace(data, edge_attr=torch.cat([data.edge_attr, dynamic], dim=1))

    def history_provenance(self):
        return {"source_branch": "jingjing/accident_history",
                "source_commit": "d3b5ed7e1aa8cd197be4573e4b8ef05a6bb013dc",
                "target_unit": "physical", "groups": list(self.groups),
                "features": self.feature_names, "missing_indicators": True,
                "common_warmup_months": self.min_history,
                "forecast_protocol": "rolling one-month-ahead inputs; fixed model weights during validation/test",
                "scaler": "log1p count features; mean/std from eligible training physical road-months",
                "coverage_assumptions": self.history.source_audit,
                "scaler_mean": self._mean.tolist(), "scaler_std": self._std.tolist()}
