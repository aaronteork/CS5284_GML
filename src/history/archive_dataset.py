"""Dataset adapter for manifest-validated physical-road history archives."""
from dataclasses import replace

import torch

from src.data.history import PhysicalHistory


class ArchiveHistoryDataset:
    """Append archived history features without changing the benchmark loader."""

    def __init__(self, base, history_dir, mode):
        self.base = base
        self.history = PhysicalHistory(
            history_dir=history_dir,
            state_dir=base.dir,
            edge_index=base.edge_index,
            road_keys=base._road_key,
            accidents=base._accidents,
            weather_months=base.available_months,
            train_years=base.train_years,
            mode=mode,
        )
        self.history_mode = mode
        self.available_months = self.history.months
        self._history_months = set(self.available_months)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def months(self, years):
        years = set(years)
        return [(year, month) for year, month in self.base.months(years)
                if (year, month) in self._history_months]

    def monthly(self, year, month):
        data = self.base.monthly(year, month)
        if self.history_mode == "none":
            return data
        history = torch.from_numpy(self.history.for_edges(year, month))
        return replace(data, edge_attr=torch.cat([data.edge_attr, history], dim=1))
