"""Main-pipeline dataset with OpenStreetMap columns appended to the static road features.

Same feature construction as src/osm/train.py (OsmRoadAccidentDataset), but built on the
main RoadAccidentDataset so it works with every model in src/model and can be wrapped by
the accident-history dataset. With osm_groups=() the result equals RoadAccidentDataset.
"""
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import RoadAccidentDataset
from src.osm.preprocessing import build_osm_features, osm_group_columns, osm_table_path

# Same names as src/osm/train.py. road_character_filled replaces road_character (never both).
OSM_GROUPS = ("road_character", "road_character_filled", "intersection_control",
              "cycle_pedestrian_lighting", "nearby_places", "road_shape")
# "all" in the OSM experiment = these five groups.
OSM_ALL = ("road_character", "intersection_control", "cycle_pedestrian_lighting",
           "nearby_places", "road_shape")


def expand_osm_groups(groups):
    """Turn ["all"] into the five groups; check names and the road_character conflict."""
    groups = list(OSM_ALL) if list(groups) == ["all"] else list(groups)
    unknown = [g for g in groups if g not in OSM_GROUPS]
    if unknown:
        raise ValueError(f"Unknown OSM groups {unknown}; choose from {OSM_GROUPS} or 'all'")
    if "road_character" in groups and "road_character_filled" in groups:
        raise ValueError("Use road_character or road_character_filled, not both")
    return groups


def osm_label(groups):
    """Run-name label: 'all' for the five default groups, otherwise the groups joined by '-'."""
    return "all" if tuple(groups) == OSM_ALL else "-".join(groups)


class OsmRoadAccidentDataset(RoadAccidentDataset):
    def __init__(self, *args, osm_groups=(), osm_path=None, **kwargs):
        self.osm_groups = tuple(osm_groups)
        self.osm_feature_names, self.osm_feature_groups = [], []
        data_dir = kwargs.get("data_dir", "data")
        self.osm_path = Path(osm_path) if osm_path else osm_table_path(args[0], data_dir)
        super().__init__(*args, **kwargs)

    def _load_static_edge_attr(self):
        base = super()._load_static_edge_attr()
        if not self.osm_groups:
            return base
        if not self.osm_path.is_file():
            raise FileNotFoundError(
                f"{self.osm_path} not found. Build it with osm_features_and_eda.ipynb (Stage 13) "
                "or copy it from the team data folder.")
        table = pd.read_parquet(self.osm_path)
        src, dst = self.edge_index.numpy()
        features, self.osm_feature_names, self.osm_feature_groups = build_osm_features(
            table, osm_group_columns(table, self.state, self.osm_path.parent), self.osm_groups, src, dst)
        return torch.cat([base, torch.from_numpy(features)], dim=1)
