# Pinned comparison baseline; extracted from notebooks/osm_baseline_comparison.ipynb, cell 8.
import os
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(os.environ.get("CS5284_DATA_DIR", str(Path(os.environ.get("CS5284_PROJECT_DIR", str(Path(__file__).resolve().parents[3]))) / "data")))

DATAVERSE_URL = "https://dataverse.harvard.edu"
DATAVERSE_DOI = "doi:10.7910/DVN/V71K5R"
WEATHER_COLUMNS = ["tavg", "tmin", "tmax", "prcp", "wspd", "pres"]

# Same column order as the paper's code; keys missing from a state become zero columns.
ROAD_FLAG_KEYS = [
    "oneway", "access_ramp", "bus_stop", "crossing", "disused", "elevator",
    "escape", "living_street", "motorway", "motorway_link", "primary",
    "primary_link", "residential", "rest_area", "road", "secondary",
    "secondary_link", "stairs", "tertiary", "tertiary_link", "trunk",
    "trunk_link", "unclassified", "unsurfaced",
]


def download_state(state, data_dir=DATA_DIR, zip_path=None, keep_zip=False):
    """Download and extract a state's dataset into `data_dir/<state>/`.

    Pass `zip_path` to extract an already-downloaded zip (e.g. from Google Drive)
    instead of downloading from Harvard Dataverse.
    """
    data_dir = Path(data_dir)
    state_dir = data_dir / state
    if (state_dir / "adj_matrix.pt").exists():
        return state_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    if zip_path is None:
        zip_path = data_dir / f"{state}.zip"
        if not zip_path.exists():
            _download_from_dataverse(f"{state}.zip", zip_path)
    zip_path = Path(zip_path)

    print(f"Extracting {zip_path.name}...")
    tmp_dir = data_dir / f"_extract_{state}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp_dir)

    # Zips differ in layout (`DE/...` vs files at the root), so locate the folder by its contents.
    adj_files = list(tmp_dir.rglob("adj_matrix.pt"))
    if len(adj_files) != 1:
        raise RuntimeError(f"Expected one adj_matrix.pt in {zip_path}, found {len(adj_files)}")
    shutil.move(str(adj_files[0].parent), str(state_dir))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not keep_zip and zip_path.parent == data_dir:
        zip_path.unlink()
    print(f"{state} ready at {state_dir}")
    return state_dir


def _download_from_dataverse(filename, dest):
    url = f"{DATAVERSE_URL}/api/datasets/:persistentId/?persistentId={DATAVERSE_DOI}"
    print(f"Looking up {filename} on Harvard Dataverse...", flush=True)
    with _open_dataverse(url, filename, dest) as response:
        files = json.load(response)["data"]["latestVersion"]["files"]
    file_ids = {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files}
    if filename not in file_ids:
        raise ValueError(f"{filename} not on Dataverse; available: {sorted(file_ids)}")

    file_url = f"{DATAVERSE_URL}/api/access/datafile/{file_ids[filename]}"
    tmp = dest.with_suffix(".part")
    print(f"Downloading {filename}...")
    with _open_dataverse(file_url, filename, dest) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    tmp.rename(dest)


def _open_dataverse(url, filename, dest):
    try:
        return urllib.request.urlopen(url, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError(
                f"Harvard Dataverse returned HTTP 403 for {filename}. "
                "Download the ZIP in your browser from https://doi.org/10.7910/DVN/V71K5R "
                f"and place it at {dest}, then rerun. In the Colab notebook, this maps to "
                f"My Drive/cs5284_project/data/{filename}. "
                "The loader will extract the local ZIP without contacting Dataverse."
            ) from exc
        raise


@dataclass
class MonthlyData:
    """Inputs and labels for one month.

    The graph is directed: two-way roads appear as two edges, one-way roads as one.
    Labels are per physical road (undirected), so an accident road's reverse
    direction is never treated as accident-free.
    """
    year: int
    month: int
    x: torch.Tensor           # [num_nodes, F] node features (embeddings + normalised weather)
    edge_index: torch.Tensor  # [2, num_edges] directed edges, for message passing
    edge_attr: torch.Tensor   # [num_edges, D] edge features
    road_edge: torch.Tensor   # [num_roads] one representative directed edge per road
    y: torch.Tensor           # [num_roads] accident count this month

    @property
    def road_index(self):
        return self.edge_index[:, self.road_edge]

    @property
    def road_attr(self):
        return self.edge_attr[self.road_edge]

    def to(self, device):
        return MonthlyData(
            self.year, self.month,
            *(t.to(device) for t in (self.x, self.edge_index, self.edge_attr, self.road_edge, self.y)),
        )


class RoadAccidentDataset:
    """Merged feature pipeline; only Node2Vec inclusion varies in the ablation.

    Targets remain physical roads; the graph is directed for message passing.
    Continuous weather/AADT statistics and missing-value filling use training
    years. Missing traffic years carry forward only an earlier available report.
    """

    def __init__(self, state, train_years, data_dir=DATA_DIR,
                 node_embedding_path=None, download=True):
        from .cache import load_node_cache, load_static_cache
        self.state, self.train_years = state, list(train_years)
        self.dir = Path(data_dir) / state
        self.target_unit = "physical"
        if download:
            download_state(state, data_dir)
        adj = torch.load(self.dir / "adj_matrix.pt", map_location="cpu", weights_only=False).coalesce()
        self.num_nodes = adj.shape[0]
        self.edge_index = adj.indices().long()
        src, dst = self.edge_index.numpy()
        self._edge_key = src * self.num_nodes + dst
        road_key = np.minimum(src, dst) * self.num_nodes + np.maximum(src, dst)
        self._road_key, road_edge = np.unique(road_key, return_index=True)
        self.road_edge = torch.from_numpy(road_edge).long()
        self.num_roads = len(self._road_key)
        self.static_edge_attr = self._load_static_edge_attr()
        self._accidents = self._load_accidents()
        self._node_cache, month_files = load_node_cache(self.dir, WEATHER_COLUMNS, self.num_nodes)
        self.available_months = [tuple(map(int, p.stem.split("_")[-2:])) for p in month_files]
        self._month_position = {ym: i for i, ym in enumerate(self.available_months)}
        self.node_coordinates = torch.from_numpy(np.array(self._node_cache["coords"]))
        static = load_static_cache(self.dir, self.edge_index, self.num_nodes)
        coords = self._standardize_static(self.node_coordinates)
        centrality = self._standardize_static(torch.log1p(torch.from_numpy(static["centrality"]) * 1e3))
        self.static_node_features = torch.cat([coords, torch.from_numpy(static["degree"]), centrality], dim=1)
        self.node_embeddings = None
        if node_embedding_path is not None:
            embeddings = torch.from_numpy(np.load(node_embedding_path)).float()
            if embeddings.ndim != 2 or embeddings.shape[0] != self.num_nodes or not torch.isfinite(embeddings).all():
                raise ValueError("Invalid or misaligned node embeddings")
            self.node_embeddings = self._standardize_static(embeddings)
        train_months = self.months(self.train_years)
        if not train_months:
            raise ValueError("Training years contain no weather months")
        self._weather_mean, self._weather_std = self._train_stats(
            torch.cat([self._raw_weather(y, m) for y, m in train_months]))
        self._traffic_files = {int(p.stem.split("_")[-1]): p
            for p in (self.dir / "Edges").glob("edge_features_traffic_*.pt")}
        self._traffic_mean, self._traffic_std = self._train_stats(
            torch.cat([self._raw_traffic(y) for y in self.train_years]))

    def months(self, years):
        return [(y, m) for y, m in self.available_months if y in set(years)]

    def monthly(self, year, month):
        raw_weather = self._raw_weather(year, month)
        weather_missing = (~torch.isfinite(raw_weather)).float()
        weather = torch.nan_to_num((raw_weather - self._weather_mean) / self._weather_std,
                                   nan=0.0, posinf=0.0, neginf=0.0)
        inputs = [self.static_node_features, weather, weather_missing]
        if self.node_embeddings is not None:
            inputs.append(self.node_embeddings)
        x = torch.cat(inputs, dim=1)
        raw_traffic = self._raw_traffic(year)
        missing = (~torch.isfinite(raw_traffic)).float()
        traffic = torch.nan_to_num((raw_traffic - self._traffic_mean) / self._traffic_std,
                                   nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.cat([self.static_edge_attr, traffic, missing], dim=1)
        y = torch.zeros(self.num_roads)
        road_ids, counts = self._accidents.get((year, month), (np.array([], dtype=np.int64), np.array([])))
        y[torch.from_numpy(road_ids)] = torch.from_numpy(counts).float()
        return MonthlyData(year, month, x, self.edge_index, edge_attr, self.road_edge, y)

    def _edge_values(self, sparse):
        """Align raw COO observations, averaging duplicates while ignoring NaNs."""
        src, dst = sparse._indices().numpy()
        keys = src * self.num_nodes + dst
        pos = np.clip(np.searchsorted(self._edge_key, keys), 0, len(self._edge_key) - 1)
        values = sparse._values().float()
        valid = torch.from_numpy(self._edge_key[pos] == keys) & torch.isfinite(values)
        positions = torch.from_numpy(pos)[valid]
        sums, counts = torch.zeros(len(self._edge_key)), torch.zeros(len(self._edge_key))
        sums.index_add_(0, positions, values[valid])
        counts.index_add_(0, positions, torch.ones(len(positions)))
        out = torch.full_like(sums, float("nan"))
        out[counts > 0] = sums[counts > 0] / counts[counts > 0]
        return out

    def _load_static_edge_attr(self):
        features = torch.load(self.dir / "Edges" / "edge_features.pt", map_location="cpu", weights_only=False)
        features = {str(k): v for k, v in features.items()}
        columns = [torch.nan_to_num(self._edge_values(features[key])) if key in features
                   else torch.zeros(len(self._edge_key)) for key in ROAD_FLAG_KEYS]
        length = self._edge_values(features["length"])
        if (length[torch.isfinite(length)] < 0).any():
            raise ValueError("Road lengths must be nonnegative")
        columns.append(self._standardize_static(torch.log1p(length).view(-1, 1)).squeeze(1))
        return torch.stack(columns, dim=1)

    def _load_accidents(self):
        acc = pd.read_csv(self.dir / "accidents_monthly.csv")
        a, b = acc["node_1_idx"].to_numpy(), acc["node_2_idx"].to_numpy()
        key = np.minimum(a, b) * self.num_nodes + np.maximum(a, b)
        pos = np.clip(np.searchsorted(self._road_key, key), 0, self.num_roads - 1)
        found = self._road_key[pos] == key
        if not found.all():
            print(f"Warning: {(~found).sum()} accident rows don't match a road; dropped")
        acc = pd.DataFrame({"year": acc["year"].to_numpy()[found], "month": acc["month"].to_numpy()[found],
            "road": pos[found], "count": acc["acc_count"].to_numpy()[found]})
        if (acc["count"] < 0).any():
            raise ValueError("Accident counts must be nonnegative")
        acc = acc.groupby(["year", "month", "road"], as_index=False)["count"].sum()
        return {(int(y), int(m)): (g["road"].to_numpy(), g["count"].to_numpy())
                for (y, m), g in acc.groupby(["year", "month"])}

    @lru_cache(maxsize=None)
    def _raw_weather(self, year, month):
        return torch.from_numpy(np.array(self._node_cache["weather"][self._month_position[(year, month)]]))

    @lru_cache(maxsize=None)
    def _raw_traffic(self, year):
        earlier = [y for y in self._traffic_files if y <= year]
        if not earlier:
            return torch.full((len(self._edge_key), 1), float("nan"))
        path = self._traffic_files[max(earlier)]
        raw = self._edge_values(torch.load(path, map_location="cpu", weights_only=False)["AADT"])
        if (raw[torch.isfinite(raw)] < 0).any():
            raise ValueError("AADT must be nonnegative")
        return torch.log1p(raw).view(-1, 1)

    @staticmethod
    def _train_stats(values):
        finite = torch.isfinite(values)
        count = finite.sum(dim=0).clamp(min=1)
        mean = torch.where(finite, values, 0).sum(dim=0) / count
        variance = torch.where(finite, values - mean, 0).square().sum(dim=0) / count
        std = variance.sqrt()
        return mean, torch.where(std > 0, std, torch.ones_like(std))

    @classmethod
    def _standardize_static(cls, values):
        mean, std = cls._train_stats(values)
        return torch.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
