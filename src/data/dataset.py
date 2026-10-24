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

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DATAVERSE_URL = "https://dataverse.harvard.edu"
DATAVERSE_DOI = "doi:10.7910/DVN/V71K5R"
CENTRALITY_URL = ("https://raw.githubusercontent.com/VirtuosoResearch/ML4RoadSafety/"
                  "main/ml_for_road_safety/embeddings/centrality")

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


def download_centrality(state, data_dir=DATA_DIR):
    """Download the paper's precomputed degree/betweenness node features for a state."""
    dest = Path(data_dir) / "embeddings" / "centrality" / f"{state}.npy"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading centrality features for {state}...")
        urllib.request.urlretrieve(f"{CENTRALITY_URL}/{state}.npy", dest)
    return dest


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
    """Monthly road-accident data for one state.

    Normalisation statistics for weather and traffic volume are computed from
    `train_years` only, so validation and test months don't leak into training.
    """

    def __init__(self, state, train_years, data_dir=DATA_DIR,
                 node_embedding_path=None, use_traffic_volume=False, download=True):
        self.state = state
        self.dir = Path(data_dir) / state
        if download:
            download_state(state, data_dir)

        self.use_traffic_volume = use_traffic_volume

        adj = torch.load(self.dir / "adj_matrix.pt", weights_only=False).coalesce()
        self.num_nodes = adj.shape[0]
        self.edge_index = adj.indices().long()
        src, dst = self.edge_index.numpy()
        self._edge_key = src * self.num_nodes + dst

        road_key = np.minimum(src, dst) * self.num_nodes + np.maximum(src, dst)
        self._road_key, road_edge = np.unique(road_key, return_index=True)
        self.road_edge = torch.from_numpy(road_edge).long()
        self.num_roads = len(self._road_key)

        self.static_edge_attr = self._load_static_edge_attr()

        self.node_embeddings = None
        if node_embedding_path is not None:
            embeddings = torch.from_numpy(np.load(node_embedding_path)).float()
            if embeddings.shape[0] != self.num_nodes:
                raise ValueError(f"Embeddings have {embeddings.shape[0]} rows, graph has {self.num_nodes} nodes")
            mean, std = self._train_stats(embeddings)
            self.node_embeddings = (embeddings - mean) / std

        self._accidents = self._load_accidents()
        self.available_months = sorted(
            (int(y), int(m)) for y, m in
            (f.stem.split("_")[-2:] for f in (self.dir / "Nodes").glob("node_features_*_*.csv"))
        )

        self.train_years = list(train_years)
        self._weather_mean, self._weather_std = self._train_stats(
            torch.cat([self._raw_weather(y, m) for y, m in self.available_months if y in self.train_years])
        )
        if use_traffic_volume:
            self._traffic_mean, self._traffic_std = self._train_stats(
                torch.cat([self._raw_traffic(y) for y in self.train_years])
            )

    def months(self, years):
        return [(y, m) for y, m in self.available_months if y in set(years)]

    def monthly(self, year, month):
        if (year, month) not in self.available_months:
            raise ValueError(f"No weather data for {self.state} {year}-{month:02d}")

        weather = (self._raw_weather(year, month) - self._weather_mean) / self._weather_std
        x = weather if self.node_embeddings is None else torch.cat([self.node_embeddings, weather], dim=1)

        edge_attr = self.static_edge_attr
        if self.use_traffic_volume:
            traffic = (self._raw_traffic(year) - self._traffic_mean) / self._traffic_std
            edge_attr = torch.cat([edge_attr, traffic], dim=1)

        y = torch.zeros(self.num_roads)
        road_ids, counts = self._accidents.get((year, month), (np.array([], dtype=np.int64), np.array([])))
        y[torch.from_numpy(road_ids)] = torch.from_numpy(counts).float()

        return MonthlyData(year, month, x, self.edge_index, edge_attr, self.road_edge, y)

    def _edge_values(self, sparse):
        """Align a sparse [N, N] tensor's values to `edge_index` order (NaN where absent)."""
        sparse = sparse.coalesce()
        s, d = sparse.indices().numpy()
        pos = np.searchsorted(self._edge_key, s * self.num_nodes + d)
        pos = np.clip(pos, 0, len(self._edge_key) - 1)
        found = self._edge_key[pos] == s * self.num_nodes + d
        out = torch.full((len(self._edge_key),), float("nan"))
        out[torch.from_numpy(pos[found])] = sparse.values().float()[torch.from_numpy(found)]
        return out

    def _load_static_edge_attr(self):
        features = torch.load(self.dir / "Edges" / "edge_features.pt", weights_only=False)
        features = {str(k): v for k, v in features.items()}

        columns = []
        for key in ROAD_FLAG_KEYS:
            if key in features:
                columns.append(torch.nan_to_num(self._edge_values(features[key]), nan=0.0))
            else:
                columns.append(torch.zeros(len(self._edge_key)))

        length = self._edge_values(features["length"])
        length = torch.where(torch.isnan(length), length.nanmean(), length)
        columns.append((length - length.mean()) / length.std())
        return torch.stack(columns, dim=1)

    def _load_accidents(self):
        acc = pd.read_csv(self.dir / "accidents_monthly.csv")
        a, b = acc["node_1_idx"].to_numpy(), acc["node_2_idx"].to_numpy()
        key = np.minimum(a, b) * self.num_nodes + np.maximum(a, b)
        pos = np.clip(np.searchsorted(self._road_key, key), 0, self.num_roads - 1)
        found = self._road_key[pos] == key
        if not found.all():
            print(f"Warning: {(~found).sum()} of {len(acc)} accident rows don't match a road; dropped")

        acc = pd.DataFrame({
            "year": acc["year"].to_numpy()[found],
            "month": acc["month"].to_numpy()[found],
            "road": pos[found],
            "count": acc["acc_count"].to_numpy()[found],
        }).groupby(["year", "month", "road"], as_index=False)["count"].sum()

        return {
            (int(y), int(m)): (g["road"].to_numpy(), g["count"].to_numpy())
            for (y, m), g in acc.groupby(["year", "month"])
        }

    @lru_cache(maxsize=None)
    def _raw_weather(self, year, month):
        df = pd.read_csv(self.dir / "Nodes" / f"node_features_{year}_{month}.csv")[WEATHER_COLUMNS]
        df = df.fillna(df.mean()).fillna(0)
        return torch.tensor(df.to_numpy(), dtype=torch.float32)

    @lru_cache(maxsize=None)
    def _raw_traffic(self, year):
        path = self.dir / "Edges" / f"edge_features_traffic_{year}.pt"
        if not path.exists():
            return torch.zeros(len(self._edge_key), 1)
        aadt = self._edge_values(torch.load(path, weights_only=False)["AADT"])
        mean = aadt.nanmean()
        aadt = torch.where(torch.isnan(aadt), 0.0 if torch.isnan(mean) else mean, aadt)
        return aadt.view(-1, 1)

    @staticmethod
    def _train_stats(values):
        mean, std = values.mean(dim=0), values.std(dim=0)
        std[std == 0] = 1
        return mean, std
