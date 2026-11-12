"""OSM preparation methods extracted from osm_features_and_eda.ipynb.

No notebook setup, package installation, Drive mounting or experiments execute on
import. Geospatial dependencies are optional until a geometry method is called.
Defaults preserve pipeline 3.1, including its unit-less-speed-as-mph assumption.
Use prepare_features explicitly to build cached features; source methods and
their limitations are documented in docs/osm_integration.md.
"""
import ast, gc, importlib, json, os, re, shutil, subprocess, sys, time, warnings, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from src.reporting import apply_report_style as apply_chart_style
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import chi2, norm
from IPython.display import Markdown, display

class _OptionalModule:
    def __init__(self, name):
        self.name = name
    def __getattr__(self, attribute):
        try:
            return getattr(importlib.import_module(self.name), attribute)
        except ImportError as exc:
            raise ImportError("OSM rebuilding needs requirements-osm.txt; saved report evidence does not.") from exc

gpd = _OptionalModule("geopandas")
shapely = _OptionalModule("shapely")
def STRtree(*args, **kwargs):
    return shapely.STRtree(*args, **kwargs)
def shape(*args, **kwargs):
    return importlib.import_module("shapely.geometry").shape(*args, **kwargs)

FIGURE_DIR = None
ENVIRONMENT = {}

# Source: osm_features_and_eda.ipynb, cell 4 (zero-based).
STATES = ["DE"]

FORCE_REBUILD = False

PIPELINE_VERSION = "3.1"

USABLE_MONTHS = {"DE": ("2009-01", "2022-09")}

SPLIT = {
    "DE": {"train": ("2009-01", "2012-12"), "val": ("2013-01", "2017-12"), "test": ("2018-01", "2022-09")},
    "MD": {"train": ("2015-01", "2017-12"), "val": ("2018-01", "2019-12"), "test": ("2020-01", "2022-12")},
}

HEADLINE_WINDOW = "full"

NODE_SNAP_MAX_M = 20

NODE_ID_COORD_TOL_M = 25

ROAD_NEAREST_MAX_M = 30

ROAD_LEN_RATIO_RANGE = (0.6, 1.6)

MIDBLOCK_NODE_MAX_M = 15

PLACE_RADII_M = (100, 250, 500, 1000)

RARE_CATEGORY_MIN_ROADS = 50

IMPUTE_FEATURES = {"road_character_maxspeed_mph": "speed", "road_character_lanes": "lanes", "road_character_surface": "surface"}

IMPUTE_FOLDS = 5

IMPUTE_MIN_GAIN = 0.05

IMPUTE_MIN_ACCURACY = 0.60

IMPUTE_NEIGHBOUR_EXTRA = 0.01

SPEED_TOLERANCE_MPH = 5

MIN_ACCIDENTS_FOR_VERDICT = 100

MIN_EFFECT_RATIO = 1.25

MIN_ROADS_FOR_ROAD_TYPE = 100

RANDOM_SEED = 0


# Source: osm_features_and_eda.ipynb, cell 6 (zero-based).
STATE_COLORS = {"DE": "#2a78d6", "MD": "#eb6834"}

SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID_COLOR = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

SEQ_CMAP = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)

def save_figure(fig, name, figure_dir=None):
    """Save a figure as a PNG in `figure_dir` (created if needed); does nothing if `figure_dir` is None."""
    figure_dir = FIGURE_DIR if figure_dir is None else figure_dir
    if figure_dir is None:
        return
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figure_dir) / f"{name}.png", dpi=200, bbox_inches="tight")


# Source: osm_features_and_eda.ipynb, cell 10 (zero-based).
def environment_record():
    """Versions of the libraries that matter for reproducing the numbers."""
    import statsmodels, scipy, matplotlib as mpl
    record = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
              "statsmodels": statsmodels.__version__, "matplotlib": mpl.__version__}
    for name in ("osmnx", "geopandas", "shapely", "pyarrow"):
        try:
            record[name] = importlib.import_module(name).__version__
        except Exception:
            record[name] = "not installed"
    return record


# Source: osm_features_and_eda.ipynb, cell 12 (zero-based).
def as_text_list(value):
    """Turn an OSM tag value into a list of lowercase strings.

    OSM tags can hold one value ("25 mph"), several joined by osmnx into a list (["25 mph", "30 mph"]),
    or a string that looks like a list. Missing values give an empty list.
    """
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(v).strip().lower() for v in value]
    if value is None or pd.isna(value):
        return []
    text = str(value).strip().lower()
    if text.startswith("["):
        try:
            return [str(v).strip().lower() for v in ast.literal_eval(text)]
        except (ValueError, SyntaxError):
            pass
    return [text]

def parse_maxspeed_mph(value):
    """Speed limit in mph. Takes the highest if several; unit-less numbers are read as mph; km/h is converted.
    Words such as "none" or "signals" give NaN.

    Note: the OSM wiki (Key:maxspeed) says a value without a unit is km/h. We read it as mph because US limits are posted in mph;
    Stage 7 measures how many tags this affects and whether their values look like mph limits (assumption A10 in Stage 11).
    """
    speeds = []
    for item in as_text_list(value):
        match = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?", item)
        if match:
            number, unit = float(match.group(1)), match.group(2)
            speeds.append(number / 1.609344 if unit in ("km/h", "kmh", "kph") else number)
    return max(speeds) if speeds else np.nan

def parse_lanes(value):
    """Number of lanes. Takes the highest number if several are listed; missing gives NaN."""
    numbers = [float(n) for item in as_text_list(value) for n in re.findall(r"\d+(?:\.\d+)?", item)]
    return max(numbers) if numbers else np.nan

def first_text(value):
    """First value of a text tag, lowercase; missing gives NaN."""
    items = as_text_list(value)
    return items[0] if items else np.nan

def tag_is_set(value):
    """1.0 if the tag is present and not a "no"-type value (e.g. bridge=yes or bridge=viaduct), else 0.0."""
    items = as_text_list(value)
    return float(any(item not in ("no", "none", "false", "0", "") for item in items))

NO_VALUES = ("no", "none", "false", "0", "")

def yes_no_or_untagged(items):
    """From a list of tag values: 1.0 if any says the thing is present, 0.0 if all say no, NaN if untagged."""
    if not items:
        return np.nan
    return float(any(item not in NO_VALUES for item in items))

def road_tag_values(osm_roads, name):
    """One OSM road tag as a column; if no road has the tag at all, an empty column (with a notice)."""
    if name not in osm_roads.columns:
        print(f"notice: no OSM road has the tag '{name}'")
        return pd.Series(np.nan, index=range(len(osm_roads)), dtype=object)
    return osm_roads[name].reset_index(drop=True)

def node_tag_values(osm_nodes, name):
    """One OSM node tag as a column indexed by OSM node id; empty if no node has the tag."""
    if name not in osm_nodes.columns:
        print(f"notice: no OSM node has the tag '{name}'")
        return pd.Series(np.nan, index=osm_nodes.index, dtype=object)
    return osm_nodes[name]

def single_surface(value):
    """One surface for a road. A merged road whose OSM pieces have different surfaces is "mixed".

    Why: osmnx's simplification merges consecutive OSM pieces into one road and keeps their different tag values as a list
    built from a set, whose order can change between runs. Taking the first value (as v7 did) made the surface of such roads
    depend on the run. "mixed" is what OSM actually says about them, and it does not depend on order (assumption A17).
    """
    items = sorted(set(as_text_list(value)))
    if not items:
        return np.nan
    return items[0] if len(items) == 1 else "mixed"

def most_common_value(values):
    """One value per row for a tag that can hold a list (e.g. the OSM road's `highway`), independent of the list's order.

    Rows with one value keep it. Rows with several take the one most common in the state among single-valued rows
    (ties: alphabetical). Takes a Series of tag values; returns a Series of single values (NaN where untagged).
    """
    lists = values.map(lambda value: sorted(set(as_text_list(value))))
    counts = pd.Series([items[0] for items in lists if len(items) == 1], dtype=object).value_counts()
    return lists.map(lambda items: np.nan if not items else sorted(items, key=lambda item: (-counts.get(item, 0), item))[0])


# Source: osm_features_and_eda.ipynb, cell 13 (zero-based).
EARTH_RADIUS_M = 6371000.0

def straight_line_distance_m(lat1, lon1, lat2, lon2):
    """Distance in metres along the Earth's surface between two lat/lon points (haversine). Works on numbers or arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))

_SPATIAL_INDEX = {}

def nearest_osm_node(osm_nodes, x, y):
    """The OSM node closest to each point, found with a KD-tree on the node positions.

    Does what osmnx's nearest_nodes does, but needs only the node table, not the (very large) graph.
    Takes `osm_nodes` (positions in metres) and the points' x and y in the same metre coordinates.
    Returns (OSM node ids, distances in metres), one per point.
    """
    key = ("nodes", id(osm_nodes), len(osm_nodes))
    if key not in _SPATIAL_INDEX:
        _SPATIAL_INDEX[key] = cKDTree(np.column_stack([osm_nodes.geometry.x.to_numpy(), osm_nodes.geometry.y.to_numpy()]))
    distance, position = _SPATIAL_INDEX[key].query(np.column_stack([x, y]))
    return osm_nodes.index.to_numpy()[position], distance

def nearest_osm_road(osm_roads, x, y):
    """The OSM road closest to each point, found with a spatial index on the road shapes.

    Does what osmnx's nearest_edges does, but needs only the road table. The two directions of a two-way road
    have the same shape, so one of them is returned.
    Returns (position of the closest road in `osm_roads`, distance in metres), one per point.
    """
    key = ("roads", id(osm_roads), len(osm_roads))
    if key not in _SPATIAL_INDEX:
        _SPATIAL_INDEX[key] = STRtree(osm_roads.geometry.to_numpy())
    pairs, distance = _SPATIAL_INDEX[key].query_nearest(shapely.points(x, y), return_distance=True, all_matches=False)
    position = np.full(len(x), -1)
    nearest_distance = np.full(len(x), np.nan)
    position[pairs[0]] = pairs[1]  # row 0: the point, row 1: the closest road
    nearest_distance[pairs[0]] = distance
    return position, nearest_distance


# Source: osm_features_and_eda.ipynb, cell 14 (zero-based).
def poisson_interval(k, exposure):
    """Exact 95% Poisson interval for a rate: k events over `exposure` (arrays or numbers). Returns (low, high)."""
    k = np.asarray(k, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    low = np.where(k > 0, chi2.ppf(0.025, 2 * np.maximum(k, 1)) / 2 / exposure, 0.0)
    high = chi2.ppf(0.975, 2 * k + 2) / 2 / exposure
    return low, high

def weighted_trend(positions, log_ratio, se_log):
    """Straight-line trend of log(rate ratio) against the level's position (0, 1, 2, ...), weighted by 1/se^2.

    Why: to say in words whether the rate rises or falls as a feature's value increases, without reading every bar.
    Returns (slope per step as a ratio, p-value). The slope is exp(b), so 1.2 means +20% per step.
    Needs at least 3 levels with a usable standard error; otherwise returns (nan, nan).
    """
    x = np.asarray(positions, dtype=float)
    y = np.asarray(log_ratio, dtype=float)
    se = np.asarray(se_log, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se) & (se > 0)
    if ok.sum() < 3:
        return np.nan, np.nan
    x, y, w = x[ok], y[ok], 1 / se[ok] ** 2
    x_bar = np.sum(w * x) / np.sum(w)                     # weighted mean position
    sxx = np.sum(w * (x - x_bar) ** 2)                    # weighted sum of squares of the positions
    if sxx <= 0:
        return np.nan, np.nan
    slope = np.sum(w * (x - x_bar) * (y - np.sum(w * y) / np.sum(w))) / sxx
    # residual-based standard error (does not assume the reported se are exactly right)
    intercept = np.sum(w * y) / np.sum(w) - slope * x_bar
    residual = y - intercept - slope * x
    dof = ok.sum() - 2                                    # degrees of freedom of the straight-line fit
    scale = max(np.sum(w * residual ** 2) / dof, 1.0)   # never smaller than the reported precision
    slope_se = np.sqrt(scale / sxx)
    return float(np.exp(slope)), float(2 * (1 - norm.cdf(abs(slope / slope_se))))


# Source: osm_features_and_eda.ipynb, cell 15 (zero-based).
def show_markdown(text):
    """Show text as formatted Markdown in the notebook."""
    display(Markdown(text))

def format_percent(value, digits=1):
    """0.1234 -> '12.3%'. NaN -> 'n/a'."""
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:.{digits}f}%"

def memory_report(label):
    """Print how much memory this notebook's process is using right now, to find which stage needs the most."""
    import resource

    import psutil
    now = psutil.Process().memory_info().rss / 1e9
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux reports kB
    print(f"[memory] {label}: now {now:.1f} GB, highest so far {peak:.1f} GB")

def grouped_horizontal_bars(ax, table, states, xlabel, value_format="{:.1%}"):
    """Horizontal bars, one group per row of `table` and one bar per state, with the value at the end of each bar."""
    positions = np.arange(len(table))
    x_max = max(float(np.nanmax(table[list(states)].to_numpy())) * 1.2, 0.05)
    offsets = (-0.2, 0.2) if len(states) == 2 else (0.0,)
    for offset, state in zip(offsets, states):
        ax.barh(positions + offset, table[state].to_numpy(), height=0.34, color=STATE_COLORS[state], label=state)
        for y, value in zip(positions + offset, table[state].to_numpy()):
            ax.text(value + 0.012 * x_max, y, value_format.format(value), va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(positions)
    ax.set_yticklabels(table.index)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True)

def table_to_markdown(table, float_format="{:,.3g}"):
    """A small DataFrame as a Markdown table (so wide text wraps nicely in the notebook)."""
    index_name = " / ".join(str(n) for n in table.index.names if n) if table.index.nlevels > 1 else str(table.index.name or "")
    columns = [index_name] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for index, row in table.iterrows():
        cells = [" / ".join(str(part) for part in index) if isinstance(index, tuple) else str(index)]
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    cells.append("")
                elif float(value).is_integer() and abs(value) >= 1000:   # counts: 116,196, not 1.16e+05
                    cells.append(f"{value:,.0f}")
                else:
                    cells.append(float_format.format(value))
            elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value).replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Source: osm_features_and_eda.ipynb, cell 19 (zero-based).
STATE_NAMES = {"DE": "delaware", "MD": "maryland"}

ROAD_HIGHWAY_VALUES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                       "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
                       "residential", "living_street", "road"]

PLACE_AMENITY_VALUES = ["school", "bar", "pub", "restaurant", "fast_food", "parking"]

PLACE_HIGHWAY_VALUES = ["bus_stop"]

def run_osmium(arguments):
    if shutil.which("osmium") is None:
        raise RuntimeError("Install osmium-tool before rebuilding OSM features (Colab/Linux/WSL).")
    subprocess.run(["osmium", *map(str, arguments)], check=True)


def find_state_pbf(state, data_dir):
    """The Geofabrik .pbf for a state in the Drive data folder (e.g. delaware-261018.osm.pbf).

    If several dated files exist, the latest date is used.
    """
    candidates = sorted(data_dir.glob(f"{STATE_NAMES[state]}-*.osm.pbf"))
    if not candidates:
        raise FileNotFoundError(f"No {STATE_NAMES[state]}-*.osm.pbf in {data_dir}. Download it from Geofabrik first.")
    print("using", candidates[-1].name)
    return candidates[-1]

def prepare_osm_roads_zip(state, data_dir, local_dir, overwrite=False):
    """Make `{state}_osm_roads.zip` in the Drive data folder from the state's Geofabrik .pbf.

    Keeps only the dataset's road types (with the points they use and those points' tags) and writes them as .osm XML,
    which osmnx can read. If the zip already exists and overwrite is False, nothing happens. Returns the zip's path.
    """
    zip_path = data_dir / f"{state}_osm_roads.zip"
    if zip_path.exists() and not overwrite:
        print(f"{zip_path.name} already exists; skipping")
        return zip_path

    pbf = find_state_pbf(state, data_dir)
    filtered = local_dir / f"{state}_roads.osm.pbf"
    xml = local_dir / f"{state}_roads.osm"

    # Keep only ways tagged with one of the road types (osmium keeps the nodes they use by default).
    run_osmium(["tags-filter", pbf, "w/highway=" + ",".join(ROAD_HIGHWAY_VALUES), "-o", filtered, "--overwrite"])
    # Convert to XML, the format osmnx reads.
    run_osmium(["cat", filtered, "-o", xml, "--overwrite"])

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(xml, arcname=xml.name)
    for path in (filtered, xml, zip_path):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
    return zip_path

def prepare_osm_places_zip(state, data_dir, local_dir, overwrite=False):
    """Make `{state}_osm_places.zip` in the Drive data folder: schools, bars, restaurants, bus stops, parking.

    Places can be points or areas (a school is often an outline), so they are exported as GeoJSON and the centre of
    each is used later. If the zip already exists and overwrite is False, nothing happens. Returns the zip's path.
    """
    zip_path = data_dir / f"{state}_osm_places.zip"
    if zip_path.exists() and not overwrite:
        print(f"{zip_path.name} already exists; skipping")
        return zip_path

    pbf = find_state_pbf(state, data_dir)
    filtered = local_dir / f"{state}_places.osm.pbf"
    geojson = local_dir / f"{state}_places.geojsonseq"

    # Keep nodes, ways and relations (nwr) with any of the place tags.
    run_osmium(["tags-filter", pbf, "nwr/amenity=" + ",".join(PLACE_AMENITY_VALUES),
                "nwr/highway=" + ",".join(PLACE_HIGHWAY_VALUES), "-o", filtered, "--overwrite"])
    # One GeoJSON feature per line; areas become polygons.
    run_osmium(["export", filtered, "-f", "geojsonseq", "-o", geojson, "--overwrite"])

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(geojson, arcname=geojson.name)
    for path in (filtered, geojson, zip_path):
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
    return zip_path

def load_osm_drive_network_from_zip(state, data_dir, local_dir):
    """Build the OSM road network for a state from the zip saved on Drive.

    Returns `osm_graph` (osmnx graph in metres), `osm_nodes` (one row per OSM node with its tags) and
    `osm_roads` (one row per directed OSM road, index = start node, end node, key).
    """
    import osmnx as ox
    print("osmnx version:", ox.__version__)

    # Unpack the XML into local storage, unless it is already there (utils.read_data may have done it).
    xml_path = local_dir / f"{state}_roads.osm"
    if not xml_path.exists():
        with zipfile.ZipFile(data_dir / f"{state}_osm_roads.zip") as archive:
            archive.extractall(local_dir)

    # Ask for the extra tags BEFORE building; osmnx drops tags it was not told to keep.
    # Keep only the tags our features use: fewer tags means a smaller graph in memory.
    ox.settings.useful_tags_way = ["highway", "maxspeed", "lanes", "surface", "bridge", "tunnel", "junction", "oneway",
                                   "lit", "sidewalk", "cycleway", "cycleway:left", "cycleway:right", "cycleway:both"]
    ox.settings.useful_tags_node = ["highway", "crossing", "traffic_calming"]

    # simplify=True merges road bends so a road runs between intersections.
    # retain_all=True keeps disconnected pieces of the network instead of only the largest one.
    osm_graph = ox.graph_from_xml(xml_path, simplify=True, retain_all=True)

    # Convert coordinates to metres so distances and thresholds are in metres.
    osm_graph = ox.project_graph(osm_graph)

    osm_nodes, osm_roads = ox.graph_to_gdfs(osm_graph)
    return osm_graph, osm_nodes, osm_roads

def build_osm_tables(state, data_dir, local_dir):
    """Build the OSM nodes and roads tables in a separate process and save them, so the big graph never stays in memory.

    The work runs in a child process; when it exits, all its memory goes back to the system.
    IMPORTANT: run this BEFORE loading the dataset (utils.read_data), so the child has the memory to itself.
    Saves `{state}_osm_nodes.pkl` and `{state}_osm_roads.pkl` in the local folder (skipped if both already exist).
    """
    import multiprocessing

    nodes_path = local_dir / f"{state}_osm_nodes.pkl"
    roads_path = local_dir / f"{state}_osm_roads.pkl"
    if nodes_path.exists() and roads_path.exists():
        print(f"{state} OSM tables already built; skipping")
        return

    prepare_osm_roads_zip(state, data_dir, local_dir)
    memory_report("before the OSM build starts")

    context = "spawn" if os.name == "nt" else "fork"
    process = multiprocessing.get_context(context).Process(
        target=_build_osm_worker, args=(state, data_dir, local_dir, nodes_path, roads_path))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"OSM build failed (exit code {process.exitcode}). -9 means it ran out of memory.")
    memory_report("after the OSM build")

def load_osm_tables(state, local_dir):
    """Read the OSM nodes and roads tables saved by `build_osm_tables`. Returns (osm_nodes, osm_roads)."""
    return pd.read_pickle(local_dir / f"{state}_osm_nodes.pkl"), pd.read_pickle(local_dir / f"{state}_osm_roads.pkl")

def saved_feature_columns(state, data_dir):
    """Column names of a state's saved feature file, or None if there is no file."""
    path = data_dir / f"{state}_osm_matched_roads.parquet"
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).schema.names

def saved_features_are_current(state, data_dir):
    """True if the state's saved feature file exists and was built by this PIPELINE_VERSION (an older file is rebuilt)."""
    meta_path = data_dir / f"{state}_osm_matching_meta.json"
    if not (data_dir / f"{state}_osm_matched_roads.parquet").exists() or not meta_path.exists():
        return False
    saved = json.loads(meta_path.read_text()).get("pipeline_version")
    if saved != PIPELINE_VERSION:
        print(f"{state}: saved features are version {saved}, this notebook is {PIPELINE_VERSION}: they will be rebuilt")
        return False
    return True


# Source: osm_features_and_eda.ipynb, cell 23 (zero-based).
def load_dataset_node_and_road_tables(info):
    """Turn the dataset's graph into two plain tables, one row per node and one per road.

    Takes `info`, the dict that `utils.read_data` returns for one state (e.g. data["DE"]).
    Returns:
        nodes: one row per intersection (node_id, lat, lon, node_idx).
        roads: one row per directed road, in the same order as `adj`
               (node_1_idx, node_2_idx, length_m, road_u, road_v, has_reverse, u_lt_v).
    The dataset's lat and lon columns are swapped; this function fixes that.
    """
    # The road network as a sparse matrix. coalesce() puts its entries in a fixed order.
    adj = info["adj"].coalesce()

    # Node files exist for every month; positions are the same in all of them, so take the first.
    nodes = pd.read_csv(next((info["filepath"] / "Nodes").glob("node_features_*.csv")))[["node_id", "lat", "lon"]]

    # The release has the two columns swapped: the column named lat holds longitudes and vice versa.
    nodes = nodes.rename(columns={"lat": "lon", "lon": "lat"})

    # Check the swap is fixed. DE and MD both lie between 37-41 N and 74-80 W.
    # A plain +/-90 check would not catch a swap, because -75 is also a valid latitude.
    assert nodes["lat"].between(37, 41).all() and nodes["lon"].between(-80, -74).all()

    # One row per matrix row, otherwise the node numbering cannot be trusted.
    assert len(nodes) == adj.shape[0]

    # The dataset refers to nodes by position (0, 1, 2, ...). Assumes the file rows are in that order.
    nodes["node_idx"] = np.arange(len(nodes))

    # Start and end node of every directed road, in the matrix's order. Keep this order.
    src, dst = adj.indices().numpy()
    roads = pd.DataFrame({"node_1_idx": src, "node_2_idx": dst})

    # Road lengths are stored in their own sparse matrix; turn it into a start / end / length table.
    length_matrix = info["edges"]["length"].coalesce()
    length_index = length_matrix.indices().numpy()
    length = pd.DataFrame({"node_1_idx": length_index[0], "node_2_idx": length_index[1], "length_m": length_matrix.values().numpy()})

    # Attach each road's length. how="left" keeps every road; the assert checks no rows were added or lost.
    n_before = len(roads)
    roads = roads.merge(length, how="left", on=["node_1_idx", "node_2_idx"])
    assert len(roads) == n_before, "length merge changed the number of roads"

    # Undirected road key: 17->18 and 18->17 both become (17, 18).
    roads["road_u"] = np.minimum(roads.node_1_idx, roads.node_2_idx)
    roads["road_v"] = np.maximum(roads.node_1_idx, roads.node_2_idx)

    # True if the opposite direction also exists, i.e. the road is two-way.
    pair = set(zip(roads.node_1_idx, roads.node_2_idx))
    roads["has_reverse"] = [(b, a) in pair for a, b in zip(roads.node_1_idx, roads.node_2_idx)]

    # True if the start node has the lower index. Crashes mostly sit on this direction.
    roads["u_lt_v"] = roads.node_1_idx < roads.node_2_idx
    return nodes, roads

EXPECTED_LENGTH_RATIO = (1.0, 1.3)

def check_dataset_road_lengths(nodes, roads, verbose=True):
    """Check that the dataset's node order matches its road lengths.

    Roads refer to nodes by position. If the node file were in a different order, every coordinate we attach to a road
    would be wrong, and so would the OSM matching. A road's recorded length should be close to the straight-line distance
    between its ends. Returns the length / straight-line ratio for every road; raises an error if the median is outside
    EXPECTED_LENGTH_RATIO.
    """
    coverage = roads.length_m.notna().mean()
    assert coverage > 0.99, "too many roads without a length"

    lat, lon = nodes.lat.to_numpy(), nodes.lon.to_numpy()
    start, end = roads.node_1_idx.to_numpy(), roads.node_2_idx.to_numpy()
    straight = straight_line_distance_m(lat[start], lon[start], lat[end], lon[end])
    ratio = roads.length_m.to_numpy() / np.where(straight > 0, straight, np.nan)

    median = np.nanmedian(ratio)
    if verbose:
        print(f"roads with a length: {coverage:.4f}; median length / straight-line: {median:.3f} (expected {EXPECTED_LENGTH_RATIO})")
    if not (EXPECTED_LENGTH_RATIO[0] <= median <= EXPECTED_LENGTH_RATIO[1]):
        raise RuntimeError("Node order looks wrong: STOP and ask before continuing.")
    return ratio

def dataset_highway_class(info, roads):
    """The dataset's own road type for every road, as a text label.

    The dataset stores road type as 14 separate sparse matrices (one per type). We want one label per road so we can
    group results by road type. Returns a Series aligned with `roads` ("motorway", "residential", ...; "none" if no type
    was found). If a road has several types the last one found is used for this label, so the label is only a convenience for
    matching checks and gap-filling (assumption A8); roads with several types are EXCLUDED from the analysis (Stage 12), which uses `.attrs["multi_type"]`.
    The counts of roads with no type or several types are kept in the Series' `.attrs` (`n_no_type`, `n_several_types`).
    """
    n_nodes = info["adj"].shape[0]
    road_key = roads["node_1_idx"].to_numpy().astype(np.int64) * n_nodes + roads["node_2_idx"].to_numpy()
    order = np.argsort(road_key)
    sorted_keys = road_key[order]

    label = np.full(len(roads), "none", dtype=object)
    n_types = np.zeros(len(roads), dtype=int)
    for name, matrix in info["edges"].items():
        if name in ("oneway", "length"):
            continue  # not road types
        matrix = matrix.coalesce()
        index, value = matrix.indices().numpy(), matrix.values().numpy()
        keys = index[0].astype(np.int64)[value > 0] * n_nodes + index[1][value > 0]

        # Find each key among the roads (a key that is not a road is skipped).
        position = np.minimum(np.searchsorted(sorted_keys, keys), len(roads) - 1)
        found = sorted_keys[position] == keys
        rows = order[position[found]]
        label[rows] = str(name)
        n_types[rows] += 1

    result = pd.Series(label, index=roads.index, name="highway_class")
    result.attrs["n_no_type"] = int((n_types == 0).sum())
    result.attrs["n_several_types"] = int((n_types > 1).sum())
    result.attrs["multi_type"] = (n_types > 1)   # boolean array aligned with `roads`, used to leave these roads out of the trend analysis
    return result

def accident_totals_in_period(info, roads, first_month, last_month):
    """Total accidents on each directed road between two months (inclusive), e.g. "2015-01" to "2017-12".

    Every label-based number goes through this function, so the window is always explicit.
    Returns (an array aligned with `roads`, the length of the period in years).
    """
    accidents = info["accidents"]
    month_start = pd.to_datetime(accidents[["year", "month"]].assign(day=1))
    in_period = (month_start >= pd.Timestamp(first_month + "-01")) & (month_start <= pd.Timestamp(last_month + "-01"))

    totals = accidents[in_period].groupby(["node_1_idx", "node_2_idx"])["acc_count"].sum().rename("acc_total").reset_index()
    merged = roads[["node_1_idx", "node_2_idx"]].merge(totals, how="left", on=["node_1_idx", "node_2_idx"])
    assert len(merged) == len(roads), "accident merge changed the number of roads"

    n_months = (pd.Period(last_month, "M") - pd.Period(first_month, "M")).n + 1
    return merged["acc_total"].fillna(0).to_numpy(), n_months / 12

def monthly_accident_totals(info):
    """Total accidents per calendar month (index = first day of the month)."""
    accidents = info["accidents"]
    month_start = pd.to_datetime(accidents[["year", "month"]].assign(day=1))
    return accidents.groupby(month_start)["acc_count"].sum().sort_index()

def read_benchmark(drive_path, local_path, states):
    """Load the benchmark for the given states only: the team's `utils.read_data`, restricted to `states`.

    Why: `utils.read_data` always loads both Delaware and Maryland and unpacks every zip in the data folder, so it fails
    when one state's zip is absent (here MD.zip) and wastes time on the other state's OSM zips. This does the same steps,
    with the same file names and the same result for each state, but only for `states`. The team's utils.py is not changed.
    Returns {state: {"filepath", "adj", "edges", "edge_features_traffic", "accidents"}}.
    """
    import torch
    data = {}
    for state in states:
        zip_path = Path(drive_path) / f"{state}.zip"
        if not (Path(local_path) / state / "adj_matrix.pt").exists():
            if not zip_path.exists():
                raise FileNotFoundError(f"{zip_path} not found: put the benchmark zip for {state} in the Drive data folder")
            print(f"Extracting {zip_path.name} to {local_path}...")
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(local_path)
        info = {"filepath": Path(local_path) / state}
        info["adj"] = torch.load(info["filepath"] / "adj_matrix.pt", map_location="cpu", weights_only=False).coalesce()
        info["edges"] = torch.load(info["filepath"] / "Edges" / "edge_features.pt", map_location="cpu", weights_only=False)
        info["edge_features_traffic"] = {int(path.stem.split("_")[-1]): torch.load(path, map_location="cpu", weights_only=False)
                                         for path in sorted((info["filepath"] / "Edges").glob("edge_features_traffic_*.pt"))}
        info["accidents"] = pd.read_csv(info["filepath"] / "accidents_monthly.csv")
        data[state] = info
        print(f"{state}: loaded from {info['filepath']}")
    return data


# Source: osm_features_and_eda.ipynb, cell 29 (zero-based).
def detect_partial_months(monthly, window_months=12, threshold=0.5):
    """Months whose total is below `threshold` times the median of the other months within `window_months` either side."""
    flagged = []
    for month, value in monthly.items():
        near = monthly.drop(month)
        near = near[(near.index >= month - pd.DateOffset(months=window_months)) & (near.index <= month + pd.DateOffset(months=window_months))]
        if len(near) >= 3 and value < threshold * near.median():
            flagged.append(month)
    return flagged

def configured_excluded_months(monthly, state):
    """Months present in the data but outside USABLE_MONTHS[state] (all months are usable if the state has no entry)."""
    if state not in USABLE_MONTHS:
        return []
    first, last = (pd.Timestamp(m + "-01") for m in USABLE_MONTHS[state])
    return [m for m in monthly.index if m < first or m > last]

def direction_evidence(roads, totals):
    """How one-sided the accident records are on two-way roads.

    Returns a dict with: the share of two-way roads with accidents whose accidents sit on only one of the two directions,
    and the share of two-way-road accidents recorded on the direction going from the lower to the higher node index.
    """
    two_way = roads["has_reverse"].to_numpy()
    frame = pd.DataFrame({"u": roads["road_u"].to_numpy()[two_way], "v": roads["road_v"].to_numpy()[two_way],
                          "acc": totals[two_way], "lower_to_higher": roads["u_lt_v"].to_numpy()[two_way]})
    per_road = frame.groupby(["u", "v"]).agg(total=("acc", "sum"), sides=("acc", lambda s: int((s > 0).sum())))
    with_accidents = per_road[per_road["total"] > 0]
    return {"one_sided_share": float((with_accidents["sides"] == 1).mean()) if len(with_accidents) else np.nan,
            "lower_to_higher_share": float(frame.loc[frame["lower_to_higher"], "acc"].sum() / max(frame["acc"].sum(), 1)),
            "two_way_roads_with_accidents": int(len(with_accidents))}

def coordinate_swap_evidence(info):
    """Share of nodes whose raw `lat` column holds a longitude (-80 to -74) and raw `lon` holds a latitude (37 to 41)."""
    raw = pd.read_csv(next((info["filepath"] / "Nodes").glob("node_features_*.csv")))[["lat", "lon"]]
    return float((raw["lat"].between(-80, -74) & raw["lon"].between(37, 41)).mean())

def excluded_month_rows(info, excluded):
    """How many accident rows, and accidents, fall in the excluded months. Returns (rows, accidents)."""
    accidents = info["accidents"]
    month_start = pd.to_datetime(accidents[["year", "month"]].assign(day=1))
    in_excluded = month_start.isin(excluded)
    return int(in_excluded.sum()), float(accidents.loc[in_excluded, "acc_count"].sum())

def build_cleaning_log(state, info, nodes, roads, length_ratio, highway_class, monthly, flagged):
    """One row per cleaning or checking decision on the benchmark: its type, what we did, why, the measured evidence,
    how much data it touched, and its effect.

    Type is "change" (alters the data or what the analysis uses), "check" (verifies only; the run stops if it fails) or "rule" (a convention).
    Returns a DataFrame indexed by the decision.
    """
    excluded = configured_excluded_months(monthly, state)
    full_first, full_last = USABLE_MONTHS.get(state, (monthly.index.min().strftime("%Y-%m"), monthly.index.max().strftime("%Y-%m")))
    totals_all, _ = accident_totals_in_period(info, roads, monthly.index.min().strftime("%Y-%m"), monthly.index.max().strftime("%Y-%m"))
    direction = direction_evidence(roads, totals_all)
    accidents = info["accidents"]
    bad_counts = int((accidents["acc_count"] <= 0).sum() + accidents["acc_count"].isna().sum())
    excluded_rows, excluded_accidents = excluded_month_rows(info, excluded)
    physical_roads = roads.groupby(["road_u", "road_v"]).ngroups
    multi = np.asarray(highway_class.attrs.get("multi_type", np.zeros(len(roads), dtype=bool)))
    multi_accident_share = totals_all[multi].sum() / max(totals_all.sum(), 1)

    def format_months(months):
        return ", ".join(m.strftime("%Y-%m") for m in months) if len(months) <= 6 else f"{len(months)} months ({months[0]:%Y-%m} to {months[-1]:%Y-%m})"

    rows = [
        ("change", "Swap latitude and longitude columns", "The release names the columns the wrong way round; positions were unusable otherwise.",
         f"{format_percent(coordinate_swap_evidence(info))} of nodes have a longitude in the `lat` column. After the swap all nodes fall inside 37-41 N, 74-80 W (asserted).",
         f"all {len(nodes):,} nodes", "All coordinates, matching and place counts depend on it."),
        ("check", "Validate node order", "Roads refer to nodes by row position; a shuffled node file would silently ruin every match.",
         f"Median recorded length / straight-line distance = {np.nanmedian(length_ratio):.3f} (expected {EXPECTED_LENGTH_RATIO[0]}-{EXPECTED_LENGTH_RATIO[1]}); "
         f"{format_percent(roads['length_m'].notna().mean(), 2)} of roads have a length.", "none (check only)", "The run stops if the check fails."),
        ("change", "Drop partial accident months", "Months with a fraction of the usual records would look like a real fall in risk.",
         f"Detector flags: {format_months(flagged) if flagged else 'none'}. Configured exclusions: {format_months(excluded) if excluded else 'none'}. "
         f"Usable window {full_first} to {full_last}.",
         f"{excluded_rows:,} accident rows ({excluded_accidents:,.0f} accidents) in {len(excluded)} month(s)" if excluded else "none",
         "A flagged month inside the window is a real drop (for example spring 2020), not a gap, and is kept."),
        ("change", "Merge the two directions of a two-way road", "Accidents are recorded on one direction only, so the other direction always shows zero.",
         f"Of {direction['two_way_roads_with_accidents']:,} two-way roads with accidents, {format_percent(direction['one_sided_share'])} have accidents on only one direction. "
         f"{format_percent(direction['lower_to_higher_share'])} of two-way-road accidents are on the lower-to-higher-index direction.",
         f"{len(roads):,} directed roads become {physical_roads:,} physical roads (analysis only)",
         "Directed roads are merged into physical roads for all analysis; OSM features are the same in both directions."),
        ("change", "Leave roads with more than one type out of the analysis", "The dataset stores road type as 14 matrices; some roads appear in more than one, and the comparisons need one type per road.",
         f"{highway_class.attrs.get('n_several_types', 0):,} roads have more than one type, {highway_class.attrs.get('n_no_type', 0):,} have none.",
         f"{int(multi.sum()):,} directed roads, {format_percent(multi_accident_share)} of accidents (analysis only)",
         "They stay in the feature table and the benchmark's own type columns are untouched. Matching checks and gap-filling use the last type found as a label (assumption A8)."),
        ("check", "Accident counts are positive", "Zero, negative or missing counts would corrupt rates.",
         f"{bad_counts:,} rows with count <= 0 or missing out of {len(accidents):,}.", f"{bad_counts:,} rows", "None expected; any would be dropped."),
        ("rule", "Missing stays missing", "A road without a speed-limit tag is unknown, not 'no speed limit'.",
         "Tag-based features are NaN when the tag is absent or the road is unmatched. For node flags (signals, stop signs, crossings) 0 means 'not mapped there', which is weaker than 'absent'.",
         "every untagged or unmatched road (measured in Stage 9)", "The analysis uses known values only (Stage 12)."),
        ("rule", "How OSM tags are read", "OSM tags are free text: '25 mph', '35', '80 km/h', 'none', or a list.",
         "Unit-less speeds are read as mph, km/h is converted, the highest value is kept when several are listed, words give missing; a road with several surfaces is 'mixed' (self-test in Stage 1.6).",
         "counted in Stage 7 (OSM tag cleaning)", "Defines what every speed-limit and lane value means."),
    ]
    columns = ["type", "decision", "why", "evidence", "data affected", "effect / note"]
    return pd.DataFrame(rows, columns=columns).set_index("decision")


# Source: osm_features_and_eda.ipynb, cell 32 (zero-based).
def compare_dataset_and_osm_sizes(nodes, roads, osm_nodes, osm_roads):
    """Compare how many nodes and roads the dataset and OpenStreetMap have (ratios near 1 mean similar networks).

    Returns a dict with the four counts and the OSM / dataset ratios.
    """
    sizes = {"dataset_nodes": len(nodes), "osm_nodes": len(osm_nodes), "dataset_roads": len(roads), "osm_roads": len(osm_roads)}
    sizes["node_ratio"] = sizes["osm_nodes"] / sizes["dataset_nodes"]
    sizes["road_ratio"] = sizes["osm_roads"] / sizes["dataset_roads"]
    return sizes

def match_dataset_nodes_to_osm_nodes(nodes, osm_graph, osm_nodes):
    """Find, for every dataset node, the OSM node it corresponds to.

    Takes dataset `nodes`, `osm_graph` (no longer used; may be None) and `osm_nodes`.
    Returns a copy of `nodes` with extra columns:
        id_in_osm: the dataset's node_id exists in OSM.
        id_dist_m: distance between the dataset position and that OSM node (NaN if not in OSM).
        nearest_osm_id, nearest_dist_m: the closest OSM node by position, and how far it is.
        node_match: "id", "snap" (nearest node) or "none".
        osm_id: the matched OSM node id, or -1 when node_match is "none".
    """
    nodes = nodes.copy()

    # Dataset positions in the same metre-based coordinates as the OSM graph.
    points = gpd.GeoSeries(gpd.points_from_xy(nodes.lon, nodes.lat), crs=4326).to_crs(osm_nodes.crs)
    x, y = points.x.to_numpy(), points.y.to_numpy()

    # Nearest OSM node to every dataset node, by position, and the distance to it in metres.
    nearest_id, nearest_dist = nearest_osm_node(osm_nodes, x, y)
    nodes["nearest_osm_id"] = nearest_id
    nodes["nearest_dist_m"] = nearest_dist

    # Does the dataset's own OSM id exist in the OSM file?
    nodes["id_in_osm"] = nodes["node_id"].isin(osm_nodes.index)

    # For those, how far apart are the dataset position and the OSM node with that id? (NaN for ids not in OSM)
    osm_x = osm_nodes.geometry.x.reindex(nodes["node_id"]).to_numpy()
    osm_y = osm_nodes.geometry.y.reindex(nodes["node_id"]).to_numpy()
    nodes["id_dist_m"] = np.hypot(x - osm_x, y - osm_y)

    # Decision: trust the id if the positions agree, else the nearest node if it is close enough.
    id_ok = nodes["id_in_osm"] & (nodes["id_dist_m"] <= NODE_ID_COORD_TOL_M)
    snap_ok = ~id_ok & (nodes["nearest_dist_m"] <= NODE_SNAP_MAX_M)
    nodes["node_match"] = np.select([id_ok, snap_ok], ["id", "snap"], "none")
    nodes["osm_id"] = np.where(id_ok, nodes["node_id"], np.where(snap_ok, nodes["nearest_osm_id"], -1))
    return nodes

def match_dataset_roads_to_osm_roads(nodes, roads, osm_graph, osm_roads):
    """Find, for every dataset road, the OSM road it corresponds to (tiers: exact, reverse, nearest, unmatched).

    Takes `nodes` (with `osm_id` from `match_dataset_nodes_to_osm_nodes`), dataset `roads`, and `osm_roads`.
    Returns a copy of `roads` (same row order) with extra columns:
        osm_u, osm_v: OSM ids of the road's start and end nodes (-1 if the node had no match).
        match_type: "exact", "reverse", "nearest" or "unmatched".
        match_dist_m: 0 for exact and reverse; distance to the OSM road for nearest.
        osm_row: position of the matched road in `osm_roads` (-1 if unmatched). Use it to copy tags.
        osm_length_m: the matched OSM road's length.   len_ratio: osm_length_m / the dataset's length_m.
        osm_highway: the matched OSM road's own `highway` tag (for checking the match; NaN if unmatched).
        cand_dist_m, cand_len_ratio: for roads that reached the nearest tier, the distance and length ratio of the
            nearest candidate even when it was REJECTED. Lets us ask how many more roads a looser threshold would match.
    """
    roads = roads.copy()
    assert (roads.index == np.arange(len(roads))).all(), "roads must keep a default 0..N-1 index"
    n_roads = len(roads)

    # OSM ids of each road's two end nodes (-1 where the node was not matched).
    osm_id = nodes["osm_id"].to_numpy()
    roads["osm_u"] = osm_id[roads["node_1_idx"].to_numpy()]
    roads["osm_v"] = osm_id[roads["node_2_idx"].to_numpy()]

    # The OSM roads as a flat table, remembering each road's position in `osm_roads`.
    osm_table = pd.DataFrame({
        "u": osm_roads.index.get_level_values("u"),
        "v": osm_roads.index.get_level_values("v"),
        "osm_length_m": osm_roads["length"].to_numpy(),
        "osm_row": np.arange(len(osm_roads)),
    })

    # Results, one entry per dataset road.
    match_type = np.full(n_roads, "unmatched", dtype=object)
    osm_row = np.full(n_roads, -1)
    osm_length = np.full(n_roads, np.nan)
    match_dist = np.full(n_roads, np.nan)
    cand_dist = np.full(n_roads, np.nan)
    cand_ratio = np.full(n_roads, np.nan)

    def match_by_node_pair(todo, start_col, end_col):
        """OSM roads running from `start_col` to `end_col`; for parallel roads keep the closest length."""
        pairs = todo[[start_col, end_col, "length_m"]].reset_index(names="road")
        candidates = pairs.merge(osm_table, left_on=[start_col, end_col], right_on=["u", "v"])
        candidates["length_error"] = (candidates["osm_length_m"] - candidates["length_m"]).abs()
        best = candidates.sort_values("length_error").drop_duplicates("road")
        return best["road"].to_numpy(), best["osm_row"].to_numpy(), best["osm_length_m"].to_numpy()

    # Tier 1, exact: same direction as the dataset road.
    rows, found, lengths = match_by_node_pair(roads, "osm_u", "osm_v")
    match_type[rows], osm_row[rows], osm_length[rows], match_dist[rows] = "exact", found, lengths, 0.0

    # Tier 2, reverse: only for roads not yet matched; look for the OSM road running the other way.
    todo = roads[match_type == "unmatched"]
    rows, found, lengths = match_by_node_pair(todo, "osm_v", "osm_u")
    match_type[rows], osm_row[rows], osm_length[rows], match_dist[rows] = "reverse", found, lengths, 0.0

    # Tier 3, nearest: use the road's midpoint, in the same metre coordinates as the OSM graph.
    points = gpd.GeoSeries(gpd.points_from_xy(nodes.lon, nodes.lat), crs=4326).to_crs(osm_roads.crs)
    x, y = points.x.to_numpy(), points.y.to_numpy()
    start, end = roads["node_1_idx"].to_numpy(), roads["node_2_idx"].to_numpy()
    mid_x, mid_y = (x[start] + x[end]) / 2, (y[start] + y[end]) / 2

    left = np.flatnonzero(match_type == "unmatched")
    found, nearest_dist = nearest_osm_road(osm_roads, mid_x[left], mid_y[left])  # position of the closest road in osm_roads
    found_length = osm_roads["length"].to_numpy()[found]
    ratio = found_length / roads["length_m"].to_numpy()[left]
    cand_dist[left], cand_ratio[left] = nearest_dist, ratio                       # kept even for rejected candidates
    accept = (found >= 0) & (nearest_dist <= ROAD_NEAREST_MAX_M) & \
             (ratio >= ROAD_LEN_RATIO_RANGE[0]) & (ratio <= ROAD_LEN_RATIO_RANGE[1])
    rows = left[accept]
    match_type[rows], osm_row[rows] = "nearest", found[accept]
    osm_length[rows], match_dist[rows] = found_length[accept], nearest_dist[accept]

    roads["match_type"] = match_type
    roads["match_dist_m"] = match_dist
    roads["osm_row"] = osm_row
    roads["osm_length_m"] = osm_length
    roads["len_ratio"] = osm_length / np.where(roads["length_m"] > 0, roads["length_m"], np.nan)
    roads["cand_dist_m"] = cand_dist
    roads["cand_len_ratio"] = cand_ratio

    osm_highway = most_common_value(road_tag_values(osm_roads, "highway")).to_numpy(dtype=object)   # independent of list order (A17)
    roads["osm_highway"] = np.where(osm_row >= 0, osm_highway[np.maximum(osm_row, 0)], None)
    return roads

def matching_diagnostics(nodes, roads, sizes):
    """Everything worth keeping about the matching, as plain numbers (saved in the metadata so it survives reloading)."""
    def quantiles(series):
        q = series.dropna().quantile([0.05, 0.5, 0.95])
        return {"p05": float(q.iloc[0]), "p50": float(q.iloc[1]), "p95": float(q.iloc[2])} if len(series.dropna()) else {}

    by_tier = {}
    for tier, group in roads.groupby("match_type"):
        entry = {"roads": int(len(group))}
        if tier != "unmatched":
            entry.update({"len_ratio": quantiles(group["len_ratio"]),
                          "share_ratio_0.8_1.25": float(group["len_ratio"].between(0.8, 1.25).mean())})
        by_tier[tier] = entry
    return {
        "osm_sizes": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in sizes.items()},
        "node_match_share": nodes["node_match"].value_counts(normalize=True).round(4).to_dict(),
        "dataset_node_id_found_in_osm": float(nodes["id_in_osm"].mean()),
        "id_found_but_position_differs": int((nodes["id_in_osm"] & (nodes["node_match"] != "id")).sum()),
        "nearest_node_distance_m": quantiles(nodes["nearest_dist_m"]),
        "road_match_share": roads["match_type"].value_counts(normalize=True).round(4).to_dict(),
        "road_match_by_tier": by_tier,
    }


# Source: osm_features_and_eda.ipynb, cell 35 (zero-based).
FEATURE_GROUP_PREFIXES = ("road_character", "intersection_control", "cycle_pedestrian_lighting", "nearby_places", "road_shape")

def feature_group_of_column(column):
    """The feature group a column belongs to (its prefix), or None for identifiers and diagnostics."""
    return next((group for group in FEATURE_GROUP_PREFIXES if column.startswith(group + "_")), None)

TAGGED_NODE_KINDS = ["signal", "stop", "give_way", "crossing", "traffic_calming"]

PLACE_TYPE_OF_TAG = {"school": "school", "bar": "bar_pub", "pub": "bar_pub", "restaurant": "food",
                     "fast_food": "food", "parking": "parking", "bus_stop": "bus_stop"}

def build_road_character_features(roads, osm_roads):
    """road_character features: what kind of road it is, from the OSM road's own tags.

    Why: speed limit, lanes and surface describe the road and are not in the dataset
    (the dataset only has road type, one-way and length).

    How: each tag is parsed once per OSM road, then copied to the dataset roads through `osm_row`.
    Matched roads get the value (NaN if the tag is missing in OSM); unmatched roads get NaN.

    Takes: `roads` (after matching) and `osm_roads`.

    Returns: a DataFrame with the same rows as `roads` and these columns:
        road_character_maxspeed_mph: speed limit in mph (highest if OSM lists several).
        road_character_lanes: number of lanes (highest if OSM lists several).
        road_character_surface: surface type; categories with fewer than RARE_CATEGORY_MIN_ROADS roads become "other".
        road_character_bridge, road_character_tunnel: 1 if the road is on a bridge / in a tunnel, 0 otherwise.
    """
    matched = (roads["osm_row"] >= 0).to_numpy()
    safe_rows = np.where(matched, roads["osm_row"].to_numpy(), 0)  # row 0 stands in for unmatched roads

    def copy_to_dataset_roads(osm_values):
        return np.where(matched, np.asarray(osm_values, dtype=object)[safe_rows], np.nan)

    features = pd.DataFrame(index=roads.index)
    features["road_character_maxspeed_mph"] = copy_to_dataset_roads(road_tag_values(osm_roads, "maxspeed").map(parse_maxspeed_mph))
    features["road_character_lanes"] = copy_to_dataset_roads(road_tag_values(osm_roads, "lanes").map(parse_lanes))
    features["road_character_surface"] = copy_to_dataset_roads(road_tag_values(osm_roads, "surface").map(single_surface))   # several surfaces: "mixed" (A17)
    features["road_character_bridge"] = copy_to_dataset_roads(road_tag_values(osm_roads, "bridge").map(tag_is_set))
    features["road_character_tunnel"] = copy_to_dataset_roads(road_tag_values(osm_roads, "tunnel").map(tag_is_set))
    numeric = ["road_character_maxspeed_mph", "road_character_lanes", "road_character_bridge", "road_character_tunnel"]
    features[numeric] = features[numeric].astype(float)

    # Fold rare surface types into "other" (keeping missing values as missing).
    counts = features["road_character_surface"].value_counts()
    rare = counts[counts < RARE_CATEGORY_MIN_ROADS].index
    features["road_character_surface"] = features["road_character_surface"].where(~features["road_character_surface"].isin(rare), "other")
    return features


# Source: osm_features_and_eda.ipynb, cell 37 (zero-based).
def build_intersection_control_features(nodes, roads, osm_roads, osm_nodes, osm_graph):
    """intersection_control features: what controls traffic at each end of the road, and how busy the junctions are.

    Why: crashes cluster at intersections, and signals, stop signs and crossings change the risk there.

    How: for every matched road we find the OSM road's two end nodes, work out which one is the
    dataset road's start (the one closer to the dataset start node), and read the node tags there.
    Matched roads get 1 or 0 (tag present or not); unmatched roads get NaN.

    Takes: `nodes`, `roads` (after matching), and `osm_roads`, `osm_nodes`, `osm_graph`.

    Returns: a DataFrame with the same rows as `roads` and these columns:
        intersection_control_{signal,stop,give_way,crossing}_{start,end}: 1 if that node carries the tag.
        intersection_control_roundabout: 1 if the OSM road is part of a roundabout.

    Limitation: osmnx's simplification removes nodes that are not intersections or dead ends, and
    with them their tags. A signal or crossing in the middle of a block is therefore missing here;
    only tags on intersection and dead-end nodes are seen.
    """
    matched = (roads["osm_row"] >= 0).to_numpy()
    safe_rows = np.where(matched, roads["osm_row"].to_numpy(), 0)  # row 0 stands in for unmatched roads

    # The two OSM nodes at the ends of each matched OSM road (-1 where unmatched).
    osm_u = osm_roads.index.get_level_values("u").to_numpy()[safe_rows]
    osm_v = osm_roads.index.get_level_values("v").to_numpy()[safe_rows]
    osm_u, osm_v = np.where(matched, osm_u, -1), np.where(matched, osm_v, -1)

    # Which OSM end is the dataset road's start? The one nearer to the dataset start node.
    points = gpd.GeoSeries(gpd.points_from_xy(nodes.lon, nodes.lat), crs=4326).to_crs(osm_nodes.crs)
    x, y = points.x.to_numpy(), points.y.to_numpy()
    start_node = roads["node_1_idx"].to_numpy()
    osm_x, osm_y = osm_nodes.geometry.x, osm_nodes.geometry.y
    dist_to_u = np.hypot(x[start_node] - osm_x.reindex(osm_u).to_numpy(), y[start_node] - osm_y.reindex(osm_u).to_numpy())
    dist_to_v = np.hypot(x[start_node] - osm_x.reindex(osm_v).to_numpy(), y[start_node] - osm_y.reindex(osm_v).to_numpy())
    u_is_start = dist_to_u <= dist_to_v
    start_osm = np.where(u_is_start, osm_u, osm_v)
    end_osm = np.where(u_is_start, osm_v, osm_u)

    # Node tags. OSM stores the kind of control in the node's `highway` tag.
    node_highway = node_tag_values(osm_nodes, "highway").map(as_text_list)
    node_flags = {
        "signal": node_highway.map(lambda items: "traffic_signals" in items),
        "stop": node_highway.map(lambda items: "stop" in items),
        "give_way": node_highway.map(lambda items: "give_way" in items),
        "crossing": node_highway.map(lambda items: "crossing" in items) | node_tag_values(osm_nodes, "crossing").notna(),
    }

    features = pd.DataFrame(index=roads.index)
    for name, flag in node_flags.items():
        for end, ids in (("start", start_osm), ("end", end_osm)):
            value = flag.astype(float).reindex(ids).to_numpy()  # NaN for id -1
            features[f"intersection_control_{name}_{end}"] = np.where(matched, value, np.nan)

    # Roundabout: a property of the OSM road, parsed once per OSM road then copied across.
    on_roundabout = road_tag_values(osm_roads, "junction").map(
        lambda v: float(any(item in ("roundabout", "circular") for item in as_text_list(v)))).to_numpy()
    features["intersection_control_roundabout"] = np.where(matched, on_roundabout[safe_rows], np.nan)
    return features

def read_tagged_nodes_from_xml(xml_path):
    """List every point in the OSM XML file that carries a traffic-control tag.

    Why: the simplified osmnx network lost most signals, stops and crossings that sit mid-block.
    The XML file still has them, so we read them from there.

    Takes: path to the road XML (e.g. /content/data/DE_roads.osm).
    Returns: a DataFrame with one row per tagged point: osm_id, lat, lon, and one True/False column
    per kind in TAGGED_NODE_KINDS (signal = traffic_signals, stop, give_way, crossing, traffic_calming).
    """
    import xml.etree.ElementTree as ET

    rows = []
    for _, element in ET.iterparse(xml_path, events=("end",)):
        if element.tag == "node":
            tags = {tag.get("k"): tag.get("v") for tag in element.findall("tag")}
            highway = tags.get("highway")
            kinds = {
                "signal": highway == "traffic_signals",
                "stop": highway == "stop",
                "give_way": highway == "give_way",
                "crossing": highway == "crossing" or "crossing" in tags,
                "traffic_calming": "traffic_calming" in tags,
            }
            if any(kinds.values()):
                rows.append({"osm_id": int(element.get("id")), "lat": float(element.get("lat")),
                             "lon": float(element.get("lon")), **kinds})
        if element.tag in ("node", "way"):
            element.clear()  # free memory; the file is large

    tagged_nodes = pd.DataFrame(rows)
    print(f"tagged points in the file: {len(tagged_nodes):,}")
    print(tagged_nodes[TAGGED_NODE_KINDS].sum().to_string())
    return tagged_nodes

def count_lost_tagged_nodes_per_road(tagged_nodes, osm_graph, osm_nodes, osm_roads):
    """For tagged points that simplification removed, count how many sit on each OSM road.

    Why: a removed point lay in the middle of one road, so it belongs to that road.
    Points that survived (at intersections) are already covered by the start/end flags, so skip them.

    How: find the nearest OSM road to each removed point; keep it if within MIDBLOCK_NODE_MAX_M.
    The two directions of a two-way road share a count, so roads are keyed by their (smaller, larger) node ids.

    Takes: `tagged_nodes` from `read_tagged_nodes_from_xml`, `osm_graph`, `osm_nodes`, `osm_roads`.
    Returns: a table indexed by (road_u, road_v) with one count column per kind.
    Prints how many points were removed, how many were attached, and how far they were from a road.
    """
    lost = tagged_nodes[~tagged_nodes["osm_id"].isin(osm_nodes.index)].copy()

    # Nearest OSM road to each removed point, in metre coordinates.
    points = gpd.GeoSeries(gpd.points_from_xy(lost.lon, lost.lat), crs=4326).to_crs(osm_roads.crs)
    position, distance = nearest_osm_road(osm_roads, points.x.to_numpy(), points.y.to_numpy())
    start = osm_roads.index.get_level_values("u").to_numpy()[position]
    end = osm_roads.index.get_level_values("v").to_numpy()[position]
    lost["road_u"], lost["road_v"] = np.minimum(start, end), np.maximum(start, end)
    lost["distance_m"] = distance

    near = lost[lost["distance_m"] <= MIDBLOCK_NODE_MAX_M]
    print(f"tagged points removed by simplification: {len(lost):,}")
    print(f"attached to a road within {MIDBLOCK_NODE_MAX_M} m: {len(near):,}")
    print("distance to the nearest road (m):")
    print(lost["distance_m"].describe(percentiles=[.5, .9, .99]).round(2).to_string())
    print("\nattached, by kind:")
    print(near[TAGGED_NODE_KINDS].sum().to_string())
    return near.groupby(["road_u", "road_v"])[TAGGED_NODE_KINDS].sum()

def build_midblock_control_counts(roads, osm_roads, midblock_counts):
    """intersection_control features: how many signals / stops / give-way signs / crossings / traffic-calming points lie mid-block.

    Takes: `roads` (after matching), `osm_roads`, and the table from `count_lost_tagged_nodes_per_road`.
    Returns: a DataFrame with the same rows as `roads` and columns intersection_control_{kind}_midblock (a count; 0 if none,
    NaN for unmatched roads). Both directions of a two-way road get the same count.
    """
    matched = (roads["osm_row"] >= 0).to_numpy()
    safe_rows = np.where(matched, roads["osm_row"].to_numpy(), 0)  # row 0 stands in for unmatched roads

    # The matched OSM road's (smaller, larger) end-node ids, which is the key used in midblock_counts.
    osm_u = osm_roads.index.get_level_values("u").to_numpy()[safe_rows]
    osm_v = osm_roads.index.get_level_values("v").to_numpy()[safe_rows]
    keys = pd.MultiIndex.from_arrays([np.minimum(osm_u, osm_v), np.maximum(osm_u, osm_v)])

    features = pd.DataFrame(index=roads.index)
    for kind in TAGGED_NODE_KINDS:
        counts = midblock_counts[kind].reindex(keys).fillna(0).to_numpy()
        features[f"intersection_control_{kind}_midblock"] = np.where(matched, counts, np.nan)
    return features

def midblock_recovery_stats(tagged_nodes, osm_nodes, midblock_counts):
    """How many traffic-control points the simplified OSM network lost, and how many we recovered onto roads.

    Takes the full list of tagged points (`read_tagged_nodes_from_xml`), the OSM node table and the mid-block counts.
    Returns a table with one row per kind: tagged points in the file, still present as nodes (end flags see them),
    removed by simplification, and attached to a road within MIDBLOCK_NODE_MAX_M (summed counts).
    """
    kept = tagged_nodes["osm_id"].isin(osm_nodes.index)
    table = pd.DataFrame({
        "in_file": tagged_nodes[TAGGED_NODE_KINDS].sum(),
        "kept_as_nodes": tagged_nodes.loc[kept, TAGGED_NODE_KINDS].sum(),
        "removed_by_simplification": tagged_nodes.loc[~kept, TAGGED_NODE_KINDS].sum(),
        "recovered_mid_block": midblock_counts[TAGGED_NODE_KINDS].sum(),
    })
    table["share_removed"] = table["removed_by_simplification"] / table["in_file"].replace(0, np.nan)
    table["share_of_removed_recovered"] = table["recovered_mid_block"] / table["removed_by_simplification"].replace(0, np.nan)
    return table


# Source: osm_features_and_eda.ipynb, cell 39 (zero-based).
def build_cycle_pedestrian_lighting_features(roads, osm_roads):
    """cycle_pedestrian_lighting features: infrastructure for cyclists and pedestrians, and street lighting.

    Why: the brief expects these tags to be rare on a drive network; we measure how rare.

    How: like road_character, each OSM road is parsed once and the result copied to the dataset roads through
    `osm_row`. Each feature has three states: 1 = tagged and present, 0 = tagged "no",
    NaN = not tagged at all (or the road was unmatched). So the fill rate is the share that is not NaN.

    Takes: `roads` (after matching) and `osm_roads`.

    Returns: a DataFrame with the same rows as `roads` and these columns:
        cycle_pedestrian_lighting_cycleway_any: any of cycleway, cycleway:left, cycleway:right, cycleway:both says yes.
        cycle_pedestrian_lighting_sidewalk_any: the `sidewalk` tag says yes (includes "separate", a sidewalk mapped as its own way).
        cycle_pedestrian_lighting_lit: the `lit` tag says yes.

    Limitation: only the plain `sidewalk` tag was requested from OSM, not sidewalk:left / sidewalk:right.
    """
    matched = (roads["osm_row"] >= 0).to_numpy()
    safe_rows = np.where(matched, roads["osm_row"].to_numpy(), 0)  # row 0 stands in for unmatched roads

    def copy_to_dataset_roads(osm_values):
        return np.where(matched, np.asarray(osm_values, dtype=float)[safe_rows], np.nan)

    def tag_lists(name):
        return road_tag_values(osm_roads, name).map(as_text_list)

    # Cycleway can be tagged under four keys; pool the values of all four for each OSM road.
    cycleway_keys = ["cycleway", "cycleway:left", "cycleway:right", "cycleway:both"]
    pooled = [sum(values, []) for values in zip(*[tag_lists(key) for key in cycleway_keys])]

    features = pd.DataFrame(index=roads.index)
    features["cycle_pedestrian_lighting_cycleway_any"] = copy_to_dataset_roads([yes_no_or_untagged(items) for items in pooled])
    features["cycle_pedestrian_lighting_sidewalk_any"] = copy_to_dataset_roads(tag_lists("sidewalk").map(yes_no_or_untagged))
    features["cycle_pedestrian_lighting_lit"] = copy_to_dataset_roads(tag_lists("lit").map(yes_no_or_untagged))
    return features


# Source: osm_features_and_eda.ipynb, cell 41 (zero-based).
def read_osm_places(state, data_dir, local_dir):
    """Read the schools, bars, restaurants, bus stops and parking from `{state}_osm_places.zip`.

    Why: these are the raw material for the nearby_places "surroundings" features.

    Takes: state code, the Drive data folder, the Colab local folder.
    Returns: a DataFrame with one row per place: lat, lon, and place_type (school, bar_pub, food, parking, bus_stop).
    Areas (a school outline, a parking lot) are reduced to their centre. Prints how many of each type.
    """
    geojson = local_dir / f"{state}_places.geojsonseq"
    if not geojson.exists():
        with zipfile.ZipFile(data_dir / f"{state}_osm_places.zip") as archive:
            archive.extractall(local_dir)

    rows = []
    with open(geojson) as file:
        for line in file:
            line = line.strip().lstrip("\x1e")  # GeoJSON sequences may start each record with a separator character
            if not line:
                continue
            feature = json.loads(line)
            tags = feature.get("properties", {})
            place_type = PLACE_TYPE_OF_TAG.get(tags.get("amenity")) or PLACE_TYPE_OF_TAG.get(tags.get("highway"))
            if place_type is None or feature.get("geometry") is None:
                continue
            centre = shape(feature["geometry"]).centroid
            rows.append({"lat": centre.y, "lon": centre.x, "place_type": place_type})

    places = pd.DataFrame(rows)
    print(f"places read: {len(places):,}")
    print(places["place_type"].value_counts().to_string())
    return places

def build_nearby_places_features(places, tagged_nodes, nodes, roads, metric_crs):
    """nearby_places features: how many places of each type are near a road, and how far the nearest is.

    Why: schools, bars, bus stops and so on mean pedestrians, traffic and turning cars near the road.
    These need no road tags, and every road gets a value (also unmatched ones), because they use only
    the road's position.

    How: each road is represented by the midpoint of its two end nodes. For each place type we count the
    places within PLACE_RADII_M of that midpoint, and find the distance to the nearest one.
    Crossings and traffic signals come from `tagged_nodes` (the full list, including the ones simplification removed).

    Takes: `places` from `read_osm_places`, `tagged_nodes` from `read_tagged_nodes_from_xml`, `nodes`, `roads`,
    and `metric_crs` (the metre-based coordinate system of the OSM graph, e.g. osm_roads.crs).

    Returns: a DataFrame with the same rows as `roads` and these columns for each place type
    (school, bar_pub, food, parking, bus_stop, crossing, signal):
        nearby_places_{type}_{100,250,500,1000}m: how many within each distance in PLACE_RADII_M.
        nearby_places_{type}_nearest_m: distance to the nearest one in metres (NaN if there are none at all).
    """
    def to_metres(lon, lat):
        points = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(metric_crs)
        return np.column_stack([points.x.to_numpy(), points.y.to_numpy()])

    # Road midpoints, in metres.
    start, end = roads["node_1_idx"].to_numpy(), roads["node_2_idx"].to_numpy()
    node_xy = to_metres(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    midpoints = (node_xy[start] + node_xy[end]) / 2

    # Locations of each kind of place.
    locations = {kind: places.loc[places["place_type"] == kind, ["lon", "lat"]] for kind in places["place_type"].unique()}
    locations["crossing"] = tagged_nodes.loc[tagged_nodes["crossing"], ["lon", "lat"]]
    locations["signal"] = tagged_nodes.loc[tagged_nodes["signal"], ["lon", "lat"]]

    features = pd.DataFrame(index=roads.index)
    for kind, lonlat in locations.items():
        if len(lonlat) == 0:
            continue
        tree = cKDTree(to_metres(lonlat["lon"].to_numpy(), lonlat["lat"].to_numpy()))
        for radius in PLACE_RADII_M:
            features[f"nearby_places_{kind}_{radius}m"] = tree.query_ball_point(midpoints, r=radius, return_length=True)
        features[f"nearby_places_{kind}_nearest_m"] = tree.query(midpoints)[0]
    return features


# Source: osm_features_and_eda.ipynb, cell 43 (zero-based).
def build_road_shape_features(nodes, roads):
    """road_shape feature: how curvy each road is. No OSM tags are needed.

    Why: bends reduce sight lines, the benchmark does not carry this, and every road gets a value (unlike most OSM tags).

    Takes: `nodes` and `roads` (from `load_dataset_node_and_road_tables`).

    Returns: a DataFrame with the same rows as `roads` and one column:
        road_shape_sinuosity: recorded road length / straight-line distance between its ends (1 = straight; NaN if the ends coincide).
    """
    start, end = roads["node_1_idx"].to_numpy(), roads["node_2_idx"].to_numpy()
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    straight = straight_line_distance_m(lat[start], lon[start], lat[end], lon[end])
    features = pd.DataFrame(index=roads.index)
    features["road_shape_sinuosity"] = roads["length_m"].to_numpy() / np.where(straight > 0, straight, np.nan)
    return features


# Source: osm_features_and_eda.ipynb, cell 45 (zero-based).
US_MPH_PATTERN_VALUES = (15, 25, 35, 45, 55, 65, 75)

def osm_tag_cleaning_evidence(osm_roads):
    """How the raw speed-limit and lane tags were read, counted on the OSM road pieces.

    Why: the parsers turn free text into numbers; this records how many values each rule touched, and tests the unit-less = mph rule.
    Takes: `osm_roads`. Returns a dict (JSON-safe) with counts for maxspeed and lanes, the most common unit-less speed values,
    and the share of unit-less and of explicit-mph values that end in 5 (the US mph pattern).
    """
    def unit_of(item):
        number = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?\s*$", item)
        if number is None:
            return "words", None
        return ("km/h" if number.group(2) in ("km/h", "kmh", "kph") else number.group(2) or "none"), float(number.group(1))

    speed_items = road_tag_values(osm_roads, "maxspeed").map(as_text_list)
    tagged = speed_items[speed_items.map(len) > 0]
    units = tagged.map(lambda items: [unit_of(item) for item in items])
    has_unit = lambda unit: units.map(lambda pairs: any(u == unit for u, _ in pairs))
    unitless_values = pd.Series([value for pairs in units for u, value in pairs if u == "none"], dtype=float)
    mph_values = pd.Series([value for pairs in units for u, value in pairs if u == "mph"], dtype=float)
    ends_in_5 = lambda values: float((values % 10 == 5).mean()) if len(values) else float("nan")
    speed = {"osm_roads": int(len(speed_items)), "tagged": int(len(tagged)),
             "with_mph": int(has_unit("mph").sum()), "with_kmh_converted": int(has_unit("km/h").sum()),
             "unitless_read_as_mph": int(has_unit("none").sum()), "words_only_set_missing": int(units.map(lambda pairs: all(u == "words" for u, _ in pairs)).sum()),
             "several_values_highest_kept": int((tagged.map(len) > 1).sum()),
             "unitless_top_values": {f"{k:g}": int(n) for k, n in unitless_values.value_counts().head(8).items()},
             "unitless_share_ending_in_5": ends_in_5(unitless_values), "explicit_mph_share_ending_in_5": ends_in_5(mph_values)}

    several = lambda name: int(road_tag_values(osm_roads, name).map(lambda v: len(set(as_text_list(v))) > 1).sum())
    speed["surface_tagged"] = int(road_tag_values(osm_roads, "surface").map(lambda v: len(as_text_list(v)) > 0).sum())
    speed["surface_several_values_mixed"] = several("surface")
    speed["highway_several_values"] = several("highway")
    lane_items = road_tag_values(osm_roads, "lanes").map(as_text_list)
    lanes_tagged = lane_items[lane_items.map(len) > 0]
    parsed = lanes_tagged.map(parse_lanes)
    lanes = {"tagged": int(len(lanes_tagged)), "parsed": int(parsed.notna().sum()), "unusable_set_missing": int(parsed.isna().sum()),
             "several_values_highest_kept": int((lanes_tagged.map(len) > 1).sum())}
    return {"maxspeed": speed, "lanes": lanes}

def tag_cleaning_table(evidence):
    """The tag-cleaning counts as a small table for display."""
    speed, lanes = evidence["maxspeed"], evidence["lanes"]
    rows = [("maxspeed", "OSM road pieces with the tag", speed["tagged"], speed["tagged"] / max(speed["osm_roads"], 1)),
            ("maxspeed", "with 'mph' (kept)", speed["with_mph"], speed["with_mph"] / max(speed["tagged"], 1)),
            ("maxspeed", "without a unit (read as mph, A10)", speed["unitless_read_as_mph"], speed["unitless_read_as_mph"] / max(speed["tagged"], 1)),
            ("maxspeed", "in km/h (converted to mph)", speed["with_kmh_converted"], speed["with_kmh_converted"] / max(speed["tagged"], 1)),
            ("maxspeed", "words only, e.g. 'none' (set to missing)", speed["words_only_set_missing"], speed["words_only_set_missing"] / max(speed["tagged"], 1)),
            ("maxspeed", "several values (highest kept)", speed["several_values_highest_kept"], speed["several_values_highest_kept"] / max(speed["tagged"], 1)),
            ("lanes", "OSM road pieces with the tag", lanes["tagged"], float("nan")),
            ("lanes", "parsed to a number", lanes["parsed"], lanes["parsed"] / max(lanes["tagged"], 1)),
            ("lanes", "unusable (set to missing)", lanes["unusable_set_missing"], lanes["unusable_set_missing"] / max(lanes["tagged"], 1)),
            ("lanes", "several values (highest kept)", lanes["several_values_highest_kept"], lanes["several_values_highest_kept"] / max(lanes["tagged"], 1)),
            ("surface", "OSM road pieces with the tag", speed.get("surface_tagged", 0), float("nan")),
            ("surface", "several different surfaces (set to 'mixed', A17)", speed.get("surface_several_values_mixed", 0),
             speed.get("surface_several_values_mixed", 0) / max(speed.get("surface_tagged", 0), 1)),
            ("highway", "several road types (most common in the state kept, A17)", speed.get("highway_several_values", 0), float("nan"))]
    return pd.DataFrame(rows, columns=["tag", "rule", "OSM road pieces", "share of tagged"]).set_index(["tag", "rule"])


# Source: osm_features_and_eda.ipynb, cell 47 (zero-based).
ROAD_ID_COLUMNS = ["node_1_idx", "node_2_idx", "road_u", "road_v", "has_reverse", "u_lt_v",
                   "length_m", "match_type", "match_dist_m", "len_ratio"]

DIAGNOSTIC_COLUMNS = ["osm_highway", "cand_dist_m", "cand_len_ratio"]

FEATURE_COLUMN_PATTERN = re.compile("(" + "|".join(FEATURE_GROUP_PREFIXES) + ")_")

def configured_thresholds():
    """The thresholds currently set in the config cell, for the metadata."""
    return {"NODE_SNAP_MAX_M": NODE_SNAP_MAX_M, "NODE_ID_COORD_TOL_M": NODE_ID_COORD_TOL_M,
            "ROAD_NEAREST_MAX_M": ROAD_NEAREST_MAX_M, "ROAD_LEN_RATIO_RANGE": ROAD_LEN_RATIO_RANGE,
            "MIDBLOCK_NODE_MAX_M": MIDBLOCK_NODE_MAX_M, "PLACE_RADII_M": PLACE_RADII_M,
            "RARE_CATEGORY_MIN_ROADS": RARE_CATEGORY_MIN_ROADS}

def save_road_features(state, nodes, roads, road_features, data_dir, diagnostics):
    """Save the feature table and a metadata file to the Drive data folder.

    The table has one row per directed road, in the same order as `adj.coalesce().indices()`, with the road identifiers
    in ROAD_ID_COLUMNS and DIAGNOSTIC_COLUMNS, then every feature column (road_character_ ... road_shape_). Values are raw: missing stays NaN,
    nothing is imputed or scaled. The metadata records the OSM snapshot date, versions, thresholds, matching and recovery
    diagnostics, which columns belong to which group, and the fill rate of every column.
    Returns the paths of the parquet file and the metadata file.
    """
    assert len(road_features) == len(roads) and (road_features.index == roads.index).all(), "features not aligned with roads"
    feature_columns = [c for c in road_features.columns if FEATURE_COLUMN_PATTERN.match(c)]
    table = pd.concat([roads[ROAD_ID_COLUMNS + DIAGNOSTIC_COLUMNS].reset_index(drop=True),
                       road_features[feature_columns].reset_index(drop=True)], axis=1)
    assert not table.columns.duplicated().any(), "duplicate column names"

    parquet_path = data_dir / f"{state}_osm_matched_roads.parquet"
    table.to_parquet(parquet_path, index=False)

    # OSM snapshot date from the Geofabrik file name (e.g. delaware-261018.osm.pbf -> 2026-10-18).
    pbf = find_state_pbf(state, data_dir)
    date = re.search(r"-(\d{2})(\d{2})(\d{2})\.osm\.pbf$", pbf.name)
    snapshot = f"20{date.group(1)}-{date.group(2)}-{date.group(3)}" if date else None

    metadata = {
        "state": state, "pipeline_version": PIPELINE_VERSION, "rows": len(table),
        "row_order": "same as adj.coalesce().indices() of the dataset",
        "osm_source_file": pbf.name, "osm_snapshot_date": snapshot,
        "environment": ENVIRONMENT, "thresholds": configured_thresholds(),
        "node_match_share": nodes["node_match"].value_counts(normalize=True).round(4).to_dict(),
        "road_match_share": roads["match_type"].value_counts(normalize=True).round(4).to_dict(),
        "diagnostics": diagnostics,
        "groups": {group: [c for c in feature_columns if c.startswith(group + "_")]
                   for group in FEATURE_GROUP_PREFIXES if any(c.startswith(group + "_") for c in feature_columns)},
        "fill_rate": table[feature_columns].notna().mean().round(4).to_dict(),
        "notes": [
            "Values are raw; missing is NaN. Unmatched roads have NaN for every OSM-tag feature (road_character, intersection_control, cycle_pedestrian_lighting) but not for nearby_places and road_shape.",
            "OSM is a snapshot from the date above; accident labels span about 2009-2022.",
            "Both directions of a two-way road share the OSM-derived values.",
        ],
    }
    meta_path = data_dir / f"{state}_osm_matching_meta.json"
    with open(meta_path, "w") as file:
        json.dump(metadata, file, indent=2, default=str)
    print(f"saved {len(table):,} rows x {len(feature_columns)} features (plus identifiers and diagnostics) to {parquet_path.name} and {meta_path.name}")
    return parquet_path, meta_path

def build_state_features(state, data, data_dir, local_dir, save=True):
    """Run the whole feature pipeline for one state and save the result.

    Steps: dataset tables -> length check -> OSM tables -> node matching -> road matching -> features road_character, intersection_control (end flags
    and mid-block counts), cycle_pedestrian_lighting, nearby_places, road_shape -> diagnostics -> save. For a large state, build the OSM tables first (Stage 2.1).
    Returns a dict with `nodes`, `roads` (after matching), `road_features`, `highway_class` and `diagnostics`.
    """
    memory_report(f"{state} start")
    info = data[state]

    print(f"\n===== {state}: dataset tables")
    nodes, roads = load_dataset_node_and_road_tables(info)
    check_dataset_road_lengths(nodes, roads)

    print(f"\n===== {state}: OSM tables")
    prepare_osm_roads_zip(state, data_dir, local_dir)
    prepare_osm_places_zip(state, data_dir, local_dir)
    build_osm_tables(state, data_dir, local_dir)  # skipped if the tables are already built
    osm_nodes, osm_roads = load_osm_tables(state, local_dir)
    osm_graph = None  # not kept: lookups use the tables
    tag_cleaning = osm_tag_cleaning_evidence(osm_roads)
    sizes = compare_dataset_and_osm_sizes(nodes, roads, osm_nodes, osm_roads)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in sizes.items()})

    print(f"\n===== {state}: matching")
    nodes = match_dataset_nodes_to_osm_nodes(nodes, osm_graph, osm_nodes)
    roads = match_dataset_roads_to_osm_roads(nodes, roads, osm_graph, osm_roads)
    highway_class = dataset_highway_class(info, roads)
    diagnostics = matching_diagnostics(nodes, roads, sizes)
    diagnostics["tag_cleaning"] = tag_cleaning
    print("node match:", diagnostics["node_match_share"], "| road match:", diagnostics["road_match_share"])
    memory_report("after matching")

    print(f"\n===== {state}: features")
    road_character = build_road_character_features(roads, osm_roads)
    intersection_control = build_intersection_control_features(nodes, roads, osm_roads, osm_nodes, osm_graph)
    tagged_nodes = read_tagged_nodes_from_xml(local_dir / f"{state}_roads.osm")
    midblock_counts = count_lost_tagged_nodes_per_road(tagged_nodes, osm_graph, osm_nodes, osm_roads)
    intersection_control_midblock = build_midblock_control_counts(roads, osm_roads, midblock_counts)
    diagnostics["midblock_recovery"] = midblock_recovery_stats(tagged_nodes, osm_nodes, midblock_counts).round(4).to_dict(orient="index")
    cycle_pedestrian_lighting = build_cycle_pedestrian_lighting_features(roads, osm_roads)

    # The OSM network is the biggest object in memory and nothing below needs it, so free it before the place counts.
    metric_crs = osm_roads.crs
    del osm_nodes, osm_roads
    gc.collect()
    memory_report("after freeing the OSM network")

    road_shape = build_road_shape_features(nodes, roads)
    places = read_osm_places(state, data_dir, local_dir)
    nearby_places = build_nearby_places_features(places, tagged_nodes, nodes, roads, metric_crs)
    road_features = pd.concat([roads[["node_1_idx", "node_2_idx"]], road_character, intersection_control, intersection_control_midblock, cycle_pedestrian_lighting, nearby_places, road_shape], axis=1)
    memory_report("after all features")

    if save:
        print(f"\n===== {state}: save")
        save_road_features(state, nodes, roads, road_features, data_dir, diagnostics)
    return {"nodes": nodes, "roads": roads, "road_features": road_features, "highway_class": highway_class,
            "diagnostics": diagnostics, "has_new_diagnostics": True, "built_now": True}

def load_saved_state_features(state, data, data_dir):
    """Read a state's saved feature table back from Drive, instead of rebuilding it.

    Returns a dict like `build_state_features`'s: `roads`, `road_features`, `highway_class`, `diagnostics` (from the
    metadata file, empty if absent) and `has_new_diagnostics` (False for files saved by the older notebook).
    `roads` has the identifier and match columns; `osm_row` is only a placeholder (-1 = unmatched, 0 = matched)
    because the real position in the OSM table is not saved. `nodes` is not included (the analysis does not use it).
    """
    table = pd.read_parquet(data_dir / f"{state}_osm_matched_roads.parquet")
    has_new = all(c in table.columns for c in DIAGNOSTIC_COLUMNS)
    keep = ROAD_ID_COLUMNS + [c for c in DIAGNOSTIC_COLUMNS if c in table.columns]
    roads = table[keep].copy()
    for column in DIAGNOSTIC_COLUMNS:
        if column not in roads.columns:
            roads[column] = np.nan
    roads["osm_row"] = np.where(roads["match_type"] == "unmatched", -1, 0)
    feature_columns = [c for c in table.columns if FEATURE_COLUMN_PATTERN.match(c)]
    road_features = table[["node_1_idx", "node_2_idx"] + feature_columns].copy()
    highway_class = dataset_highway_class(data[state], roads)

    diagnostics, meta_path = {}, data_dir / f"{state}_osm_matching_meta.json"
    if meta_path.exists():
        with open(meta_path) as file:
            diagnostics = json.load(file).get("diagnostics", {})
    print(f"loaded {len(table):,} saved rows for {state}" + ("" if has_new else "  (older file: diagnostic columns missing; set FORCE_REBUILD = True to add them)"))
    return {"roads": roads, "road_features": road_features, "highway_class": highway_class,
            "diagnostics": diagnostics, "has_new_diagnostics": has_new, "built_now": False}

def get_state_features(state, data, data_dir, local_dir, force=False):
    """The saved features if they exist on Drive (and `force` is False), otherwise build them (slow) and save them."""
    if saved_features_are_current(state, data_dir) and not force:
        return load_saved_state_features(state, data, data_dir)
    return build_state_features(state, data, data_dir, local_dir)


# Source: osm_features_and_eda.ipynb, cell 50 (zero-based).
FEATURE_CATALOGUE = [
    ("road_character", "speed limit", "`maxspeed` tag on the matched OSM road", "parsed to mph; km/h converted; unit-less read as mph (A10); highest kept if several",
     "mph", "yes", "empty = not tagged or unmatched (unknown)", "road_character_maxspeed_mph"),
    ("road_character", "lanes", "`lanes` tag on the matched OSM road", "number parsed; highest kept if several", "lanes", "yes", "empty = not tagged or unmatched", "road_character_lanes"),
    ("road_character", "surface", "`surface` tag on the matched OSM road", "one value; a road whose pieces have different surfaces is 'mixed' (A17); types on fewer than %d roads become 'other'" % RARE_CATEGORY_MIN_ROADS,
     "category", "yes", "empty = not tagged or unmatched", "road_character_surface"),
    ("road_character", "bridge, tunnel", "`bridge`, `tunnel` tags on the matched OSM road", "1 if the tag has any value other than 'no'", "0/1", "yes",
     "0 = not tagged as bridge/tunnel; empty = unmatched", "road_character_bridge"),
    ("intersection_control", "control at the road's ends", "node tags `highway=traffic_signals / stop / give_way / crossing` or `crossing=*` at the OSM road's end nodes",
     "read at the end nearer the benchmark road's start, and at the other end", "0/1", "yes", "0 = not mapped there (not necessarily absent); empty = unmatched", "intersection_control_signal_start"),
    ("intersection_control", "roundabout", "`junction=roundabout` or `circular` on the matched OSM road", "1 if part of a roundabout", "0/1", "yes",
     "0 = not tagged; empty = unmatched", "intersection_control_roundabout"),
    ("intersection_control", "mid-block control points", "the same node tags, plus `traffic_calming=*`, on points that simplification removed",
     "points within %d m attached to the nearest road and counted" % MIDBLOCK_NODE_MAX_M, "count", "yes", "0 = none mapped; empty = unmatched", "intersection_control_signal_midblock"),
    ("cycle_pedestrian_lighting", "cycleway", "`cycleway`, `cycleway:left`, `:right`, `:both` on the matched OSM road", "1 if any says a cycle lane exists, 0 if tagged no",
     "0/1", "yes", "empty = not tagged (unknown) or unmatched", "cycle_pedestrian_lighting_cycleway_any"),
    ("cycle_pedestrian_lighting", "sidewalk", "`sidewalk` on the matched OSM road", "1 if it says a sidewalk exists (includes 'separate'), 0 if tagged no",
     "0/1", "yes", "empty = not tagged or unmatched", "cycle_pedestrian_lighting_sidewalk_any"),
    ("cycle_pedestrian_lighting", "street lighting", "`lit` on the matched OSM road", "1 if lit, 0 if tagged no", "0/1", "yes", "empty = not tagged or unmatched",
     "cycle_pedestrian_lighting_lit"),
    ("nearby_places", "schools, bars and pubs, food, parking", "`amenity=school`, `bar`, `pub`, `restaurant`, `fast_food`, `parking` in the state extract (areas reduced to their centre)",
     "counted within 100, 250, 500, 1,000 m of the road's midpoint, plus the distance to the nearest", "count, m", "no (position only)",
     "0 = none mapped within the radius", "nearby_places_school_250m"),
    ("nearby_places", "bus stops", "`highway=bus_stop` in the state extract", "as above", "count, m", "no", "0 = none mapped within the radius", "nearby_places_bus_stop_250m"),
    ("nearby_places", "crossings, signals", "every tagged crossing and signal point in the road XML (including those simplification removed)", "as above",
     "count, m", "no", "0 = none mapped within the radius", "nearby_places_signal_250m"),
    ("road_shape", "curviness", "no OSM tag: the benchmark's own road length and node coordinates", "length / straight-line distance between the ends (1 = straight)",
     "ratio", "no", "empty only if the two ends coincide", "road_shape_sinuosity"),
]

def feature_catalogue(results):
    """The catalogue as a table, with the share of directed roads that have a value in each state."""
    columns = ["group", "feature", "OSM source", "how it is computed", "unit", "needs a matched road", "what empty or 0 means"]
    table = pd.DataFrame([row[:7] for row in FEATURE_CATALOGUE], columns=columns)
    for state, result in results.items():
        features = result["road_features"]
        table[f"roads with a value ({state})"] = [features[row[7]].notna().mean() if row[7] in features.columns else np.nan for row in FEATURE_CATALOGUE]
    return table.set_index(["group", "feature"])


# Source: osm_features_and_eda.ipynb, cell 54 (zero-based).
MATCH_TIERS = ["exact", "reverse", "nearest", "unmatched"]

def road_match_shares(results):
    """Share of dataset roads at each tier, one column per state."""
    return pd.DataFrame({s: results[s]["roads"]["match_type"].value_counts(normalize=True) for s in results}).reindex(MATCH_TIERS).fillna(0)

def length_ratio_by_tier(roads):
    """For each matched tier: number of roads, the 5th/50th/95th percentile of OSM length / dataset length, and the share within 0.8 to 1.25."""
    rows = {}
    for tier in MATCH_TIERS[:3]:
        ratio = roads.loc[roads["match_type"] == tier, "len_ratio"].dropna()
        if len(ratio):
            rows[tier] = {"roads": len(ratio), "p05": ratio.quantile(.05), "median": ratio.median(), "p95": ratio.quantile(.95),
                          "share_0.8_to_1.25": ratio.between(0.8, 1.25).mean()}
    return pd.DataFrame(rows).T

def base_road_type(series):
    """Road type with the '_link' suffix removed (a slip road counts as its parent type)."""
    return series.astype(object).where(series.notna(), None).map(lambda v: v.replace("_link", "") if isinstance(v, str) else v)

def road_type_agreement(result):
    """Per matched tier: how often the dataset's road type equals the matched OSM road's `highway` tag.

    Returns (table by tier, the most common disagreements in the nearest tier) or (None, None) if the saved file has no `osm_highway` column.
    """
    roads = result["roads"]
    if not result.get("has_new_diagnostics", False):
        return None, None
    frame = pd.DataFrame({"tier": roads["match_type"].to_numpy(), "dataset": base_road_type(result["highway_class"]).to_numpy(),
                          "osm": base_road_type(roads["osm_highway"]).to_numpy()})
    frame = frame[frame["tier"].isin(MATCH_TIERS[:3]) & frame["osm"].notna() & (frame["dataset"] != "none")]
    frame["agree"] = frame["dataset"] == frame["osm"]
    by_tier = frame.groupby("tier").agg(roads_checked=("agree", "size"), agree=("agree", "mean")).reindex(MATCH_TIERS[:3])
    wrong = frame[(frame["tier"] == "nearest") & ~frame["agree"]].groupby(["dataset", "osm"]).size().sort_values(ascending=False).head(6)
    return by_tier, wrong.rename("roads").reset_index()

def threshold_sensitivity(roads, distances=(10, 20, 30, 50, 75), ratio_ranges=((0.8, 1.25), (0.7, 1.4), (0.6, 1.6), (0.5, 2.0))):
    """Overall match rate if the nearest-road tier used another distance limit (rows) and length-ratio range (columns).

    Uses the stored nearest candidate of every road that reached the nearest tier, so no OSM files are needed.
    Returns a table of match rates, or None if the saved file lacks the candidate columns.
    """
    if roads["cand_dist_m"].notna().sum() == 0:
        return None
    base = int(roads["match_type"].isin(["exact", "reverse"]).sum())
    reached = roads[roads["match_type"].isin(["nearest", "unmatched"]) & roads["cand_dist_m"].notna()]
    table = pd.DataFrame({f"{lo:g}-{hi:g}": {d: (base + int(((reached["cand_dist_m"] <= d) & reached["cand_len_ratio"].between(lo, hi)).sum())) / len(roads)
                                                for d in distances} for lo, hi in ratio_ranges})
    table.index.name = "max distance (m)"
    table.columns.name = "length ratio range"
    return table


# Source: osm_features_and_eda.ipynb, cell 59 (zero-based).
COVERAGE_TAGS = {"speed limit": "road_character_maxspeed_mph", "lanes": "road_character_lanes", "surface": "road_character_surface",
                 "cycleway": "cycle_pedestrian_lighting_cycleway_any", "sidewalk": "cycle_pedestrian_lighting_sidewalk_any", "lit": "cycle_pedestrian_lighting_lit"}

MAJOR_ROAD_TYPES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link"]

def full_window(state, info):
    """First and last usable month of a state as 'YYYY-MM' (the configured window, or every month in the accident file)."""
    if state in USABLE_MONTHS:
        return USABLE_MONTHS[state]
    months = pd.to_datetime(info["accidents"][["year", "month"]].assign(day=1))
    return months.min().strftime("%Y-%m"), months.max().strftime("%Y-%m")

def get_window(state, name, info):
    """('YYYY-MM', 'YYYY-MM') for the analysis window `name`: 'train' (SPLIT), 'test' (SPLIT) or 'full' (all usable months)."""
    return full_window(state, info) if name == "full" else SPLIT[state][name]

def coverage_frame(result, totals):
    """One row per directed road: road type, matched flag, accidents, and True/False for 'has a value' for each tag."""
    features, roads = result["road_features"], result["roads"]
    frame = pd.DataFrame({"road_type": result["highway_class"].to_numpy(), "matched": (roads["match_type"] != "unmatched").to_numpy(),
                          "accidents": totals, "length_m": roads["length_m"].to_numpy()})
    for label, column in COVERAGE_TAGS.items():
        frame[label] = features[column].notna().to_numpy()
    return frame

def overall_coverage(frame):
    """Per tag: share of all roads with a value, share of matched roads, share of accidents on roads with a value, share of major roads."""
    major = frame["road_type"].isin(MAJOR_ROAD_TYPES)
    rows = {}
    for label in COVERAGE_TAGS:
        rows[label] = {"all roads": frame[label].mean(), "matched roads": frame.loc[frame["matched"], label].mean(),
                       "weighted by accidents": (frame[label] * frame["accidents"]).sum() / frame["accidents"].sum(),
                       "major roads": frame.loc[major, label].mean()}
    return pd.DataFrame(rows).T

def coverage_by_class(frame, top=8):
    """For the `top` road types by accidents: share of roads, share of accidents, match rate, and fill of each tag (share of ALL roads of that type)."""
    grouped = frame.groupby("road_type")
    table = pd.DataFrame({"share_of_roads": grouped.size() / len(frame),
                          "share_of_accidents": grouped["accidents"].sum() / frame["accidents"].sum(),
                          "match_rate": grouped["matched"].mean()})
    for label in COVERAGE_TAGS:
        table[label] = grouped[label].mean()
    return table.sort_values("share_of_accidents", ascending=False).head(top)

def unmatched_profile(frame):
    """What the unmatched roads are, and the ceiling on what matching them could add to each tag's fill.

    Returns (summary dict, by-type table, ceiling table). The ceiling assumes every unmatched road could be matched AND would then have the
    same tag rate as matched roads of its own type: an optimistic upper bound.
    """
    unmatched = ~frame["matched"]
    summary = {"share_of_roads": float(unmatched.mean()), "share_of_km": float(frame.loc[unmatched, "length_m"].sum() / frame["length_m"].sum()),
               "share_of_accidents": float(frame.loc[unmatched, "accidents"].sum() / frame["accidents"].sum()),
               "median_length_m_unmatched": float(frame.loc[unmatched, "length_m"].median()), "median_length_m_matched": float(frame.loc[~unmatched, "length_m"].median())}
    grouped = frame.groupby("road_type")
    by_type = pd.DataFrame({"roads": grouped.size(), "unmatched_share": 1 - grouped["matched"].mean(),
                            "accidents_per_road_unmatched": frame[unmatched].groupby("road_type")["accidents"].mean(),
                            "accidents_per_road_matched": frame[~unmatched].groupby("road_type")["accidents"].mean()})
    by_type = by_type.sort_values("roads", ascending=False).head(8)
    ceiling = {}
    for label in COVERAGE_TAGS:
        matched_fill = frame[frame["matched"]].groupby("road_type")[label].mean()
        n_unmatched = frame[unmatched].groupby("road_type").size()
        acc_unmatched = frame[unmatched].groupby("road_type")["accidents"].sum()
        gain_roads = (n_unmatched * matched_fill.reindex(n_unmatched.index).fillna(0)).sum() / len(frame)
        gain_accidents = (acc_unmatched * matched_fill.reindex(acc_unmatched.index).fillna(0)).sum() / frame["accidents"].sum()
        ceiling[label] = {"fill now (roads)": frame[label].mean(), "most extra from matching (points of roads)": gain_roads,
                          "fill now (accidents)": (frame[label] * frame["accidents"]).sum() / frame["accidents"].sum(),
                          "most extra from matching (points of accidents)": gain_accidents}
    return summary, by_type, pd.DataFrame(ceiling).T

def fill_by_group(result):
    """For each feature group: number of columns and the lowest / median / highest fill among ALL roads."""
    features = result["road_features"].drop(columns=["node_1_idx", "node_2_idx"])
    fill = features.notna().mean()
    rows = {}
    for group in [g for g in FEATURE_GROUP_PREFIXES if any(c.startswith(g + "_") for c in features.columns)]:
        values = fill[[c for c in features.columns if c.startswith(group + "_")]]
        rows[group] = {"columns": len(values), "lowest": values.min(), "median": values.median(), "highest": values.max()}
    table = pd.DataFrame(rows).T
    table.insert(0, "what", [FEATURE_GROUP_NAMES[g] for g in table.index])
    return table

FEATURE_GROUP_NAMES = {"road_character": "road character (OSM tags)", "intersection_control": "intersection control", "cycle_pedestrian_lighting": "cycle, pedestrian and lighting tags",
                       "nearby_places": "nearby places", "road_shape": "road shape (curviness)"}

def matching_funnel(result, totals):
    """How many benchmark roads (and how many accidents) reach each step: matched, by tier, and with each tag.

    Accidents are counted on the direction where they are recorded, over the full usable period.
    Returns a table with one row per step: roads, share of roads, share of accidents.
    """
    roads, features = result["roads"], result["road_features"]
    match_type = roads["match_type"].to_numpy()
    total_accidents = max(totals.sum(), 1)
    steps = [("All benchmark roads (directed)", np.ones(len(roads), dtype=bool)),
             ("Matched to an OSM road (any tier)", match_type != "unmatched"),
             ("   exact", match_type == "exact"), ("   reverse", match_type == "reverse"), ("   nearest (the weaker tier)", match_type == "nearest"),
             ("Not matched", match_type == "unmatched")]
    for label, column in COVERAGE_TAGS.items():
        steps.append((f"Matched and has a {label} tag", features[column].notna().to_numpy()))
    place_columns = [c for c in features.columns if c.startswith("nearby_places_")]
    steps.append(("Has nearby-place features (no match needed)", features[place_columns].notna().all(axis=1).to_numpy()))
    steps.append(("Has a curviness value (no match needed)", features["road_shape_sinuosity"].notna().to_numpy()))
    return pd.DataFrame([{"step": label, "roads": int(mask.sum()), "share of roads": mask.mean(), "share of accidents": totals[mask].sum() / total_accidents}
                         for label, mask in steps]).set_index("step")


# Source: osm_features_and_eda.ipynb, cell 68 (zero-based).
SPEED_STEP_MPH = 5

def physical_attribute_table(roads, road_features, highway_class):
    """One row per PHYSICAL road (both directions merged) with the road type and the raw values of the features to fill."""
    frame = pd.DataFrame({"road_u": roads["road_u"].to_numpy(), "road_v": roads["road_v"].to_numpy(),
                          "road_type": base_road_type(highway_class).to_numpy()})
    for column in IMPUTE_FEATURES:
        frame[column] = road_features[column].to_numpy()
    return frame.groupby(["road_u", "road_v"]).first()

def neighbour_matrix(road_u, road_v, class_codes):
    """Sparse matrix neighbour_weights: neighbour_weights[i, j] > 0 when physical roads i and j share a node AND have the same road type (diagonal removed)."""
    n = len(road_u)
    n_classes = int(class_codes.max()) + 1
    columns = np.concatenate([road_u.astype(np.int64) * n_classes + class_codes, road_v.astype(np.int64) * n_classes + class_codes])
    codes, _ = pd.factorize(columns)
    rows = np.concatenate([np.arange(n), np.arange(n)])
    incidence = sparse.csr_matrix((np.ones(len(rows)), (rows, codes)), shape=(n, codes.max() + 1))
    neighbour_weights = (incidence @ incidence.T).tocsr()
    neighbour_weights.setdiag(0)
    neighbour_weights.eliminate_zeros()
    return neighbour_weights

def snap_estimate(kind, values):
    """Round an estimate to what the tag can look like: speed to the nearest 5 mph, lanes to a whole number >= 1."""
    if kind == "speed":
        return np.clip(np.round(values / SPEED_STEP_MPH) * SPEED_STEP_MPH, SPEED_STEP_MPH, 85)
    if kind == "lanes":
        return np.clip(np.round(values), 1, 12)
    return values

def neighbour_estimate(kind, values, known, neighbour_weights, n_categories=0):
    """For every road: the typical value among its same-type touching neighbours that have a known value (NaN if none).

    Numbers: the mean of the neighbours' values, snapped. Surface: the most common neighbour category.
    `values` are floats (category codes for surface), `known` says which entries may be used as sources.
    """
    if kind == "surface":
        counts = np.column_stack([neighbour_weights @ ((values == c) & known).astype(float) for c in range(n_categories)])
        best = counts.argmax(axis=1).astype(float)
        best[counts.sum(axis=1) == 0] = np.nan
        return best
    total = neighbour_weights @ np.where(known, values, 0.0)
    weight = neighbour_weights @ known.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        estimate = np.where(weight > 0, total / weight, np.nan)
    return snap_estimate(kind, estimate)

def typical_value(kind, values, known, class_codes, n_categories=0):
    """The state-wide typical value and the typical value of each road type (NaN-free: types with no known value get the state-wide one).

    Returns (state_value, array with one typical value per road).
    """
    if kind == "surface":
        state_value = float(np.bincount(values[known].astype(int), minlength=n_categories).argmax())
        table = pd.crosstab(class_codes[known], values[known].astype(int))
        per_class = pd.Series(table.columns[table.to_numpy().argmax(axis=1)].to_numpy().astype(float), index=table.index) if len(table) else pd.Series(dtype=float)
    else:
        state_value = float(snap_estimate(kind, np.median(values[known])))
        per_class = pd.Series(values[known]).groupby(class_codes[known]).median()
        per_class = pd.Series(snap_estimate(kind, per_class.to_numpy()), index=per_class.index)
    return state_value, per_class.reindex(class_codes).fillna(state_value).to_numpy()

def fill_strategies(kind, values, known, class_codes, neighbour_weights, n_categories=0):
    """All three filling methods at once, using only entries marked `known` as information.

    Returns a dict with arrays over all roads: `state` (state-wide value), `type` (road type value), `full` (neighbours then road type),
    and `source` for `full`: 1 = one-hop neighbours, 2 = two-hop, 3 = road type, 4 = state-wide value.
    """
    state_value, type_values = typical_value(kind, values, known, class_codes, n_categories)
    first = neighbour_estimate(kind, values, known, neighbour_weights, n_categories)
    got_first = np.isfinite(first) & ~known
    values_2 = np.where(known, values, np.where(got_first, first, np.nan))
    second = neighbour_estimate(kind, np.nan_to_num(values_2, nan=-1.0), np.isfinite(values_2), neighbour_weights, n_categories)
    got_second = np.isfinite(second) & ~known & ~got_first

    full = type_values.copy()
    source = np.where(np.isin(class_codes, class_codes[known]), 3, 4)
    full[got_second], source[got_second] = second[got_second], 2
    full[got_first], source[got_first] = first[got_first], 1
    return {"state": np.full(len(values), state_value), "type": type_values, "full": full, "source": source}

def encode_feature(series, kind):
    """(values as floats with NaN for missing, category names or None). Surface text becomes category codes 0, 1, 2, ..."""
    if kind == "surface":
        categories = sorted(series.dropna().unique())
        lookup = {name: code for code, name in enumerate(categories)}
        return series.map(lookup).astype(float).to_numpy(), categories
    return series.astype(float).to_numpy(), None

def is_correct(kind, predicted, truth):
    """True where an estimate counts as right: speed within SPEED_TOLERANCE_MPH, lanes and surface exactly."""
    if kind == "speed":
        return np.abs(predicted - truth) <= SPEED_TOLERANCE_MPH
    return predicted == truth

def masked_cross_validation(kind, values, class_codes, neighbour_weights, n_categories, folds, seed):
    """Hide each known value once (stratified by road type), fill it without that value, and keep both for scoring.

    Returns a DataFrame with one row per known road: road type code, the truth, the three estimates and the source of the `full` one.
    """
    known = np.isfinite(values)
    known_index = np.flatnonzero(known)
    rng = np.random.default_rng(seed)
    order = rng.permutation(known_index)
    rank_in_type = pd.Series(np.arange(len(order))).groupby(class_codes[order]).cumcount().to_numpy()
    fold = np.full(len(values), -1)
    fold[order] = rank_in_type % folds
    parts = []
    for k in range(folds):
        hidden = fold == k
        visible = known & ~hidden
        result = fill_strategies(kind, np.where(visible, values, np.nan), visible, class_codes, neighbour_weights, n_categories)
        parts.append(pd.DataFrame({"type_code": class_codes[hidden], "truth": values[hidden], "state": result["state"][hidden],
                                   "type": result["type"][hidden], "full": result["full"][hidden], "source": result["source"][hidden]}))
    return pd.concat(parts, ignore_index=True)

METHOD_LABELS = {"state": "state-wide typical value", "type": "road type", "full": "neighbours + road type"}

def score_validation(kind, records):
    """Accuracy (and mean absolute error for numbers) of each method on the hidden values; accuracy of `full` by where its estimate came from."""
    scores = {}
    for method in METHOD_LABELS:
        correct = is_correct(kind, records[method].to_numpy(), records["truth"].to_numpy())
        scores[method] = {"accuracy": float(correct.mean())}
        if kind != "surface":
            scores[method]["mean_abs_error"] = float(np.abs(records[method] - records["truth"]).mean())
    by_source = records.assign(correct=is_correct(kind, records["full"].to_numpy(), records["truth"].to_numpy())).groupby("source").agg(
        hidden=("correct", "size"), accuracy=("correct", "mean"))
    return pd.DataFrame(scores).T, by_source

def decide_method(scores):
    """The gate. Returns (method to use: 'full', 'type' or 'none', plain-language reason)."""
    baseline = scores.loc["state", "accuracy"]
    passing = {m: scores.loc[m, "accuracy"] for m in ("type", "full")
               if scores.loc[m, "accuracy"] - baseline >= IMPUTE_MIN_GAIN and scores.loc[m, "accuracy"] >= IMPUTE_MIN_ACCURACY}
    if not passing:
        best = max(scores.loc[m, "accuracy"] for m in ("type", "full"))
        return "none", f"best method reaches {best:.0%}; needs +{IMPUTE_MIN_GAIN:.0%} over the state-wide value ({baseline:.0%}) and at least {IMPUTE_MIN_ACCURACY:.0%}"
    if "full" in passing and passing["full"] >= passing.get("type", 0) + IMPUTE_NEIGHBOUR_EXTRA:
        return "full", f"neighbours + road type reaches {passing['full']:.0%} against {baseline:.0%} for the state-wide value"
    method = max(passing, key=passing.get)
    return method, f"{METHOD_LABELS[method]} reaches {passing[method]:.0%} against {baseline:.0%} for the state-wide value"

def validation_by_road_type(kind, records, type_names):
    """Hidden-value accuracy of each method, by road type, with the number of hidden values."""
    table = records.assign(**{m: is_correct(kind, records[m].to_numpy(), records["truth"].to_numpy()) for m in METHOD_LABELS})
    out = table.groupby("type_code").agg(hidden=("truth", "size"), state=("state", "mean"), type=("type", "mean"), full=("full", "mean"))
    out.index = [type_names[int(c)] for c in out.index]
    return out.sort_values("hidden", ascending=False)

def fill_state(state, result, folds=IMPUTE_FOLDS, seed=RANDOM_SEED):
    """Validate and apply the gap-filling for one state.

    Returns a dict with `validation` (per feature: scores, by_source, by_type, decision, reason), `filled` (a DataFrame aligned with the
    directed roads: `<feature>_filled`, `<feature>_is_est`, `<feature>_est_source`), and `physical` (the physical-road table).
    """
    roads, features, classes = result["roads"], result["road_features"], result["highway_class"]
    physical = physical_attribute_table(roads, features, classes)
    type_names = sorted(physical["road_type"].dropna().unique())
    class_codes = physical["road_type"].map({name: code for code, name in enumerate(type_names)}).fillna(len(type_names)).astype(int).to_numpy()
    type_names = type_names + ["unknown"]
    neighbour_weights = neighbour_matrix(physical.index.get_level_values("road_u").to_numpy(), physical.index.get_level_values("road_v").to_numpy(), class_codes)

    directed_key = pd.MultiIndex.from_arrays([roads["road_u"].to_numpy(), roads["road_v"].to_numpy()])
    validation, filled = {}, pd.DataFrame(index=roads.index)
    for column, kind in IMPUTE_FEATURES.items():
        values, categories = encode_feature(physical[column], kind)
        n_categories = len(categories) if categories else 0
        known = np.isfinite(values)
        if known.sum() < 50 * IMPUTE_FOLDS:
            validation[column] = {"decision": "none", "reason": f"only {int(known.sum())} known values; too few to validate"}
            decision = "none"
            estimate = np.full(len(values), np.nan); source = np.zeros(len(values), dtype=int)
        else:
            records = masked_cross_validation(kind, values, class_codes, neighbour_weights, n_categories, folds, seed)
            scores, by_source = score_validation(kind, records)
            decision, reason = decide_method(scores)
            validation[column] = {"scores": scores, "by_source": by_source, "by_type": validation_by_road_type(kind, records, type_names),
                                  "decision": decision, "reason": reason, "kind": kind, "n_known": int(known.sum())}
            applied = fill_strategies(kind, values, known, class_codes, neighbour_weights, n_categories)
            estimate, source = applied["full" if decision == "full" else "type"], applied["source"]
            if decision == "type":
                source = np.where(known, 0, np.where(np.isin(class_codes, class_codes[known]), 3, 4))
            # mix of where the real estimates come from (decides how much to trust the overall accuracy)
            missing = ~known
            if decision != "none":
                mix = pd.Series(source[missing]).value_counts(normalize=True).sort_index()
                validation[column]["real_source_mix"] = mix
                if decision == "full":
                    accuracy_by_source = by_source["accuracy"]
                    validation[column]["expected_accuracy_on_missing"] = float(sum(mix.get(s, 0) * accuracy_by_source.get(s, np.nan) for s in mix.index if s in accuracy_by_source.index))
                else:
                    validation[column]["expected_accuracy_on_missing"] = float(scores.loc["type", "accuracy"])
                few = validation[column]["by_type"]
                small_types = set(few.index[few["hidden"] < 30])
                share_in_small = float(np.isin(class_codes[missing], [i for i, name in enumerate(type_names) if name in small_types]).mean())
                validation[column]["missing_share_in_types_with_few_validation_cases"] = share_in_small
                # Second gate: the accuracy that matters is the one on the roads that are really missing (mostly minor roads),
                # not the accuracy on the tagged roads used for validation.
                expected = validation[column]["expected_accuracy_on_missing"]
                if not expected >= IMPUTE_MIN_ACCURACY:
                    decision = "none"
                    validation[column]["decision"] = "none"
                    validation[column]["reason"] = (f"{reason}; but the expected accuracy on the roads that are really missing is {expected:.0%}, "
                                                    f"below the required {IMPUTE_MIN_ACCURACY:.0%}, so it is not filled")
        if decision == "none":
            filled_values = values.copy()
            is_est = np.zeros(len(values), dtype=bool)
            source_name = np.where(np.isfinite(values), "known", "none")
        else:
            filled_values = np.where(np.isfinite(values), values, estimate)
            is_est = ~np.isfinite(values) & np.isfinite(filled_values)
            names = np.array(["known", "neighbour", "neighbour2", "road_type", "state_value"], dtype=object)
            source_name = np.where(is_est, names[np.clip(source, 0, 4)], np.where(np.isfinite(values), "known", "none"))
        if categories:
            decoded = np.where(np.isfinite(filled_values), np.array(categories + [None], dtype=object)[np.where(np.isfinite(filled_values), filled_values, len(categories)).astype(int)], None)
            filled_values = decoded
        physical_out = pd.DataFrame({f"{column}_filled": filled_values, f"{column}_is_est": is_est.astype(float), f"{column}_est_source": source_name},
                                    index=physical.index)
        mapped = physical_out.reindex(directed_key)
        mapped.index = roads.index
        filled = pd.concat([filled, mapped], axis=1)
    return {"validation": validation, "filled": filled, "physical": physical}

def coverage_after_filling(state_fill, result, totals):
    """For each filled feature: share of directed roads (and of accidents) that are known, estimated and still missing."""
    rows = {}
    for column in IMPUTE_FEATURES:
        known = result["road_features"][column].notna().to_numpy()
        estimated = state_fill["filled"][f"{column}_is_est"].to_numpy() == 1
        for basis, weights in (("roads", np.ones(len(known))), ("accidents", totals)):
            total = weights.sum()
            rows[(column, basis)] = {"known": (weights * known).sum() / total, "estimated": (weights * estimated).sum() / total,
                                     "still missing": (weights * ~(known | estimated)).sum() / total}
    table = pd.DataFrame(rows).T
    table.index.names = ["feature", "share of"]
    return table


# Source: osm_features_and_eda.ipynb, cell 71 (zero-based).
FEATURE_LABELS = {"road_character_maxspeed_mph": "speed limit", "road_character_lanes": "lanes", "road_character_surface": "surface"}

def validation_table(fills):
    """One row per state and feature: accuracy of each method on hidden values, and the gate decision."""
    rows = []
    for state, fill in fills.items():
        for column, entry in fill["validation"].items():
            row = {"state": state, "feature": FEATURE_LABELS[column], "decision": {"full": "neighbours + road type", "type": "road type only", "none": "not filled"}[entry["decision"]],
                   "reason": entry["reason"]}
            if "scores" in entry:
                row.update({"known values": entry["n_known"]})
                for method in METHOD_LABELS:
                    row[METHOD_LABELS[method]] = entry["scores"].loc[method, "accuracy"]
                if "expected_accuracy_on_missing" in entry:
                    row["expected accuracy on the missing roads"] = entry["expected_accuracy_on_missing"]
            rows.append(row)
    return pd.DataFrame(rows).set_index(["state", "feature"])


# Source: osm_features_and_eda.ipynb, cell 81 (zero-based).
def expected_accidents(levels, strata_columns, n_years):
    """How many accidents each road would have if only its stratum mattered.

    Why: a feature can look risky just because the roads that have it are long, or are major roads. To control for
    that, roads are grouped into strata (for example length band x road type). Within a stratum every road is given
    the same rate per km-year, so a road's expected accidents = stratum rate x its own km-years.

    Takes: the table from `add_analysis_levels`, the stratum columns, and the period length in years.
    Returns: an array aligned with `levels`.
    """
    km_years = levels["length_m"] / 1000 * n_years
    stratum_accidents = levels.groupby(strata_columns, observed=True, dropna=False)["accidents"].transform("sum")
    stratum_exposure = km_years.groupby([levels[c] for c in strata_columns], observed=True, dropna=False).transform("sum")
    return (stratum_accidents / stratum_exposure * km_years).to_numpy()

def rate_by_level(levels, variable, n_years):
    """Accident rate per km-year for each level of one variable, with a 95% confidence interval.

    Why: raw counts mislead because long roads collect more accidents. A rate per kilometre of road per year
    puts roads of different length on one scale.

    How: for each level, rate = accidents / (total km x years). The interval is the exact Poisson interval:
    lower = chi2.ppf(0.025, 2k) / 2 / E and upper = chi2.ppf(0.975, 2k + 2) / 2 / E, with k accidents and E km-years.

    Takes: the table from `add_analysis_levels`, the variable name, and the length of the period in years.
    Returns: one row per level: roads, accidents, km, rate, interval, and the share of roads with any accident.
    """
    grouped = levels.groupby(variable, observed=True)
    result = pd.DataFrame({
        "roads": grouped.size(),
        "accidents": grouped["accidents"].sum(),
        "km": grouped["length_m"].sum() / 1000,
        "share_roads_with_accident": grouped["accidents"].apply(lambda s: (s > 0).mean()),
    })
    exposure = result["km"] * n_years
    k = result["accidents"]
    result["rate"] = k / exposure
    result["ci_low"] = np.where(k > 0, chi2.ppf(0.025, 2 * k) / 2 / exposure, 0.0)
    result["ci_high"] = chi2.ppf(0.975, 2 * k + 2) / 2 / exposure
    return result

def build_physical_road_table(roads, road_features, highway_class, accident_total):
    """One row per PHYSICAL road: the two directions of a two-way road are merged.

    Why: the dataset stores each accident on only one direction of a two-way road (brief section 1b), so the
    other direction always shows zero. Summing both directions removes that artefact.

    How: accidents are added over both directions; length is the mean; for the features, the larger value of
    the two directions is used (OSM features are the same in both; "at the start / end" features are combined
    into "at either end").

    Takes: `roads` (after matching), `road_features`, `highway_class`, and the accident totals per directed road.
    Returns: a DataFrame indexed by (road_u, road_v) with the columns used by `add_analysis_levels`.
    """
    f = road_features
    directed = pd.DataFrame({
        "road_u": roads["road_u"].to_numpy(), "road_v": roads["road_v"].to_numpy(),
        "length_m": roads["length_m"].to_numpy(), "accidents": accident_total,
        "highway_class": highway_class.to_numpy(), "matched": (roads["osm_row"] >= 0).to_numpy(),
        # road_character
        "speed_mph": f["road_character_maxspeed_mph"].to_numpy(), "lanes": f["road_character_lanes"].to_numpy(),
        "surface": f["road_character_surface"].to_numpy(), "bridge": f["road_character_bridge"].to_numpy(),
        # intersection_control: at either end of the road, and mid-block
        "signal_at_end": np.fmax(f["intersection_control_signal_start"], f["intersection_control_signal_end"]),
        "stop_at_end": np.fmax(f["intersection_control_stop_start"], f["intersection_control_stop_end"]),
        "crossing_at_end": np.fmax(f["intersection_control_crossing_start"], f["intersection_control_crossing_end"]),
        "signal_midblock": f["intersection_control_signal_midblock"].to_numpy(), "stop_midblock": f["intersection_control_stop_midblock"].to_numpy(),
        "crossing_midblock": f["intersection_control_crossing_midblock"].to_numpy(), "calming_midblock": f["intersection_control_traffic_calming_midblock"].to_numpy(),
        "roundabout": f["intersection_control_roundabout"].to_numpy(),
        # cycle_pedestrian_lighting
        "cycleway": f["cycle_pedestrian_lighting_cycleway_any"].to_numpy(), "sidewalk": f["cycle_pedestrian_lighting_sidewalk_any"].to_numpy(), "lit": f["cycle_pedestrian_lighting_lit"].to_numpy(),
        # nearby_places
        "crossings_250m": f["nearby_places_crossing_250m"].to_numpy(), "parking_250m": f["nearby_places_parking_250m"].to_numpy(),
        "food_250m": f["nearby_places_food_250m"].to_numpy(), "signals_250m": f["nearby_places_signal_250m"].to_numpy(),
        "school_nearest_m": f["nearby_places_school_nearest_m"].to_numpy(),
        # road_shape
        "sinuosity": f["road_shape_sinuosity"].to_numpy(),
    })
    how = {column: "max" for column in directed.columns if column not in ("road_u", "road_v")}
    how.update({"length_m": "mean", "accidents": "sum", "highway_class": "first", "surface": "first"})
    return directed.groupby(["road_u", "road_v"]).agg(how)


# Source: osm_features_and_eda.ipynb, cell 82 (zero-based).
LEVEL_ORDER = {
    "length": ["<50 m", "50-100 m", "100-200 m", "200-500 m", "0.5-1 km", ">1 km"],
    "speed_limit": ["<=20", "25", "30", "35", "40", "45", "50", "55+"],
    "lanes": ["1", "2", "3", "4+"],
    "signal_midblock": ["0", "1", "2+"], "stop_midblock": ["0", "1", "2+"], "crossing_midblock": ["0", "1", "2+"], "calming_midblock": ["0", "1", "2+"],
    "crossings_within_250m": ["0", "1-2", "3-5", "6+"], "parking_within_250m": ["0", "1-2", "3-5", "6+"],
    "food_within_250m": ["0", "1-2", "3-5", "6+"], "signals_within_250m": ["0", "1-2", "3-5", "6+"],
    "nearest_school": ["<250 m", "250-500 m", "0.5-1 km", ">1 km"],
    "sinuosity": ["straight", "<5% longer", "5-20% longer", ">20% longer"],
}

BINARY_VARIABLES = ["bridge", "signal_at_end", "stop_at_end", "crossing_at_end", "roundabout", "cycleway", "sidewalk", "lit"]

VARIABLE_LABELS = {
    "speed_limit": "speed limit (mph)", "lanes": "lanes", "surface": "surface", "bridge": "on a bridge",
    "signal_at_end": "signal at a road end", "stop_at_end": "stop sign at a road end", "crossing_at_end": "crossing at a road end", "roundabout": "roundabout",
    "signal_midblock": "signals mid-block", "stop_midblock": "stop signs mid-block", "crossing_midblock": "crossings mid-block", "calming_midblock": "traffic calming mid-block",
    "cycleway": "cycleway tagged", "sidewalk": "sidewalk tagged", "lit": "street lighting",
    "crossings_within_250m": "crossings within 250 m", "parking_within_250m": "parking within 250 m", "food_within_250m": "food places within 250 m",
    "signals_within_250m": "signals within 250 m", "nearest_school": "distance to nearest school", "sinuosity": "curviness (length vs straight line)",
    "road_type": "road type",
}

FEATURE_GROUP_OF_VARIABLE = {
    "speed_limit": "road_character", "lanes": "road_character", "surface": "road_character", "bridge": "road_character",
    "signal_at_end": "intersection_control", "stop_at_end": "intersection_control", "crossing_at_end": "intersection_control", "roundabout": "intersection_control",
    "signal_midblock": "intersection_control", "stop_midblock": "intersection_control", "crossing_midblock": "intersection_control", "calming_midblock": "intersection_control",
    "cycleway": "cycle_pedestrian_lighting", "sidewalk": "cycle_pedestrian_lighting", "lit": "cycle_pedestrian_lighting",
    "crossings_within_250m": "nearby_places", "parking_within_250m": "nearby_places", "food_within_250m": "nearby_places",
    "signals_within_250m": "nearby_places", "nearest_school": "nearby_places",
    "sinuosity": "road_shape",
}

def add_analysis_levels(table):
    """Group each feature into a few labelled levels so rates can be compared (for example speed limit "25", "30", "35").

    Takes the table from `build_physical_road_table`. Returns a DataFrame with the same rows plus `accidents` and `length_m`, and one column per
    analysis variable. **A value that is missing (no tag, or the road did not match OSM) stays missing (NaN)**; it is never turned into a level.
    Dense variables (nearby places, shape) have a value on every road; flags from OSM nodes exist on matched roads only; tag variables exist where the tag exists.
    """
    def binned(values, edges, labels):
        return pd.cut(values, edges, labels=labels).astype(object)

    def yes_no(values):
        return pd.Series(np.where(values > 0, "yes", "no"), index=values.index).where(values.notna())

    def count_bins(values):
        return binned(values, [-1, 0, 2, 5, np.inf], LEVEL_ORDER["crossings_within_250m"])

    def at_most_two(values):
        return binned(values, [-1, 0, 1, np.inf], LEVEL_ORDER["signal_midblock"])

    surface_top = table["surface"].value_counts().index[:4]
    levels = pd.DataFrame(index=table.index)
    levels["length_m"], levels["accidents"] = table["length_m"], table["accidents"]

    levels["length"] = binned(table["length_m"], [0, 50, 100, 200, 500, 1000, np.inf], LEVEL_ORDER["length"])
    levels["road_type"] = table["highway_class"]
    levels["speed_limit"] = binned(table["speed_mph"], [0, 20, 25, 30, 35, 40, 45, 50, 200], LEVEL_ORDER["speed_limit"])
    levels["lanes"] = binned(table["lanes"], [0, 1, 2, 3, 100], LEVEL_ORDER["lanes"])
    levels["surface"] = table["surface"].where(table["surface"].isin(surface_top), "other").where(table["surface"].notna())
    levels["bridge"] = yes_no(table["bridge"])
    for name in ("signal_at_end", "stop_at_end", "crossing_at_end", "roundabout"):
        levels[name] = yes_no(table[name])
    for name in ("signal_midblock", "stop_midblock", "crossing_midblock", "calming_midblock"):
        levels[name] = at_most_two(table[name])
    for name in ("cycleway", "sidewalk", "lit"):
        levels[name] = yes_no(table[name])

    levels["crossings_within_250m"] = count_bins(table["crossings_250m"])
    levels["parking_within_250m"] = count_bins(table["parking_250m"])
    levels["food_within_250m"] = count_bins(table["food_250m"])
    levels["signals_within_250m"] = count_bins(table["signals_250m"])
    levels["nearest_school"] = binned(table["school_nearest_m"], [-1, 250, 500, 1000, np.inf], LEVEL_ORDER["nearest_school"])
    levels["sinuosity"] = binned(table["sinuosity"], [0, 1.001, 1.05, 1.2, np.inf], LEVEL_ORDER["sinuosity"])
    return levels

def controlled_ratio_by_level(levels, variable, n_years, strata_columns):
    """Observed / expected accidents for each level of one variable, controlling for the strata.

    A ratio of 1 means the level has as many accidents as roads of the same strata (length band, and road type) usually do; 1.5 means 50% more.
    Only rows where `variable` has a value are used, so the expected rate comes from comparable roads that also have a value.

    Two 95% intervals are returned:
      ci_low / ci_high: robust (use this one). Accidents cluster on a few roads, so counts are far more spread out than Poisson.
        se(log ratio) = sqrt(sum over roads of (observed - ratio x expected)^2) / observed.   `se_log` is returned too.
      poisson_low / poisson_high: the plain exact Poisson interval, kept to show how much too narrow it is.
    Returns one row per level: observed, expected, ratio, se_log and the intervals.
    """
    levels = levels[levels[variable].notna()]
    frame = pd.DataFrame({"level": levels[variable].to_numpy(), "observed": levels["accidents"].to_numpy(),
                          "expected": expected_accidents(levels, strata_columns, n_years)})
    sums = frame.groupby("level", observed=True)[["observed", "expected"]].sum()
    ratio = sums["observed"] / sums["expected"]

    frame["residual_sq"] = (frame["observed"] - frame["level"].map(ratio) * frame["expected"]) ** 2
    se_log = np.sqrt(frame.groupby("level", observed=True)["residual_sq"].sum()) / sums["observed"].replace(0, np.nan)

    result = sums.assign(ratio=ratio, se_log=se_log, ci_low=ratio * np.exp(-1.96 * se_log), ci_high=ratio * np.exp(1.96 * se_log))
    k = sums["observed"]
    result["poisson_low"] = np.where(k > 0, chi2.ppf(0.025, 2 * np.maximum(k, 1)) / 2 / sums["expected"], 0.0)
    result["poisson_high"] = chi2.ppf(0.975, 2 * k + 2) / 2 / sums["expected"]
    return result

def controlled_ratios_for_all_variables(levels, n_years):
    """Controlled ratios for every variable, on known values only, with two controls side by side.

    `_len`: controls for road length only. `_len_type`: controls for length and road type together, which answers "does this feature add
    anything beyond road type and length?". `road_type` itself is only controlled for length.
    `known_share` and `known_accident_share`: how many of all roads, and of all accidents, have a value for the variable, i.e. how much
    of the state the finding speaks for.
    `differs` is True only when ALL hold: at least MIN_ACCIDENTS_FOR_VERDICT accidents; the robust interval excludes 1; the ratio is
    outside 1/MIN_EFFECT_RATIO .. MIN_EFFECT_RATIO.
    """
    variables = [c for c in levels.columns if c not in ("length_m", "accidents", "length")]
    parts = []
    for variable in variables:
        known = levels[variable].notna()
        if levels.loc[known, variable].nunique() < 2:
            continue
        subset = levels[known]
        by_length = controlled_ratio_by_level(levels, variable, n_years, ["length"])
        part = pd.DataFrame({"variable": variable, "roads": subset.groupby(variable, observed=True).size(), "accidents": by_length["observed"],
                             "ratio_len": by_length["ratio"], "low_len": by_length["ci_low"], "high_len": by_length["ci_high"]})
        if variable != "road_type":
            by_both = controlled_ratio_by_level(levels, variable, n_years, ["length", "road_type"])
            part["ratio_len_type"], part["low_len_type"], part["high_len_type"] = by_both["ratio"], by_both["ci_low"], by_both["ci_high"]
            part["se_len_type"] = by_both["se_log"]
            part["poisson_low"], part["poisson_high"] = by_both["poisson_low"], by_both["poisson_high"]
        else:
            part["ratio_len_type"] = part["low_len_type"] = part["high_len_type"] = part["se_len_type"] = np.nan
            part["poisson_low"], part["poisson_high"] = by_length["poisson_low"], by_length["poisson_high"]
        part["se_len"] = by_length["se_log"]
        part["known_share"] = known.mean()
        part["known_accident_share"] = subset["accidents"].sum() / max(levels["accidents"].sum(), 1)
        part.index.name = "level"
        parts.append(part.reset_index())
    table = pd.concat(parts, ignore_index=True)

    ratio = table["ratio_len_type"].fillna(table["ratio_len"])
    low = table["low_len_type"].fillna(table["low_len"])
    high = table["high_len_type"].fillna(table["high_len"])
    table["differs"] = ((table["accidents"] >= MIN_ACCIDENTS_FOR_VERDICT) & ((low > 1) | (high < 1))
                        & ((ratio > MIN_EFFECT_RATIO) | (ratio < 1 / MIN_EFFECT_RATIO)))
    table["ratio_used"], table["low_used"], table["high_used"] = ratio, low, high
    table["se_used"] = table["se_len_type"].fillna(table["se_len"])
    return table

def describe_trend(rows, order):
    """Direction of a variable's rate across its ordered levels, from the judged levels only.

    'rises' / 'falls': the rate changes by at least the minimum effect from the lowest to the highest judged level AND the weighted straight-line trend
    is clear (p < 0.05; for a two-level variable, the levels clearly differ). 'mixed': some level clearly differs but there is no steady direction.
    'flat': no level clearly differs. 'too few levels': fewer than two levels with enough accidents.
    """
    judged = rows[rows["accidents"] >= MIN_ACCIDENTS_FOR_VERDICT].copy()
    judged["position"] = judged["level"].map({level: i for i, level in enumerate(order)})
    judged = judged.dropna(subset=["position"]).sort_values("position")
    if len(judged) < 2:
        return {"direction": "too few levels", "first_level": None, "last_level": None, "first_ratio": np.nan, "last_ratio": np.nan, "slope": np.nan, "p_value": np.nan}
    first, last = judged.iloc[0], judged.iloc[-1]
    end_to_end = last["ratio_used"] / first["ratio_used"]
    slope, p_value = (np.nan, np.nan)
    if len(judged) >= 3:
        slope, p_value = weighted_trend(judged["position"], np.log(judged["ratio_used"]), judged["se_used"])
        steady = np.isfinite(p_value) and p_value < 0.05
    else:
        steady = bool(rows["differs"].any())
    if steady and end_to_end >= MIN_EFFECT_RATIO:
        direction = "rises"
    elif steady and end_to_end <= 1 / MIN_EFFECT_RATIO:
        direction = "falls"
    elif rows["differs"].any():
        direction = "mixed"
    else:
        direction = "flat"
    return {"direction": direction, "first_level": first["level"], "last_level": last["level"], "first_ratio": first["ratio_used"],
            "last_ratio": last["ratio_used"], "slope": slope, "p_value": p_value}

UNORDERED_VARIABLES = ("surface", "road_type")

def verdict_by_variable(controlled):
    """One line per variable: coverage, the trend direction, and the strongest level.

    Returns a table with: how many roads / accidents have a known value, the number of levels judged and that differ, the strongest level
    with its ratio and interval, the trend direction (ordered variables) and a verdict.
    """
    judged = controlled[(controlled["accidents"] >= MIN_ACCIDENTS_FOR_VERDICT)].copy()
    judged["distance"] = np.abs(np.log(judged["ratio_used"]))
    strongest = judged.sort_values("distance", ascending=False).groupby("variable").head(1).set_index("variable")
    summary = pd.DataFrame({
        "known_share_of_roads": controlled.groupby("variable")["known_share"].first(),
        "known_share_of_accidents": controlled.groupby("variable")["known_accident_share"].first(),
        "levels_judged": judged.groupby("variable").size(),
        "levels_that_differ": controlled.groupby("variable")["differs"].sum(),
    })
    summary["strongest_level"] = strongest["level"]
    summary["ratio"], summary["ci_low"], summary["ci_high"] = strongest["ratio_used"], strongest["low_used"], strongest["high_used"]
    empty = {"direction": "unordered", "first_level": None, "last_level": None, "first_ratio": np.nan, "last_ratio": np.nan, "slope": np.nan, "p_value": np.nan}
    trends = {variable: (empty if variable in UNORDERED_VARIABLES else describe_trend(controlled[controlled["variable"] == variable], LEVEL_ORDER[variable]))
              for variable in summary.index}
    summary["trend"] = pd.Series({v: t["direction"] for v, t in trends.items()})
    summary["from_level"], summary["from_ratio"] = pd.Series({v: t["first_level"] for v, t in trends.items()}), pd.Series({v: t["first_ratio"] for v, t in trends.items()})
    summary["to_level"], summary["to_ratio"] = pd.Series({v: t["last_level"] for v, t in trends.items()}), pd.Series({v: t["last_ratio"] for v, t in trends.items()})
    summary["trend_p"] = pd.Series({v: t["p_value"] for v, t in trends.items()})
    summary["verdict"] = np.where(summary["levels_that_differ"] > 0, "meaningful difference", "no meaningful difference")
    return summary.sort_values("ratio", key=lambda s: np.abs(np.log(s)), ascending=False)

def analyse_state_window(state, window, result, info):
    """Controlled ratios, verdicts and trends for one state in one accident window ('train' or 'full')."""
    first, last = get_window(state, window, info)
    totals, years = accident_totals_in_period(info, result["roads"], first, last)
    physical = build_physical_road_table(result["roads"], result["road_features"], result["highway_class"], totals)
    # Roads with more than one road type cannot be put in one comparison group, so they are left out of the trend analysis (the comparison groups need one type per road).
    multi = result["highway_class"].attrs.get("multi_type")
    left_out = {"roads": 0, "accidents": 0.0, "road_share": 0.0, "accident_share": 0.0}
    if multi is not None and multi.any():
        key = pd.MultiIndex.from_arrays([result["roads"]["road_u"].to_numpy(), result["roads"]["road_v"].to_numpy()])
        flagged = pd.Series(multi, index=key).groupby(level=[0, 1]).any().reindex(physical.index).fillna(False).to_numpy().astype(bool)
        left_out = {"roads": int(flagged.sum()), "accidents": float(physical.loc[flagged, "accidents"].sum()),
                    "road_share": float(flagged.mean()), "accident_share": float(physical.loc[flagged, "accidents"].sum() / max(physical["accidents"].sum(), 1))}
        physical = physical[~flagged]
    levels = add_analysis_levels(physical)
    controlled = controlled_ratios_for_all_variables(levels, years)
    return {"window": window, "first": first, "last": last, "years": years, "levels": levels, "controlled": controlled, "left_out_multi_type": left_out,
            "verdict": verdict_by_variable(controlled), "accidents": float(levels["accidents"].sum()), "physical_roads": len(levels)}


# Source: osm_features_and_eda.ipynb, cell 87 (zero-based).
def within_road_type_table(analyses_window, variables=("speed_limit", "lanes"), top_types=4):
    """Controlled ratio of each variable by value inside each of the main road types (length held fixed), per state.

    Takes the headline analyses. Returns one row per state, variable, road type and level with at least MIN_ACCIDENTS_FOR_VERDICT accidents:
    accidents, ratio, robust interval, and `clear` (interval excludes 1 and the ratio is outside 1/MIN_EFFECT_RATIO .. MIN_EFFECT_RATIO).
    """
    rows = []
    for state, analysis in analyses_window.items():
        levels, years = analysis["levels"], analysis["years"]
        types = levels.groupby("road_type")["accidents"].sum().sort_values(ascending=False).index[:top_types]
        for variable in variables:
            for road_type in types:
                subset = levels[levels["road_type"] == road_type]
                if subset[variable].nunique() < 2:
                    continue
                table = controlled_ratio_by_level(subset, variable, years, ["length"])
                for level in LEVEL_ORDER[variable]:
                    if level in table.index and table.loc[level, "observed"] >= MIN_ACCIDENTS_FOR_VERDICT:
                        r = table.loc[level]
                        clear = (r["ci_low"] > 1 or r["ci_high"] < 1) and (r["ratio"] > MIN_EFFECT_RATIO or r["ratio"] < 1 / MIN_EFFECT_RATIO)
                        rows.append({"state": state, "variable": VARIABLE_LABELS[variable], "road type": road_type, "level": level, "accidents": r["observed"],
                                     "ratio": r["ratio"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "clear": bool(clear)})
    return pd.DataFrame(rows)


# Source: osm_features_and_eda.ipynb, cell 91 (zero-based).
DENSE_MODEL_FEATURES = ["bridge", "signal_at_end", "stop_at_end", "crossing_at_end", "roundabout", "signal_midblock", "stop_midblock",
                        "crossing_midblock", "calming_midblock", "crossings_within_250m", "parking_within_250m", "food_within_250m",
                        "signals_within_250m", "nearest_school", "sinuosity"]

TAG_MODEL_FEATURES = ["speed_limit", "lanes", "surface", "cycleway", "sidewalk", "lit"]

TAG_MODEL_CONTROLS = ["signals_within_250m"]

def common_references(levels_list, variables):
    """For each variable, the level that is most common across all the given tables (known values only)."""
    references = {}
    for variable in variables:
        parts = [levels[variable].value_counts() for levels in levels_list if variable in levels.columns]
        if parts:
            counts = pd.concat(parts).groupby(level=0).sum()
            if len(counts):
                references[variable] = counts.idxmax()
    return references

NOT_ESTIMABLE_INTERVAL_WIDTH = 1000

def fit_rate_model(levels, n_years, features, references, model_name):
    """One Poisson model of accident counts on the given features plus road length and road type; rows with any missing value are dropped.

    Returns one row per variable level: roads, accidents, rate_ratio against the reference level, 95% interval, p_value, `estimable` (False when the interval is absurdly wide, e.g. a level fully explained by another feature; its estimates are NaN), `clear` (at least
    MIN_ACCIDENTS_FOR_VERDICT accidents, interval excludes 1, and ratio outside 1/MIN_EFFECT_RATIO .. MIN_EFFECT_RATIO), `is_control` and `model`. The table's `.attrs` hold the number of roads and accidents used.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    needed = ["length", "road_type"] + list(features)
    data = levels.dropna(subset=needed).copy()
    type_counts = data["road_type"].map(data["road_type"].value_counts())
    data["road_type"] = data["road_type"].where(type_counts >= MIN_ROADS_FOR_ROAD_TYPE, "rare_type")
    km_years = data["length_m"] / 1000 * n_years

    variables = [v for v in needed if data[v].nunique() >= 2]
    reference = {v: (references.get(v) if references.get(v) in set(data[v].unique()) else data[v].value_counts().idxmax()) for v in variables}
    terms = [f"C({v}, Treatment(reference='{reference[v]}'))" for v in variables]
    fit = smf.glm("accidents ~ " + " + ".join(terms), data=data, family=sm.families.Poisson(), offset=np.log(km_years)).fit(cov_type="HC1")

    interval = fit.conf_int()
    rows = []
    for name in fit.params.index:
        match = re.match(r"C\((\w+), Treatment\(reference='(.*?)'\)\)\[T\.(.*)\]$", name)
        if match is None:  # the intercept
            continue
        variable, level = match.group(1), match.group(3)
        rows.append({"variable": variable, "level": level, "reference": match.group(2), "roads": int((data[variable] == level).sum()),
                     "accidents": float(data.loc[data[variable] == level, "accidents"].sum()),
                     "rate_ratio": np.exp(fit.params[name]), "ci_low": np.exp(interval.loc[name, 0]), "ci_high": np.exp(interval.loc[name, 1]),
                     "p_value": fit.pvalues[name], "is_control": variable in ("length", "road_type"), "model": model_name})
    result = pd.DataFrame(rows)
    # A level that is almost perfectly explained by another feature (e.g. every 55 mph road is a motorway) has no usable estimate: the model
    # returns an interval from 0 to infinity. Report those as not estimable (NaN) instead of printing a meaningless number.
    width = result["ci_high"] / result["ci_low"].replace(0, np.nan)
    result["estimable"] = np.isfinite(width) & (width < NOT_ESTIMABLE_INTERVAL_WIDTH)
    result.loc[~result["estimable"], ["rate_ratio", "ci_low", "ci_high", "p_value"]] = np.nan
    result["clear"] = ((result["accidents"] >= MIN_ACCIDENTS_FOR_VERDICT) & ((result["ci_low"] > 1) | (result["ci_high"] < 1))
                       & ((result["rate_ratio"] > MIN_EFFECT_RATIO) | (result["rate_ratio"] < 1 / MIN_EFFECT_RATIO)))
    result.attrs.update({"n_roads": int(len(data)), "n_accidents": float(data["accidents"].sum())})
    return result

def fit_all_models(levels, n_years, references):
    """The dense joint model plus one model per sparse tag (each on the roads where the tag is known). Tag models keep only their own tag's rows."""
    parts = [fit_rate_model(levels, n_years, DENSE_MODEL_FEATURES, references, "dense")]
    sizes = {"dense": (parts[0].attrs["n_roads"], parts[0].attrs["n_accidents"])}
    for tag in TAG_MODEL_FEATURES:
        if levels[tag].notna().sum() < 500 or levels.loc[levels[tag].notna(), tag].nunique() < 2:
            continue
        try:
            table = fit_rate_model(levels, n_years, [tag] + TAG_MODEL_CONTROLS, references, f"tag: {tag}")
        except Exception as error:   # a very small or degenerate subset can fail to fit; say so rather than stopping
            print(f"  tag model for {tag} could not be fitted ({type(error).__name__}: {error})")
            continue
        sizes[tag] = (table.attrs["n_roads"], table.attrs["n_accidents"])
        parts.append(table[table["variable"] == tag])
    result = pd.concat(parts, ignore_index=True)
    result.attrs["model_sizes"] = sizes
    return result

def compare_joint_models(joint_a, joint_b, name_a, name_b):
    """Put two states' joint-model results side by side to see which findings replicate.

    Rate ratios are only comparable when both models used the same reference level for the variable (the most common
    level can differ between states), so rows with different references are marked and left out of the counts.

    Takes: two tables from `fit_joint_rate_model` and the names to label them.
    Returns: one row per variable level with both rate ratios, `same_direction` (both above 1 or both below 1) and
    `clear_in_both`. Prints how many of the levels that are clear in the first state also hold in the second.
    """
    merged = joint_a.merge(joint_b, on=["variable", "level"], suffixes=(f"_{name_a}", f"_{name_b}"))
    merged["same_reference"] = merged[f"reference_{name_a}"] == merged[f"reference_{name_b}"]
    ratio_a, ratio_b = merged[f"rate_ratio_{name_a}"], merged[f"rate_ratio_{name_b}"]
    merged["same_direction"] = (ratio_a - 1) * (ratio_b - 1) > 0
    merged["clear_in_both"] = merged[f"clear_{name_a}"] & merged[f"clear_{name_b}"]

    comparable = merged[merged["same_reference"]]
    clear_a = comparable[comparable[f"clear_{name_a}"]]
    print(f"levels compared: {len(comparable)} ({len(merged) - len(comparable)} skipped: different reference level)")
    print(f"clear in {name_a}: {len(clear_a)}; same direction in {name_b}: {int(clear_a['same_direction'].sum())}; "
          f"also clear in {name_b}: {int(clear_a['clear_in_both'].sum())}")
    columns = ["variable", "level", f"reference_{name_a}", f"rate_ratio_{name_a}", f"rate_ratio_{name_b}",
               f"clear_{name_a}", f"clear_{name_b}", "same_direction", "same_reference"]
    return merged[columns]


# Source: osm_features_and_eda.ipynb, cell 94 (zero-based).
def effect_table(models_window, first, second):
    """Both states' effects side by side for feature levels (not controls) with the same reference level."""
    tables = {s: models_window[s].query("not is_control").set_index(["variable", "level"]) for s in (first, second)}
    both = tables[first].join(tables[second], how="inner", lsuffix=f"_{first}", rsuffix=f"_{second}")
    both = both[both[f"reference_{first}"] == both[f"reference_{second}"]].copy()
    both["group"] = [FEATURE_GROUP_OF_VARIABLE.get(v, "?") for v, _ in both.index]
    both["same_direction"] = (both[f"rate_ratio_{first}"] - 1) * (both[f"rate_ratio_{second}"] - 1) > 0
    both["replicates"] = both[f"clear_{first}"] & both[f"clear_{second}"] & both["same_direction"]
    return both

def plot_effect_summary(models_window, first, second, max_rows=40):
    """Figure 5: grouped forest plot of feature effects in both states."""
    apply_chart_style()
    both = effect_table(models_window, first, second)
    both = both[both[f"clear_{first}"] | both[f"clear_{second}"]]
    both = both.assign(order=both["group"], sort=both[f"rate_ratio_{first}"]).sort_values(["order", "sort"])
    if len(both) > max_rows:   # keep the strongest if there are very many
        strength = np.abs(np.log(both[[f"rate_ratio_{first}", f"rate_ratio_{second}"]])).max(axis=1)
        both = both.loc[strength.sort_values(ascending=False).index[:max_rows]].sort_values(["order", "sort"])
    n_rows = len(both) + both["group"].nunique()
    fig, ax = plt.subplots(figsize=(11.5, 0.33 * n_rows + 2.0), layout="constrained")
    y, ticks, labels, current = 0, [], [], None
    for (variable, level), line in both.iterrows():
        if line["group"] != current:
            current = line["group"]
            ax.text(0.16, y, FEATURE_GROUP_NAMES.get(current, current).upper(), fontsize=8.5, fontweight="bold", color=INK_SECONDARY, va="center")
            y += 1
        for offset, state in zip((-0.15, 0.15), (first, second)):
            ratio, low, high = line[f"rate_ratio_{state}"], line[f"ci_low_{state}"], line[f"ci_high_{state}"]
            colour, filled = STATE_COLORS[state], line[f"clear_{state}"]
            ax.plot([low, high], [y + offset] * 2, color=colour, lw=1.6, solid_capstyle="round")
            ax.plot(ratio, y + offset, "o", ms=7, mfc=colour if filled else SURFACE, mec=colour, mew=1.6)
        if line["replicates"]:
            ax.text(5.6, y, "✓", fontsize=11, color=INK, va="center", ha="center")
        ticks.append(y); labels.append(f"{VARIABLE_LABELS.get(variable, variable)}: {level}  (vs {line[f'reference_{first}']})")
        y += 1
    ax.axvline(1, color=INK_SECONDARY, lw=1)
    ax.set_xscale("log"); ax.set_xlim(0.15, 6.5)
    ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x")); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9); ax.set_ylim(y - 0.3, -0.8)
    ax.xaxis.grid(True)
    ax.set_xlabel("accident rate ratio per km, other features held fixed (log scale)")
    n_rep = int(both["replicates"].sum())
    ax.set_title(f"Figure 5. {n_rep} of {len(both)} feature levels shown are clear and point the same way in both states (✓)")
    handles = [Line2D([], [], color=STATE_COLORS[s], marker="o", ms=7, lw=1.6, label=s) for s in (first, second)]
    handles.append(Line2D([], [], color=INK_SECONDARY, marker="o", ms=7, mfc=SURFACE, lw=0, label="not clearly different"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9)
    save_figure(fig, "fig7_effect_summary")
    plt.show()
    return both

def effect_table_one_state(model_window, window_comparison):
    """One state's joint-model effects for feature levels, with the training-years check.

    Takes the state's headline model table and the table from `compare_joint_models(train, full)`.
    Returns one row per (variable, level): the model columns plus `group` and `holds_in_training_years`
    (clear in both windows, same direction, same reference level).
    """
    table = model_window.query("not is_control").set_index(["variable", "level"])
    check = window_comparison.set_index(["variable", "level"])
    holds = check["same_reference"] & check["clear_train"] & check["clear_full"] & check["same_direction"]
    table["holds_in_training_years"] = holds.reindex(table.index).fillna(False).astype(bool)
    table["group"] = [FEATURE_GROUP_OF_VARIABLE.get(variable, "?") for variable, _ in table.index]
    return table

def plot_effect_summary_one_state(effects, state):
    """Figure 5 (one state): feature levels that clearly differ, other features held fixed; a tick marks those that also hold in the training years."""
    apply_chart_style()
    shown = effects[effects["clear"]].assign(order=lambda t: t["group"]).sort_values(["order", "rate_ratio"])
    if not len(shown):
        print("No feature level is clearly different.")
        return
    n_rows = len(shown) + shown["group"].nunique()
    fig, ax = plt.subplots(figsize=(11.5, 0.33 * n_rows + 2.0), layout="constrained")
    y, ticks, labels, current = 0, [], [], None
    for (variable, level), line in shown.iterrows():
        if line["group"] != current:
            current = line["group"]
            ax.text(0.16, y, FEATURE_GROUP_NAMES.get(current, current).upper(), fontsize=8.5, fontweight="bold", color=INK_SECONDARY, va="center")
            y += 1
        ax.plot([line["ci_low"], line["ci_high"]], [y, y], color=STATE_COLORS[state], lw=1.6, solid_capstyle="round")
        ax.plot(line["rate_ratio"], y, "o", ms=7, color=STATE_COLORS[state])
        if line["holds_in_training_years"]:
            ax.text(5.6, y, "✓", fontsize=11, color=INK, va="center", ha="center")
        ticks.append(y); labels.append(f"{VARIABLE_LABELS.get(variable, variable)}: {level}  (vs {line['reference']})")
        y += 1
    ax.axvline(1, color=INK_SECONDARY, lw=1)
    ax.set_xscale("log"); ax.set_xlim(0.15, 6.5)
    ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:g}x")); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9); ax.set_ylim(y - 0.3, -0.8)
    ax.xaxis.grid(True)
    ax.set_xlabel("accident rate ratio per km, other features held fixed (log scale)")
    n_hold = int(shown["holds_in_training_years"].sum())
    ax.set_title(f"Figure 5. {len(shown)} feature levels clearly differ in {state}; {n_hold} also hold in the paper's training years (✓)")
    save_figure(fig, "fig5_effect_summary")
    plt.show()

def unclear_levels_table(effects):
    """Feature levels that are NOT clearly different, with the reason (so nothing is hidden by the figure)."""
    rest = effects[~effects["clear"]].copy()
    reasons = []
    for _, row in rest.iterrows():
        if not row["estimable"]:
            reasons.append("not estimable (explained by another feature)")
        elif row["accidents"] < MIN_ACCIDENTS_FOR_VERDICT:
            reasons.append(f"fewer than {MIN_ACCIDENTS_FOR_VERDICT} accidents")
        elif row["ci_low"] <= 1 <= row["ci_high"]:
            reasons.append("interval includes 1")
        else:
            reasons.append(f"effect within {1 / MIN_EFFECT_RATIO:.2f}-{MIN_EFFECT_RATIO}x")
    rest["reason"] = reasons
    rest.index = [f"{VARIABLE_LABELS.get(variable, variable)}: {level} (vs {r})" for (variable, level), r in zip(rest.index, rest["reference"])]
    return rest[["group", "accidents", "rate_ratio", "ci_low", "ci_high", "reason"]]


# Source: osm_features_and_eda.ipynb, cell 98 (zero-based).
def summarise_feature_groups(models_window, stability_states, stability_windows, first, second):
    """One row per feature group: levels tested, clear in each state, clear in both with the same direction, and also stable in the paper's training years."""
    rows = {}
    if second is None:   # one state: tested, clear, and also clear in the training years
        table = models_window[first].query("not is_control")
        table = table[table["variable"].isin(FEATURE_GROUP_OF_VARIABLE)].assign(group=lambda t: t["variable"].map(FEATURE_GROUP_OF_VARIABLE))
        rows[f"levels_tested_{first}"] = table.groupby("group").size()
        rows[f"clear_{first}"] = table.groupby("group")["clear"].sum()
        windows = stability_windows[stability_windows["variable"].isin(FEATURE_GROUP_OF_VARIABLE)].copy()
        rows["also_hold_in_training_years"] = windows["replicated"].groupby(windows["variable"].map(FEATURE_GROUP_OF_VARIABLE)).sum()
        summary = pd.DataFrame(rows).fillna(0).astype(int)
        summary.insert(0, "what", [FEATURE_GROUP_NAMES[g] for g in summary.index])
        return summary
    for state in (first, second):
        table = models_window[state].query("not is_control")
        table = table[table["variable"].isin(FEATURE_GROUP_OF_VARIABLE)].assign(group=lambda t: t["variable"].map(FEATURE_GROUP_OF_VARIABLE))
        rows[f"levels_tested_{state}"] = table.groupby("group").size()
        rows[f"clear_{state}"] = table.groupby("group")["clear"].sum()
    both = stability_states[stability_states["variable"].isin(FEATURE_GROUP_OF_VARIABLE)].copy()
    both["group"] = both["variable"].map(FEATURE_GROUP_OF_VARIABLE)
    rows["clear_in_both_states"] = both["replicated"].groupby(both["group"]).sum()
    stable = set()
    for state, table in stability_windows.groupby("state"):
        stable |= {(state, v, l) for v, l, r in zip(table["variable"], table["level"], table["replicated"]) if r}
    both["stable_over_time"] = [all(((s, v, l) in stable) for s in (first, second)) for v, l in zip(both["variable"], both["level"])]
    rows["also_hold_in_training_years"] = (both["replicated"] & both["stable_over_time"]).groupby(both["group"]).sum()
    summary = pd.DataFrame(rows).fillna(0).astype(int)
    summary.insert(0, "what", [FEATURE_GROUP_NAMES[g] for g in summary.index])
    return summary


# Source: osm_features_and_eda.ipynb, cell 100 (zero-based).
def data_dictionary_entries():
    """Return {column: (group, kind, unit, meaning, missing_rule)} for every column the pipeline can write."""
    entries = {}
    add = lambda column, *info: entries.__setitem__(column, info)

    for column, meaning in (("node_1_idx", "start node of the directed road (row position in the dataset's node table)"),
                            ("node_2_idx", "end node of the directed road"), ("road_u", "smaller of the two node indices (key of the physical road)"),
                            ("road_v", "larger of the two node indices"), ("length_m", "road length recorded in the dataset")):
        add(column, "id", "id" if column != "length_m" else "numeric", "m" if column == "length_m" else "", meaning, "never missing")
    add("has_reverse", "id", "binary", "", "the opposite direction also exists (two-way road)", "never missing")
    add("u_lt_v", "id", "binary", "", "start node has the lower index (accidents are recorded mostly on this direction)", "never missing")
    add("match_type", "diagnostic", "category", "", "how the road was matched to OSM: exact, reverse, nearest, unmatched", "never missing")
    add("match_dist_m", "diagnostic", "numeric", "m", "distance to the OSM road for nearest matches (0 for exact and reverse)", "NaN for unmatched")
    add("len_ratio", "diagnostic", "numeric", "", "OSM road length / dataset road length", "NaN for unmatched")
    add("osm_highway", "diagnostic", "category", "", "the matched OSM road's own highway tag", "NaN for unmatched")
    add("cand_dist_m", "diagnostic", "numeric", "m", "distance of the nearest OSM candidate for roads that reached the nearest tier (also when rejected)", "NaN if exact/reverse matched")
    add("cand_len_ratio", "diagnostic", "numeric", "", "length ratio of that candidate", "NaN if exact/reverse matched")
    add("has_osm_match", "diagnostic", "binary", "", "1 if the road matched an OSM road (any tier)", "never missing")

    add("road_character_maxspeed_mph", "road_character", "numeric", "mph", "speed limit from the OSM road's maxspeed (highest if several; km/h converted)", "NaN: no parsable maxspeed tag, or road unmatched (unknown, not zero)")
    add("road_character_lanes", "road_character", "count", "lanes", "number of lanes from the OSM lanes tag (highest if several)", "NaN: no lanes tag, or unmatched")
    add("road_character_surface", "road_character", "category", "", "surface type ('mixed' if the road's OSM pieces have different surfaces); categories on fewer than %d roads are folded into 'other'" % RARE_CATEGORY_MIN_ROADS, "NaN: no surface tag, or unmatched")
    add("road_character_bridge", "road_character", "binary", "", "1 if the OSM road is tagged as a bridge", "0 = not tagged as bridge; NaN only for unmatched")
    add("road_character_tunnel", "road_character", "binary", "", "1 if the OSM road is tagged as a tunnel", "0 = not tagged as tunnel; NaN only for unmatched")
    for kind, name in (("signal", "traffic signals"), ("stop", "a stop sign"), ("give_way", "a give-way sign"), ("crossing", "a pedestrian crossing")):
        for end in ("start", "end"):
            add(f"intersection_control_{kind}_{end}", "intersection_control", "binary", "", f"1 if the OSM node at the road's {end} carries {name}", "0 = not mapped there (not necessarily absent in reality); NaN for unmatched")
    add("intersection_control_roundabout", "intersection_control", "binary", "", "1 if the OSM road is part of a roundabout", "0 = not tagged; NaN for unmatched")
    for kind in TAGGED_NODE_KINDS:
        add(f"intersection_control_{kind}_midblock", "intersection_control", "count", "points", f"{kind.replace('_', ' ')} points in the middle of the road, recovered from the XML after simplification removed them (within {MIDBLOCK_NODE_MAX_M} m of the road)", "0 = none found; NaN for unmatched")
    add("cycle_pedestrian_lighting_cycleway_any", "cycle_pedestrian_lighting", "binary", "", "any cycleway tag (cycleway, :left, :right, :both) says a cycle lane exists", "NaN = not tagged (unknown), or unmatched; 0 = tagged 'no'")
    add("cycle_pedestrian_lighting_sidewalk_any", "cycle_pedestrian_lighting", "binary", "", "the sidewalk tag says there is a sidewalk (includes 'separate')", "NaN = not tagged, or unmatched; 0 = tagged 'no'")
    add("cycle_pedestrian_lighting_lit", "cycle_pedestrian_lighting", "binary", "", "the lit tag says the road is lit", "NaN = not tagged, or unmatched; 0 = tagged 'no'")
    place_text = {"school": "schools", "bar_pub": "bars and pubs", "food": "restaurants and fast food", "parking": "parking places", "bus_stop": "bus stops",
                  "crossing": "pedestrian crossings", "signal": "traffic signals"}
    for kind, text in place_text.items():
        for radius in PLACE_RADII_M:
            add(f"nearby_places_{kind}_{radius}m", "nearby_places", "count", "places", f"number of {text} within {radius} m of the road's midpoint", "never missing")
        add(f"nearby_places_{kind}_nearest_m", "nearby_places", "distance", "m", f"distance from the road's midpoint to the nearest of {text}", "NaN only if the state has none at all")
    add("road_shape_sinuosity", "road_shape", "numeric", "ratio", "road length / straight-line distance between its ends (1 = straight)", "NaN if the two ends coincide")
    for column, text in (("road_character_maxspeed_mph", "speed limit"), ("road_character_lanes", "lane count"), ("road_character_surface", "surface")):
        add(f"{column}_filled", "road_character_filled", "numeric" if column != "road_character_surface" else "category", "mph" if column == "road_character_maxspeed_mph" else "",
            f"{text}: the real OSM value where known, otherwise a validated estimate (see {column}_is_est)", "NaN if the estimate method failed the gate (Stage 10) for this state")
        add(f"{column}_is_est", "road_character_filled", "binary", "", f"1 if the {text} in {column}_filled is an estimate, 0 if it is the real OSM value", "never missing")
        add(f"{column}_est_source", "diagnostic", "category", "", "known / neighbour / neighbour2 / road_type / state_value / none: where the estimate came from", "never missing")
    return entries


# Source: osm_features_and_eda.ipynb, cell 101 (zero-based).
MODEL_READY_ID_COLUMNS = ["node_1_idx", "node_2_idx", "road_u", "road_v", "has_reverse", "u_lt_v", "length_m", "match_type", "has_osm_match"]

def build_model_ready_table(state, result, state_fill):
    """Keys, raw features (unscaled, missing kept) and the filled columns with their flags, in the dataset's adjacency order."""
    roads, features = result["roads"], result["road_features"]
    ids = roads[["node_1_idx", "node_2_idx", "road_u", "road_v", "has_reverse", "u_lt_v", "length_m", "match_type"]].copy().reset_index(drop=True)
    ids["has_osm_match"] = (ids["match_type"] != "unmatched").astype(float)
    raw = features.drop(columns=["node_1_idx", "node_2_idx"]).reset_index(drop=True)
    filled = state_fill["filled"].reset_index(drop=True)
    table = pd.concat([ids, raw, filled], axis=1)
    assert len(table) == len(roads) and not table.columns.duplicated().any()
    assert not any("accident" in c.lower() for c in table.columns), "a label-derived column slipped into the feature table"
    return table

def redundant_pairs(table, threshold=0.9, sample=100_000, seed=RANDOM_SEED):
    """Pairs of numeric feature columns that move together (Spearman |rho| at or above the threshold), strongest first, on a random sample of roads."""
    numeric = table.select_dtypes("number").drop(columns=[c for c in table.columns if c in MODEL_READY_ID_COLUMNS or c.endswith("_is_est")], errors="ignore")
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if len(numeric) > sample:
        numeric = numeric.sample(sample, random_state=seed)
    corr = numeric.corr(method="spearman")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().rename("spearman").reset_index()
    pairs.columns = ["feature_a", "feature_b", "spearman"]
    return pairs[pairs["spearman"].abs() >= threshold].sort_values("spearman", key=abs, ascending=False).reset_index(drop=True)

PREPROCESSING_RECIPE = [
    "Features are RAW. Fit every scaler, encoder and imputer on the TRAINING years' roads only (SPLIT in the config), then apply to validation and test.",
    "Directed rows: OSM-derived values are identical in both directions of a road; the start/end columns swap meaning between the two directions.",
    "Counts and distances (nearby_places_*_{100,250,500,1000}m, nearby_places_*_nearest_m, intersection_control_*_midblock): apply log1p, then standardise. A nearest-distance is NaN only if a state has no place of that kind; fill with the maximum plus a flag.",
    "Road length, node degree and road type are NOT repeated here: take them from the benchmark's own features. road_shape holds only curviness (road_shape_sinuosity).",
    "Binary flags in intersection_control: NaN means the road was not matched (check has_osm_match). Fill with 0 and keep has_osm_match as an input. A 0 means 'not mapped there', which is weaker than 'absent'.",
    "cycle_pedestrian_lighting flags are three-state: 1 yes, 0 tagged 'no', NaN not tagged. Encode as three categories rather than filling; they are too sparse to fill.",
    "road_character speed limit, lanes, surface: use the *_filled columns (with *_is_est) for models, or the raw columns with a missing indicator. Encode surface one-hot (categories include 'other'). Never mix: an estimate is not a measurement.",
    "A feature group is defined by its prefix (road_character, road_character_filled, intersection_control, cycle_pedestrian_lighting, nearby_places, road_shape); the meta file lists the columns of each.",
    "Many nearby_places columns are near-duplicates across radii (see redundant_pairs in the meta file). Choose one or two radii per place type, or reduce them, before comparing group importance.",
    "No feature uses accident counts, and no feature was selected, dropped or filled using them.",
]

def write_hand_off(state, result, state_fill, fill_validation, data_dir):
    """Write the model-ready parquet and its metadata for one state; returns the table and the metadata dict."""
    table = build_model_ready_table(state, result, state_fill)
    pairs = redundant_pairs(table)
    parquet_path = data_dir / f"{state}_osm_features.parquet"
    table.to_parquet(parquet_path, index=False)

    feature_columns = [c for c in table.columns if c not in MODEL_READY_ID_COLUMNS and not c.endswith("_est_source")]
    groups = {"road_character": [c for c in feature_columns if c.startswith("road_character_") and not c.endswith(("_filled", "_is_est"))],
              "road_character_filled": [c for c in feature_columns if c.startswith("road_character_") and c.endswith(("_filled", "_is_est"))]}
    for group in ("intersection_control", "cycle_pedestrian_lighting", "nearby_places", "road_shape"):
        groups[group] = [c for c in feature_columns if c.startswith(group + "_")]
    metadata = {
        "state": state, "pipeline_version": PIPELINE_VERSION, "rows": len(table), "row_order": "same as adj.coalesce().indices() of the dataset",
        "keys": ["node_1_idx", "node_2_idx"], "id_columns": MODEL_READY_ID_COLUMNS,
        "feature_groups_for_ablation": groups,
        "diagnostic_columns_not_features": [c for c in table.columns if c.endswith("_est_source")],
        "gap_filling": {FEATURE_LABELS[c]: {"decision": e["decision"], "reason": e["reason"]} for c, e in fill_validation.items()},
        "preprocessing_recipe": PREPROCESSING_RECIPE,
        "redundant_pairs_spearman_0.9": pairs.head(40).round(3).to_dict(orient="records"),
        "split": SPLIT[state], "split_note": "the paper's Table 7 (arXiv 2311.00164)",
        "thresholds": configured_thresholds(), "environment": ENVIRONMENT,
        "fill_rate": table[feature_columns].notna().mean().round(4).to_dict(),
    }
    meta_path = data_dir / f"{state}_osm_features_meta.json"
    with open(meta_path, "w") as file:
        json.dump(metadata, file, indent=2, default=str)
    return table, metadata

def _build_osm_worker(state, data_dir, local_dir, nodes_path, roads_path):
    graph, nodes, roads = load_osm_drive_network_from_zip(state, data_dir, local_dir)
    nodes.to_pickle(nodes_path)
    roads.to_pickle(roads_path)


def prepare_features(data_dir, cache_dir, states=("DE",), force=False):
    """Explicitly run feature construction, gap validation and hand-off export.

    Requires the benchmark, OSM snapshots/caches and optional geospatial packages.
    This operation is deliberately not run by the final report.
    """
    global ENVIRONMENT
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ENVIRONMENT = environment_record()
    for state in states:
        if force or not saved_features_are_current(state, data_dir):
            build_osm_tables(state, data_dir, cache_dir)
    data = read_benchmark(data_dir, cache_dir, states)
    output = {}
    for state in states:
        result = get_state_features(state, data, data_dir, cache_dir, force=force)
        filled = fill_state(state, result)
        table, metadata = write_hand_off(state, result, filled, filled["validation"], data_dir)
        output[state] = {"result": result, "filled": filled, "table": table, "metadata": metadata}
    return output
