"""Reusable dataset audit and plots extracted from ``eda.ipynb``.

Importing this module does not mount Drive, load data, change directories or show
figures. Call ``load_data(root / 'data')`` and pass its result to the functions
below. Plot functions return Matplotlib figures; table functions return pandas
DataFrames. Node CSVs are streamed for audits to avoid retaining all weather
observations in memory. The source notebook is retained as the original analysis.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


STATE_NAMES = {"DE": "Delaware", "MD": "Maryland"}
WEATHER = ("tavg", "tmin", "tmax", "prcp", "wspd", "pres")
WEATHER_LABELS = ("Mean temperature", "Minimum temperature", "Maximum temperature",
                  "Precipitation", "Wind speed", "Air pressure")
SPLITS = {
    "DE": {"train": (2009, 2012), "validation": (2013, 2017), "test": (2018, 2022)},
    "MD": {"train": (2015, 2017), "validation": (2018, 2019), "test": (2020, 2022)},
}
DATASET_DESCRIPTION = (
    "ML4RoadSafety (Nippani et al., NeurIPS 2023) combines official US accident "
    "records, OpenStreetMap road networks, monthly weather and available annual "
    "average daily traffic (AADT). Source: https://arxiv.org/abs/2311.00164; "
    "dataset: https://doi.org/10.7910/DVN/V71K5R."
)
TASKS = {
    "classification": "Whether a physical road records at least one accident in a month.",
    "regression": "The total number of accidents on a physical road in a month.",
}


def _sparse_entries(tensor):
    # Keep duplicate observations until feature averaging (coalesce would sum).
    tensor = tensor.detach().cpu()
    if tensor.layout != torch.sparse_coo:
        tensor = tensor.to_sparse_coo()
    return tensor._indices().numpy(), tensor._values().numpy()


def _load_tensor(path):
    # Benchmark files contain trusted PyTorch-serialised dictionaries/tensors.
    return torch.load(path, map_location="cpu", weights_only=False)


def _feature_series(tensor, n, physical=False):
    (u, v), values = _sparse_entries(tensor)
    keys = np.minimum(u, v) * n + np.maximum(u, v) if physical else u * n + v
    return pd.Series(np.where(np.isfinite(values), values, np.nan), index=keys).groupby(level=0).mean()


@dataclass
class StateData:
    """One state's graph and labels, with file references for large features."""

    state: str
    root: Path
    n_nodes: int
    edges: np.ndarray
    roads: np.ndarray
    accidents: pd.DataFrame
    counts: pd.Series
    node_files: dict
    traffic_files: dict
    edge_features: dict
    months: pd.DatetimeIndex


def load_data(data_dir, states=("DE", "MD"), exclude_months=None):
    """Load extracted benchmark folders, with no download or ZIP extraction.

    ``data_dir/DE`` must contain ``adj_matrix.pt``, ``accidents_monthly.csv``,
    ``Nodes/node_features_YEAR_MONTH.csv`` and ``Edges/edge_features.pt``.
    DE October 2022 is excluded by default to match the source notebook.
    An empty ``exclude_months`` mapping disables that exclusion.
    Month coverage is the union of label and weather months, so a month with a
    weather file but no positive accident rows still contributes zero targets.
    """
    exclusions = {"DE": {(2022, 10)}} if exclude_months is None else exclude_months
    data = {}
    for state in states:
        root = Path(data_dir) / state
        if not (root / "adj_matrix.pt").exists():
            raise FileNotFoundError(f"Missing {root / 'adj_matrix.pt'}. Extract the benchmark dataset first.")
        adj = _load_tensor(root / "adj_matrix.pt").coalesce()
        edges = adj.indices()[:, adj.values() != 0].numpy()
        n = adj.shape[0]
        roads = np.unique(np.minimum(*edges) * n + np.maximum(*edges))
        accidents = pd.read_csv(root / "accidents_monthly.csv")
        excluded = set(exclusions.get(state, ()))
        keep = [(int(y), int(m)) not in excluded for y, m in zip(accidents.year, accidents.month)]
        accidents = accidents.loc[keep].copy()
        accidents["date"] = pd.to_datetime(accidents[["year", "month"]].assign(day=1))
        a, b = accidents.node_1_idx, accidents.node_2_idx
        accidents["road"] = np.minimum(a, b) * n + np.maximum(a, b)
        counts = accidents.groupby(["date", "road"])["acc_count"].sum()
        node_files = {}
        for path in sorted((root / "Nodes").glob("node_features_*.csv")):
            match = re.fullmatch(r"node_features_(\d{4})_(\d{1,2})", path.stem)
            if match:
                year, month = map(int, match.groups())
                if (year, month) not in excluded:
                    node_files[pd.Timestamp(year, month, 1)] = path
        if not node_files:
            raise FileNotFoundError(f"No monthly node-feature CSVs found in {root / 'Nodes'}")
        months = pd.DatetimeIndex(sorted(set(node_files) | set(accidents.date)))
        traffic = {int(p.stem.rsplit("_", 1)[1]): p
                   for p in sorted((root / "Edges").glob("edge_features_traffic_*.pt"))}
        data[state] = StateData(state, root, n, edges, roads, accidents, counts,
                                node_files, traffic, _load_tensor(root / "Edges" / "edge_features.pt"), months)
    return data


def monthly_accidents(d):
    """Total recorded accidents, including observed months with no label rows."""
    return d.accidents.groupby("date").acc_count.sum().reindex(d.months, fill_value=0)


def dataset_summary(data):
    """Graph size, target population, observation window and traffic coverage."""
    rows = []
    for state, d in data.items():
        edge_keys = pd.Index(d.edges[0] * d.n_nodes + d.edges[1])
        observed = pd.Index([], dtype="int64")
        for path in d.traffic_files.values():
            values = _feature_series(_load_tensor(path)["AADT"], d.n_nodes).dropna()
            observed = observed.union(values.index.intersection(edge_keys))
        keys = observed.to_numpy(dtype=np.int64)
        a, b = keys // d.n_nodes, keys % d.n_nodes
        covered = np.unique(np.minimum(a, b) * d.n_nodes + np.maximum(a, b))
        lengths = _feature_series(d.edge_features["length"], d.n_nodes).reindex(edge_keys)
        degree = np.bincount(d.edges[0], minlength=d.n_nodes)
        rows.append({"State": STATE_NAMES.get(state, state),
                     "Period": f"{d.months.min():%Y-%m} to {d.months.max():%Y-%m}",
                     "Months": len(d.months), "Nodes": d.n_nodes,
                     "Directed edges": len(edge_keys), "Physical roads": len(d.roads),
                     "Accidents/month": monthly_accidents(d).mean(),
                     "Mean degree": degree.mean(), "Max degree": degree.max(),
                     "Mean length (m)": lengths.mean(),
                     "Traffic coverage (%)": 100 * len(covered) / len(d.roads)})
    return pd.DataFrame(rows).set_index("State")


def split_summary(data):
    """Calendar boundaries plus actual label/weather month availability."""
    rows = []
    for state, d in data.items():
        for split, (first, last) in SPLITS[state].items():
            months = d.months[(d.months.year >= first) & (d.months.year <= last)]
            rows.append({"State": STATE_NAMES[state], "Split": split,
                         "Years": f"{first}–{last}", "Observed months": len(months),
                         "Months with weather files": sum(m in d.node_files for m in months)})
    return pd.DataFrame(rows).set_index(["State", "Split"])


def audit_data(data):
    """Reproduce label, graph, month, weather and feature integrity checks.

    Returns a dictionary of ``checks``, per-variable ``weather_missingness``
    and annual ``weather_coverage`` tables. Checks are diagnostic counts, not
    automatic corrections. Missing accident rows are not proof of no accidents
    when a whole source month is unavailable; inspect month coverage first.
    """
    checks, missing_rows, coverage_rows = [], [], []
    for state, d in data.items():
        def add(check, value):
            checks.append({"State": STATE_NAMES[state], "Check": check, "Count": int(value)})

        df = d.accidents
        count = pd.to_numeric(df.acc_count, errors="coerce")
        add("Missing/non-numeric accident counts", count.isna().sum())
        add("Negative accident counts", count.lt(0).sum())
        add("Infinite accident counts", np.isinf(count).sum())
        add("Fractional accident counts", (np.isfinite(count) & count.mod(1).ne(0)).sum())
        add("Invalid months", (~df.month.isin(range(1, 13))).sum())
        add("Duplicate directed road-month rows", df.duplicated(["year", "month", "node_1_idx", "node_2_idx"]).sum())
        a, b = df.node_1_idx, df.node_2_idx
        valid = a.between(0, d.n_nodes - 1) & b.between(0, d.n_nodes - 1)
        keys = pd.Index(d.edges[0] * d.n_nodes + d.edges[1])
        add("Accident endpoints outside graph", (~valid).sum())
        add("Accident rows absent from graph", (~(a[valid] * d.n_nodes + b[valid]).isin(keys)).sum())
        expected = pd.date_range(d.months.min(), d.months.max(), freq="MS")
        add("Calendar months without weather files", len(expected.difference(pd.DatetimeIndex(d.node_files))))
        add("Calendar months without accident rows (inspect)", len(expected.difference(pd.DatetimeIndex(df.date.unique()))))
        missing = pd.Series(0, index=WEATHER, dtype="int64")
        observations = 0
        reference = None
        node_issues = {k: 0 for k in ("Incorrect node counts", "Duplicate node IDs", "Different node ordering",
                                     "Infinite weather values", "Inconsistent temperatures", "Negative precipitation",
                                     "Negative wind speed", "Non-positive air pressure", "Invalid coordinate ranges")}
        for date, path in d.node_files.items():
            frame = pd.read_csv(path)
            node_issues["Incorrect node counts"] += len(frame) != d.n_nodes
            if reference is None:
                reference = frame.node_id
            node_issues["Duplicate node IDs"] += frame.node_id.duplicated().any()
            node_issues["Different node ordering"] += not frame.node_id.equals(reference)
            weather = frame[list(WEATHER)]
            missing += weather.isna().sum()
            observations += len(frame)
            node_issues["Infinite weather values"] += np.isinf(weather.to_numpy()).sum()
            node_issues["Inconsistent temperatures"] += ((frame.tmin > frame.tavg) | (frame.tavg > frame.tmax) | (frame.tmin > frame.tmax)).sum()
            node_issues["Negative precipitation"] += frame.prcp.lt(0).sum()
            node_issues["Negative wind speed"] += frame.wspd.lt(0).sum()
            node_issues["Non-positive air pressure"] += frame.pres.le(0).sum()
            node_issues["Invalid coordinate ranges"] += (~frame.lat.between(-90, 90) | ~frame.lon.between(-180, 180)).sum()
            coverage_rows.append({"State": STATE_NAMES[state], "Year": date.year,
                                  "Observations": len(frame), **weather.notna().sum().to_dict()})
        for check, value in node_issues.items():
            add(check, value)
        for column, label in zip(WEATHER, WEATHER_LABELS):
            missing_rows.append({"State": STATE_NAMES[state], "Feature": label,
                                 "Missing": missing[column], "Missing (%)": 100 * missing[column] / observations})
        features = dict(d.edge_features)
        for year, path in d.traffic_files.items():
            features.update({f"{name} ({year})": t for name, t in _load_tensor(path).items()})
        for name, tensor in features.items():
            if tuple(tensor.shape) != (d.n_nodes, d.n_nodes):
                add(f"{name}: wrong matrix shape", 1)
                continue
            (u, v), values = _sparse_entries(tensor)
            add(f"{name}: missing/non-finite observations", (~np.isfinite(values)).sum())
            add(f"{name}: negative observations", (values < 0).sum())
            add(f"{name}: nonzero entries outside graph", (~pd.Index(u[values != 0] * d.n_nodes + v[values != 0]).isin(keys)).sum())
    annual = pd.DataFrame(coverage_rows).groupby(["State", "Year"]).sum()
    annual[list(WEATHER)] = annual[list(WEATHER)].div(annual.Observations, axis=0) * 100
    return {"checks": pd.DataFrame(checks),
            "weather_missingness": pd.DataFrame(missing_rows).set_index(["State", "Feature"]),
            "weather_coverage": annual[list(WEATHER)]}


def accident_distribution(data):
    """Include zero-count physical roads in every observed month."""
    rows = []
    for state, d in data.items():
        positive = d.counts[d.counts > 0]
        total = len(d.roads) * len(d.months)
        rows.append({"State": STATE_NAMES[state], "Road-months": total,
                     "Zero accidents (%)": 100 * (total - len(positive)) / total,
                     "Mean count": positive.sum() / total,
                     **{label: value for label, value in zip(
                         ("0", "1", "2", "3", "4", "5+"),
                         (total - len(positive), positive.eq(1).sum(), positive.eq(2).sum(),
                          positive.eq(3).sum(), positive.eq(4).sum(), positive.ge(5).sum()))}})
    return pd.DataFrame(rows).set_index("State")


def concentration(data, fractions=(0.01, 0.05, 0.10)):
    """Retrospective concentration across the full observed period, not accuracy."""
    rows = []
    for state, d in data.items():
        totals = d.counts.groupby("road").sum().reindex(d.roads, fill_value=0).sort_values(ascending=False)
        for fraction in fractions:
            if not 0 < fraction <= 1:
                raise ValueError("fractions must be in (0, 1]")
            k = max(1, int(len(totals) * fraction))
            share = totals.iloc[:k].sum() / totals.sum() if totals.sum() else np.nan
            rows.append({"State": STATE_NAMES[state], "Top roads (%)": 100 * fraction,
                         "Accident share (%)": 100 * share})
    return pd.DataFrame(rows)


def historical_persistence(data, training_only=False):
    """Next-month association; optional training restriction for feature design."""
    rows = []
    for state, d in data.items():
        months = d.months
        if training_only:
            first, last = SPLITS[state]["train"]
            months = months[(months.year >= first) & (months.year <= last)]
        positive = d.counts[d.counts > 0]
        sets = {date: set(group.index.get_level_values("road")) for date, group in positive.groupby(level="date")}
        repeat = previous_positive = new = previous_zero = 0
        available = set(months)
        for date in months:
            previous_date = date - pd.offsets.MonthBegin(1)
            if previous_date not in available:
                continue
            current, previous = sets.get(date, set()), sets.get(previous_date, set())
            repeat += len(current & previous)
            previous_positive += len(previous)
            new += len(current - previous)
            previous_zero += len(d.roads) - len(previous)
        p_repeat = repeat / previous_positive if previous_positive else np.nan
        p_new = new / previous_zero if previous_zero else np.nan
        rows.append({"State": STATE_NAMES[state], "After accident (%)": 100 * p_repeat,
                     "After zero (%)": 100 * p_new,
                     "Relative likelihood": p_repeat / p_new if p_new > 0 else np.nan})
    return pd.DataFrame(rows).set_index("State")


def _risk(d, training_only=True):
    months = d.months
    if training_only:
        first, last = SPLITS[d.state]["train"]
        months = months[(months.year >= first) & (months.year <= last)]
    if not len(months):
        raise ValueError(f"No months in the requested window for {d.state}")
    counts = d.counts[d.counts.index.get_level_values("date").isin(months)]
    return counts.groupby("road").sum().reindex(d.roads, fill_value=0) / len(months)


def _axes(data, height=3.4):
    fig, axes = plt.subplots(1, len(data), figsize=(5.4 * len(data), height),
                             squeeze=False, layout="constrained")
    return fig, axes.ravel()


def plot_monthly_accidents(data):
    """Monthly totals with chronological split boundaries."""
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 2.5 * len(data)),
                             squeeze=False, layout="constrained")
    for ax, (state, d) in zip(axes.ravel(), data.items()):
        totals = monthly_accidents(d)
        ax.plot(totals.index, totals.values)
        val, test = (pd.Timestamp(SPLITS[state][s][0], 1, 1) for s in ("validation", "test"))
        for date in (val, test):
            ax.axvline(date, color="grey", linestyle=":")
        for label, start, end in (("Train", totals.index.min(), val), ("Validation", val, test),
                                  ("Test", test, totals.index.max())):
            ax.text(start + (end - start) / 2, .95, label, transform=ax.get_xaxis_transform(), ha="center", va="top")
        ax.set(title=STATE_NAMES[state], xlabel="Month", ylabel="Recorded accidents")
        ax.grid(alpha=.3)
    return fig


def plot_seasonality(data):
    fig, axes = _axes(data)
    for ax, (state, d) in zip(axes, data.items()):
        totals = monthly_accidents(d)
        seasonal = totals.groupby(totals.index.month).mean()
        ax.plot(seasonal.index, seasonal.values, marker="o")
        ax.set(title=STATE_NAMES[state], xlabel="Calendar month", ylabel="Mean recorded accidents", xticks=range(1, 13))
    return fig


def plot_accident_distribution(data):
    table = accident_distribution(data)
    fig, axes = _axes(data)
    labels = ["0", "1", "2", "3", "4", "5+"]
    for ax, state in zip(axes, data):
        row = table.loc[STATE_NAMES[state]]
        ax.bar(labels, 100 * row[labels].astype(float) / row["Road-months"])
        ax.set(title=STATE_NAMES[state], xlabel="Accidents per physical road-month",
               ylabel="Road-months (%)", yscale="log")
    return fig


def plot_concentration(data):
    fig, axes = _axes(data)
    for ax, (state, d) in zip(axes, data.items()):
        totals = d.counts.groupby("road").sum().reindex(d.roads, fill_value=0).sort_values(ascending=False)
        cumulative = totals.cumsum() / totals.sum() * 100 if totals.sum() else totals * np.nan
        ax.plot(np.linspace(0, 100, len(totals) + 1), np.r_[0, cumulative])
        ax.set(title=STATE_NAMES[state], xlabel="Roads ranked by total accidents (%)",
               ylabel="Cumulative accidents (%)", xlim=(0, 100), ylim=(0, 100))
        ax.grid(alpha=.3)
    return fig


def plot_risk_profile(data):
    """Compact report figure: count sparsity and retrospective concentration."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    table = accident_distribution(data)
    labels = ["0", "1", "2", "3", "4", "5+"]
    width = .8 / len(data)
    for i, (state, d) in enumerate(data.items()):
        name = STATE_NAMES[state]
        row = table.loc[name]
        left.bar(np.arange(6) - .4 + width * (i + .5),
                 100 * row[labels].astype(float) / row["Road-months"], width, label=name)
        totals = d.counts.groupby("road").sum().reindex(d.roads, fill_value=0).sort_values(ascending=False)
        cumulative = totals.cumsum() / totals.sum() * 100 if totals.sum() else totals * np.nan
        right.plot(np.linspace(0, 100, len(totals) + 1), np.r_[0, cumulative], label=name)
    left.set(xticks=range(6), xticklabels=labels, yscale="log", xlabel="Accidents per physical road-month", ylabel="Road-months (%)")
    right.set(xlabel="Roads ranked by total accidents (%)", ylabel="Cumulative accidents (%)", xlim=(0, 100), ylim=(0, 100))
    left.legend()
    right.legend()
    return fig


def plot_neighbour_risk(data, sample=5000):
    """Training-period spatial association; return figure and correlations."""
    fig, axes = _axes(data)
    correlations = {}
    for ax, (state, d) in zip(axes, data.items()):
        risk = _risk(d)
        a, b = d.roads // d.n_nodes, d.roads % d.n_nodes
        keep = a != b
        a, b, values = a[keep], b[keep], risk.to_numpy()[keep]
        degree = np.bincount(np.r_[a, b], minlength=d.n_nodes)
        totals = np.bincount(np.r_[a, b], weights=np.r_[values, values], minlength=d.n_nodes)
        neighbours = degree[a] + degree[b] - 2
        valid = neighbours > 0
        neighbour_risk = (totals[a] + totals[b] - 2 * values)[valid] / neighbours[valid]
        points = pd.DataFrame({"road": np.log1p(values[valid]), "neighbours": np.log1p(neighbour_risk)})
        correlations[STATE_NAMES[state]] = points.corr().iloc[0, 1]
        shown = points.sample(min(sample, len(points)), random_state=0)
        ax.scatter(shown.neighbours, shown.road, s=5, alpha=.2)
        ax.set(title=STATE_NAMES[state], xlabel="Neighbour mean monthly accidents (log1p)", ylabel="Road mean monthly accidents (log1p)")
    return fig, pd.Series(correlations, name="Training-period correlation")


def plot_length_risk(data):
    fig, axes = _axes(data)
    for ax, (state, d) in zip(axes, data.items()):
        lengths = _feature_series(d.edge_features["length"], d.n_nodes, physical=True)
        summary = pd.DataFrame({"length": lengths.reindex(d.roads), "accidents": _risk(d)}).dropna()
        summary = summary[summary.length > 0]
        summary["group"] = pd.qcut(summary.length, 5, duplicates="drop")
        groups = summary.groupby("group", observed=True)[["length", "accidents"]].mean()
        ax.plot(groups.length, groups.accidents, marker="o")
        ax.set(title=STATE_NAMES[state], xlabel="Mean road length within group (m)", ylabel="Mean accidents per road-month")
    return fig


def plot_road_type_risk(data, min_roads=100):
    fig, axes = _axes(data, height=4.5)
    for ax, (state, d) in zip(axes, data.items()):
        categories = {}
        for name, tensor in d.edge_features.items():
            if name in ("length", "oneway"):
                continue
            series = _feature_series(tensor, d.n_nodes, physical=True)
            categories[name] = pd.Series(1, index=series[series > 0].index)
        flags = pd.DataFrame(categories).reindex(d.roads).fillna(0)
        types = flags.loc[flags.sum(axis=1) == 1].idxmax(axis=1)
        summary = pd.DataFrame({"type": types, "accidents": _risk(d).reindex(types.index)})
        groups = summary.groupby("type").accidents.agg(["size", "mean"])
        groups = groups[groups["size"] >= min_roads].sort_values("mean")
        ax.barh(groups.index.str.replace("_", " "), groups["mean"])
        ax.set(title=STATE_NAMES[state], xlabel="Mean accidents per road-month", ylabel="Road type")
    return fig


def plot_weather_coverage(audit):
    """Use the cached audit result rather than re-reading monthly node CSVs."""
    table = audit["weather_coverage"]
    states = table.index.get_level_values("State").unique()
    fig, axes = plt.subplots(len(states), 1, figsize=(10, 2.8 * len(states)),
                             squeeze=False, layout="constrained")
    for ax, state in zip(axes.ravel(), states):
        annual = table.loc[state]
        for column, label in zip(WEATHER, WEATHER_LABELS):
            ax.plot(annual.index, annual[column], marker=".", label=label)
        ax.set(title=state, xlabel="Year", ylabel="Available observations (%)", ylim=(0, 105))
    axes[0, 0].legend(ncol=3, fontsize=8)
    return fig


def osm_coverage(features):
    """Summarise an OSM hand-off DataFrame without counting reverse rows twice.

    Pass a table loaded from ``STATE_osm_features.parquet`` by the OSM notebook
    (reading Parquet requires an optional engine such as pyarrow). Raw feature
    coverage is measured before imputation. Values in filled/estimated columns
    are excluded; a recorded zero is an observation, not a missing value.
    """
    frame = features.copy()
    frame["_u"] = np.minimum(frame.node_1_idx, frame.node_2_idx)
    frame["_v"] = np.maximum(frame.node_1_idx, frame.node_2_idx)
    frame = frame.drop_duplicates(["_u", "_v"])
    prefixes = ("road_character_", "intersection_control_", "cycle_pedestrian_lighting_", "nearby_places_", "road_shape_")
    columns = [c for c in frame if c.startswith(prefixes) and not c.endswith(("_filled", "_is_est"))]
    rows = [{"Feature": c, "Known roads (%)": 100 * frame[c].notna().mean()} for c in columns]
    if "has_osm_match" in frame:
        rows.insert(0, {"Feature": "Matched to OSM", "Known roads (%)": 100 * frame.has_osm_match.fillna(0).gt(0).mean()})
    return pd.DataFrame(rows, columns=["Feature", "Known roads (%)"]).set_index("Feature")


def show_risk_map(project_dir, **kwargs):
    """Explicit opt-in wrapper for the existing standalone interactive map."""
    from src.visualization.risk_map import show_risk_map as show
    return show(project_dir, **kwargs)
