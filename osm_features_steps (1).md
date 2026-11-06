# Task brief: add OpenStreetMap features to the road-safety benchmark + run EDA

Owner: Vidurveer (role C). Course project: CS5284 Graph ML, Group 01.
Goal: predict monthly accident counts per road segment (NeurIPS'23 road-safety benchmark, Nippani et al., arXiv 2311.00164). Role C adds richer OSM road features, measures them by ablation, and hands a clean feature table to the modelling code.

## 0. Working rules for you (Claude Code)

- The user prefers to be consulted before assumptions are made. Where this brief says **ASK**, stop and ask. Do not guess.
- Do **not** edit teammates' files (`scripts/utils.py`, `eda.ipynb`, `interactive_plot_draft.ipynb`). Add new files only.
- Match the existing style: Colab, Google Drive paths, `utils.read_data`, seaborn `whitegrid`/`talk` theme, colours `{"DE": "tab:blue", "MD": "tab:orange"}`.
- Keep functions in `scripts/osm_features.py` (importable from notebooks); keep notebooks thin.
- Run Delaware (DE) first. It is small (49,023 nodes, 116,196 directed roads). Only then Maryland (MD: 250,565 nodes, 580,526 roads).
- Nothing below has been run against real data yet. Treat all code as a skeleton and fix it as you go.
- Report numbers honestly, including bad ones (low match rates, sparse tags).

## 1. What the existing code already tells us (verified from the team's notebooks)

- Project layout on Drive: `MyDrive/cs5284_project/{data,scripts}`. `data/` holds `DE.zip` and `MD.zip`. `utils.read_data(DATA_DIR, LOCAL_DATA_DIR)` extracts them to `/content/data/{DE,MD}` and returns, per state: `filepath`, `adj`, `edges`, `edge_features_traffic` (dict by year), `accidents`.
- `adj` is a **directed** sparse COO tensor (`adj.coalesce().indices()` gives `src, dst`). It is not symmetric. One row per directed road.
- `accidents` columns: `year, month, node_1_idx, node_2_idx, acc_count`. It is keyed by directed road. Aaron groups duplicates with a `groupby(...).sum()`; do the same.
- `edges` is a dict of sparse NxN tensors. `edges["length"].coalesce()` works. **Inspect the other keys** (`print(list(info["edges"].keys()))`) before designing features, because OSM-derived features should not duplicate what is already there (e.g. highway type, oneway).
- `Nodes/node_features_{year}_{month}.csv` contains `node_id` (the raw OSM node ID) plus weather columns. **Confirmed: the `lat` and `lon` columns are swapped in the release** (DE rows show lat = -75.49, lon = 39.81). Aaron's map code fixes it with `lat = nodes["lon"]`, `lon = nodes["lat"]`. Do the same, and never use the columns raw.
- Aaron's EDA results to build on:
  - DE: 2,761 accidents/month; 68.6% of roads never have an accident; about 98% of roads have zero accidents in any given month.
  - MD: 9,149 accidents/month; 77.3% never have an accident.
  - AADT (traffic volume) covers only 1.77% of directed DE roads and 0.76% of MD roads.
  - History signal is very strong: P(accident next month | accident this month) is 23% (DE) and 20% (MD) versus about 1.5% and 1.1% otherwise. Relative risk is 15x (DE) and 19x (MD). This matters when judging how much OSM can add on top of history.

## 1b. Findings from the verification checks (already run in Colab on DE and MD, 2026-10-20)

These change how the work below should be done. Apply the rules in the last column everywhere.

| Check | Result | Rule for our code |
|---|---|---|
| **Incomplete months (DE)** | 2022-10 has 23 accidents (median month 2,820), so it is a partial month. 2009 totals 19,097 vs 31,276 in 2010 (partial year or ramp-up). Other years: 2010-13 about 31-32k, 2014-19 rising to about 37k, 2020 dip 31,657 (COVID), 2021 37,749, 2022 27,013 (Jan-Sep only). | **Drop DE 2009 and DE 2022-10.** Usable DE range is **2010-01 to 2022-09**. The 2020-04 dip (both states) is real COVID; keep it. Expect a modest train-to-test level shift in DE (about +19% from 2010-13 to 2016-19). |
| **Incomplete months (MD)** | Only 2020-04 is low (COVID). First months about 8.2-8.5k, last months about 9.6-10.3k. | No cleaning needed. |
| **Accident direction on two-way roads** | The reverse edge has accidents **0.0%** of the time (median ratio 0.00) in both states. About 88% (DE) / 92% (MD) of directed edges have a reverse edge; 71.6% (DE) / 73.5% (MD) of edges with accidents are on two-way roads. Among accident edges, `u<v` is 84.2% (DE) / 85.9% (MD). | Each crash is stored on **one fixed direction** of its road. Always carry an **undirected road key** `(min(u,v), max(u,v))` next to the directed key. Never treat the unlabelled direction of a two-way road as a "true negative". |
| **Extreme edges** | Top-5 edges hold only 1.27% (DE) / 0.48% (MD) of all accidents. They are 0.9-7.9 km long (one MD edge is 228 m). DE max is 1,692 accidents on a 1,812 m edge. | Heavy tail, but not dominated by a few artifact edges. Use **rates per km**, not raw counts, and always control for length. |

### Why only one direction gets crashes (from reading the repo, `datasets/3_extract_nearest_street.py`)

For each crash the script scores every edge by `|dist(crash, node1) + dist(crash, node2) - edge_straight_length|` and takes the minimum. The score is identical for A->B and B->A, and the tie is broken by `min_df.loc[0]`, the **first row of the edges file**. So the labelled direction is an artifact of file order. If every two-way road went to one direction (71.6% of accident edges) and one-way roads split evenly, we would expect about 85.8% with `u<v`. Observed: 84.2% and 85.9%.

Consequences (the percentages are my estimates from the check outputs, not measured directly):
- Roughly 65k physical roads sit behind DE's 116k directed edges. About **44% of directed edges can never be positive**.
- DE's "68.6% of roads never have an accident" is nearer **55% per physical road**.
- Aaron's persistence statistics (15-19x relative risk) are **still valid**, because the direction is fixed per road.
- A model on directed edges cannot know which direction is the labelled one, so it will score both directions about equally. Half of its two-way predictions are structurally wrong. The paper also tests on all edges, so its reported numbers are affected too.

### A labelling bias toward long edges (hypothesis from the code, **not yet measured**)

The score above treats each edge as a straight line between its two nodes (curvature ignored) and works in raw lon/lat degrees. For a crash at lateral offset `h` from the middle of a straight segment of length `L`, the score is about `2h^2 / L`. The same offset scores lower on a long segment, so a long chord can beat a nearer short street. For example, a crash 20 m from a 100 m street scores about 8 m, and a 2 km chord up to about 90 m away scores the same or better. This fits the long top edges, but long highways also really do collect many crashes, so it is not proof.

Consequences:
- **Length may predict accidents partly through the labelling**, not only through risk. That is probably why length is such a strong baseline feature.
- **OSM features that correlate with long roads** (road class, speed limit, lanes) may look informative for the same reason. Always report results with and without controlling for length and highway class.
- Labels are noisy, which caps what any model can achieve.
- We cannot fix it, because the benchmark ships only monthly aggregated counts, not raw crash coordinates.

### Decisions

| Decision | Status |
|---|---|
| Drop DE 2009 and DE 2022-10 from all analysis | **Decided** (it is a fact about the data) |
| Output table carries both the directed key and the undirected road key | **Decided** |
| Collapse two-way roads into one row for our own modelling and evaluation (sum both directions), while keeping the directed setup for the paper reproduction | **Open: user to discuss with Aaron (loader) and Fung Wei (models)**. Do not change shared code before that. |
| Control for length and highway class when reading any OSM effect | **Decided** |

## 2. Deliverables

1. `scripts/osm_features.py`: functions to pull OSM, match to benchmark roads, build features, save, and load aligned to `adj` order.
2. `{DATA_DIR}/osm/{STATE}_osm_edge_features.parquet` plus `{STATE}_osm_meta.json`.
3. `osm_eda.ipynb`: the EDA (section 7).
4. `OSM_FEATURES_README.md`: a one-page data dictionary (columns, groups, fill rates, snapshot date, known limits).
5. A short list of figures for the report (section 7.5).

## 3. Step 1: Build the benchmark tables

```python
import numpy as np, pandas as pd, torch

def benchmark_tables(info):
    adj = info["adj"].coalesce()
    nodes = pd.read_csv(next((info["filepath"] / "Nodes").glob("node_features_*.csv")))[["node_id", "lat", "lon"]]
    nodes = nodes.rename(columns={"lat": "lon", "lon": "lat"})   # release has them swapped
    assert nodes["lat"].between(-90, 90).all() and nodes["lon"].between(-180, 180).all()
    assert len(nodes) == adj.shape[0]
    nodes["node_idx"] = np.arange(len(nodes))

    src, dst = adj.indices().numpy()
    edges = pd.DataFrame({"node_1_idx": src, "node_2_idx": dst})   # row order == adj order, keep it
    L = info["edges"]["length"].coalesce()
    li = L.indices().numpy()
    length = pd.DataFrame({"node_1_idx": li[0], "node_2_idx": li[1], "length_m": L.values().numpy()})
    edges = edges.merge(length, how="left", on=["node_1_idx", "node_2_idx"])
    return nodes, edges
```

Sanity gates (print and check before continuing):
- `edges.length_m.notna().mean() > 0.99`.
- **Index-order gate:** compute the haversine distance between the endpoints of each edge. `length_m / straight_line` should have a median in about [1.0, 1.3]. If it does not, the node index order assumption is wrong. **STOP and ASK.**

## 4. Step 2: Pull OSM

- Install and pin: `pip install osmnx` (use 2.x; print `ox.__version__`). Record the pull date.
- Build the study polygon from the benchmark nodes: a convex hull buffered by about 1 km (buffer in a projected CRS, then convert back to EPSG:4326).
- Request the extra tags **before** downloading, otherwise osmnx drops them:

```python
import osmnx as ox
ox.settings.useful_tags_way += ["lit", "surface", "sidewalk", "cycleway", "cycleway:left",
                                "cycleway:right", "cycleway:both", "bicycle"]
ox.settings.useful_tags_node += ["crossing", "traffic_calming"]
G = ox.graph_from_polygon(poly_ll, network_type="drive", simplify=True, retain_all=True)
G = ox.project_graph(G)                      # metres
n_gdf, e_gdf = ox.graph_to_gdfs(G)           # e_gdf index = (u, v, key)
```

- The Harvard Dataverse networks the benchmark is built on are OSMnx `drive` networks as far as we know. **Verify** by comparing node and edge counts with the benchmark (DE: 49,023 nodes and 116,196 directed edges). Report the ratio.
- If Overpass times out for MD: **ASK** the user before switching to a Geofabrik extract (`maryland-latest.osm.pbf`) because it adds dependencies.

## 5. Step 3: Map benchmark nodes to OSM nodes

OSM node IDs ship with the benchmark (`node_id`), so try the ID join first. IDs can drift between the benchmark's OSM date and today, so fall back to a spatial snap.

```python
import geopandas as gpd
pts = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lon, nodes.lat), crs=4326).to_crs(n_gdf.crs)

nodes["id_in_osm"] = nodes["node_id"].isin(n_gdf.index)
snap_id, snap_dist = ox.distance.nearest_nodes(G, pts.geometry.x.values, pts.geometry.y.values, return_dist=True)
nodes["snap_osm_id"], nodes["snap_dist_m"] = snap_id, snap_dist

# for ID matches, sanity-check coordinates agree (flag if > 25 m apart)
# final mapping: ID if present and consistent, else snap if snap_dist_m <= 20, else NaN
nodes["osm_id"] = np.where(nodes.id_in_osm, nodes.node_id, np.where(nodes.snap_dist_m <= 20, nodes.snap_osm_id, np.nan))
nodes["node_match"] = np.select([nodes.id_in_osm, nodes.snap_dist_m <= 20], ["id", "snap"], "none")
```

Report: share of nodes by `node_match`, and the snap-distance distribution. If `id_in_osm` is below about 90%, say so clearly (it means the benchmark's OSM snapshot is old).

## 6. Step 4: Match edges (3 tiers) and build features

### 6.1 Edge matching

Add `osm_u`, `osm_v` to the benchmark edges from the node mapping. Then:

```python
e = e_gdf.reset_index()                                   # columns: u, v, key, length, highway, ...
cand = edges.merge(e, left_on=["osm_u", "osm_v"], right_on=["u", "v"], how="left", suffixes=("", "_osm"))
cand["len_err"] = (cand["length"] - cand["length_m"]).abs()
tier1 = cand.sort_values("len_err").drop_duplicates(["node_1_idx", "node_2_idx"])   # parallel edges: closest length wins
```

- **Tier `exact`:** the directed pair `(osm_u, osm_v)` exists in OSM.
- **Tier `reverse`:** only `(osm_v, osm_u)` exists. Accept, but flag it (a direction mismatch).
- **Tier `nearest`:** for the rest, use `ox.distance.nearest_edges(G, mid_x, mid_y, return_dist=True)` on the segment midpoint (projected coordinates). Accept only if distance <= 30 m **and** OSM length / benchmark length is within [0.6, 1.6]. Make thresholds constants at the top of the module.
- **Tier `unmatched`:** everything else. Leave features as NaN and keep the row.

Columns to keep on the output: `match_type`, `match_dist_m` (0 for exact), `len_ratio`.

### 6.2 Feature groups (prefix columns by group so ablations are one line)

| Prefix | Group | Features |
|---|---|---|
| `g1_` | Road character | `maxspeed_mph`, `lanes`, `surface`, `bridge`, `tunnel`, `oneway` |
| `g2_` | Intersection control | signals / stop / give-way / crossing at the source node and at the destination node; `street_count` of both nodes; roundabout (`junction`) |
| `g3_` | VRU infrastructure | `cycleway_any`, `sidewalk_any`, `lit` |
| `g4_` | Surroundings | counts of schools, bars/pubs, restaurants/fast food, bus stops, parking, crossings within 100 m and 250 m of the midpoint; landuse share |

Notes:
- `maxspeed` arrives as `"25 mph"`, `"50"`, or a list like `"['25 mph','30 mph']"`. Parse it, take the max of a list, treat unit-less US values as mph.
- Node tags: `n_gdf["highway"]` holds `traffic_signals`, `stop`, `crossing`, `give_way`, and so on. Join to the edge via `osm_u` and `osm_v`.
- G4: `ox.features_from_polygon(poly_ll, tags={"amenity": [...], "highway": ["bus_stop", "crossing", "traffic_signals"]})`, project the geometry centroids, then count with `scipy.spatial.cKDTree(poi_xy).query_ball_point(mid_xy, r, return_length=True)`.
- Under the `drive` network, `sidewalk` and `cycleway` exist only as tags on the road way. Expect very low fill rates for G3. This is a finding to report, not something to hide. **Do not** pull separate footway or cycleway networks without asking.
- **Missing values:** keep raw columns. For the model-ready columns impute missing numerics with the median **per `highway` class**, and add a `*_missing` flag column for each. Never impute using validation or test rows; compute medians on the training period's roads (the roads are the same set in every period, so this mainly means: do not look at labels).
- Drop categories with fewer than about 50 roads into `other`.

### 6.3 Save

- Parquet keyed by `(node_1_idx, node_2_idx)`, same row order as `adj.coalesce().indices()`. Also include these columns (see section 1b): `road_u = min(u,v)`, `road_v = max(u,v)`, `has_reverse` (a reverse edge exists in `adj`), `u_lt_v`. OSM features are identical for both directions of a two-way road.
- `{STATE}_osm_meta.json`: OSM pull date, osmnx version, thresholds, node and edge match shares, fill rate per column.
- Also provide this loader so the model code never touches raw OSM:

```python
def load_osm_edge_attr(state_info, parquet_path, groups=("g1", "g2", "g3", "g4"), train_stats=None):
    """Return (edge_attr [E, F] float tensor aligned to adj.coalesce().indices(), feature_names, train_stats)."""
```

Standardise numeric columns with **training-period statistics only** (the features are static, so "training statistics" means fit the scaler on the training roads and reuse for val/test; roads are the same, so there is no leakage, but keep the habit and document it).

## 7. Step 5: Exploratory analysis (`osm_eda.ipynb`)

All label-based analyses use the **training period only**. Do not look at validation or test labels when choosing features.

**First apply the cleaning from section 1b:** drop DE 2009 and DE 2022-10 before computing anything, and keep every split inside DE 2010-01 to 2022-09. Compute every label statistic two ways: per directed edge (as in the paper) and per physical road (sum both directions via `road_u, road_v`).

**ASK** the user for the agreed temporal split. The paper uses train 2015-17, val 2018-19, test 2020-22 for MD; the DE split is not yet confirmed. Put it in one config cell: `SPLIT = {"DE": {...}, "MD": {...}}`.

### 7.1 Coverage and match quality
- Match shares (`exact`, `reverse`, `nearest`, `unmatched`) overall and by highway class; snap distance histogram.
- Fill rate per OSM column, overall and by highway class (heatmap: column x road class).
- Sanity: compare matched OSM `highway` with the benchmark's highway feature. Report the agreement rate.

### 7.2 Distributions
- Histograms for numeric features, bar charts for categorical ones. Check outliers (e.g. `lanes > 8`, `maxspeed > 80`).
- Check the lengths: `length_m` vs OSM `length`.

### 7.3 Target shape (training period)
- Edge-level total accidents (log1p histogram), share of roads with at least one accident, share with an accident per month. Aaron already has these; reuse the numbers, do not redo the plots unless needed. The new part is the **per-physical-road version** (section 1b): the zero-accident share should drop from 68.6% to roughly 55% in DE.
- Check that the sum of both directions' counts equals the directed total (sanity), and that no physical road has accidents on both directions.

### 7.4 Feature vs risk (the core of the EDA)
- **Use rates, not counts:** accidents per km-year = `sum(acc_count) / (length_m / 1000 * n_years)`, grouped by feature level (highway class, speed bin, lanes, signal at either end, junction type, cycleway, lit, POI-count bins).
- Add 95% exact Poisson intervals. For count `k` and exposure `E` (km-years): lower = `chi2.ppf(0.025, 2k) / 2 / E`, upper = `chi2.ppf(0.975, 2k + 2) / 2 / E`.
- Also show the share of roads with at least one accident in the period, because the counts are dominated by a few very busy roads (max 1,692 on one DE road).
- **Length confound (important, see 1b):** plot accidents per km against `length_m` (binned), and show the highway-class and speed-limit rate panels both **raw and within length bins**. If the effect of a feature vanishes within length bins, it was mostly the long-edge labelling bias. Also report the share of accidents that fall on edges longer than 1 km.
- **Confounding check:** cross-tabulate speed limit vs highway class. Correlation matrix (Spearman) and VIF among numeric features. Where AADT exists (1-3% of roads), compare it with `highway` and `maxspeed`.
- **Spatial view:** Plotly scattermapbox of accident rate and one or two features (follow `interactive_plot_draft.ipynb`; sample about 3,000 roads). Optional: Moran's I on edge midpoints.
- **Temporal sanity:** monthly totals and seasonality are already in `eda.ipynb`; only add what is OSM-specific. In MD, the 2020-22 test window contains the COVID dip, so mention this as a shift when reading any "OSM helps" result.

### 7.5 Quick utility check (a sanity preview, not the real experiment)
- Task: predict "any accident in the validation period" per road, from static features. Model: `sklearn.ensemble.HistGradientBoostingClassifier` (no extra dependencies). Metrics: PR-AUC and Recall@K (K = number of true positives), and compare with the prevalence baseline.
- Unit of analysis: **one row per physical road** (summed over both directions), not per directed edge, so no row is structurally zero.
- Controls: (a0) `length` only, (a) `highway` + `length` + `oneway` (what the paper already has), then (b) add each group `g1..g4` one at a time, then (c) drop each group from the full set. Report (a0) too: if length alone gets most of the way, say so.
- The headline question for the report: **does OSM add anything beyond highway type and length?** If a group adds nothing here, say so.
- Do not call this the ablation study. The real ablations run through Fung Wei's GraphSAGE pipeline once it works.

### 7.6 Figures to save (for the report)
1. Match and coverage chart (7.1).
2. Rate-by-feature panel with intervals (7.4).
3. One map of risk vs one feature.
4. Group-ablation bar chart from 7.5.

## 8. Step 6: Hand-off and open interfaces

- The shared interface agreed in the plan: a feature table keyed by road segment (and month for time-varying features) and a single run function that reads a config. OSM features are static, so the table is keyed by `(node_1_idx, node_2_idx)` only. Tell Fung Wei the loader's name and the group prefixes so a config can switch groups on and off.
- **ASK** before changing anything in the shared `utils.py` (e.g. adding a loader there).
- Raise with Aaron and Fung Wei (via the user) the **directed vs undirected** question from section 1b. Until they agree, keep the directed table as the default and ship the undirected key columns alongside it.
- Post a short status note for the Friday weekly update: match rate, fill rates, best and worst groups so far.

## 9. Known limitations (write these into the README and report)

- **Snapshot mismatch:** OSM is today's snapshot but labels span about 2009-2022. Infrastructure added later (a signal, a bike lane) can create spurious associations. Main run uses one pinned snapshot; **ASK** whether to attempt an Overpass historical snapshot (`[date:"YYYY-MM-DDT00:00:00Z"]`) as a sensitivity check.
- **Tag sparsity:** OSM tags such as `maxspeed`, `lanes`, `lit`, `cycleway` are often missing, especially on residential roads. Missingness correlates with road class.
- **Matching noise:** the benchmark collapses parallel edges (`drop_duplicates` on node pairs) and drops edge geometry, so some segments will not match exactly.
- **Directedness:** the benchmark is directed. OSM features describe the road, not the direction, so both directions of a two-way road receive the same values. Labels sit on only one direction of each two-way road (section 1b), so about 44% of DE directed edges can never be positive.
- **Snapping bias (hypothesis):** the benchmark's crash-to-street assignment (straight-line chord score, ties by file order) likely favours long edges and ignores road curvature. Length and length-correlated OSM features may be rewarded partly by the labelling. Raw crash coordinates are not available, so this cannot be corrected.
- **Labels:** the benchmark counts generic accidents; there is no pedestrian, cyclist or severity split.
- **Data coverage:** DE 2009 and 2022-10 are partial and are dropped; the 2020 COVID dip is kept and is a distribution shift for any 2020-22 test window.

## 10. Definition of done

- DE parquet + meta + README exist on Drive; MD follows if time allows.
- Gates in sections 3-5 pass or are explained.
- `osm_eda.ipynb` runs top to bottom in Colab and produces the four figures.
- The loader returns an `edge_attr` tensor aligned with `adj` and one run with `groups=()` vs `groups=("g1",)` works end to end in a smoke test.
- A short list of "what surprised us" for the user to put in the Friday update. Two items are already known: the one-direction labelling and the likely long-edge bias (section 1b).
- Every label statistic in the EDA is reported both per directed edge and per physical road, and the length-control results are shown next to the uncontrolled ones.
