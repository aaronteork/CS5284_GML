# Accident-history inputs

## Data and road identity

Paths are relative to the project root. The [source EDA notebook](../notebooks/accident_history_eda.ipynb) reconstructs the directed-edge archive. The [final report](../Final_Report.ipynb) and [training notebook](../notebooks/train_accident_history.ipynb) use physical-road targets.

| Input | Role |
|---|---|
| `data/DE/adj_matrix.pt` | Directed graph and edge identity |
| `data/DE/Nodes/node_features_YYYY_M.csv` | Coordinates and monthly weather |
| `data/DE/Edges/edge_features.pt` | Road attributes and length |
| `data/DE/Edges/edge_features_traffic_YYYY.pt` | Annual traffic estimates through 2019, with earlier reports carried forward where applicable |
| `data/DE/accidents_monthly.csv` | Monthly crash counts by endpoints |
| `data/accident_history/outputs/monthly_features/*.npz` | Ordered-edge archives with target counts stored separately from features |
| `edge_mapping.csv`, `manifest.json`, `monthly_coverage_audit.csv` in the same outputs directory | Edge alignment, source hashes and usable months |

The physical-road key is `min(u,v) * num_nodes + max(u,v)`. Sum crash counts over available directions once; count a self-loop once. If opposite directions record two and three crashes in the same month, the next month's lag is five and the recurrence feature counts one crash-positive month. The archive adapter rebuilds physical-road history from the archived labels and checks them against the CSV. Selecting one direction's precomputed inputs would give a different series.

Neighbours are other physical roads sharing either endpoint. A road with no neighbours has a zero neighbour mean when the window is available. Missing windows remain missing. The physical features are broadcast to both directed edges for message passing, and each physical road receives one prediction.

## Features and preprocessing

The ten raw features comprise the previous-month count, the 3/6/12-month means, the twelve-month lag, the fraction of crash-positive months in the previous year, the neighbour averages of lag 1 and the three-month mean, the neighbour count and the number of usable prior months.

| History mode | Added columns | Total edge inputs without OSM |
|---|---|---:|
| `none` | None | 27 |
| `own` | Six own-road features, coverage and six missingness indicators | 40 |
| `all` | Ten features and eight missingness indicators | 45 |

The original 27 columns comprise 24 road indicators, length, traffic and a traffic missingness indicator. Node inputs comprise two coordinates, 14 degree indicators, centrality, six weather values and six missingness indicators (29 columns, or 157 with Node2Vec). The training notebook uses `--history-mode`. The older `--history-groups` and `--history-min-months` interface supports earlier experiments and is a separate configuration.

All crash windows end before the forecast month. Incomplete windows are kept missing until encoding. The scaler uses observed values in the 2009–2012 training months, imputes the training mean and adds missingness indicators. Including the warm-up months yields 48 training, 60 validation and 57 test months. The archive adapter checks labels and month availability for the no-history control as well. The physical-history cache under `data/DE/cache/physical_history` uses memory-mapped arrays; it is rebuilt when the construction or data fingerprint changes.

## Training and evaluation

The full-history experiment uses Adam (learning rate 0.001), two width-256 layers, zero dropout, balanced positive/zero road sampling, 100 epochs and three model seeds. Input dimensions follow the dataset. A separate NumPy generator controls month ordering and negative samples. Completed epoch checkpoints include the model, optimizer, best state, training history and random states. Resume checks the experiment identity before continuing.

Classification selects the highest validation AUROC; count regression selects the lowest validation MAE. Evaluation clips negative predicted counts to zero. The main results use the full road population; the balanced evaluation is supplementary. Smoke tests run two epochs on two training and two validation months and skip test evaluation. Simple count references use zero, lag 1 and the three- and twelve-month means.

Forecasts assume timely prior crash reports and retain the benchmark's observed target-month weather. The original directed-edge EDA explored data through 2018. The final report uses training-only physical-road associations and reports the held-out predictive comparisons separately.
