# CS5284: Road Accident Prediction with Graph Neural Networks

This repository contains our CS5284 group project on predicting monthly road-segment accident occurrence and counts. It combines a modified reproduction of ML4RoadSafety baselines with richer OpenStreetMap (OSM) road context, past accident history, and graph model comparisons.

**Start with [Final_Report.ipynb](Final_Report.ipynb)** for the complete study: data exploration, preprocessing, methodology, experimental results, interactive risk visualisation, limitations, and references. The final report focuses on Delaware; the repository also includes Maryland baseline experiments.

## Models and experiments

The final report compares four architectures across classification and count regression:

- **MLP:** a predictor without message passing.
- **GraphSAGE:** an edge-aware neighbourhood aggregation model.
- **GatedGCN:** a graph network with gated node and edge updates.
- **Graph Transformer:** sparse, edge-aware attention over connected roads.

The code also implements **GCN** and **GAT** for additional baseline experiments. Model implementations are in [`src/model/`](src/model/).

The main comparisons use baseline features, added OSM context, added accident history, and both additions together. Node2Vec embeddings provide structural inputs, with separate embedding ablations. Training and evaluation use chronological splits; accident-history features use only prior months. These are modified reproductions, so the original paper's scores are contextual references rather than directly comparable results. See [baseline differences](docs/baseline_differences.md).

## Data sources

- **ML4RoadSafety (Nippani et al., 2023):** road graphs, accident records, weather, road attributes, and available traffic estimates. See the [paper](https://arxiv.org/abs/2311.00164), [original project](https://github.com/VirtuosoResearch/ML4RoadSafety), and [Harvard Dataverse dataset](https://doi.org/10.7910/DVN/V71K5R).
- **OpenStreetMap via Geofabrik:** additional road attributes, intersection controls, pedestrian/cycling context, nearby places, and road geometry. The final study uses the Delaware extract dated **18 October 2026** (`delaware-261018.osm.pbf`), as documented in the report. See the [Geofabrik Delaware download page](https://download.geofabrik.de/north-america/us/delaware.html). Data are credited to Geofabrik GmbH and OpenStreetMap contributors and distributed under the Open Database Licence.
- **Accident-history features:** derived from the benchmark's monthly accident records for each physical road and neighbouring roads; they are not a separate external dataset. See [history construction and integration](docs/accident_history.md).

Raw datasets, generated feature archives, embeddings, prediction tensors, and trained checkpoints are excluded from Git. The notebooks and source code expect these inputs in local or Google Drive data folders.

## Repository guide

- [`Final_Report.ipynb`](Final_Report.ipynb): final narrative, analyses, results, and visualisations.
- [`notebooks/`](notebooks/): exploratory analysis, feature preparation, training, and comparison workflows. Key entry points are [four-model baseline training](notebooks/train_baseline+gated_gcn+graph_transformer.ipynb), [accident-history training](notebooks/train_accident_history.ipynb), [OSM feature preparation](notebooks/osm_features_and_eda.ipynb), and [OSM comparisons](notebooks/osm_baseline_comparison.ipynb).
- [`src/`](src/): shared data loading, models, training, evaluation, reporting, OSM and history pipelines, and risk-map visualisation.
- [`results/`](results/): selected experiment metrics, configuration/provenance, and training histories/curves, including baseline, OSM, history, and combined-input runs. The final report identifies the runs used in its comparisons; the full-history results are in [`results/history_100_epochs/`](results/history_100_epochs/).
- [`docs/`](docs/): supporting methodology and setup notes, including the [accident-history Colab guide](docs/colab_history_guide.md).
- [`scripts/`](scripts/): notebook packaging, local validation, and experiment utilities.
- [`tests/`](tests/): synthetic-data checks for preprocessing, evaluation, history features, and training outputs.

For running experiments, install [`requirements.txt`](requirements.txt) and follow the relevant notebook's setup and data instructions. The final report is the main entry point for understanding the completed project.

For the interactive accident map, run the **Spatial exploration of recorded accidents** cell in Section 3.1.5 of the final report. It colours road segments by recorded counts, predictions, or prediction errors, with monthly, quarterly, and yearly views. The cell exports `results/visualizations/road_safety_full_eda.html`. Recorded counts require the benchmark data; prediction layers additionally require `seed*_test_predictions.pt` exports. The basemap needs an internet connection.
