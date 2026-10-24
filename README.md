# CS5284_GML: merged baseline pipeline

Branch `feature/merged-baselines` combines `feature/baseline` (43025f1) and
`baseline-fw` (4f926d1) in one `src/`, `data/`, `notebooks/` structure.
It is a modified reproduction; see [differences](docs/baseline_differences.md).

## Shared pipeline

Every model receives coordinates, in/out-degree encodings, approximate
betweenness, six weather variables, road-type flags, log road length, log AADT
and missing-value indicators. Continuous weather/traffic statistics use training
years only. Missing traffic years carry forward an earlier report, never a future
one. Sparse feature duplicates are averaged while ignoring missing observations.
Monthly and structural features are cached on disk and reused across experiments.

The learned models are a predictor-only MLP and the paper-style edge-aware
GraphSAGE. Each is trained with and without the same fixed Node2Vec embeddings,
for classification and regression, across seeds 0/1/2. Physical-road targets,
corrected negative sampling, nonnegative regression clipping and full-network
validation/test evaluation are retained. AUROC is averaged monthly using
positive-road counts; MAE includes all available road-months, including zero-
accident months. Pooled/balanced and top-risk metrics provide extra diagnostics.

## Colab

1. Push this branch to GitHub, then upload `notebooks/train_colab.ipynb` to Colab.
2. Select a GPU runtime and run cells in order. The clone cell targets
   `feature/merged-baselines` and privately prompts for a GitHub token if needed.
3. Keep `STATE = "DE"` to begin. Setup creates a Python 3.12 environment using
   the single `requirements.txt`; no automatic unit tests are run.
4. The embedding cell uses the GPU. Initial structural caching computes
   approximate betweenness on CPU once and can take several minutes.
5. The training cell runs eight configurations, each for 100 epochs and 3 seeds.
   Change `STATE` to `MD` and rerun setup onward to train Maryland.

`data/`, `weights/` and `results/` persist in
`My Drive/cs5284_project/`. If Dataverse returns HTTP 403, download the ZIP from
https://doi.org/10.7910/DVN/V71K5R and put it in that Drive `data/` folder.
Existing folders do not need to be cleared. Re-running the same training run
name overwrites its outputs; incomplete training is not resumed automatically.
Completed-seed metrics are saved immediately to survive a later interruption.

## Local commands

Use Python 3.10-3.12 and install `requirements.txt` in a virtual environment:

```bash
python -m pip install -r requirements.txt
python -m src.data.node2vec --state DE
python -m src.model.train --state DE --model mlp --task classification --without-node2vec
python -m src.model.train --state DE --model graphsage --task classification
```

The embedding trainer uses CUDA when available, otherwise CPU. It is adapted
from baseline-fw's custom PyTorch skip-gram/negative-sampling implementation,
with 20-step walks, 10 walks per node, 128 dimensions and one negative per pair.
This is different from the original karateclub/gensim embedding configuration.
Generation metadata identifies the implementation and parameters; caches from
another backend are rejected rather than silently reused. New embeddings live
under `data/embeddings/node2vec_merged/`; older original embeddings are preserved.

## Outputs

- `data/<state>/cache/`: weather, coordinates, degree and betweenness caches.
- `data/embeddings/node2vec_merged/<state>_128.npy` and `.json`: embeddings/provenance.
- `weights/<state>_<model>_<task>_<n2v|no_n2v>/seed<k>.pt`: best validation checkpoint.
- `results/<run>/seed<k>_metrics.json`: each completed seed's metrics/configuration.
- `results/<run>/metrics.json`: mean/std and completion status for all seeds.
- `results/<run>/seed<k>_test_predictions.pt`: months, predictions, actual counts,
  road endpoint indices, node longitude/latitude and target-unit metadata.

Data, outputs, environments and caches are ignored by Git. Neither parent branch
contained saved final AUROC/MAE results; the merged pipeline must be trained before
claiming improvements. The old feature branch remains available for comparison.
