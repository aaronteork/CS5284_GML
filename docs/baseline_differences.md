# Merged baseline: differences from the paper

This branch integrates feature/baseline (43025f1) and baseline-fw (4f926d1).
It is one merged pipeline, with a controlled with/without-Node2Vec ablation.
The methodological differences below prevent direct accuracy comparisons with
the paper. Completed three-seed Delaware reports are now included in `results/`;
they document the merged pipeline rather than a strict reproduction.

## What remains aligned

- Train/validation/test years from Table 7, for Delaware and Maryland.
- Two-layer, width-256 edge-aware GraphSAGE and per-road endpoint predictor;
  the MLP uses the predictor without a learned graph encoder.
- Adam, learning rate 0.001, 100 epochs, no dropout, validation every five epochs.
- 1:1 positive/negative training sampling; skip training months with fewer than
  ten accident roads. Each seed starts from a fresh encoder and predictor.
- Primary AUROC is computed per month and weighted by positive-road count.
  Pooled MAE is equivalent to sample-count-weighted monthly MAE when examples match.

## Combined feature and embedding changes

- **Custom GPU embeddings:** the active implementation is adapted from
  baseline-fw's PyTorch Node2Vec-style trainer, not karateclub. It uses 128
  dimensions, uniform unweighted walks (p=q=1), 20 nodes per walk, 10 walks per
  node, forward context window 5, one uniformly sampled negative per pair,
  Adam learning rate 0.01, batch size 1024 and one epoch, seed 0. The original
  karateclub setup used 80-node walks and gensim CBOW training. This changes
  the walk corpus and embedding objective. The original implementation remains
  in the parent feature branch, not as a second pipeline here.
- **Graph sampling correction:** reverse edges are added and deduplicated
  before GPU sampling. This fixes baseline-fw's duplication of already-present
  reverse edges, which could overweight bidirectional neighbours.
- **Richer shared node inputs:** standardised longitude/latitude, one-hot
  in/out degrees, approximate directed betweenness (500 sampled source nodes,
  seed 0, transformed by log1p(value * 1000)), and weather missing indicators.
- **Weather:** six columns including pressure; observed training values define
  normalisation. Missing inputs are replaced with the training mean (zero after
  standardisation) and accompanied by missingness indicators.
- **Road/traffic transforms:** log1p length and AADT, standardised; traffic and
  its missing indicator are always included. Unreported years carry forward only
  an earlier available report. This differs from both raw features and the
  reference notebook's endpoint-clamped traffic-year lookup.
- **Node2Vec ablation:** every MLP/GraphSAGE and classification/regression pair
  runs both with and without embeddings. Other features, splits and model seeds
  stay identical. The same cached embedding seed is held fixed across model runs.
- **Caching:** persistent monthly and structural caches with source-file
  signatures. This changes IO performance rather than the intended feature values.

## Data handling and targets

- Road features are looked up by actual graph edge ID, correcting the original
  trainer's position-based lookup in monthly accident lists.
- Raw sparse feature observations with duplicate indices are averaged while
  ignoring NaNs. Prior COO coalescing summed duplicates, altering feature values.
- Opposite directions are merged into one physical-road target, with counts
  summed. Message passing remains directed. The assumption that physical-road
  targets suit the labels still needs explicit validation in the data audit.
- Accident-free sampling includes all genuinely zero-count physical roads. It
  does not use the original code's endpoint-membership exclusion filter.
- Training months are shuffled every epoch. One optimiser step is taken per
  month rather than 32,768-positive mini-batches; differences matter for large months.
- Only months with available weather files are evaluated. Exact split coverage
  is recorded in metrics JSON; calendar-year boundaries alone do not establish
  that every month is present.

## Evaluation and reporting

- Full-network physical-road evaluation is primary; the original code evaluated
  accident edges plus the negatives remaining after its endpoint-based filter.
  Our balanced population is a supplementary diagnostic, not a paper equivalent.
- All available validation/test months are included, even with fewer than ten
  positive roads. Zero-accident months contribute to MAE and prediction exports;
  single-class months have undefined AUROC and are excluded only from that average.
- Regression outputs are clipped at zero during validation, test and prediction
  export. Training uses raw outputs. This intentional post-processing cannot
  worsen MAE/MSE for nonnegative targets, and gains must not be attributed solely
  to model architecture. Clipping can also introduce ties in regression risk scores.
- One checkpoint is selected per task by all-road validation AUROC or MAE;
  auxiliary metrics come from that checkpoint, rather than each choosing an epoch.
  Selection is separate for each configuration and seed, among evaluated epochs;
  validation runs every five epochs and at the final epoch. There is no early stopping.
- sklearn's AUROC correctly handles ties, unlike baseline-fw's rank implementation.
- Additional diagnostics include pooled AUROC, AP labelled PR-AUC, threshold
  precision/recall, monthly top-1% precision/recall, MSE and positive/negative MAE.
  AP is already in the paper's code; our implementation currently omits F1.
- Top-risk precision includes accident-free months as zero; top-risk recall
  omits those months where recall is undefined. Other auxiliary classification
  metrics are pooled rather than averaged like the primary AUROC.
- Three model seeds have separate saved checkpoints, prediction files and metrics;
  combined means/stds are written after every completed seed. Embedding variability
  is not included in the model-seed standard deviation.
- Coordinates and target-unit metadata accompany prediction exports for the map.
- Per-seed CSV histories persist training loss after each completed epoch and
  validation AUROC/MAE at evaluation checks (default every 5 epochs plus final).
  Separate curve panels mark the best validation epoch; test metrics are never
  used for checkpoint selection or plotted as an epoch-by-epoch tuning curve.
  Run metadata records the configuration, splits, embedding provenance, package
  versions and Git commit/dirty flag. These are reporting additions, not changes
  to the training objective or evaluation schedule.

## Scope and evidence

We cover MLP and GraphSAGE on DE/MD, not all eight states or the paper's other
architectures, multitask/transfer learning, or repeated hyperparameter search.
The Colab notebook separates training into four model/task cells, each covering
both Node2Vec settings and three seeds: 24 training runs per state.

The committed Delaware export contains all eight configurations, each with three
completed seeds (0/1/2) and histories covering epochs 1-100: 88 JSON/CSV/PNG reports.
All report 49,023 nodes, 65,252 physical-road targets, and 48/60/57 available
train/validation/test months. The histories, metrics, curves and metadata were
checked for completeness; full training was not independently rerun locally.
No Maryland results are included yet. Prediction tensors and checkpoints are
excluded from Git. The original two-model training notebook has been retired;
use `notebooks/train_baseline+gated_gcn+graph_transformer.ipynb` for the current
four-model baseline workflow.

All saved DE runs identify source commit `526b30e` and set `git_dirty: true`.
The exact uncommitted changes in that training checkout were not captured by
the metadata, so the commit alone does not establish an exact source snapshot.
The three seeds vary model training while holding the embedding seed fixed;
their standard deviation therefore does not measure embedding variability.
Comparisons with the paper remain qualified by the feature, embedding, target
and evaluation-population differences above, especially for full-network MAE.

The baseline-fw notebook saved preprocessing timings (DE embeddings 4 s, MD 27 s)
but no final AUROC/MAE: its training/result cells were unexecuted. Those timings
are not benchmarks for the merged implementation or the paper's original setup.
