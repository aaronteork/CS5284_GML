# Accident-history experiment reports

This snapshot contains the completed, non-smoke `history=all` experiments copied
from the local `train_historydata 2/experiments/results` export on 2026-11-11.
The eight runs cover MLP, GraphSAGE, GatedGCN, and Graph Transformer for
classification and regression, each with Node2Vec enabled and seeds 0, 1, and 2.
The standard training configuration uses 256 hidden channels, 100 epochs,
learning rate 0.001, and validation every five epochs. Per-run JSON records the
exact saved configuration and provenance; per-seed metrics and epoch histories
are included where available.

The results are test-set metrics. Classification is primarily evaluated with
AUROC and PR-AUC; regression uses MAE/MSE, while AP and top-1% metrics describe
risk ranking from predicted accident counts. The graph keeps directed edges for
message passing, while labels and history use physical roads with both
directions' monthly accident counts combined.

This snapshot does **not** establish the gain from adding history. The matching
`history=none` runs are incomplete in the source export, so
`comparison.csv` has no paired history deltas. Complete matching baselines must
be run with the same code, data, seeds, and training settings before reporting
history's contribution. The older root-level benchmark reports remain intact
and are not treated as these matched baselines.

Only lightweight reports are committed here: JSON metrics/metadata, CSV epoch
histories, and PNG learning curves. Raw data, embeddings, prediction tensors,
model weights, and resume checkpoints are excluded.
