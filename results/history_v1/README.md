# Accident-history experiment reports

This snapshot contains the completed, non-smoke `history=all` experiments copied
from the local `train_historydata 2/experiments/results` export on 2026-11-11.
The eight runs cover MLP, GraphSAGE, GatedGCN, and Graph Transformer for
classification and regression, each with Node2Vec enabled and seeds 0, 1, and 2.
The runs use 256 hidden channels, learning rate 0.001, validation every five
epochs, and Node2Vec. Classification runs used 30 epochs; regression used 30
epochs for MLP and 40 for the graph models. Per-run JSON records the saved
configuration and provenance; per-seed metrics and epoch histories are included.

The results are test-set metrics. Classification is primarily evaluated with
AUROC and PR-AUC; regression uses MAE/MSE, while AP and top-1% metrics describe
risk ranking from predicted accident counts. The graph keeps directed edges for
message passing, while labels and history use physical roads with both
directions' monthly accident counts combined.

`comparison.csv` compares these runs with the committed original benchmark
results in the repository root where a model/task/Node2Vec reference exists.
MLP and GraphSAGE have such references; GatedGCN and Graph Transformer do not.
The original runs used 100 epochs, while these history runs used 30 or 40, and
the code/data provenance is not identical. The deltas are contextual
comparisons, not paired estimates of the effect of adding history. No new
`history=none` training runs are required for this table.

Only lightweight reports are committed here: JSON metrics/metadata, CSV epoch
histories, and PNG learning curves. Raw data, embeddings, prediction tensors,
model weights, and resume checkpoints are excluded.
