# Baseline: differences from the paper

Comparison of our baseline (`src/`) with Nippani et al. (NeurIPS 2023) and its official code ([ML4RoadSafety](https://github.com/VirtuosoResearch/ML4RoadSafety)).

Matching the paper: train/validation/test years (Table 7), intersection features (centrality + weather for the MLP), road features (24 road-type flags + length), weather filling and training-year normalisation, the GraphSAGE layer (edge features in messages, mean with self-loop, L2-normalised), the 2-layer predictor on both endpoints plus road features, 1:1 sampling of accident-free roads during training, skipping months with fewer than 10 accident roads, Adam, learning rate 0.001, 100 epochs, hidden size 256, dropout 0, and validation every 5 epochs.

**Types**

- **Bug fix**: the paper's code does something unintended; we do what the paper describes.
- **Implementation choice**: a deliberate difference in how something is done.
- **Evaluation**: a difference in how results are scored or reported.
- **Not done yet**: part of the replication that's still outstanding.
- **Scope**: what we chose to cover.

| # | Area | Paper / official code | Ours | Type | Explanation |
| --- | --- | --- | --- | --- | --- |
| 1 | Road features | Looked up with `edge_attr[perm]`, a position in the month's accident list | Looked up by the road's ID | Bug fix | The paper's code gives each road the features of an unrelated road, so its road features are effectively noise. |
| 2 | Sampling accident-free roads | Excludes any road whose two endpoints both appear in that month's accident roads | Any road with no accident that month | Bug fix | The paper's filter drops roads next to accident roads, leaving easier, far-away negatives that likely inflate AUROC. |
| 3 | Seeds | Only the predictor is reset between runs; the GNN keeps its trained weights | A fresh model for every seed | Bug fix | The paper's later seeds start from an already-trained encoder, so they aren't independent runs. |
| 4 | Unit of prediction | Directed edges (a two-way road is two edges) | Physical roads (two directions merged) | Implementation choice | Accidents are recorded against one direction only, so we predict once per road instead of counting a two-way road twice. |
| 5 | Month order | Calendar order every epoch | Shuffled every epoch | Implementation choice | Shuffling avoids the model drifting towards whichever months it saw last. |
| 6 | Centrality features | Used raw (degree around 2–6, betweenness around 0.0001) | Standardised per column | Implementation choice | Putting features on a common scale makes the MLP easier to train and shouldn't change what it can learn. |
| 7 | Traffic volume | Enabled in the example script | Off by default (`--use-traffic-volume` to enable) | Implementation choice | It's missing for over 98% of roads, so we leave it out unless a run shows it helps. |
| 8 | Regression output | Raw predictions, which can be negative | Clipped at 0 when evaluating | Implementation choice | A negative accident count is meaningless, and clipping can only reduce error. |
| 9 | Batching within a month | Mini-batches of 32,768 accident roads | One step per month | Implementation choice | Equivalent here, since no Delaware or Maryland month has more than 32,768 accident roads. |
| 10 | Node2Vec training | `karateclub` (gensim CBOW, hierarchical softmax) | Same walk settings, trained as skip-gram with negative sampling in PyTorch | Implementation choice | gensim produced corrupted embeddings on large inputs in our environment and `karateclub` won't install with current libraries, so the embeddings will be similar but not identical. |
| 11 | Best epoch | Chosen separately for each metric | One epoch (best validation AUROC or MAE) for all metrics | Evaluation | Picking a different epoch per metric is optimistic, because one model can't be at its best epoch for every metric at once. |
| 12 | Combining months | Metric per month, then a weighted average | All months pooled, then one metric | Evaluation | Both are reasonable, but they give slightly different numbers. |
| 13 | Roads scored at test time | Accident roads plus all negatives left after the biased filter in row 2 | `all`: every road; `balanced`: accident roads plus an equal random sample | Evaluation | `all` reflects real use and is the closer match for MAE; `balanced` is easier to compare across models. |
| 14 | Model selection | The paper's own scoring | `balanced` validation scores by default (`--select-protocol all` to change) | Evaluation | Selection on `all` is the closer match to the paper and should be used for the like-for-like comparison. |
| 15 | Metrics | AUROC, MAE, plus F1, average precision, precision and recall at 0.5 | Also PR-AUC, precision/recall on the top 1% riskiest roads each month, and MAE split by accident and accident-free roads | Evaluation | The extra metrics show how useful the model is as a monthly risk list, which AUROC alone doesn't. |
| 16 | Node2Vec embeddings | Used as GraphSAGE's intersection features | Code written but not yet generated or checked | Not done yet | Until they exist, GraphSAGE runs use centrality features and aren't comparable to the paper. |
| 17 | Full runs | 100 epochs × 3 seeds per model and state | Only 2-epoch smoke tests so far | Not done yet | No replication numbers should be reported until the full runs are done. |
| 18 | Hyperparameters | Grid search over layers, width, learning rate and epochs | The paper's final values, no search | Scope | Re-tuning isn't needed to reproduce the paper's reported configuration. |
| 19 | States | Eight US states | Delaware and Maryland | Scope | The two smaller states keep training affordable and are enough to check the paper's findings. |
