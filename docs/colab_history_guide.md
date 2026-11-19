# Run the Delaware accident-history experiments in Colab

## Prepare the project

Open [the training notebook](../notebooks/train_accident_history.ipynb) in a Colab GPU runtime. Upload the project to `MyDrive/train_historydata` with its `data`, `src`, `scripts`, `notebooks` and `tests` directories, or upload a zip rooted at `train_historydata/` as `MyDrive/train_historydata_colab.zip`. The notebook copies code and inputs to the runtime and writes checkpoints and results directly to Drive.

Provide the Delaware data under `data/DE/` and the directed-edge history exports under `data/accident_history/outputs/`. To regenerate the archive from the benchmark, run [the source EDA](../notebooks/accident_history_eda.ipynb). It reads `DE/` under `CS5284_DATA_DIR` (default: the project's `data/`). It writes to `results/history_eda/` by default. Set `CS5284_HISTORY_EDA_OUTPUT` to a fresh output directory to prepare a separate export, then place that archive under the training path.

1. Mount Drive and run the Python 3.11 environment setup. Training runs in the isolated interpreter. The optional [runtime check](../notebooks/check_colab_requirements.ipynb) inspects the current kernel without installing packages.
2. Run the tests and data preparation. Expect 49,023 nodes, 116,196 directed edges, 65,252 physical roads, 48/60/57 months and 45 edge columns with full history.
3. Run configuration and monitoring, then profile the current GPU. The profile estimates 100-epoch training time per seed from a small set of training and validation months. It excludes installation, preprocessing and initial Node2Vec generation. Allow space for the data, caches and checkpoints (roughly 10–20 GB). If GPU memory is insufficient, use a larger GPU or change the width consistently across the comparisons under a fresh tag. Colab hardware and session limits vary. [Colab FAQ](https://research.google.com/colaboratory/faq.html)
4. Run the smoke tests before launching the full experiments. They write separate smoke-run directories and do not evaluate test data.

## Train, monitor and resume

The default configuration is classification, full history, Node2Vec enabled, width 256, 100 epochs and seeds 0–2. Each architecture has its own cell. MLP, GraphSAGE, GatedGCN and Graph Transformer together require 12 runs per task. Set `TASKS = ['regression']` and rerun configuration for the count models, or enable both tasks for all 24 runs. For a matched history comparison, keep the seeds, split, hyperparameters and Node2Vec setting fixed.

Training refreshes a loss/validation plot at evaluation time. Each run's curves can also be reconstructed from its saved CSVs. Under `MyDrive/train_historydata/experiments/`:

- `weights/<run>/seed0.pt` contains the validation-selected weights; `seed0_last.pt` contains resumable training state.
- `results/<run>/` contains run metadata, epoch CSVs, convergence plots, per-seed metrics and test predictions, and the cross-seed summary in `metrics.json`.

After disconnection, repeat setup and configuration and rerun the model cell with the original settings. Completed seeds are skipped and interrupted seeds resume at the next complete epoch. Use one writer per run directory. To increase a completed run's budget with the same code and data, keep the original tag and seeds, increase `EPOCHS`, and set `EXTEND_EXISTING = True`. The previous budget's results and weights are backed up under `epoch_snapshots/`. For new experiments leave extension disabled.

## Compare and export

Run the summary cell to write `experiments/results/comparison.csv`. Classification uses full-network AUROC, AP and precision among the top 1% of roads. Regression uses MAE, including separate errors on crash-positive and zero-crash roads. Choose checkpoints and model settings on validation. The optional simple baselines default to validation as well.

The optional GraphSAGE classification comparison uses independent run names for `none`, `own` and `all`. Enable its training flag to train these configurations or its download flag to summarise existing runs. The export includes scores, paired changes by seed and the available curves. The [100-epoch results used in the final report](../results/history_100_epochs) compare full history with the archived controls; comparisons using different implementations or training budgets provide contextual evidence.

Training uses the same observed target-month weather as the benchmark. History advances one month at a time using earlier recorded crashes. Deployment with delayed reports or unavailable weather would require a corresponding change to every input condition.
