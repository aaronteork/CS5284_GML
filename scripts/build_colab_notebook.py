"""Generate the portable Colab entry point without executing any training."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip().splitlines(True)})


md("""
# Accident-history experiments on Delaware

Place the project and data in `MyDrive/train_historydata` or upload `train_historydata_colab.zip` to MyDrive. Select a GPU runtime. Code and data are read locally on Colab, and checkpoints and metrics are written to Drive. The graph retains directed edges. Targets and history use unordered endpoints, and all history windows end before the target month.

The default experiment uses full history, Node2Vec, 100 epochs and three seeds. Run one model at a time to check the outputs before proceeding.
""")
code("""
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import json, os, shutil, subprocess, sys, zipfile

DRIVE_ROOT = Path('/content/drive/MyDrive')
DRIVE_PROJECT = DRIVE_ROOT / 'train_historydata'
WORK_DIR = Path('/content/train_historydata')
ARCHIVE = DRIVE_ROOT / 'train_historydata_colab.zip'
OUTPUT_DIR = DRIVE_PROJECT / 'experiments'

if (DRIVE_PROJECT / 'src/model/train.py').exists():
    # Code/data stay local during training; do not copy old training outputs.
    shutil.copytree(DRIVE_PROJECT, WORK_DIR, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('experiments', '__pycache__', '.git', 'local_smoke*'))
elif ARCHIVE.exists():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        for info in z.infolist():
            target = (Path('/content') / info.filename).resolve()
            if not target.is_relative_to(WORK_DIR):
                raise ValueError(f'Unexpected archive path: {info.filename}')
        z.extractall('/content')
else:
    raise FileNotFoundError('Upload the entire train_historydata folder or train_historydata_colab.zip to MyDrive.')
assert (WORK_DIR / 'data/DE/adj_matrix.pt').exists()
assert (WORK_DIR / 'data/accident_history/outputs/monthly_features/history_2022-09.npz').exists()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
print('Code/data:', WORK_DIR)
print('Persistent results/checkpoints:', OUTPUT_DIR)
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
""")
md("""
## 1. Install the training environment

Create an isolated Python 3.11 environment and install the project dependencies. The first installation requires internet access. Training commands use TRAIN_PYTHON throughout.
""")
code(r"""
TRAIN_PYTHON = '/content/train_history_env/bin/python'

def run_live(command, on_line=None):
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with subprocess.Popen([str(x) for x in command], cwd=WORK_DIR, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            if on_line is not None:
                on_line(line)
        result = process.wait()
    if result:
        raise subprocess.CalledProcessError(result, command)

run_live([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
if not Path(TRAIN_PYTHON).exists():
    run_live([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '--seed', '/content/train_history_env'])
run_live([TRAIN_PYTHON, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])
run_live([TRAIN_PYTHON, '-c', "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'Select a GPU runtime'; print(torch.cuda.get_device_name())"])
""")
md("""
## 2. Test and validate the data

Run synthetic unit tests, check the monthly archives and build reusable caches. Expected Delaware dimensions are 49,023 nodes, 116,196 directed edges and 65,252 physical-road targets. The chronological splits contain 48, 60 and 57 months. Without Node2Vec each node has 29 inputs. Each edge has 27 baseline inputs or 45 with full history.
""")
code("""
run_live([TRAIN_PYTHON, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
run_live([TRAIN_PYTHON, '-u', '-m', 'src.prepare', '--report', str(OUTPUT_DIR / 'data_validation.json')])
""")
md("""
## 3. Configure experiments

The defaults train full-history classifiers. Set TASKS to ['regression'] for count forecasts or include both tasks to run them together. Select 'own' for own-road history or 'none' for a control. Use a new experiment tag when changing settings. To extend an already completed run with matching code, inputs and seeds, retain its tag, increase EPOCHS and set EXTEND_EXISTING to True. The previous results and weights are backed up before extension.
""")
code(r"""
TASKS = ['classification']
HISTORY_MODES = ['all']
USE_NODE2VEC = True
SEEDS = [0, 1, 2]
EXTEND_EXISTING = False
EPOCHS = 100
HIDDEN_CHANNELS = 256
EXPERIMENT_TAG = 'v1'

def train_command(model, task, history, smoke=False):
    embedding = 'n2v' if USE_NODE2VEC else 'no_n2v'
    name = f'{EXPERIMENT_TAG}_DE_{model}_{task}_{embedding}_hist_{history}'
    command = [TRAIN_PYTHON, '-u', '-m', ('scripts.extend_accident_run' if EXTEND_EXISTING and not smoke else 'src.model.train'), '--model', model,
               '--task', task, '--history-mode', history, '--epochs', str(EPOCHS),
               '--hidden-channels', str(HIDDEN_CHANNELS), '--seeds', *map(str, SEEDS),
               '--device', 'cuda', '--output-dir', str(OUTPUT_DIR), '--run-name', name, '--resume']
    if not USE_NODE2VEC:
        command += ['--without-node2vec']
    if smoke:
        command += ['--smoke-test']
    return command

print('Full seed runs:', 4 * len(TASKS) * len(HISTORY_MODES) * len(SEEDS))
""")
md(r"""
### Monitor convergence

Each validation refresh shows epoch losses and measured validation scores. Stars mark the best validation checkpoint for each seed. Curves are also saved to the run directory on Drive.
""")
code(r"""
import re
from io import BytesIO
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import Image, display

def save_convergence_plot(command, task, pending=None):
    name = command[command.index('--run-name') + 1]
    folder = OUTPUT_DIR / 'results' / name
    metric = 'valid_mae' if task == 'regression' else 'valid_auroc'
    fig, (loss_ax, val_ax) = plt.subplots(1, 2, figsize=(10, 3.5))
    try:
        for index, seed in enumerate(SEEDS):
            path = folder / f'seed{seed}_history.csv'
            history = pd.read_csv(path) if path.exists() else pd.DataFrame()
            # The trainer prints the validation line just before appending its CSV row.
            if pending and seed == pending[0]:
                _, epoch, loss, score = pending
                if history.empty or epoch not in history['epoch'].values:
                    history = pd.concat([history, pd.DataFrame([{
                        'epoch': epoch, 'train_loss': loss, metric: score}])])
            if history.empty:
                continue
            color = f'C{index % 10}'
            loss_ax.plot(history['epoch'], history['train_loss'], color=color, label=f'seed {seed}')
            valid = history.dropna(subset=[metric]).set_index('epoch')[metric]
            if len(valid):
                best = valid.idxmin() if task == 'regression' else valid.idxmax()
                val_ax.plot(valid.index, valid, '.-', color=color, label=f'seed {seed}')
                val_ax.plot(best, valid.loc[best], '*', ms=13, color=color)
        if not loss_ax.lines:
            return None
        loss_ax.set_title('Training loss')
        val_ax.set_title('Validation ' + ('MAE (lower is better)' if task == 'regression'
                                         else 'AUROC (higher is better)'))
        for ax in (loss_ax, val_ax):
            ax.set_xlabel('Epoch')
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend()
        fig.suptitle(name)
        fig.tight_layout()
        with BytesIO() as buffer:
            fig.savefig(buffer, format='png', dpi=105)
            png = buffer.getvalue()
        target = folder / 'convergence.png'
        with NamedTemporaryFile(dir=folder, suffix='.png', delete=False) as tmp:
            tmp.write(png)
        Path(tmp.name).replace(target)
        return png
    finally:
        plt.close(fig)

validation_line = re.compile(
    r'\[seed (\d+)\] epoch\s+(\d+) \| loss ([\d.eE+-]+) \| '
    r'valid (?:MAE|AUROC) \([^)]+\) ([\d.eE+-]+)'
)

def run_model(model):
    for task in TASKS:
        for history in HISTORY_MODES:
            command = train_command(model, task, history)
            print(f'Running {model} / {task} / history={history} / Node2Vec={USE_NODE2VEC}', flush=True)
            chart = widgets.Image(format='png')
            display(chart)

            def refresh(line):
                match = validation_line.search(line)
                if match:
                    seed, epoch, loss, score = match.groups()
                    chart.value = save_convergence_plot(command, task, (int(seed), int(epoch),
                                                                   float(loss), float(score))) or b''

            run_live(command, on_line=refresh)
            chart.value = save_convergence_plot(command, task) or b''
""")
md("""
## 4. Profile GPU time and memory

Measure training and validation on a small subset of months. The report estimates the cost of 100 epochs per seed and records peak GPU memory. If memory is insufficient, select a larger GPU or use a common smaller width and a new tag for all configurations.
""")
code("""
command = [TRAIN_PYTHON, '-u', '-m', 'src.profile', '--device', 'cuda',
           '--hidden-channels', str(HIDDEN_CHANNELS), '--report', str(OUTPUT_DIR / 'hardware_profile.json')]
if USE_NODE2VEC:
    command += ['--with-node2vec']
run_live(command)
""")
md("""
## 5. Smoke-test the four models

Each model trains for two epochs on two training months and validates on two months. Results use a separate smoke-test directory and do not enter the test comparison.
""")
code("""
for model in ['mlp', 'graphsage', 'gatedgcn', 'graph_transformer']:
    run_live(train_command(model, TASKS[0], 'all', smoke=True))
""")
for number, model in enumerate(['mlp', 'graphsage', 'gatedgcn', 'graph_transformer'], 6):
    md(f"""
## {number}. Train {model}

Run the selected tasks, history settings and seeds. After a disconnect, repeat setup and configuration with the original settings, then run this cell to resume from the last complete epoch or skip completed seeds.
""")
    code(f"run_model('{model}')")
md("""
### Optional GraphSAGE classification ablation

Compare the same three seeds and training settings with no history, own-road history and full history. To train, enable RUN_ABLATION. To view and download an existing comparison after reconnecting, restore the original experiment tag, epoch budget and embedding setting, then enable DOWNLOAD_ABLATION. The final report compares full history with archived no-history controls; this optional experiment has its own run names.
""")
code(r"""
RUN_ABLATION = False
DOWNLOAD_ABLATION = False
results_root = OUTPUT_DIR / "results"
embedding = "n2v" if USE_NODE2VEC else "no_n2v"
prefix = f"{EXPERIMENT_TAG}_ablation_DE_graphsage_classification_{embedding}_e{EPOCHS}"
ablation_modes = ("none", "own", "all")
ablation_runs = {mode: f"{prefix}_hist_{mode}" for mode in ablation_modes}

if RUN_ABLATION:
    for mode, name in ablation_runs.items():
        command = train_command("graphsage", "classification", mode)
        command[command.index("--run-name") + 1] = name
        run_live(command)
        save_convergence_plot(command, "classification")
""")
code(r"""
if RUN_ABLATION or DOWNLOAD_ABLATION:
    ablation_rows = []
    reference_identity = None
    for mode, name in ablation_runs.items():
        result_path = results_root / name / "metrics.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing summary for {mode}: {result_path}")
        saved = json.loads(result_path.read_text())
        if not saved["complete"] or saved["smoke_only"] or sorted(r["seed"] for r in saved["runs"]) != sorted(SEEDS):
            raise ValueError(f"Incomplete ablation run: {name}")
        identity = dict(saved["identity"])
        identity["settings"] = {k: v for k, v in identity["settings"].items()
                                if k not in ("history_mode", "run_name")}
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise ValueError(f"Source, data or training settings differ: {name}")
        for run in saved["runs"]:
            test = run["test"]["all"]
            ablation_rows.append({"History": mode, "Seed": run["seed"], "AUROC": test["AUROC"],
                                  "AP": test["PR-AUC"], "Precision@top1%": test["precision@top1%"]})

    ablation_scores = pd.DataFrame(ablation_rows).set_index(["History", "Seed"])
    def mean_sd(values):
        return f"{values.mean():.4f} ± {values.std(ddof=0):.4f}"

    ablation_summary = ablation_scores.groupby(level="History").agg(mean_sd).loc[list(ablation_modes)]
    display(ablation_summary)
    contrasts = {}
    for before, after in [("none", "own"), ("own", "all"), ("none", "all")]:
        paired = ablation_scores.loc[after] - ablation_scores.loc[before]
        contrasts[f"{after} − {before}"] = paired.apply(mean_sd)
    ablation_differences = pd.DataFrame(contrasts).T
    display(ablation_differences)
""")
code(r"""
if DOWNLOAD_ABLATION:
    from zipfile import ZipFile, ZIP_DEFLATED
    from google.colab import files

    archive_path = results_root.parent / f"{prefix}_report.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, table in (("seed_scores", ablation_scores), ("summary", ablation_summary),
                            ("paired_differences", ablation_differences)):
            archive.writestr(f"{name}.csv", table.to_csv())
        for folder_name in ablation_runs.values():
            folder = results_root / folder_name
            for path in [folder / "metrics.json", folder / "run_metadata.json",
                         folder / "convergence.png", *sorted(folder.glob("seed*_history.csv"))]:
                if path.is_file():
                    archive.write(path, path.relative_to(results_root))
    print("Archive on Drive:", archive_path)
    files.download(str(archive_path))
""")
md(r"""
### Review saved curves

Read both tasks from the selected experiment tag and history settings. This cell can be used after reconnecting without running training again.
""")
code(r"""
for model in ['mlp', 'graphsage', 'gatedgcn', 'graph_transformer']:
    for task in ['regression', 'classification']:
        for history in HISTORY_MODES:
            command = train_command(model, task, history)
            png = save_convergence_plot(command, task)
            if png:
                display(Image(data=png))
            else:
                print('No training history for', command[command.index('--run-name') + 1])
""")
md("""
## 10. Compare completed experiments

The summary excludes smoke tests and incomplete runs. It compares compatible runs and includes archived reference scores where available. Positive AUROC changes and negative MAE changes favour the history setting. Separate experiments with different training budgets or implementations provide contextual comparisons. Select models on validation data. To run regression, update TASKS, rerun the configuration and run each model cell.
""")
code("""
run_live([TRAIN_PYTHON, '-m', 'src.compare', '--results-dir', str(OUTPUT_DIR / 'results')])
import pandas as pd
table = OUTPUT_DIR / 'results/comparison.csv'
if table.exists():
    display(pd.read_csv(table))
""")
md("""
## 11. Optional simple regression baselines

Compare zero counts, the previous month and trailing three- and twelve-month means. Use validation for model selection. Set SPLIT to 'test' for the final evaluation once the choices are fixed.
""")
code("""
SPLIT = 'valid'
run_live([TRAIN_PYTHON, '-u', '-m', 'src.simple_baselines', '--split', SPLIT,
          '--output', str(OUTPUT_DIR / f'simple_baselines_{SPLIT}.json')])
""")
md("""
## Output files and interpretation

Drive stores best and last checkpoints under `experiments/weights/<run>/` and per-seed metrics, curves and test predictions under `experiments/results/<run>/`. The original benchmark uses observed target-month weather. History uses earlier months only, including prior test months under a rolling forecast. The original exploratory analysis included 2018; the final report uses training-only history associations.
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell['id'] = f'history-cell-{i}'
path = ROOT / 'notebooks/train_accident_history.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(path)
