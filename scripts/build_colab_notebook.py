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
# Accident-history experiments: MLP → GraphSAGE → GatedGCN → Graph Transformer

先将整个 `train_historydata` 文件夹放在 **MyDrive/train_historydata**，或上传
`train_historydata_colab.zip` 到 MyDrive。选择 GPU runtime，然后按顺序运行。

这是修改后的独立项目：**不要再 clone 上游仓库**。
数据在 Colab 本地磁盘读取；checkpoint 和结果每个 epoch 直接写入 Drive。
图保留有向边，标签和历史按无序端点合并。历史只使用目标月以前的事故。

初版默认：四模型、分类、Node2Vec、完整历史特征、100 epochs、3 seeds。
每个模型有单独的运行单元；建议先完成 MLP，确认后继续其他模型。
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
## 1. 安装环境
使用隔离的 Python 3.11 环境和项目固定依赖。第一次安装需要联网。
不要用 notebook 内核直接 import 训练模块；后续均由 TRAIN_PYTHON 执行，避免依赖混用。
""")
code("""
TRAIN_PYTHON = '/content/train_history_env/bin/python'

def run_live(command):
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with subprocess.Popen([str(x) for x in command], cwd=WORK_DIR, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
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
## 2. 测试和完整数据检查
测试使用微型合成数据；prepare 检查全部原始档案、方向合并、月份和缺失值，并生成缓存。
预期：49,023 nodes；116,196 directed edges；65,252 road targets；48/60/57 个月；
无 Node2Vec 时节点 29 维；道路 baseline 27 维，完整历史 45 维。
""")
code("""
run_live([TRAIN_PYTHON, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
run_live([TRAIN_PYTHON, '-u', '-m', 'src.prepare', '--report', str(OUTPUT_DIR / 'data_validation.json')])
""")
md("""
## 3. 固定实验设置
先保持以下默认值。若需要两类任务一并运行，设置 TASKS=['classification', 'regression']。
默认只训练完整历史配置；分类和回归结果会与仓库中已有原始基准分数作上下文比较。
如需自身历史子组消融，可将 HISTORY_MODES 改为 ['own']。不要为了该比较重跑无历史组。
修改层宽/学习率/epoch 等后需新的 EXPERIMENT_TAG，旧实验不会被静默覆盖。
""")
code("""
TASKS = ['classification']
HISTORY_MODES = ['all']
USE_NODE2VEC = True
SEEDS = [0, 1, 2]
EPOCHS = 100
HIDDEN_CHANNELS = 256
EXPERIMENT_TAG = 'v1'

def train_command(model, task, history, smoke=False):
    embedding = 'n2v' if USE_NODE2VEC else 'no_n2v'
    name = f'{EXPERIMENT_TAG}_DE_{model}_{task}_{embedding}_hist_{history}'
    command = [TRAIN_PYTHON, '-u', '-m', 'src.model.train', '--model', model,
               '--task', task, '--history-mode', history, '--epochs', str(EPOCHS),
               '--hidden-channels', str(HIDDEN_CHANNELS), '--seeds', *map(str, SEEDS),
               '--device', 'cuda', '--output-dir', str(OUTPUT_DIR), '--run-name', name, '--resume']
    if not USE_NODE2VEC:
        command += ['--without-node2vec']
    if smoke:
        command += ['--smoke-test']
    return command

def run_model(model):
    for task in TASKS:
        for history in HISTORY_MODES:
            print(f'\\nRunning {model} / {task} / history={history}', flush=True)
            run_live(train_command(model, task, history))

print('Full seed runs:', 4 * len(TASKS) * len(HISTORY_MODES) * len(SEEDS))
""")
md("""
## 4. 实测 GPU 时间和显存
此单元仅在训练集做短训练，并用少量验证月份估时；不评估测试集。
报告给出每个模型 100 epochs/单 seed 的估算和显存峰值。它比静态时间表更可靠。
若某模型 OOM，选择更大显存 GPU；或把所有对照都统一改为 128 维并更换实验标签。
""")
code("""
command = [TRAIN_PYTHON, '-u', '-m', 'src.profile', '--device', 'cuda',
           '--hidden-channels', str(HIDDEN_CHANNELS), '--report', str(OUTPUT_DIR / 'hardware_profile.json')]
if USE_NODE2VEC:
    command += ['--with-node2vec']
run_live(command)
""")
md("""
## 5. 四模型 smoke test
每个模型只用两个训练月份、两个验证月份，跑两个 epochs；不评估测试集。
日志和 checkpoint 存在带 `_smoke` 后缀的独立目录，不进入正式对比表。
""")
code("""
for model in ['mlp', 'graphsage', 'gatedgcn', 'graph_transformer']:
    run_live(train_command(model, TASKS[0], 'all', smoke=True))
""")
for number, model in enumerate(['mlp', 'graphsage', 'gatedgcn', 'graph_transformer'], 6):
    md(f"""
## {number}. 正式训练 {model}
本单元依次运行指定任务、历史配置和三个种子。每个 epoch 保存恢复状态。
断线后重新连接，运行环境准备和配置单元，再重跑本单元：自动续训或跳过已完成 seed。
""")
    code(f"run_model('{model}')")
md("""
## 10. 汇总正式结果
只有完成的正式实验进入 comparison.csv。表格会使用仓库里兼容的原始 MLP/GraphSAGE
结果作为参考；原始运行用了 100 epochs，历史运行用了 30/40 epochs，因此差值仅作上下文比较。
AUROC 差值为正表示历史结果更高，MAE 差值为负表示历史结果更低；缺少原始结果的模型不显示 delta。
要继续回归：在第 3 步设 TASKS=['regression']，重新运行第 3 步，再依次运行四模型单元。
不要依据测试结果不断调参；选择和调参使用验证集。
""")
code("""
run_live([TRAIN_PYTHON, '-m', 'src.compare', '--results-dir', str(OUTPUT_DIR / 'results')])
import pandas as pd
table = OUTPUT_DIR / 'results/comparison.csv'
if table.exists():
    display(pd.read_csv(table))
""")
md("""
## 11. 可选：固定简单参照
回归实验需比较全零、上一月计数、过去 3/12 月均值，避免被稀疏标签的低 MAE 误导。
默认只检查验证集。全部实验方案确定、需要最终报告时，将 SPLIT 改为 'test' 再执行一次。
这些参照直接预测物理道路计数，不经过训练采样。
""")
code("""
SPLIT = 'valid'
run_live([TRAIN_PYTHON, '-u', '-m', 'src.simple_baselines', '--split', SPLIT,
          '--output', str(OUTPUT_DIR / f'simple_baselines_{SPLIT}.json')])
""")
md("""
## 结果位置与解释
Drive 的 `train_historydata/experiments/weights/<run>/`：最佳 checkpoint、每 epoch 的 last checkpoint。
`experiments/results/<run>/`：每 seed 的曲线、指标、测试预测及配置/数据/源码指纹。

保留了原 benchmark 的目标月观测天气。因此本实验是在同等原始输入条件下比较新增历史，
不能宣称所有输入在月初均已可用。历史严格过去，但后续测试月份可使用此前测试月份已发生的事故。
原历史 EDA 查看过 2018 年；报告需披露这与测试期的重叠。旧仓库分数是上下文参考，不是配对消融。
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell['id'] = f'history-cell-{i}'
path = ROOT / 'notebooks/train_accident_history.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(path)
