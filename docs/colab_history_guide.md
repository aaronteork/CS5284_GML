# Colab 操作指南

## 1. 上传和打开

选择一种方式：

- **文件夹**：上传整个 `train_historydata` 到 `MyDrive/train_historydata`，保留 data/src/notebooks 等结构。
- **ZIP**：上传 `train_historydata_colab.zip` 到 `MyDrive/train_historydata_colab.zip`，notebook 自动解压到 `/content/train_historydata`。

在 Colab 选择上传 notebook，打开本地 **`notebooks/train_accident_history.ipynb`**。也可从 Drive 打开同一个文件。不要使用 reference 中的原 notebook，不需要 GitHub token，不再 clone 上游。

## 2. 选择 GPU

Runtime → Change runtime type → GPU。单张 **T4 16 GB** 可作为起点，**L4 24 GB** 有更多显存余量；无需多卡或 TPU。

建议至少约 12 GB 系统 RAM，24 GB 更宽裕；为依赖、数据、缓存、checkpoint 和预测预留 10–20 GB 空间。GPU 负责前后向，CPU 负责预处理和指标。

这不是实测显存保证。GatedGCN/Transformer 的完整图中间张量更多，第 4 步 profile 会测当前设备的峰值。OOM 时优先换大显存 GPU；也可把**所有对照统一改为 128 维并使用新实验标签**，不能只缩小某个历史实验。

显存规格参考 [Google Cloud GPU 说明](https://docs.cloud.google.com/compute/docs/gpus/)。Colab GPU 类型、时长和额度不保证固定，见 [官方 FAQ](https://research.google.com/colaboratory/faq.html)。不要假定一个免费 session 能完成全部实验。

## 3. 按 notebook 顺序准备

1. **挂载 Drive**。代码/数据复制到 `/content` 本地磁盘，checkpoint 和结果直接写 Drive。
2. **安装环境**。创建隔离 Python 3.11 环境，安装 requirements，并确认 CUDA。首次需要联网。
3. **测试和 prepare**。检查所有档案，构建可复用缓存。预期：49,023 nodes、116,196 directed edges、65,252 roads、48/60/57 个月、完整历史边输入 45 维。
4. **固定配置**。保留默认值开始。
5. **GPU profile**。实测训练/验证时间、显存，生成当前设备的耗时估计，不评估测试集。
6. **四模型 smoke test**。每模型两个 epochs、两个训练/验证月份，不评估测试集。日志在独立 `_smoke` 目录，不进入正式结果表。

先只检查 MLP 时，可把 smoke 单元的模型列表改为 `['mlp']`。之后再检查其他三种。

## 4. 先跑哪个

推荐 **MLP → GraphSAGE → GatedGCN → Graph Transformer**。默认：

```python
TASKS = ['classification']
HISTORY_MODES = ['none', 'all']
USE_NODE2VEC = True
SEEDS = [0, 1, 2]
EPOCHS = 100
HIDDEN_CHANNELS = 256
EXPERIMENT_TAG = 'v1'
```

先只运行 MLP 正式训练单元。它先跑无历史三个 seed，再跑完整历史三个 seed，共六次。确认能产生结果表后，再依次运行其他三个模型单元。

四模型分类总共 **4 × 2 × 3 = 24 次**。匹配种子的月份排列和负样本序列相同。

分类完成后设 `TASKS=['regression']`，重新执行配置单元，再依次运行四模型。回归额外 24 次，合计 **48 次**。也可一开始同时设两类任务，但每个模型单元会连续训练 12 次。

需要自身历史消融时，把 HISTORY_MODES 改为 `['none','own','all']`；同配置已完成 none/all 会跳过，新增 own。

要考察 Node2Vec，统一设 `USE_NODE2VEC=True` 后重跑两边。第一次自动生成嵌入；不同 run 名不会覆盖无嵌入结果。两种嵌入设置都跑会将任务数再翻倍。

## 5. 路径与结果

活动代码的数据路径相对于项目根目录，不再使用 `/Users/...`：

```text
data/DE/
data/accident_history/outputs/
```

Notebook 开头根据实际上传位置设置 DRIVE_PROJECT 或 ZIP 路径即可。默认持久化结果：

```text
/content/drive/MyDrive/train_historydata/experiments/
├── weights/<run>/
│   ├── seed0.pt           # 验证最优模型
│   └── seed0_last.pt      # 完整 epoch 的恢复状态
└── results/<run>/
    ├── run_metadata.json
    ├── seed0_history.csv
    ├── seed0_history.png
    ├── seed0_metrics.json
    ├── seed0_test_predictions.pt
    └── metrics.json
```

不要删除 experiments，不要在两个 session 同时写同一个 run。无需复制本地 smoke checkpoint 到正式结果目录。

## 6. 断线续训

重新连接 GPU，运行挂载、环境和配置单元，保持原设置及 EXPERIMENT_TAG，重跑中断模型单元。

Notebook 已带 `--resume`：完成的 seed 自动跳过，中断的 seed 恢复上个完整 epoch 的模型、优化器和随机状态。中间未完成的 epoch 重跑。

若代码/数据/配置指纹改变，恢复会被拒绝。恢复原设置，或换新的 EXPERIMENT_TAG。修改层宽、学习率、epoch 数都视为新实验，不会静默覆盖。

## 7. 看结果

最后运行汇总，生成 `experiments/results/comparison.csv`：

- 分类主看 all AUROC，同时看 AP、top-1% precision；不是 accuracy。
- 回归看 all MAE、事故道路 MAE，以及全零/历史均值参照。
- mean/std 是三个 seed 的均值/标准差。
- delta 是历史实验减匹配 baseline：AUROC 正数好，MAE 负数好。
- smoke、未完成实验不会进入正式表；不匹配配置不会计算增益。
- reference 中的旧指标仅供参考，不能代替本次 baseline 重跑。

可选简单参照单元默认评估验证集。最终方案确定后设 SPLIT='test'，报告预先固定的全零/lag1/mean3/mean12；不要用测试集选窗口。

## 8. 时间与算力预算

**优先使用 profile 的实测值**：它计时三个训练和三个验证月份，推算 100 epochs/单 seed 的时间，并给 IO 余量。首次预处理、安装和 Node2Vec 不在估计内。CPU 指标计算和 Drive checkpoint 写入也影响总时间。

旧仓库日志中 MLP 约 **6–7 分钟/seed**，GraphSAGE 约 **10–11 分钟/seed**。这是旧特征、未记录 GPU 型号的 epoch 耗时，不含首次准备和最终测试，不能当成本次 T4 跑分。记录在 `reports/upstream_timing_reference.json`。

实测前可使用以下**宽松排期区间**：单张 T4/L4、256 维、100 epochs、单任务、单 seed、单个历史设置。这是估算，不是四模型 GPU 实测结果。

| 模型 | 暂定规划时间 |
|---|---:|
| MLP | 10–30 分钟 |
| GraphSAGE | 15–60 分钟 |
| GatedGCN | 30–120 分钟 |
| Graph Transformer | 30–120 分钟 |

四模型分类、none/all、三个 seed，共约 **8.5–33 GPU 小时**；再做回归约 **17–66 GPU 小时**。这些只用于留出排期余量，实际可能明显更快，应在运行 profile 后更新。

只做完整历史而不跑 baseline，分类 12 次，约上述预算一半，但无法在本次代码上完整归因历史增益。

Compute Units 不是 GPU 小时。用 Colab 界面当前显示的每小时消耗率乘以实测小时数估算，不存在本项目能保证的固定扣费数。

## 9. 实验解释

方向合并仅用于标签和历史身份；图消息传播保持方向。历史严格过去，允许后续测试月份使用前面测试月份已经发生的事故。预处理只在训练期拟合。

保留原目标月观测天气以保证 baseline 输入条件一致，不能宣称全部输入均在月初已知。严格部署预测需为所有对照同时调整天气可用性后重跑。

原 EDA 已查看 2018 年，与测试期重叠需披露。本地 smoke 未做测试集评估，正式性能以 Colab 完整实验为准。
