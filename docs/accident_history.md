# 已实现的数据接入

## 数据路径与输入

默认数据根目录为项目内 data，历史目录为 data/accident_history/outputs。移动项目不需修改源码。

| 文件 | 用途 |
|---|---|
| data/DE/adj_matrix.pt | 图结构、边身份 |
| data/DE/Nodes/node_features_YYYY_M.csv | 165 个月的天气、坐标 |
| data/DE/Edges/edge_features.pt | 道路属性、长度 |
| data/DE/Edges/edge_features_traffic_YYYY.pt | 2009–2019 AADT，之后沿用较早报告 |
| data/DE/accidents_monthly.csv | 按无序端点合并的监督标签 |
| data/accident_history/outputs/monthly_features/*.npz | 从分离的 y 读取月度计数，重建物理道路历史 |
| 同 outputs 下 edge_mapping.csv、manifest.json、monthly_coverage_audit.csv | 边对齐、源文件哈希、可用月份校验 |

原历史 X 保留，但其自身历史按方向定义，不能直接拿一个代表方向去预测合并后的标签。独立的归档加载器包装原有 `RoadAccidentDataset`，从月度档案中的 y 重建历史，并逐月检查合并计数与 CSV 标签一致；基础数据加载器保持不变。目标月 y 仍只作监督，只有早于目标月的计数能进入特征窗口。

节点输入：坐标 2 + 入出度编码 14 + 中心性 1 + 天气 6 + 缺失标记 6 = 29 维；Node2Vec 可加 128 维，共 157 维。

原道路输入：属性标记 24 + 长度 1 + AADT 1 + 缺失标记 1 = 27 维。

| 模式 | 新增输入 | 总道路维度 |
|---|---|---:|
| none | 无 | 27 |
| own | 自身六项、覆盖一项、自身六项缺失标记 | 40 |
| all | 历史十项、事故历史八项缺失标记 | 45 |

仓库原有的 `src/history` 分组接口也保留，以兼容此前的实验 notebook。它通过
`--history-groups` 和 `--history-min-months` 启用；事故历史实验 notebook 使用
`--history-mode none/own/all`。同一次运行不要同时启用两套历史特征接口。

十项历史：lag1、mean3、mean6、mean12、lag12、12 月非零比例、邻居 lag1 均值、邻居 mean3、邻居数、可用历史月数。

## 方向与道路身份

道路键为 min(u,v) * num_nodes + max(u,v)。两个方向共享道路键，但 edge_index 保持原始有向图。

先计算每月 physical_count = count(u,v) + count(v,u)。仅一个方向的道路只加一次，自环只加一次。标签和历史都基于这个物理道路序列。

例如两个方向同月分别发生 2、3 起事故，合并计数是 5，下月 lag1 为 5；该月只算一个非零月份，不能把两个方向的非零比例相加。

邻居为共享任一端点的不同无序连接，排除目标连接及其反向；孤立道路在历史可用时均值为零，缺历史时为 NaN。

计算后的物理道路历史映射回其所有有向边，追加到 edge_attr。GNN 消息传播仍有方向，预测头对每条物理道路只预测一次。原始方向性道路属性保留原实现。

## 时间与缺失

所有事故窗口严格使用 T 以前的计数。完整滚动窗口不足或有不可用月份时保持 NaN。

2009–2012 的实际训练月份用于拟合均值和标准差；忽略 NaN。缺失值填训练均值，标准化后为零，并附加缺失标记。验证/测试不参与拟合。保留 2009 年，训练/验证/测试保持 48/60/57 月。

none 模式也做相同的月份和标签检查，保证 ablation 覆盖一致。

物理历史缓存位于 data/DE/cache/physical_history，以 NPY mmap 读取。缓存签名包含构造源码、原数据和全部档案哈希，变化后重建。训练期缩放独立拟合，不写回原 NPZ。

## 修改的训练行为

保留架构、Adam 0.001、100 epochs、宽 256、两层、dropout 0、1:1 采样和三个种子。预测头自动按输入宽度创建。历史同时进入边消息计算和道路预测头。

负样本使用独立于模型初始化的 NumPy RNG，保证匹配实验的月份排列和负样本一致；均匀无放回的抽样规则不变。

每个 epoch 原子保存模型、优化器、最佳状态、训练记录和 NumPy/PyTorch/CUDA 随机状态。--resume 从上个完整 epoch 恢复；核对源码、原始数据、嵌入和历史指纹，拒绝错配置续训。

分类按验证 AUROC、回归按验证 MAE 选最佳权重；正式测试仅评估最佳权重。all/balanced 两种评估保留，主表使用 all。回归评估仍把负预测截为零。

--smoke-test 只用两个训练月份、两个验证月份训练两轮，不评估测试集。src.compare 排除 smoke 和未完成实验，并仅匹配相同代码/数据/参数/种子的 baseline。src.simple_baselines 提供全零、lag1、mean3、mean12 参照。

## 注意事项

原目标月观测天气、静态路网、事故报告及时可用等假设保留。原 EDA 已查看 2018 年需披露。原 SOURCE_MANIFEST 记录的是复制时哈希；活动代码已经按本文件说明修改。每次正式训练会另记实际源码指纹。
