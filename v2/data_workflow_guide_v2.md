# 因子重加权数据工作流 完整指南
# Factor Reweighting Data Workflow — Complete Guide

> **版本**: v2.1 (Rolling Per-Snapshot Pipeline)  
> **目标读者**: 数据处理、模型训练、评估流程的开发人员

---

## 目录
1. [项目背景与目标](#1-项目背景与目标)
2. [数据源结构详解](#2-数据源结构详解)
3. [核心概念与术语](#3-核心概念与术语)
4. [数据处理规则（关键约束）](#4-数据处理规则关键约束)
5. [滚动逐快照工作流程](#5-滚动逐快照工作流程)
6. [代码实现指南](#6-代码实现指南)
7. [常见错误与避坑指南](#7-常见错误与避坑指南)
8. [文件输出规范](#8-文件输出规范)
9. [脚本使用指南](#9-脚本使用指南)
10. [快速参考卡片](#10-快速参考卡片)

---

## 1. 项目背景与目标

### 1.1 核心任务
使用模型对 **1128个弱因子进行加权**，生成最终组合信号 `pred_ensemble`，用于量化回测与评估。

### 1.2 最终交付物
| 交付物 | 说明 |
|--------|------|
| 训练好的模型 | 对每个 factor key 分别训练 |
| `pred_ensemble` 信号 | 三路预测的平均值 |
| 样本内/外评估结果 | IC, ICIR, Long/Short 收益等 |
| 与 baseline 对比报告 | 神经网络 vs 树模型 |

### 1.3 绩效基准
神经网络模型必须 **优于树模型 baseline**，重点关注：
- **IC / ICIR** — 信息系数及其稳定性
- **Long / Short / LS** — 多空收益
- **IR 类指标** — 风险调整后收益

---

## 2. 数据源结构详解

### 2.1 因子数据 (`weakFactors.h5`)

#### 目录结构
```
1128_weight_factors/
├── 20181228/
│   └── weakFactors.h5
├── 20191231/
│   └── weakFactors.h5
└── 20201231/
    └── weakFactors.h5

```

#### 文件夹日期含义 ⭐
| 文件夹名称 | 含义 |
|-----------|------|
| `YYYYMMDD` | **训练截止日（Cutoff Date）** |
| 该日期之前 | 样本内 (In-Sample, IS) — 用于训练该模型 |
| 该日期之后 | 样本外 (Out-of-Sample, OOS) — 用于评估 |

**示例**：`20201231/weakFactors.h5`
- 使用 `<= 2020-12-31` 的数据训练生成
- `>= 2021-01-01` 的数据是 OOS（测试集）

#### H5 内部结构
```
weakFactors.h5
├── /0   ← 副本 0 (Replica 0)
├── /1   ← 副本 1 (Replica 1)  
└── /2   ← 副本 2 (Replica 2)
```

**关键点**：
- 3 个 key 是 **同一因子定义的不同副本**（不同随机种子/batch）
- ⚠️ **三个 key 之间无对应关系，必须分别建模**
- 最终信号 = `mean(pred_0, pred_1, pred_2)`

#### 每个 Key 的数据结构 (pandas fixed-format)
```
/0/
├── axis0            (1128,)         int64    → 因子 ID（列标签）
├── block0_values    (3203935, 1128) float32  → 因子值矩阵 ⭐
├── axis1_level0     (933,)          int64    → 日期唯一值
├── axis1_level1     (3646,)         bytes    → 证券代码唯一值
├── axis1_label0     (3203935,)      int16    → 日期索引码
└── axis1_label1     (3203935,)      int16    → 证券索引码
```

| 属性 | 数值 |
|------|------|
| 行数 | ~320 万 |
| 列数 | 1,128 |
| 数据类型 | float32 |
| 单 key 大小 | ~12.7 GB |
| 总文件大小 | ~38 GB |

### 2.2 标签数据 (`Label10.h5`)

#### 文件格式
- **格式**: PyTables TABLE format（支持切片查询）
- **行数**: ~1250 万
- **存储路径**: `/Data/table`

#### 关键字段
| 字段 | 类型 | 说明 |
|------|------|------|
| `labelDate` | int | 标签日期，**必须与因子日期对齐** |
| `code` | str | 证券代码 |
| `labelValue` | float | 标签值（预测目标） |
| `endDate` | int | 标签计算的结束日期，**用于过滤未来数据** |

---

## 3. 核心概念与术语

### 3.1 统一术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 快照/模型截止 | Snapshot / Cutoff | 文件夹日期 `YYYYMMDD`，表示模型训练的数据截止点 |
| 样本内 (IS) | In-Sample | 日期 `<= Cutoff` 的数据 |
| 样本外 (OOS) | Out-of-Sample | 日期 `> Cutoff` 的数据 |
| 副本 | Replica | H5 中的 key 0/1/2，同一因子的不同随机实现 |
| 副本均值因子 | Replica-Mean | 三个副本的平均值（默认使用） |
| 对齐 | Alignment | 因子日期与 labelDate 一一对应 |
| 未来过滤 | Future Filter | endDate > cutoff 的数据不可用 |

### 3.2 滚动快照评估图示

```
Timeline:
─────────────────────────────────────────────────────────►
   D1           D2           D3           D4       ...
   │            │            │            │
   ▼            ▼            ▼            ▼

Snapshot 20181228:  [====IS====]|---OOS--->
Snapshot 20191231:  [=======IS=======]|---OOS--->
Snapshot 20201231:  [===========IS===========]|---OOS--->
Snapshot 20211231:  [===============IS===============]|---OOS--->

对于每个 snapshot Di:
  - IS 区间: dates <= Di  (用于训练该模型)
  - OOS 区间: dates > Di  (用于评估)
```

### 3.3 滚动回测逻辑

为模拟真实部署场景（避免前视偏差）：
```
使用 Snapshot i 的模型，仅评估区间 (Di, D_{i+1}]

示例:
- 20181228 模型 → 评估 2019年全年
- 20191231 模型 → 评估 2020年全年
- 20201231 模型 → 评估 2021年全年
- 20211231 模型 → 评估 2022年全年
```

---

## 4. 数据处理规则（关键约束）

### 4.1 🚨 绝对不可违反的规则

#### 规则 1: Label 日期对齐
```
labelDate == factor_date

❌ 错误: 使用 labelDate=20220105 的标签训练 factor_date=20220104 的因子
✅ 正确: labelDate 必须严格等于 factor_date
```

#### 规则 2: 未来数据过滤
```
endDate <= current_snapshot_cutoff

❌ 错误: 使用 endDate=20220115 的标签，在 cutoff=20211231 的快照中训练
✅ 正确: 只使用 endDate <= 20211231 的数据
```

#### 规则 3: OOS 评估严格分离
```
评估必须在 OOS 数据上进行

❌ 错误: 使用训练数据评估模型效果
✅ 正确: dates > cutoff 的数据用于评估
```

#### 规则 4: 三路独立建模
```
pred_ensemble = mean(pred_key0, pred_key1, pred_key2)

❌ 错误: 将 key 0/1/2 的数据混合后统一训练
✅ 正确: 分别对每个 key 训练独立模型，最后取平均
```

#### 规则 5: 禁止跨快照合并数据 🆕
```
不同快照文件夹的数据不能合并为统一的训练集或测试集。
每个快照文件夹必须独立完成训练和评估。

❌ 错误: 将多个快照的 IS 数据合并后统一训练一个模型
❌ 错误: 将所有快照的 OOS 数据合并后统一评估（跳过逐快照评估）
✅ 正确: 逐快照独立处理 — 每个快照独立训练、预测、评估
✅ 正确: 最终聚合时，按时间顺序拼接各快照的 OOS 预测结果
```

### 4.2 读取 weakFactors.h5 的正确方式

⚠️ **严禁使用 pandas 直接读取**

```python
# ❌ 绝对不要这样做 — 会导致 MultiIndex 重建错误
df = pd.read_hdf("weakFactors.h5", key="/0", start=0, stop=1000)

# ❌ 绝对不要这样做 — 内存不足
df = pd.read_hdf("weakFactors.h5", key="/0")
```

```python
# ✅ 正确方式: 使用 h5py 手动重建索引
import h5py
import pandas as pd
import numpy as np

def load_factor_slice(h5_file, key="0", start=0, stop=1000):
    with h5py.File(h5_file, "r") as f:
        grp = f[key]
        
        # 1. 读取因子值切片
        values = grp["block0_values"][start:stop, :]
        columns = grp["axis0"][:]
        
        # 2. 读取索引组件
        level0 = grp["axis1_level0"][:]     # 日期唯一值
        level1 = grp["axis1_level1"][:]     # 证券代码唯一值
        label0 = grp["axis1_label0"][start:stop]
        label1 = grp["axis1_label1"][start:stop]
        
        # 3. 解码 bytes
        if level1.dtype.kind == 'S':
            level1 = np.array([x.decode('utf-8') for x in level1])
    
    # 4. 重建 MultiIndex
    dates = level0[label0]
    instruments = level1[label1]
    index = pd.MultiIndex.from_arrays(
        [dates, instruments], 
        names=["date", "instrument"]
    )
    
    return pd.DataFrame(values, index=index, columns=columns)
```

### 4.3 读取 Label10.h5 的正确方式

✅ **Label 文件使用 TABLE 格式，支持切片**

```python
# ✅ 正确: 分块读取并过滤
labels = pd.read_hdf("Label10.h5", key="Data", start=0, stop=500000)
labels = labels.reset_index()

# 应用过滤条件
labels = labels[
    (labels['labelDate'] >= start_date) &
    (labels['labelDate'] <= end_date) &
    (labels['endDate'] <= future_cutoff)
]
```

---

## 5. 滚动逐快照工作流程

### 5.1 ⚠️ 关键设计原则：禁止跨快照合并

> **不同快照文件夹的数据不能合并。** 每个快照文件夹包含独立的因子数据，
> 由不同的训练截止日生成。跨快照合并会破坏数据完整性和时间一致性。
>
> 正确的流程是：**逐快照独立处理** — 每个快照文件夹独立完成数据加载、
> 模型训练、OOS 预测和评估，最终按时间顺序拼接所有 OOS 结果。

### 5.2 流程总览图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    数据准备阶段 (一次性)                               │
├─────────────────────────────────────────────────────────────────────┤
│  Step 1: 枚举所有快照文件夹                                           │
│  Step 2: 对每个快照，导出 IS + OOS 的对齐 Parquet 文件                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              滚动训练与评估 (逐快照循环)                               │
├─────────────────────────────────────────────────────────────────────┤
│  FOR each snapshot Di in [D1, D2, D3, ...]:                        │
│                                                                     │
│    Step 3: 加载该快照的 IS 数据                                      │
│    Step 4: 对 key 0/1/2 分别训练独立模型                             │
│    Step 5: 加载该快照的 OOS 数据                                     │
│    Step 6: 对 key 0/1/2 分别预测                                    │
│    Step 7: 融合 pred_ensemble = mean(pred_0, pred_1, pred_2)       │
│    Step 8: 使用 LMT API 评估该快照的 OOS 结果                        │
│    Step 9: 保存模型、预测、指标到 outputs/{Di}/                      │
│                                                                     │
│  END FOR                                                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    滚动聚合阶段                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Step 10: 按时间顺序拼接所有快照的 OOS 预测                          │
│  Step 11: 在完整 OOS 时间线上运行 LMT API 聚合评估                   │
│  Step 12: 生成最终滚动回测报告                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 详细步骤说明

#### Step 1: 枚举快照

```python
from pathlib import Path

base_dir = Path("1128_weight_factors/")
snapshots = sorted([
    d.name for d in base_dir.iterdir()
    if d.is_dir() and d.name.isdigit()
])
# ['20181228', '20191231', '20201231', ...]
```

#### Step 2: 导出对齐数据 (per snapshot, 一次性)

对每个快照独立执行：
1. 读取该快照的 `weakFactors.h5` (分块, 使用 h5py)
2. 读取 `Label10.h5` 并按 cutoff 过滤 endDate
3. 按 `(date, instrument)` 对齐
4. 划分 IS / OOS
5. 保存到该快照的 Parquet 目录

每个快照的输出文件结构：
```
data/{snapshot}/
├── factors_0_is.parquet        # Key 0 IS (训练)
├── factors_0_oos.parquet       # Key 0 OOS (评估)
├── factors_1_is.parquet
├── factors_1_oos.parquet
├── factors_2_is.parquet
├── factors_2_oos.parquet
└── metadata.json               # 快照元数据
```

#### Step 3-9: 逐快照训练与评估循环

```python
# 核心滚动训练循环 (伪代码)
all_oos_predictions = []

for snapshot in snapshots:
    cutoff = int(snapshot)
    print(f"=== Processing snapshot {snapshot} ===")

    # 对每个 key 独立训练和预测
    predictions = {}
    for key in ['0', '1', '2']:
        # 加载该快照的 IS 数据
        is_df = pd.read_parquet(f'data/{snapshot}/factors_{key}_is.parquet')
        X_train = is_df.drop(['labelValue', 'endDate'], axis=1)
        y_train = is_df['labelValue']

        # 训练模型
        model = train_model(X_train, y_train)

        # 加载该快照的 OOS 数据
        oos_df = pd.read_parquet(f'data/{snapshot}/factors_{key}_oos.parquet')
        X_oos = oos_df.drop(['labelValue', 'endDate'], axis=1)

        # 预测
        predictions[key] = model.predict(X_oos)

        # 保存模型到 outputs/{snapshot}/
        save_model(model, f'outputs/{snapshot}/model_key{key}')

    # 融合预测
    pred_ensemble = mean(predictions['0'], predictions['1'], predictions['2'])

    # LMT API 评估该快照的 OOS
    lmt_results = evaluate_with_lmt_api(pred_ensemble)

    # 保存该快照的结果
    save_snapshot_report(snapshot, pred_ensemble, lmt_results)

    # 收集用于最终聚合
    all_oos_predictions.append(pred_ensemble)
```

#### Step 10-12: 滚动聚合

```python
# 拼接所有快照的 OOS 预测（按时间顺序）
combined_oos = pd.concat(all_oos_predictions).sort_index()

# 在完整 OOS 时间线上运行聚合评估
final_lmt_results = evaluate_with_lmt_api(combined_oos)

# 生成滚动回测报告
save_rolling_report(final_lmt_results, per_snapshot_results)
```

#### LMT API 评估

```python
from lmt_data_api.api import DataApi
api = DataApi()

# 准备数据格式
pred_esem = pred_ensemble.copy()
pred_esem.name = 'factor'
pred_esem.index.names = ['date', 'code']

# 获取评估指标
group_re, group_ir, group_hs = api.da_eva_group_return(
    pred_esem, "factor", alpha=1, label_period=10
)

ic_df = api.da_eva_ic(pred_esem, "factor", 10)
```

---

## 6. 代码实现指南

### 6.1 配置文件 (`config_dnn.yaml` / `config_elasticnet.yaml`)

```yaml
# 数据目录 (包含各快照子文件夹)
data_dir: "data/"

# 快照列表 (按时间顺序)
snapshots:
  - "20181228"
  - "20191231"
  - "20201231"

# 每个快照的 OOS 结束日期
snapshot_oos_end:
  "20181228": 20191231
  "20191231": 20201231
  "20201231": 20211231

# 因子 key 列表
factor_keys: ["0", "1", "2"]

# 模型参数 (DNN 示例)
model:
  hidden_sizes: [512, 256, 128, 64]
  dropout: 0.3
  activation: "leaky_relu"
  batch_norm: true

# 训练参数
training:
  epochs: 100
  batch_size: 4096
  learning_rate: 0.001
  early_stopping_patience: 15
  val_ratio: 0.15
  random_seed: 42

# LMT API 评估
evaluation:
  label_period: 10
  alpha: 1

# 输出目录 (按快照自动创建子目录)
output:
  output_dir: "outputs_dnn"
```

### 6.2 数据加载 (per-snapshot)

```python
import pandas as pd
from pathlib import Path

def load_snapshot_data(data_dir, snapshot, key, split):
    """
    加载指定快照、key、split 的数据。
    
    Args:
        data_dir: 'data/'
        snapshot: '20201231'
        key: '0', '1', '2'
        split: 'is' or 'oos'
    """
    file_path = Path(data_dir) / snapshot / f"factors_{key}_{split}.parquet"
    df = pd.read_parquet(file_path)
    
    X = df.drop(['labelValue', 'endDate'], axis=1)
    y = df['labelValue']
    
    # 去除 NaN 标签
    valid = y.notna()
    return X[valid], y[valid]

# 示例: 加载 20201231 快照的 Key 0 IS 数据
X_train, y_train = load_snapshot_data('data/', '20201231', '0', 'is')

# 示例: 加载同一快照的 Key 0 OOS 数据
X_test, y_test = load_snapshot_data('data/', '20201231', '0', 'oos')
```

### 6.3 完整训练脚本结构 (滚动逐快照)

```python
# train_model.py — 滚动逐快照训练结构

def main():
    config = load_config('config.yaml')
    snapshots = config['snapshots']
    all_oos_predictions = []

    for snapshot in snapshots:
        print(f"\n=== Snapshot {snapshot} ===")
        predictions = {}

        # 对每个 key 独立训练 & 预测
        for key in ['0', '1', '2']:
            # 加载该快照的 IS 数据
            X_train, y_train = load_snapshot_data(
                config['data_dir'], snapshot, key, 'is'
            )
            model = train(X_train, y_train)

            # 加载该快照的 OOS 数据
            X_oos, y_oos = load_snapshot_data(
                config['data_dir'], snapshot, key, 'oos'
            )
            predictions[key] = model.predict(X_oos)

            # 保存模型
            save_model(model, f'outputs/{snapshot}/model_key{key}')

        # 融合 & 评估 (该快照)
        pred_ensemble = mean(predictions['0'], predictions['1'], predictions['2'])
        lmt_results = evaluate_with_lmt_api(pred_ensemble)
        save_snapshot_report(snapshot, pred_ensemble, lmt_results)
        all_oos_predictions.append(pred_ensemble)

    # 滚动聚合
    combined_oos = pd.concat(all_oos_predictions).sort_index()
    final_results = evaluate_with_lmt_api(combined_oos)
    save_rolling_report(final_results)
```

---

## 7. 常见错误与避坑指南

### 7.1 数据读取错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ValueError: code max >= length of level` | 使用 `pd.read_hdf` 切片读取 fixed-format H5 | 改用 `h5py` 手动重建索引 |
| `MemoryError` | 一次性加载完整 DataFrame | 分块读取，使用 `CHUNK_SIZE` |
| Index 不匹配 | 因子和标签的日期/代码格式不一致 | 统一索引格式为 `(date, instrument)` |

### 7.2 逻辑错误

| 错误 | 后果 | 检查方法 |
|------|------|----------|
| 使用 `endDate > cutoff` 的标签 | 前视偏差 (lookahead bias) | 打印 `labels['endDate'].max()` 确认 |
| IS/OOS 数据泄露 | 高估模型性能 | 检查日期范围是否严格分离 |
| 混合训练 key 0/1/2 | 破坏实验设计 | 确保每个 key 独立训练 |
| 忘记取平均 | 评估不完整 | `pred_ensemble = mean(pred_0, pred_1, pred_2)` |
| 跨快照合并数据 🆕 | 数据完整性破坏 | 确保每个快照独立处理，不合并 IS 或 OOS |

### 7.3 性能优化建议

```python
# ✅ 使用 Parquet 格式（比 HDF5 快 10x+）
df.to_parquet('data.parquet', engine='pyarrow')

# ✅ 分块处理大文件
CHUNK_SIZE = 100000
for start in range(0, n_rows, CHUNK_SIZE):
    chunk = load_factor_slice(h5_file, key, start, start + CHUNK_SIZE)
    process(chunk)

# ✅ 使用 PyArrow 内存映射
import pyarrow.parquet as pq
table = pq.read_table('data.parquet', memory_map=True)
```

---

## 8. 文件输出规范

### 8.1 目录结构 (滚动逐快照)

```
project/
├── raw_data/                           # 原始 H5 文件 (只读)
│   ├── 1128_weight_factors/
│   │   ├── 20181228/weakFactors.h5
│   │   ├── 20191231/weakFactors.h5
│   │   └── 20201231/weakFactors.h5
│   └── Label10.h5
│
├── data/                               # 处理后的 Parquet (每个快照独立)
│   ├── 20181228/                       # Snapshot 2018
│   │   ├── factors_0_is.parquet        # Key 0 IS data (训练)
│   │   ├── factors_0_oos.parquet       # Key 0 OOS data (2019)
│   │   ├── factors_1_is.parquet
│   │   ├── factors_1_oos.parquet
│   │   ├── factors_2_is.parquet
│   │   ├── factors_2_oos.parquet
│   │   └── metadata.json
│   ├── 20191231/                       # Snapshot 2019 (OOS = 2020)
│   │   └── ...
│   ├── 20201231/                       # Snapshot 2020 (OOS = 2021)
│   │   └── ...
│   └── all_snapshots_metadata.json     # 全局元数据
│
├── outputs_dnn/                        # DNN 模型输出 (按快照)
│   ├── 20181228/                       # Snapshot 2018 结果
│   │   ├── model_key0.pt
│   │   ├── model_key1.pt
│   │   ├── model_key2.pt
│   │   ├── scaler_key0.pkl
│   │   ├── scaler_key1.pkl
│   │   ├── scaler_key2.pkl
│   │   ├── pred_ensemble.parquet       # 该快照的 OOS 预测
│   │   └── snapshot_report.json        # 该快照的评估报告
│   ├── 20191231/
│   │   └── ...
│   ├── 20201231/
│   │   └── ...
│   ├── pred_ensemble_all.parquet       # 所有快照 OOS 拼接
│   ├── rolling_report.json             # 滚动回测总报告
│   └── training.log
│
└── outputs_elasticnet/                 # ElasticNet 模型输出 (按快照)
    ├── 20181228/
    │   ├── model_key0.pkl
    │   ├── scaler_key0.pkl
    │   ├── coefficients_key0.csv
    │   ├── pred_ensemble.parquet
    │   └── snapshot_report.json
    ├── 20191231/
    │   └── ...
    ├── 20201231/
    │   └── ...
    ├── pred_ensemble_all.parquet
    ├── rolling_report.json
    └── training.log
```

> ⚠️ **不再有 `combined/` 目录** — 数据不跨快照合并。
> 每个快照的训练和评估完全独立。

### 8.2 Parquet 文件规范

#### `factors_{key}_is.parquet` / `factors_{key}_oos.parquet`
| 字段 | 类型 | 说明 |
|------|------|------|
| index | MultiIndex(date, instrument) | 日期 + 证券代码 |
| 0-1127 | float32 | 1128 个因子值 |
| labelValue | float64 | 对应标签值 |
| endDate | int64 | 标签的 endDate |

#### `pred_ensemble.parquet` (per-snapshot)
| 字段 | 类型 | 说明 |
|------|------|------|
| index | MultiIndex(date, instrument) | 日期 + 证券代码 |
| prediction | float64 | 三个 key 预测的均值 |

#### `pred_ensemble_all.parquet` (aggregated)
| 字段 | 类型 | 说明 |
|------|------|------|
| index | MultiIndex(date, instrument) | 日期 + 证券代码 |
| prediction | float64 | 所有快照 OOS 预测拼接（按时间排序） |

### 8.3 报告文件规范

#### `snapshot_report.json` (per-snapshot)
```json
{
  "snapshot": "20201231",
  "cutoff_date": 20201231,
  "oos_end_date": 20211231,
  "oos_date_range": [20210104, 20211230],
  "metrics_by_key": {
    "0": {"ic": 0.05, "r2": 0.001, "rmse": 0.03, ...},
    "1": {"ic": 0.04, ...},
    "2": {"ic": 0.05, ...}
  },
  "ensemble_stats": {"n_samples": 500000, "n_dates": 243},
  "lmt_api_results": {"stats_all": {...}},
  "timestamp": "2026-03-07T..."
}
```

#### `rolling_report.json` (aggregated)
```json
{
  "pipeline": "DNN Rolling Per-Snapshot (v2)",
  "snapshots_processed": ["20181228", "20191231", "20201231"],
  "per_snapshot_results": [...],
  "aggregate": {
    "n_samples": 1500000,
    "n_dates": 729,
    "date_range": [20190102, 20211230],
    "lmt_api_results": {"stats_all": {...}}
  }
}
```

---

## 9. 脚本使用指南

### 9.1 数据导出 (`export_snapshot_data.py`)

```bash
# 导出所有快照数据 (每个快照独立)
python export_snapshot_data.py

# 指定快照
python export_snapshot_data.py --snapshots 20181228 20191231 20201231

# 指定输出目录
python export_snapshot_data.py --output-dir ./processed_data
```

输出 (每个快照独立):
- `data/{snapshot}/factors_{key}_is.parquet` — 该快照的 IS 训练数据
- `data/{snapshot}/factors_{key}_oos.parquet` — 该快照的 OOS 测试数据
- `data/{snapshot}/metadata.json` — 该快照的元数据

### 9.2 DNN 训练 (`train_dnn.py`)

```bash
# 基本用法 — 自动遍历所有快照
python train_dnn.py --config config_dnn.yaml

# 使用 GPU
python train_dnn.py --config config_dnn.yaml --device cuda

# 只处理特定快照
python train_dnn.py --config config_dnn.yaml --snapshots 20201231
```

自动完成 (对每个快照):
1. 加载 IS 数据 (`data/{snapshot}/factors_{key}_is.parquet`)
2. 对 key 0/1/2 分别训练 DNN
3. 加载 OOS 数据 (`data/{snapshot}/factors_{key}_oos.parquet`)
4. 生成 ensemble 预测并保存
5. 运行 LMT API 评估并生成快照报告
6. 所有快照完成后，拼接 OOS 预测并生成滚动总报告

### 9.3 ElasticNet 训练 (`train_elasticnet.py`)

```bash
# 基本用法
python train_elasticnet.py --config config_elasticnet.yaml

# 覆盖参数
python train_elasticnet.py --config config_elasticnet.yaml --alpha 0.01 --l1_ratio 0.7

# 只处理特定快照
python train_elasticnet.py --config config_elasticnet.yaml --snapshots 20181228 20191231
```

---

## 10. 快速参考卡片

### 10.1 一句话规则

```
1. 文件夹日期 = 训练截止日 (Cutoff)
2. IS = dates <= cutoff, OOS = dates > cutoff
3. labelDate 必须等于 factor date
4. endDate 必须 <= cutoff (未来过滤)
5. Key 0/1/2 分别建模，最后取平均
6. 用 h5py 读 weakFactors.h5，不要用 pd.read_hdf
7. 评估只用 OOS 数据
8. 🆕 不同快照的数据不能合并 — 逐快照独立处理
```

### 10.2 快照选择速查表

| 如果你要... | 使用快照 | OOS 评估区间 |
|-------------|----------|--------------|
| 评估 2019 年表现 | `20181228/` | 2019-01-01 ~ 2019-12-31 |
| 评估 2020 年表现 | `20191231/` | 2020-01-01 ~ 2020-12-31 |
| 评估 2021 年表现 | `20201231/` | 2021-01-01 ~ 2021-12-31 |
| 完整滚动回测 | 所有快照 | 2019-01-01 ~ 2021-12-31 |

### 10.3 完整工作流命令

```bash
# Step 1: 导出数据 (每个快照独立)
python export_snapshot_data.py

# Step 2: 训练 ElasticNet baseline (滚动逐快照)
python train_elasticnet.py --config config_elasticnet.yaml

# Step 3: 训练 DNN (滚动逐快照)
python train_dnn.py --config config_dnn.yaml --device cuda

# Step 4: 比较结果
# 查看 outputs_*/rolling_report.json 中的聚合 LMT API 指标
# 查看 outputs_*/{snapshot}/snapshot_report.json 中的逐快照指标
```

### 10.4 代码片段速查

```python
import pandas as pd

# ✅ 加载特定快照的训练数据 (每个快照独立)
is_df = pd.read_parquet('data/20201231/factors_0_is.parquet')
X_train = is_df.drop(['labelValue', 'endDate'], axis=1)
y_train = is_df['labelValue']

# ✅ 加载同一快照的 OOS 数据
oos_df = pd.read_parquet('data/20201231/factors_0_oos.parquet')
X_test = oos_df.drop(['labelValue', 'endDate'], axis=1)
y_test = oos_df['labelValue']

# ❌ 不要跨快照合并数据
# train_all = pd.concat([...])  # 禁止！

# ✅ 融合预测
pred_ensemble = (pred_0 + pred_1 + pred_2) / 3

# ✅ LMT API 格式
pred_esem = pred_ensemble.copy()
pred_esem.name = 'factor'
pred_esem.index = pred_esem.index.rename(['date', 'code'])
```

---

**如有疑问，请先查阅本文档的"常见错误与避坑指南"章节。**
