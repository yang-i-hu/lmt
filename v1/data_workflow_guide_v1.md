# 因子重加权数据工作流 完整指南
# Factor Reweighting Data Workflow — Complete Guide

> **版本**: v2.0  
> **目标读者**: 数据处理、模型训练、评估流程的开发人员

---

## 目录
1. [项目背景与目标](#1-项目背景与目标)
2. [数据源结构详解](#2-数据源结构详解)
3. [核心概念与术语](#3-核心概念与术语)
4. [数据处理规则（关键约束）](#4-数据处理规则关键约束)
5. [完整工作流程](#5-完整工作流程)
6. [代码实现指南](#6-代码实现指南)
7. [常见错误与避坑指南](#7-常见错误与避坑指南)
8. [文件输出规范](#8-文件输出规范)
9. [快速参考卡片](#9-快速参考卡片)

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

## 5. 完整工作流程

### 5.1 流程总览图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Step 1: 枚举所有快照文件夹                                           │
│  Step 2: 对每个快照，导出对齐的 Parquet 文件                           │
│  Step 3: 划分 IS / OOS 数据集                                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        模型训练阶段                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Step 4: 使用 IS 数据训练模型                                         │
│          - 对 key 0/1/2 分别训练                                     │
│          - 每个 key 独立的模型实例                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        模型预测阶段                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Step 5: 在 OOS 数据上生成预测                                        │
│          pred_0 = model_0.predict(oos_factors_0)                    │
│          pred_1 = model_1.predict(oos_factors_1)                    │
│          pred_2 = model_2.predict(oos_factors_2)                    │
│  Step 6: 融合预测                                                    │
│          pred_ensemble = mean(pred_0, pred_1, pred_2)               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        评估阶段                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Step 7: 使用 LMT API 评估 pred_ensemble                             │
│          - IC / ICIR                                                │
│          - Long / Short / LS 收益                                    │
│          - IR 类指标                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 详细步骤说明

#### Step 1: 枚举快照

```python
from pathlib import Path

base_dir = Path("1128_weight_factors/")
snapshots = sorted([
    d.name for d in base_dir.iterdir() 
    if d.is_dir() and d.name.isdigit()
])
# ['20181228', '20191231', '20201231', '20211231', ...]
```

#### Step 2: 导出对齐数据 (per snapshot)

对每个快照执行：
1. 读取 `weakFactors.h5` (分块)
2. 读取 `Label10.h5` 并过滤
3. 按 `(date, instrument)` 对齐
4. 保存 Parquet 文件

输出文件结构：
```
data/
├── 20211231/
│   ├── factors_0.parquet           # Key 0 原始因子
│   ├── factors_1.parquet           # Key 1 原始因子
│   ├── factors_2.parquet           # Key 2 原始因子
│   ├── factors_0_aligned.parquet   # Key 0 + labels
│   ├── factors_1_aligned.parquet   # Key 1 + labels
│   ├── factors_2_aligned.parquet   # Key 2 + labels
│   ├── factors_mean.parquet        # (可选) 副本均值
│   └── labels_aligned.parquet      # 对齐后的标签
```

#### Step 3: 划分 IS / OOS

```python
cutoff = 20211231

# IS: 用于训练
is_data = aligned_data[aligned_data.index.get_level_values('date') <= cutoff]

# OOS: 用于评估
oos_data = aligned_data[aligned_data.index.get_level_values('date') > cutoff]
```

#### Step 4-6: 训练 & 预测

```python
# 伪代码
for key in ['0', '1', '2']:
    # 加载训练数据
    train_data = load_is_data(snapshot, key)
    X_train = train_data.drop(['labelValue', 'endDate'], axis=1)
    y_train = train_data['labelValue']
    
    # 训练模型
    model = train_model(X_train, y_train)
    
    # OOS 预测
    oos_data = load_oos_data(snapshot, key)
    X_oos = oos_data.drop(['labelValue', 'endDate'], axis=1)
    predictions[key] = model.predict(X_oos)

# 融合
pred_ensemble = (predictions['0'] + predictions['1'] + predictions['2']) / 3
```

#### Step 7: LMT API 评估

```python
import lmt_data_api as api

# 准备数据格式
pred_ensem = pred_ensemble.copy()
pred_ensem.name = 'factor'
pred_ensem.index.names = ['date', 'code']

# 获取评估指标
group_ir, group_ls = api.get_group_return(
    args=pred_ensem,
    factor='factor',
    alpha='alpha1',
    label_period=10
)

ic_df = api.get_daily_ic(args=pred_ensem, factor='factor', label_period=10)
```

---

## 6. 代码实现指南

### 6.1 配置文件 (`config.yaml`)

```yaml
# 数据目录
data_dir: "data/"

# 因子 key 选择: '0', '1', '2'
factor_key: "0"

# 日期范围 (格式: YYYYMMDD)
start_date: 20180102
end_date: 20211230

# 股票池文件 (每行一个代码)
universe_file: "universe.txt"

# 是否只加载对齐数据
aligned_only: true

# 是否丢弃 NaN 标签
drop_na_labels: true
```

### 6.2 DataLoader 使用示例

```python
from dataloader import FactorDataLoader

# 方式 1: 从配置文件
loader = FactorDataLoader.from_config('config.yaml')
X, y = loader.load()

# 方式 2: 直接初始化
loader = FactorDataLoader(
    data_dir='data/',
    factor_key='0',
    start_date=20180102,
    end_date=20211230,
    universe_file='universe.txt'
)
X, y = loader.load()
```

### 6.3 完整训练脚本结构

```python
# train_model.py 伪代码结构

def main():
    # 1. 加载配置
    config = load_config('config.yaml')
    
    # 2. 对每个 factor key 训练
    models = {}
    for key in ['0', '1', '2']:
        print(f"Training model for key {key}...")
        
        loader = FactorDataLoader(
            data_dir=config['data_dir'],
            factor_key=key,
            start_date=config['train_start'],
            end_date=config['train_end']
        )
        X_train, y_train = loader.load()
        
        model = train(X_train, y_train)
        models[key] = model
    
    # 3. OOS 预测
    predictions = {}
    for key in ['0', '1', '2']:
        loader = FactorDataLoader(
            data_dir=config['data_dir'],
            factor_key=key,
            start_date=config['test_start'],
            end_date=config['test_end']
        )
        X_test, _ = loader.load()
        predictions[key] = models[key].predict(X_test)
    
    # 4. 融合 & 评估
    pred_ensemble = np.mean(list(predictions.values()), axis=0)
    evaluate(pred_ensemble)
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

### 8.1 目录结构 (新版多快照)

```
project/
├── raw_data/                           # 原始 H5 文件 (只读)
│   ├── 1128_weight_factors/
│   │   ├── 20181228/weakFactors.h5
│   │   ├── 20191231/weakFactors.h5
│   │   └── 20201231/weakFactors.h5
│   └── Label10.h5
│
├── data/                               # 处理后的 Parquet
│   ├── 20181228/                       # Snapshot 2018
│   │   ├── factors_0_is.parquet        # IS data (train)
│   │   ├── factors_0_oos.parquet       # OOS data (2019)
│   │   ├── factors_1_is.parquet
│   │   ├── factors_1_oos.parquet
│   │   ├── factors_2_is.parquet
│   │   ├── factors_2_oos.parquet
│   │   └── metadata.json
│   ├── 20191231/                       # Snapshot 2019 (OOS = 2020)
│   │   └── ...
│   ├── 20201231/                       # Snapshot 2020 (OOS = 2021)
│   │   └── ...
│   └── combined/                       # 合并数据 (推荐使用)
│       ├── train_0.parquet             # 完整 IS (from 20201231)
│       ├── train_1.parquet
│       ├── train_2.parquet
│       ├── test_oos_0.parquet          # 合并 OOS (2019+2020+2021)
│       ├── test_oos_1.parquet
│       ├── test_oos_2.parquet
│       └── metadata.json
│
├── outputs_dnn_full/                   # DNN 模型输出
│   ├── model_key0.pt
│   ├── model_key1.pt
│   ├── model_key2.pt
│   ├── pred_ensemble.parquet
│   └── training_results.json
│
└── outputs_elasticnet_full/            # ElasticNet 模型输出
    ├── model_key0.pkl
    ├── model_key1.pkl
    ├── model_key2.pkl
    ├── coefficients_key0.csv
    ├── pred_ensemble.parquet
    └── training_results.json
```

### 8.2 Parquet 文件规范

#### `factors_{key}_is.parquet` / `factors_{key}_oos.parquet`
| 字段 | 类型 | 说明 |
|------|------|------|
| index | MultiIndex(date, instrument) | 日期 + 证券代码 |
| 0-1127 | float32 | 1128 个因子值 |
| labelValue | float64 | 对应标签值 |
| endDate | int64 | 标签的 endDate |

#### `train_{key}.parquet` (combined/)
| 字段 | 类型 | 说明 |
|------|------|------|
| index | MultiIndex(date, instrument) | 日期 + 证券代码 |
| 0-1127 | float32 | 1128 个因子值 |
| labelValue | float64 | 标签值 |
| endDate | int64 | 标签的 endDate |

#### `test_oos_{key}.parquet` (combined/)
| 字段 | 类型 | 说明 |
|------|------|------|
| index | MultiIndex(date, instrument) | 日期 + 证券代码 |
| 0-1127 | float32 | 1128 个因子值 |
| labelValue | float64 | 标签值 |
| endDate | int64 | 标签的 endDate |
| source_snapshot | str | 来源快照 (用于滚动回测) |

---

## 9. 脚本使用指南

### 9.1 数据导出 (export_multi_snapshot.py)

```bash
# 导出所有快照数据
python export_multi_snapshot.py

# 指定快照
python export_multi_snapshot.py --snapshots 20181228 20191231 20201231

# 指定输出目录
python export_multi_snapshot.py --output-dir ./processed_data
```

输出:
- `data/{snapshot}/factors_{key}_is.parquet` — IS 训练数据
- `data/{snapshot}/factors_{key}_oos.parquet` — OOS 测试数据
- `data/combined/train_{key}.parquet` — 合并 IS (使用最新快照)
- `data/combined/test_oos_{key}.parquet` — 合并 OOS (所有年份)

### 9.2 DNN 训练 (train_dnn_full.py)

```bash
# 基本用法
python train_dnn_full.py --config config_dnn_full_new.yaml

# 使用 GPU
python train_dnn_full.py --config config_dnn_full_new.yaml --device cuda

# 指定快照
python train_dnn_full.py --config config_dnn_full_new.yaml --snapshot 20201231
```

自动完成:
1. 加载 IS 数据 (train_{key}.parquet)
2. 对 key 0/1/2 分别训练 DNN
3. 加载 OOS 数据 (test_oos_{key}.parquet)
4. 生成 ensemble 预测
5. 运行 LMT API 评估

### 9.3 ElasticNet 训练 (train_elasticnet_full.py)

```bash
# 基本用法
python train_elasticnet_full.py --config config_elasticnet_full_new.yaml

# 覆盖参数
python train_elasticnet_full.py --config config_elasticnet_full_new.yaml --alpha 0.01 --l1_ratio 0.7
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
```

### 10.2 快照选择速查表

| 如果你要... | 使用快照 | OOS 评估区间 |
|-------------|----------|--------------|
| 评估 2019 年表现 | `20181228/` | 2019-01-01 ~ 2019-12-31 |
| 评估 2020 年表现 | `20191231/` | 2020-01-01 ~ 2020-12-31 |
| 评估 2021 年表现 | `20201231/` | 2021-01-01 ~ 2021-12-31 |

### 10.3 完整工作流命令

```bash
# Step 1: 导出数据
python export_multi_snapshot.py

# Step 2: 训练 ElasticNet (baseline)
python train_elasticnet_full.py --config config_elasticnet_full_new.yaml

# Step 3: 训练 DNN
python train_dnn_full.py --config config_dnn_full_new.yaml --device cuda

# Step 4: 比较结果
# 查看 outputs_*/training_results.json 中的 LMT API 指标
```

### 10.4 代码片段速查

```python
# 加载合并后的训练数据
import pandas as pd

train_df = pd.read_parquet('data/combined/train_0.parquet')
X_train = train_df.drop(['labelValue', 'endDate'], axis=1)
y_train = train_df['labelValue']

# 加载合并后的 OOS 测试数据
test_df = pd.read_parquet('data/combined/test_oos_0.parquet')
X_test = test_df.drop(['labelValue', 'endDate', 'source_snapshot'], axis=1)
y_test = test_df['labelValue']

# 训练/测试划分 (已在导出时完成)
# IS: data/combined/train_*.parquet
# OOS: data/combined/test_oos_*.parquet

# 融合预测
pred_ensemble = (pred_0 + pred_1 + pred_2) / 3

# LMT API 格式
pred_esem = pred_ensemble.copy()
pred_esem.name = 'factor'         git 
pred_esem.index = pred_esem.index.rename(['date', 'code'])
```

---

**如有疑问，请先查阅本文档的"常见错误与避坑指南"章节。**
