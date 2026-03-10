# 因子加权模型需求文档

## 一、目标

使用模型对 **1000+ 个因子进行加权**，生成 **最终组合信号（pred_ensemble）**，用于后续回测与评估。

---

## 二、因子数据说明

### 1. 因子文件夹

路径：

```
1128_weight_factors/
```

子目录按日期组织，例如：

```
20181228/
20201231/
20211231/
20191231/
20221230/
20231229/
```

每个日期文件夹内包含一个：

```
weakFactors.h5
```

---

### 2. weakFactors.h5 结构说明

以 `20211231` 为例：

- 文件内包含 **3 个 key：0, 1, 2**
- **这三个 key 之间没有对应关系**
- 需要 **分别建模预测**
- 最终信号对三个预测结果 **取平均**

即：

```text
final_signal = mean(pred_key0, pred_key1, pred_key2)
```

---

### 3. 样本划分说明

- `20211231` 之前的数据 → **样本内（训练集）**
- `2022` 年的数据 → **样本外（测试集）**

> 换句话说：**一个文件可生成一整年的样本外数据**

---

## 三、Label 数据说明

### 1. Label 路径

```
Label10.h5
```

---

### 2. Label 字段说明

Label 文件中包含两个关键字段：

- `LabelDate`
- `EndDate`

#### 使用规则：

1. **LabelDate 必须与子文件夹中的 date 对齐**
2. **EndDate 用于过滤未来数据**
   - `EndDate > 20211231` 的数据 **不能使用**

即：

```text
仅使用 EndDate <= 当前因子日期 的样本
```

---

## 四、模型评估

### 1. LMT 评估接口

安装公司提供的 `lmt_data_api`，将你生成的信号 `pred_ensemble` 作为输入。

示例代码：

```python
pred_ensem.name = 'factor'
pred_ensem.index.names = ['date', 'code']

group_re = group_ir, group_ls = api.get_group_return(
    args=pred_ensem,
    'factor',
    'alpha1',
    label_period=10
)

df = api.get_daily_ic(args=pred_ensem, 'factor', 10)

stats_all = pd.concat([
    df[['group0', 'group1', 'group2', 'group3', 'group4', 'group5']],
    group_re[['group0', 'group1', 'group2', 'group3', 'group4', 'group5']]
], axis=1)

stats_all.columns = [
    'IC', 'ICIR',
    'Short', 'Long', 'LS',
    'ShortIR', 'LongIR', 'LSIR',
    'ShortHS', 'LongHS'
]
```

---

### 2. 基线绩效要求

- **树模型绩效如下图所示**
- **神经网络模型绩效需要优于树模型**
- 重点关注指标：
  - **收益类指标**
  - **IR（Information Ratio）**

评估核心指标：

| 指标 | 说明 |
|------|------|
| IC | 信息系数 |
| ICIR | IC 的 IR |
| Short | 空头收益 |
| Long | 多头收益 |
| LS | 多空收益 |
| ShortIR | 空头 IR |
| LongIR | 多头 IR |
| LSIR | 多空 IR |
| ShortHS | 空头换手 |
| LongHS | 多头换手 |

---

## 五、整体流程总结

### 数据流

```
1128_weight_factors/
    └── {date}/
         └── weakFactors.h5 (keys: 0,1,2)
```

### 建模逻辑

```text
for each date:
    load weakFactors.h5
    for key in [0,1,2]:
        pred_key = model.predict(key_data)
    pred_ensemble = mean(pred_key0, pred_key1, pred_key2)
```

### 训练 / 测试划分

```text
train: date <= 20211231
test:  date >= 20220101
```

### Label 对齐规则

```text
LabelDate == date
EndDate <= date
```

---

## 六、核心注意事项（非常重要）

1. **三个 key 之间无对应关系，必须分别建模**
2. **样本外严格按年份切分**
3. **EndDate 过滤是硬约束，不能使用未来信息**
4. **最终信号必须是三路预测的平均**
5. **神经网络绩效必须超过树模型 baseline**

---

## 七、最终交付物

- 训练好的模型
- `pred_ensemble` 信号生成脚本
- 样本内 / 样本外评估结果
- 与树模型 baseline 的对比报告


write a full step by step propusoal for this work 