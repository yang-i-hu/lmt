# Copilot Doc — Importing `lmt_data_api` and Running the Standard Evaluation Snippet

## 1) Imports (exact)


### Option A (recommended): import the `api` module as `api`

```python
from lmt_data_api.api import DataApi
api = DataApi()


```


## 2) Required `pred_esem` format (must satisfy this)

`pred_esem` should be a `pd.Series` with:

- `pred_esem.name == "factor"`
- `pred_esem.index` is a MultiIndex with 2 levels:
  - `["date", "code"]`
- index must be unique: each `(date, code)` appears once
- values are numeric floats

Example schema enforcement:

```python
pred_esem = pred_esem.astype("float64")
pred_esem.name = "factor"
pred_esem.index.names = ["date", "code"]
pred_esem = pred_esem[~pred_esem.index.duplicated(keep="last")]
```

---

## 3) The exact evaluation snippet

### 3.1 Run group return + IC, then merge key columns

```python
import pandas as pd
import lmt_data_api.api as api

# 0) enforce required schema
pred_esem.name = "factor"
pred_esem.index.names = ["date", "code"]

# 1) Group return related metrics
# Returns: group_re, group_ir, group_hs (DataFrames)
# - group_re: group return series (includes group0..group9, ls)
# - group_ir: IR metrics per group (includes group0..group9, ls)
# - group_hs: holding stats (often includes group0..group9, etc.)
group_re, group_ir, group_hs = api.da_eva_group_return(
    pred_esem,
    "factor",
    alpha=1,
    label_period=10
)

# 2) IC time series
ic_df = api.da_eva_ic(
    pred_esem,
    "factor",
    10
)

# 3) Build a unified stats table (aligns with screenshot intent)
stats_all = pd.concat(
    objs=[
        ic_df,  # expected to include IC / ICIR columns (depending on implementation)
        group_re[["group0", "group9", "ls"]],
        group_ir[["group0", "group9", "ls"]],
        group_hs[["group0", "group9"]],   # adjust if your group_hs columns differ
    ],
    axis=1
)

stats_all.columns = [
    "IC", "ICIR",
    "Short", "Long", "LS",
    "ShortIr", "LongIr", "LSIR",
    "ShortHS", "LongHS"
]
```