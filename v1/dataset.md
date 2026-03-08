# weakFactors Weight-Factor Dataset Structure + Export Workflow (for Reweighting / OOS Testing)

## 0) What this dataset is
`1128_weight_factors/` contains multiple snapshots of **weak factors** produced by training a factor model on a rolling history.

Each snapshot lives under a date-named subdirectory (e.g. `20181228/`), and contains a `weakFactors.h5`.

Key idea from dataset creator:
- Folder name `YYYYMMDD` = **training cutoff date** (样本内截止日).
- For that snapshot:
  - Data **<= cutoff** was used to train the factor model (样本内 / in-sample).
  - Data **> cutoff** was NOT used to train (样本外 / out-of-sample).

So each folder is a “model trained as-of date D”.

---

## 1) Directory layout
Example:
- `1128_weight_factors/20181228/weakFactors.h5`
- `1128_weight_factors/20191231/weakFactors.h5`
- `1128_weight_factors/20201231/weakFactors.h5`
- `1128_weight_factors/20211231/weakFactors.h5`

Interpretation:
- `20181228/` snapshot is an older trained model (trained using data up to 20181228).
- `20211231/` snapshot is a newer trained model (trained using data up to 20211231).

For reweighting, we typically want to evaluate on the **OOS segment** of each snapshot:
- OOS period for snapshot D = `(D+1 ... next_cutoff)` or `(D+1 ... end_of_dataset)`.

---

## 2) Inside each `weakFactors.h5`
Each H5 has **3 top-level keys**: `0`, `1`, `2`.

We treat them as 3 *replicates* (different seeds / batches) of the same factor-generation run.
- Same factor definitions, but slightly different realized values due to randomness.

Inside each key:
- multiple datasets (DS). Each dataset corresponds to a factor block / factor family / factor id group.
- Each DS is typically a matrix with shape like:
  - `[date_or_time, security]` (most common)
  - or `[date_or_time, security, factor]` depending on how it was stored.

(Exact DS names and shapes should be printed by our inspector script.)

---

## 3) What we are building (outputs)
We want Parquet outputs that are easy for:
- downstream reweight training,
- API-driven OOS evaluation,
- and fast slicing by date / factor group.

Target outputs (per snapshot):
1) `factors_k{0,1,2}.parquet`  (raw replicas)
2) `factors_mean.parquet`      (replica-averaged factors; recommended default)
3) `labels_aligned.parquet`    (labels aligned by date/security)
4) `factors_mean_aligned.parquet` (factors + labels merged for training/testing)
5) metadata json (schema, shapes, cutoffs, date ranges, factor names)

---

## 4) Reweighting evaluation rule (核心)
We ONLY evaluate reweighting on **样本外 (OOS)**.

For each snapshot folder with cutoff D:
- OOS dates = all dates `> D` (strictly after).
- In practice, for clean backtests we often define:
  - OOS window = `(D+1 ... next_snapshot_cutoff]`
  - so each day is tested using the most recent model that would have existed at that time.

This prevents lookahead and matches “real deployment” logic.

---

## 5) Recommended workflow (Read → Normalize → Align → Export)
### Step A — Enumerate snapshots
1. List subdirs under `1128_weight_factors/`:
   - parse cutoff dates from folder names
   - sort ascending: `D1 < D2 < D3 ...`

### Step B — For each snapshot folder `Di/weakFactors.h5`
1. Open H5
2. For each key `k in {0,1,2}`:
   - read all datasets under that key
   - convert to a unified factor table/matrix with explicit axes:
     - index: `date` (or datetime)
     - columns: `security_id`
     - value: factor values (possibly multiple factors → multi-column)
3. Optional but recommended: per-date cross-sectional normalization (zscore/rank)
   - to ensure replicas are on comparable scales
4. Save raw replicas:
   - `snapshot=Di/factors_k0.parquet`, `...k1...`, `...k2...`

### Step C — Replica aggregation (default path)
Compute averaged factors:
- `factors_mean = (k0 + k1 + k2) / 3`
Save:
- `snapshot=Di/factors_mean.parquet`

### Step D — Align with labels (same date/security)
1. Load labels H5 (`label10.h5` / `SxCorrBDWCAP10.h5` etc.)
2. Convert labels to matrix/table keyed by `(date, security_id)`.
3. Join:
   - `factors_mean` with `labels` on `(date, security_id)`
4. Save:
   - `snapshot=Di/factors_mean_aligned.parquet`
   - `snapshot=Di/labels_aligned.parquet` (optional shared cache)

### Step E — Split IS vs OOS for evaluation
Given cutoff Di:
- IS dates: `<= Di` (DO NOT use for OOS evaluation)
- OOS dates: `> Di`

For “rolling snapshot” evaluation:
- Use OOS window `(Di+1 ... D_{i+1}]` for snapshot i
- For last snapshot `Dn`: OOS is `(Dn+1 ... end_date]`

Export these as convenience:
- `snapshot=Di/oos_factors_mean.parquet`
- `snapshot=Di/oos_factors_mean_aligned.parquet`

---

## 6) How workers should talk about it (terminology)
- “Snapshot / as-of model”: the folder `YYYYMMDD`
- “Cutoff date Di”: the in-sample training end date for that snapshot
- “IS / 样本内”: dates <= Di
- “OOS / 样本外”: dates > Di
- “Replica keys”: H5 top-level keys 0/1/2 (three runs)
- “Replica-mean factors”: averaged across keys 0/1/2 (default)

---

## 7) Minimal decision policy (so no one bikesheds)
For reweighting pipeline:
1. Default factor input = `factors_mean` (replica averaged).
2. Default evaluation = strictly OOS (dates > cutoff).
3. Default rolling mapping = use snapshot i for dates (Di+1 ... D_{i+1}] to avoid lookahead.

Any deviation (e.g., using replicas separately or selecting best replica) must be explicitly justified and validated OOS.
