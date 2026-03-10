# HDF5 Data Layout
## File 1: `weakFactors.h5` — Factor Values (FIXED format)

### High-level meaning
- Stores **three large pandas DataFrames** under keys:
  ```
  /0
  /1
  /2
  ```
- Each key corresponds to **one factor matrix** (semantic meaning differs, structure identical).
- This file contains the **actual weak factor values**.

---

### Shape and scale (per key `/0`, `/1`, `/2`)
- Rows: **3203935**
- Columns: **1,128**
- dtype: **float32**
- Total size per key: ~12.7 GB
- Total file size: ~38 GB

---

### Internal structure (pandas `format="fixed"`)

Each group (`/0`, `/1`, `/2`) contains:

```
axis0               (1128,)        int64    → column labels (factor IDs)
block0_items        (1128,)        int64    → same as axis0
block0_values       (3203935,1128) float32 → ACTUAL FACTOR VALUES

axis1_label0        (3203935,)     int16    → index level 0 codes
axis1_label1        (3203935,)     int16    → index level 1 codes
axis1_level0        (933,)         int64    → unique values for index level 0
axis1_level1        (3646,)        bytes   → unique values for index level 1
```

---

### Row index semantics (MultiIndex)
Each row `i` corresponds to:

```
index[i] = (
  axis1_level0[ axis1_label0[i] ],
  axis1_level1[ axis1_label1[i] ]
)
```

Typical quant interpretation:
- Level 0: trading date (~928 unique)
- Level 1: instrument / security ID (~4316 unique)

---

### CRITICAL RULES (DO NOT VIOLATE)

❌ **DO NOT** use:
```
pd.read_hdf(path, key="/0", start=..., stop=...)
```

❌ **DO NOT** load full DataFrames into memory.

Reason:
- This is **pandas fixed-format**
- Partial row slicing breaks MultiIndex reconstruction
- Leads to errors such as:
  ```
  ValueError: code max >= length of level
  ```

---

### Correct way to read this file
- Use **`h5py`**
- Slice **only**:
  - `block0_values[start:stop, :]`
  - `axis1_label0[start:stop]`
  - `axis1_label1[start:stop]`
- Load `axis1_level*` fully (they are small)
- Manually reconstruct the `MultiIndex`

This file is **read-only, sequential, and fragile**.

### Example

---

### Intended use
- Batch factor modeling
- Offline research
- One-time migration to Parquet or Zarr

---

## File 2: `Label10.h5` HDF5 — Index / Label Engine (TABLE format)

### High-level meaning
- Does **NOT** store factor values
- Stores **index metadata only**
- Enables fast slicing and filtering for pandas table-format objects

---

### Top-level layout

```
/Data
/Data/table
/Data/_i_table/index/*
```

---

### `/Data/table`
```
shape: (12,519,451,)
dtype: |V30   # packed binary rows
```

- PyTables row-oriented table
- Contains serialized index + row pointers
- **Never read directly**

---

### `/Data/_i_table/index/*` — Index accelerator

Key components:
```
indices      (23, 524288)   uint64
sorted       (23, 524288)   int64
ranges       (23, 2)        int64
bounds       (23, 127)      int64
abounds      (2944,)        int64
zbounds      (2944,)        int64
indicesLR    (524288,)      uint64
```

Meaning:
- Data split into **23 chunks**
- Each chunk has sorted indices and bounds
- Enables:
  ```
  pd.read_hdf(..., where=..., start=..., stop=...)
  ```
- Safe for slicing and boolean filtering

---

### Intended use
- Index lookup
- Fast row filtering
- Metadata only (no factor values)

---

## Relationship between the two files

- `weakFactors.h5`
  → factor values only (fixed format, unsafe slicing)

- `Data` HDF5
  → index engine only (table format, safe slicing)

They are often generated together in quant pipelines but must **never be treated the same way**.
