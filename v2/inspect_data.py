"""
Data Structure Inspector — Diagnose column/index types for Parquet compatibility.

Usage:
    python inspect_data.py
"""

import h5py
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIG — adjust paths if needed
# =============================================================================
FACTOR_FILE = Path("../1128_weight_factors/20181228/weakFactors.h5")
LABEL_FILE = Path("../Label10.h5")

# =============================================================================
# 1. Inspect weakFactors.h5
# =============================================================================
print("=" * 70)
print("INSPECTING: weakFactors.h5")
print("=" * 70)

if not FACTOR_FILE.exists():
    print(f"❌ File not found: {FACTOR_FILE.absolute()}")
else:
    with h5py.File(FACTOR_FILE, "r") as f:
        print(f"\nTop-level keys: {list(f.keys())}")

        for key_name in ["0", "1", "2"]:
            if key_name not in f:
                print(f"\n  Key /{key_name}: NOT FOUND")
                continue

            grp = f[key_name]
            print(f"\n{'='*50}")
            print(f"  Key /{key_name}")
            print(f"{'='*50}")
            print(f"  Datasets in group: {list(grp.keys())}")

            for ds_name in grp.keys():
                ds = grp[ds_name]
                print(f"\n  --- {ds_name} ---")
                print(f"    shape: {ds.shape}")
                print(f"    dtype: {ds.dtype}")

                # Show first few values
                if len(ds.shape) == 1:
                    n = min(5, ds.shape[0])
                    raw = ds[:n]
                    print(f"    first {n} raw values: {raw}")
                    print(f"    python types: {[type(v).__name__ for v in raw]}")

                    # Check if bytes
                    if ds.dtype.kind == 'S':
                        decoded = [x.decode('utf-8') if isinstance(x, bytes) else x for x in raw]
                        print(f"    decoded: {decoded}")
                    elif ds.dtype.kind == 'O':
                        print(f"    (object dtype — may contain mixed types)")

                elif len(ds.shape) == 2:
                    n_rows = min(3, ds.shape[0])
                    n_cols = min(5, ds.shape[1])
                    sample = ds[:n_rows, :n_cols]
                    print(f"    sample [{n_rows}x{n_cols}]: {sample}")

            # Specifically check axis0 (columns)
            if "axis0" in grp:
                axis0 = grp["axis0"][:]
                print(f"\n  >>> axis0 (column labels) <<<")
                print(f"    full dtype: {axis0.dtype}")
                print(f"    numpy kind: {axis0.dtype.kind}")
                print(f"    length: {len(axis0)}")
                print(f"    first 10: {axis0[:10]}")
                print(f"    last 5:  {axis0[-5:]}")
                print(f"    min/max: {axis0.min()} / {axis0.max()}")
                # Check if they are contiguous 0..N-1
                expected = np.arange(len(axis0))
                print(f"    is 0..{len(axis0)-1} contiguous: {np.array_equal(axis0, expected)}")

            # Check axis1 (index levels)
            if "axis1_level0" in grp:
                lv0 = grp["axis1_level0"][:]
                print(f"\n  >>> axis1_level0 (dates) <<<")
                print(f"    dtype: {lv0.dtype}")
                print(f"    length: {len(lv0)}")
                print(f"    first 5: {lv0[:5]}")
                print(f"    last 5:  {lv0[-5:]}")

            if "axis1_level1" in grp:
                lv1 = grp["axis1_level1"][:]
                print(f"\n  >>> axis1_level1 (instruments) <<<")
                print(f"    dtype: {lv1.dtype}")
                print(f"    numpy kind: {lv1.dtype.kind}")
                print(f"    length: {len(lv1)}")
                raw_5 = lv1[:5]
                print(f"    first 5 raw: {raw_5}")
                if lv1.dtype.kind == 'S':
                    decoded = [x.decode('utf-8') for x in raw_5]
                    print(f"    first 5 decoded: {decoded}")
                elif lv1.dtype.kind == 'O':
                    print(f"    first 5 as str: {[str(x) for x in raw_5]}")

            if "axis1_label0" in grp:
                lb0 = grp["axis1_label0"][:10]
                print(f"\n  >>> axis1_label0 (date codes, first 10) <<<")
                print(f"    dtype: {grp['axis1_label0'].dtype}")
                print(f"    values: {lb0}")

            if "axis1_label1" in grp:
                lb1 = grp["axis1_label1"][:10]
                print(f"\n  >>> axis1_label1 (instrument codes, first 10) <<<")
                print(f"    dtype: {grp['axis1_label1'].dtype}")
                print(f"    values: {lb1}")

            # Only inspect key 0 in detail (others have same structure)
            break

# =============================================================================
# 2. Inspect Label10.h5
# =============================================================================
print("\n\n" + "=" * 70)
print("INSPECTING: Label10.h5")
print("=" * 70)

if not LABEL_FILE.exists():
    print(f"❌ File not found: {LABEL_FILE.absolute()}")
else:
    import pandas as pd

    # Read a small sample
    try:
        sample = pd.read_hdf(LABEL_FILE, key="Data", start=0, stop=5)
        print(f"\nSample (5 rows):")
        print(sample)
        print(f"\nIndex:")
        print(f"  type: {type(sample.index)}")
        print(f"  names: {sample.index.names}")
        print(f"  dtype: {sample.index.dtype if hasattr(sample.index, 'dtype') else 'MultiIndex'}")
        if hasattr(sample.index, 'levels'):
            for i, (name, level) in enumerate(zip(sample.index.names, sample.index.levels)):
                print(f"  level {i} ({name}): dtype={level.dtype}, first 3={list(level[:3])}")

        print(f"\nColumns: {list(sample.columns)}")
        print(f"Column dtypes:")
        for col in sample.columns:
            print(f"  {col}: {sample[col].dtype}")

        # After reset_index
        sample_reset = sample.reset_index()
        print(f"\nAfter reset_index():")
        print(f"  Columns: {list(sample_reset.columns)}")
        for col in sample_reset.columns:
            val = sample_reset[col].iloc[0]
            print(f"  {col}: dtype={sample_reset[col].dtype}, "
                  f"sample_value={val!r}, python_type={type(val).__name__}")

    except Exception as e:
        print(f"❌ Error reading Label10.h5: {e}")

# =============================================================================
# 3. Test: build a small aligned DataFrame and check column types
# =============================================================================
print("\n\n" + "=" * 70)
print("TEST: Build small aligned DataFrame")
print("=" * 70)

if FACTOR_FILE.exists() and LABEL_FILE.exists():
    import pandas as pd

    with h5py.File(FACTOR_FILE, "r") as f:
        grp = f["0"]
        values = grp["block0_values"][:100, :]
        columns_raw = grp["axis0"][:]
        level0 = grp["axis1_level0"][:]
        level1 = grp["axis1_level1"][:]
        label0 = grp["axis1_label0"][:100]
        label1 = grp["axis1_label1"][:100]

        if level1.dtype.kind == 'S':
            level1 = np.array([x.decode('utf-8') for x in level1])

    print(f"\ncolumns_raw dtype: {columns_raw.dtype}")
    print(f"columns_raw[:5]: {columns_raw[:5]}")

    # Convert to string
    columns_str = [str(c) for c in columns_raw]
    print(f"columns_str[:5]: {columns_str[:5]}")
    print(f"columns_str type check: {[type(c).__name__ for c in columns_str[:5]]}")

    dates = level0[label0]
    instruments = level1[label1]
    index = pd.MultiIndex.from_arrays([dates, instruments], names=["date", "instrument"])

    df = pd.DataFrame(values, index=index, columns=columns_str)
    print(f"\nDataFrame built successfully:")
    print(f"  shape: {df.shape}")
    print(f"  columns dtype: {df.columns.dtype}")
    print(f"  columns[:5]: {list(df.columns[:5])}")
    print(f"  index names: {df.index.names}")
    print(f"  index level dtypes: {[df.index.get_level_values(i).dtype for i in range(df.index.nlevels)]}")

    # Add label columns (simulating alignment)
    df['labelValue'] = 0.01
    df['endDate'] = 20211231
    print(f"\n  After adding labelValue/endDate:")
    print(f"  columns dtype: {df.columns.dtype}")
    print(f"  columns[-5:]: {list(df.columns[-5:])}")
    print(f"  all columns are str: {all(isinstance(c, str) for c in df.columns)}")

    # Test parquet save
    try:
        test_path = Path("_test_parquet_check.parquet")
        df.to_parquet(test_path)
        print(f"\n  ✅ to_parquet SUCCEEDED")
        test_path.unlink()  # cleanup
    except Exception as e:
        print(f"\n  ❌ to_parquet FAILED: {e}")

        # Diagnose: find non-string columns
        non_str_cols = [c for c in df.columns if not isinstance(c, str)]
        print(f"  Non-string columns ({len(non_str_cols)}): {non_str_cols[:10]}")
        print(f"  Their types: {[type(c).__name__ for c in non_str_cols[:10]]}")

        # Try with forced conversion
        df.columns = df.columns.astype(str)
        try:
            df.to_parquet(test_path)
            print(f"  ✅ to_parquet SUCCEEDED after .columns.astype(str)")
            test_path.unlink()
        except Exception as e2:
            print(f"  ❌ Still fails: {e2}")

else:
    print("⚠️ Skipping test — data files not found")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
