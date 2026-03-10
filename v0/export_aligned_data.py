"""
Export Aligned Factor and Label Data for Training

This script exports all factor data (keys 0, 1, 2) with aligned labels 
for the date range 20180102-20211230.

Rules:
- labelDate must align with factor date
- endDate > 20211231 cannot be used (future data filter)

Output files:
- factors_0.parquet, factors_1.parquet, factors_2.parquet: Full factor matrices
- factors_0_aligned.parquet, factors_1_aligned.parquet, factors_2_aligned.parquet: Factors + labels
- labels_aligned.parquet: All valid labels
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

FACTOR_FILE = Path("../1128_weight_factors/20201231/weakFactors.h5")
LABEL_FILE = Path("../label10.h5")
OUTPUT_DIR = Path("data")

FACTOR_DATE_START = 20180102
FACTOR_DATE_END = 20211230
FUTURE_CUTOFF = 20211231  # endDate > this cannot be used

FACTOR_KEYS = ["0", "1", "2"]
CHUNK_SIZE = 100000  # Rows per chunk for factor loading
LABEL_CHUNK_SIZE = 500000  # Rows per chunk for label loading


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_factor_slice(h5_file, key="0", start=0, stop=1000):
    """
    Safely load a slice of factor data using h5py.
    Returns a pandas DataFrame with proper MultiIndex.
    """
    with h5py.File(h5_file, "r") as f:
        grp = f[key]
        
        # Load factor values slice
        values = grp["block0_values"][start:stop, :]
        
        # Load columns
        columns = grp["axis0"][:]
        
        # Load index components
        level0 = grp["axis1_level0"][:]
        level1 = grp["axis1_level1"][:]
        label0 = grp["axis1_label0"][start:stop]
        label1 = grp["axis1_label1"][start:stop]
        
        # Decode bytes if needed
        if level1.dtype.kind == 'S':
            level1 = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in level1])
    
    # Reconstruct index
    dates = level0[label0]
    instruments = level1[label1]
    index = pd.MultiIndex.from_arrays([dates, instruments], names=["date", "instrument"])
    
    # Create DataFrame
    df = pd.DataFrame(values, index=index, columns=columns)
    return df


def find_label_date_position(label_file, target_date, total_rows):
    """Binary search to find approximate row where labelDate >= target_date"""
    low, high = 0, total_rows
    
    while low < high:
        mid = (low + high) // 2
        sample = pd.read_hdf(label_file, key="Data", start=mid, stop=mid+1)
        sample = sample.reset_index()
        mid_date = sample['labelDate'].iloc[0]
        
        if mid_date < target_date:
            low = mid + 1
        else:
            high = mid
    
    return low


def load_all_valid_labels(label_file, start_date, end_date, future_cutoff, label_start_row=0):
    """
    Load all labels where:
    - labelDate >= start_date AND labelDate <= end_date
    - endDate <= future_cutoff
    """
    print("Loading all valid labels...")
    
    with pd.HDFStore(label_file, mode='r') as store:
        total_rows = store.get_storer('Data').nrows
    
    all_labels = []
    rows_loaded = 0
    rows_kept = 0
    
    for start in range(label_start_row, total_rows, LABEL_CHUNK_SIZE):
        labels = pd.read_hdf(label_file, key="Data", start=start, stop=start + LABEL_CHUNK_SIZE)
        labels = labels.reset_index()
        rows_loaded += len(labels)
        
        # Check if we've passed the date range
        if labels['labelDate'].min() > end_date:
            print(f"  Reached end of date range at row {start:,}")
            break
        
        # Filter by labelDate range
        labels = labels[(labels['labelDate'] >= start_date) & (labels['labelDate'] <= end_date)]
        
        # Filter: endDate <= future_cutoff
        labels = labels[labels['endDate'] <= future_cutoff]
        
        if len(labels) > 0:
            all_labels.append(labels)
            rows_kept += len(labels)
        
        if start % 2000000 == 0:
            print(f"  Processed {start:,} rows, kept {rows_kept:,} labels...")
    
    print(f"\n✅ Total rows scanned: {rows_loaded:,}")
    print(f"✅ Valid labels kept: {rows_kept:,}")
    
    if all_labels:
        labels_combined = pd.concat(all_labels, ignore_index=True)
        return labels_combined
    return None


def export_factor_data(factor_file, key, labels_indexed, output_dir):
    """
    Load all factor data for a key and save with aligned labels.
    
    Saves:
    - factors_{key}.parquet: Factor values with MultiIndex
    - factors_{key}_aligned.parquet: Factors + labelValue for rows with labels
    """
    print(f"\n{'='*60}")
    print(f"Processing Factor Key: /{key}")
    print("=" * 60)
    
    # Get total rows
    with h5py.File(factor_file, "r") as f:
        n_rows = f[key]["block0_values"].shape[0]
        n_cols = f[key]["block0_values"].shape[1]
    
    print(f"Total rows: {n_rows:,}, columns: {n_cols}")
    
    # Load in chunks and save
    all_chunks = []
    aligned_chunks = []
    
    for start in range(0, n_rows, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, n_rows)
        
        # Load factor slice
        factors = load_factor_slice(factor_file, key=key, start=start, stop=stop)
        all_chunks.append(factors)
        
        # Align with labels
        common_idx = factors.index.intersection(labels_indexed.index)
        if len(common_idx) > 0:
            factors_aligned = factors.loc[common_idx].copy()
            factors_aligned['labelValue'] = labels_indexed.loc[common_idx, 'labelValue']
            factors_aligned['endDate'] = labels_indexed.loc[common_idx, 'endDate']
            aligned_chunks.append(factors_aligned)
        
        if start % 500000 == 0:
            print(f"  Processed {start:,}/{n_rows:,} rows...")
    
    # Combine all chunks
    print("  Combining chunks...")
    all_factors = pd.concat(all_chunks)
    
    # Save full factors
    factors_path = output_dir / f"factors_{key}.parquet"
    all_factors.to_parquet(factors_path)
    print(f"  ✅ Factors saved: {factors_path}")
    print(f"     Shape: {all_factors.shape}")
    print(f"     Size: {factors_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
    
    # Save aligned factors with labels
    if aligned_chunks:
        all_aligned = pd.concat(aligned_chunks)
        aligned_path = output_dir / f"factors_{key}_aligned.parquet"
        all_aligned.to_parquet(aligned_path)
        print(f"  ✅ Aligned data saved: {aligned_path}")
        print(f"     Shape: {all_aligned.shape}")
        print(f"     Size: {aligned_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
        
        # Stats
        print(f"  📊 Alignment rate: {len(all_aligned) / len(all_factors) * 100:.2f}%")
        print(f"     NaN in labelValue: {all_aligned['labelValue'].isna().sum():,}")
    
    # Free memory
    del all_factors, all_chunks, aligned_chunks
    
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = datetime.now()
    
    print("=" * 60)
    print("EXPORT ALIGNED FACTOR AND LABEL DATA")
    print("=" * 60)
    print(f"\nStarted at: {start_time}")
    print(f"\nConfiguration:")
    print(f"  Factor file: {FACTOR_FILE.absolute()}")
    print(f"  Label file:  {LABEL_FILE.absolute()}")
    print(f"  Output dir:  {OUTPUT_DIR.absolute()}")
    print(f"  Date range:  {FACTOR_DATE_START} to {FACTOR_DATE_END}")
    print(f"  Future cutoff: endDate > {FUTURE_CUTOFF} excluded")
    
    # Check files exist
    if not FACTOR_FILE.exists():
        print(f"\n❌ Error: Factor file not found: {FACTOR_FILE}")
        return
    if not LABEL_FILE.exists():
        print(f"\n❌ Error: Label file not found: {LABEL_FILE}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # =========================================================================
    # Step 1: Find where 2018 data starts in label file
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Finding label data start position...")
    print("=" * 60)
    
    with pd.HDFStore(LABEL_FILE, mode='r') as store:
        total_label_rows = store.get_storer('Data').nrows
    
    print(f"Total label rows: {total_label_rows:,}")
    print(f"Searching for labelDate >= {FACTOR_DATE_START}...")
    
    label_start_pos = find_label_date_position(LABEL_FILE, FACTOR_DATE_START, total_label_rows)
    print(f"✅ Found: labelDate >= {FACTOR_DATE_START} starts at row ~{label_start_pos:,}")
    
    # =========================================================================
    # Step 2: Load all valid labels
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Loading all valid labels...")
    print("=" * 60)
    
    all_labels = load_all_valid_labels(
        LABEL_FILE,
        start_date=FACTOR_DATE_START,
        end_date=FACTOR_DATE_END,
        future_cutoff=FUTURE_CUTOFF,
        label_start_row=label_start_pos
    )
    
    if all_labels is None:
        print("❌ Error: No valid labels found!")
        return
    
    print(f"\nAll labels shape: {all_labels.shape}")
    print(f"Columns: {list(all_labels.columns)}")
    print(f"labelDate range: {all_labels['labelDate'].min()} to {all_labels['labelDate'].max()}")
    print(f"endDate range: {all_labels['endDate'].min()} to {all_labels['endDate'].max()}")
    
    # =========================================================================
    # Step 3: Create label index and save
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Creating label index and saving...")
    print("=" * 60)
    
    # Set MultiIndex: (labelDate, code) -> (date, instrument)
    labels_indexed = all_labels.set_index(['labelDate', 'code'])
    labels_indexed.index.names = ['date', 'instrument']
    
    print(f"Labels indexed shape: {labels_indexed.shape}")
    print(f"Unique dates: {labels_indexed.index.get_level_values('date').nunique()}")
    print(f"Unique instruments: {labels_indexed.index.get_level_values('instrument').nunique()}")
    
    # Save labels
    labels_output_path = OUTPUT_DIR / "labels_aligned.parquet"
    labels_indexed.to_parquet(labels_output_path)
    print(f"\n✅ Labels saved to: {labels_output_path}")
    print(f"   File size: {labels_output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Free memory
    del all_labels
    
    # =========================================================================
    # Step 4: Export factor data for each key
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Exporting factor data...")
    print("=" * 60)
    
    for key in FACTOR_KEYS:
        export_factor_data(FACTOR_FILE, key, labels_indexed, OUTPUT_DIR)
    
    # =========================================================================
    # Summary
    # =========================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    
    print(f"\n📁 Output Directory: {OUTPUT_DIR.absolute()}\n")
    
    total_size = 0
    for f in sorted(OUTPUT_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  {f.name:35s} {size_mb:>10.2f} MB")
    
    print(f"\n  {'Total':35s} {total_size:>10.2f} MB ({total_size/1024:.2f} GB)")
    
    print(f"\n⏱️  Duration: {duration}")
    print(f"✅ Finished at: {end_time}")
    
    print("""
📋 File Descriptions:
  - factors_X.parquet:         All factor values for key X (full data)
  - factors_X_aligned.parquet: Factors + labelValue for training
  - labels_aligned.parquet:    All valid labels (endDate <= 20211231)

🔧 Usage Example:
  import pandas as pd
  
  # Load aligned training data
  df = pd.read_parquet('processed_data/factors_0_aligned.parquet')
  X = df.drop(['labelValue', 'endDate'], axis=1)
  y = df['labelValue']
""")


if __name__ == "__main__":
    main()
