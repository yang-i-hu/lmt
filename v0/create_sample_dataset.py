"""
Create a small sample dataset for local development/testing.

Takes 20 days of data from factors_0_aligned.parquet for all instruments.
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = Path("raw_data/factors_0_aligned.parquet")
OUTPUT_FILE = Path("raw_data/factors_0_sample_20days.parquet")

N_DAYS = 20  # Number of days to sample

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CREATE SAMPLE DATASET")
    print("=" * 60)
    
    # Load the aligned data
    print(f"\nLoading: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    
    print(f"Full dataset shape: {df.shape}")
    print(f"Full dataset size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Get unique dates
    dates = df.index.get_level_values('date').unique().sort_values()
    print(f"\nTotal unique dates: {len(dates)}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    
    # Select first N days
    sample_dates = dates[:N_DAYS]
    print(f"\nSampling first {N_DAYS} days: {sample_dates.min()} to {sample_dates.max()}")
    
    # Filter data
    df_sample = df[df.index.get_level_values('date').isin(sample_dates)]
    
    print(f"\nSample dataset shape: {df_sample.shape}")
    print(f"Sample unique dates: {df_sample.index.get_level_values('date').nunique()}")
    print(f"Sample unique instruments: {df_sample.index.get_level_values('instrument').nunique()}")
    
    # Check label stats
    print(f"\nLabel stats:")
    print(f"  NaN count: {df_sample['labelValue'].isna().sum()}")
    print(f"  Valid count: {df_sample['labelValue'].notna().sum()}")
    
    # Save sample
    print(f"\nSaving to: {OUTPUT_FILE}")
    df_sample.to_parquet(OUTPUT_FILE)
    
    file_size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"Sample file size: {file_size_mb:.2f} MB")
    
    # Compression ratio
    original_size_mb = INPUT_FILE.stat().st_size / 1024 / 1024
    print(f"Size reduction: {original_size_mb:.2f} MB → {file_size_mb:.2f} MB ({file_size_mb/original_size_mb*100:.1f}%)")
    
    print(f"\n✅ Sample dataset created: {OUTPUT_FILE}")
    print("""
📥 Download this file to your local computer for testing:
   scp user@server:/path/to/data/factors_0_sample_20days.parquet ./

🔧 Usage:
   import pandas as pd
   df = pd.read_parquet('factors_0_sample_20days.parquet')
   X = df.drop(['labelValue', 'endDate'], axis=1)
   y = df['labelValue']
""")


if __name__ == "__main__":
    main()
