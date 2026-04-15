"""
Collect all SuperEncoder experiment results into a single file.

Scans every run_* directory under outputs_super_encoder/ and loads:
  - config_used.yaml   (hyperparameters for the run)
  - lmt_summary.csv    (aggregate LMT evaluation: IC, ICIR, group returns, etc.)
  - per-snapshot lmt_summary.csv files

Produces:
  - super_encoder_all_results.xlsx   (multi-sheet: summary, per-snapshot, configs)
  - super_encoder_all_results.csv    (flat summary with key config columns)

Usage:
    python collect_super_encoder_results.py
    python collect_super_encoder_results.py --output_dir /path/to/outputs_super_encoder
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml


def load_config(run_dir: Path) -> Dict[str, Any]:
    """Load config_used.yaml from a run directory."""
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dict into dot-separated keys."""
    flat = {}
    for k, v in config.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, f"{key}."))
        elif isinstance(v, list):
            flat[key] = str(v)
        else:
            flat[key] = v
    return flat


def load_lmt_summary(path: Path) -> pd.DataFrame:
    """Load an lmt_summary.csv file (index = dates, columns = metrics)."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    return df


def summarise_lmt(df: pd.DataFrame) -> Dict[str, float]:
    """Compute mean / std / IR for each column in a lmt_summary DataFrame."""
    if df.empty:
        return {}
    summary = {}
    for col in df.columns:
        vals = df[col].dropna()
        if vals.empty:
            continue
        mean_val = vals.mean()
        std_val = vals.std()
        summary[f"{col}_mean"] = mean_val
        summary[f"{col}_std"] = std_val
        if std_val > 1e-12:
            summary[f"{col}_ir"] = mean_val / std_val
    return summary


def collect_all_runs(output_dir: Path) -> tuple:
    """
    Walk all run_* directories and return:
      - agg_rows:      list of dicts (one per run, aggregate lmt_summary stats + config)
      - snapshot_rows:  list of dicts (one per run×snapshot, snapshot lmt_summary stats)
      - raw_agg_parts:  list of DataFrames (raw aggregate lmt_summary with run_id column)
    """
    run_dirs = sorted(output_dir.glob("run_*"))
    if not run_dirs:
        print(f"No run_* directories found in {output_dir}")
        return [], [], []

    agg_rows: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []
    raw_agg_parts: List[pd.DataFrame] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        config = load_config(run_dir)
        flat_cfg = flatten_config(config)

        # ---------- Aggregate lmt_summary ----------
        agg_lmt = load_lmt_summary(run_dir / "lmt_summary.csv")
        agg_stats = summarise_lmt(agg_lmt)

        row = {"run_id": run_id, **agg_stats, **flat_cfg}
        agg_rows.append(row)

        if not agg_lmt.empty:
            raw = agg_lmt.copy()
            raw.insert(0, "run_id", run_id)
            raw_agg_parts.append(raw)

        # ---------- Per-snapshot lmt_summary ----------
        snap_dirs = sorted(run_dir.glob("snapshot_*"))
        for snap_dir in snap_dirs:
            snap_name = snap_dir.name  # e.g. "snapshot_20181228"
            snap_lmt = load_lmt_summary(snap_dir / "lmt_summary.csv")
            snap_stats = summarise_lmt(snap_lmt)
            snapshot_rows.append({
                "run_id": run_id,
                "snapshot": snap_name,
                **snap_stats,
            })

    return agg_rows, snapshot_rows, raw_agg_parts


def build_summary_df(agg_rows: list) -> pd.DataFrame:
    """Build a flat summary DataFrame with key metrics first, then config columns."""
    if not agg_rows:
        return pd.DataFrame()
    df = pd.DataFrame(agg_rows)

    # Reorder: run_id, metric columns, then config columns
    metric_cols = [c for c in df.columns if c != "run_id" and not "." in c]
    config_cols = [c for c in df.columns if "." in c]
    ordered = ["run_id"] + sorted(metric_cols) + sorted(config_cols)
    ordered = [c for c in ordered if c in df.columns]
    return df[ordered]


def main():
    parser = argparse.ArgumentParser(
        description="Collect all SuperEncoder lmt_summary results into one file."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs_super_encoder",
        help="Path to the outputs_super_encoder directory (default: ./outputs_super_encoder)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output file base name (default: super_encoder_all_results)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: directory not found: {output_dir}")
        return

    out_base = args.out or "super_encoder_all_results"

    print(f"Scanning {output_dir} ...")
    agg_rows, snapshot_rows, raw_agg_parts = collect_all_runs(output_dir)
    print(f"  Found {len(agg_rows)} runs, {len(snapshot_rows)} snapshot results")

    if not agg_rows:
        print("Nothing to save.")
        return

    summary_df = build_summary_df(agg_rows)
    snapshot_df = pd.DataFrame(snapshot_rows) if snapshot_rows else pd.DataFrame()
    raw_agg_df = pd.concat(raw_agg_parts, ignore_index=False) if raw_agg_parts else pd.DataFrame()

    # ── CSV (flat summary) ──
    csv_path = output_dir / f"{out_base}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved flat summary → {csv_path}")

    # ── Excel (multi-sheet) ──
    try:
        xlsx_path = output_dir / f"{out_base}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            if not snapshot_df.empty:
                snapshot_df.to_excel(writer, sheet_name="per_snapshot", index=False)
            if not raw_agg_df.empty:
                raw_agg_df.to_excel(writer, sheet_name="raw_aggregate_lmt")
        print(f"  Saved Excel workbook → {xlsx_path}")
    except ImportError:
        print("  openpyxl not installed — skipping Excel output")
    except Exception as e:
        print(f"  Excel save failed: {e}")

    # ── Print top-level summary ──
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY (sorted by IC_mean desc)")
    print(f"{'='*80}")
    display_cols = ["run_id"]
    for c in ["IC_mean", "IC_ir", "ICIR_mean", "LS_mean", "LS_ir",
              "Long_mean", "Short_mean", "IR_LS_mean"]:
        if c in summary_df.columns:
            display_cols.append(c)
    if len(display_cols) > 1:
        sort_col = "IC_mean" if "IC_mean" in summary_df.columns else display_cols[1]
        print(summary_df[display_cols].sort_values(sort_col, ascending=False).to_string(index=False))
    else:
        print(summary_df.to_string(index=False))

    print(f"\nDone. {len(agg_rows)} experiments collected.")


if __name__ == "__main__":
    main()
