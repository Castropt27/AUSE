"""
ETL for Business Intelligence (Projeto 2 - BI)

This script reads Data/ScreenTime_clean.csv and prepares three CSVs for dashboards:
 - Data/BI/bi_users.csv      : user-level dataset with selected columns
 - Data/BI/bi_clusters.csv   : aggregated KPIs by cluster_profile
 - Data/BI/bi_kpis.csv       : global KPIs (single-row)

Notes:
 - No machine-learning models are trained or imported here.
 - If `cluster_profile` is missing in the cleaned CSV, a simple
   rule-based segmentation (quantile buckets on `screen_time_hours`)
   is created to produce `cluster_profile` values.
 - CSVs are written ready for Power BI / Tableau.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
INPUT_CLEAN = DATA_DIR / "ScreenTime_clean.csv"
FALLBACK_RAW = DATA_DIR / "ScreenTime vs MentalWellness.csv"
OUT_DIR = DATA_DIR / "BI"


def read_cleaned_csv() -> pd.DataFrame:
    if INPUT_CLEAN.exists():
        path = INPUT_CLEAN
    elif FALLBACK_RAW.exists():
        path = FALLBACK_RAW
        logging.warning("ScreenTime_clean.csv not found; using raw CSV as fallback.")
    else:
        raise FileNotFoundError(f"Neither {INPUT_CLEAN} nor {FALLBACK_RAW} exist.")

    logging.info(f"Reading data from {path}")
    df = pd.read_csv(path)
    logging.info(f"Read {len(df)} rows and {len(df.columns)} columns")
    return df


def ensure_cluster_profile(df: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """If `cluster_profile` exists and has values, keep it.
    Otherwise create a deterministic, rule-based cluster label using quantiles
    on `screen_time_hours` (low/medium/high by default).
    This avoids introducing ML code in the BI ETL phase.
    """
    if "cluster_profile" in df.columns and df["cluster_profile"].notna().sum() > 0:
        logging.info("Using existing 'cluster_profile' column from CSV.")
        return df

    if "screen_time_hours" not in df.columns:
        logging.warning("'screen_time_hours' missing; creating default cluster_profile=unknown")
        df["cluster_profile"] = "unknown"
        return df

    try:
        labels = ["low", "medium", "high"][:n_bins]
        df["cluster_profile"] = pd.qcut(df["screen_time_hours"], q=n_bins, labels=labels, duplicates="drop")
        # If qcut resulted in NA (e.g., constant column), fallback to cut with simple bins
        if df["cluster_profile"].isna().all():
            raise ValueError("qcut produced all NA")
        df["cluster_profile"] = df["cluster_profile"].astype(str)
        logging.info("Created 'cluster_profile' from 'screen_time_hours' quantiles.")
    except Exception:
        # fallback bins
        bins = [-np.inf, 4, 8, np.inf]
        labels = ["low", "medium", "high"]
        df["cluster_profile"] = pd.cut(df["screen_time_hours"].fillna(0), bins=bins, labels=labels)
        df["cluster_profile"] = df["cluster_profile"].astype(str)
        logging.info("Fallback: created 'cluster_profile' using fixed bins.")

    return df


def export_bi_users(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = [
        "age",
        "screen_time_hours",
        "work_screen_hours",
        "leisure_screen_hours",
        "productivity_0_100",
        "mental_wellness_index_0_100",
        "work_mode",
        "cluster_profile",
    ]

    present = [c for c in cols if c in df.columns]
    bi_users = df[present].copy()
    out_path = out_dir / "bi_users.csv"
    bi_users.to_csv(out_path, index=False)
    logging.info(f"Wrote user-level BI dataset: {out_path} ({len(bi_users)} rows)")
    return out_path


def export_bi_clusters(df: pd.DataFrame, out_dir: Path) -> Path:
    grp = df.groupby("cluster_profile")
    agg = grp.agg(
        n_users=("cluster_profile", "size"),
        productivity_mean=("productivity_0_100", "mean"),
        screen_time_mean=("screen_time_hours", "mean"),
    ).reset_index()
    # round numeric values for clearer dashboards
    agg["productivity_mean"] = agg["productivity_mean"].round(3)
    agg["screen_time_mean"] = agg["screen_time_mean"].round(3)

    out_path = out_dir / "bi_clusters.csv"
    agg.to_csv(out_path, index=False)
    logging.info(f"Wrote cluster-level BI dataset: {out_path} ({len(agg)} clusters)")
    return out_path


def export_bi_kpis(df: pd.DataFrame, clusters_df: pd.DataFrame, out_dir: Path) -> Path:
    productivity_mean = float(df["productivity_0_100"].mean()) if "productivity_0_100" in df.columns else np.nan
    screen_time_mean = float(df["screen_time_hours"].mean()) if "screen_time_hours" in df.columns else np.nan

    if not clusters_df.empty and "productivity_mean" in clusters_df.columns:
        top_cluster = clusters_df.loc[clusters_df["productivity_mean"].idxmax(), "cluster_profile"]
    else:
        top_cluster = ""

    kpis = pd.DataFrame([
        {
            "productivity_mean_global": round(productivity_mean, 3),
            "screen_time_mean_global": round(screen_time_mean, 3),
            "top_cluster_by_productivity": top_cluster,
        }
    ])

    out_path = out_dir / "bi_kpis.csv"
    kpis.to_csv(out_path, index=False)
    logging.info(f"Wrote global KPIs CSV: {out_path}")
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_cleaned_csv()
    df = ensure_cluster_profile(df)

    # Export user-level dataset
    users_path = export_bi_users(df, OUT_DIR)

    # Export aggregated cluster KPIs
    clusters_path = export_bi_clusters(df, OUT_DIR)

    # Read clusters back to compute KPIs reliably (already in memory here)
    clusters_df = pd.read_csv(clusters_path)

    # Export global KPIs
    kpis_path = export_bi_kpis(df, clusters_df, OUT_DIR)

    logging.info("ETL BI completed. Files: %s, %s, %s" % (users_path, clusters_path, kpis_path))


if __name__ == "__main__":
    main()
