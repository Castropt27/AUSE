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
import joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
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


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BI-friendly engineered features without any ML training."""
    out = df.copy()

    # user id
    if "user_id" in out.columns:
        out["user_id"] = out["user_id"]

    # age groups
    if "age" in out.columns:
        try:
            bins = [0, 18, 25, 35, 45, 55, 65, 120]
            labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
            out["age_group"] = pd.cut(out["age"].fillna(-1), bins=bins, labels=labels, right=False)
        except Exception:
            out["age_group"] = None

    # screen_time buckets (quantiles)
    if "screen_time_hours" in out.columns:
        try:
            out["screen_time_bucket"] = pd.qcut(out["screen_time_hours"], q=3, labels=["low", "medium", "high"], duplicates="drop")
        except Exception:
            out["screen_time_bucket"] = pd.cut(out["screen_time_hours"].fillna(0), bins=[-np.inf,4,8,np.inf], labels=["low","medium","high"])   

    # ratios
    if "work_screen_hours" in out.columns and "screen_time_hours" in out.columns:
        out["work_screen_ratio"] = out["work_screen_hours"] / out["screen_time_hours"].replace({0: np.nan})
    if "work_screen_hours" in out.columns and "leisure_screen_hours" in out.columns:
        out["work_leisure_ratio"] = out["work_screen_hours"] / out["leisure_screen_hours"].replace({0: np.nan})

    # outlier flag on screen_time (z-score)
    if "screen_time_hours" in out.columns:
        vals = out["screen_time_hours"].astype(float)
        z = (vals - vals.mean()) / vals.std(ddof=0)
        out["is_outlier_screen_time"] = z.abs() > 3

    # productivity bucket
    if "productivity_0_100" in out.columns:
        try:
            out["productivity_bucket"] = pd.qcut(out["productivity_0_100"], q=3, labels=["low","medium","high"], duplicates="drop")
        except Exception:
            out["productivity_bucket"] = None

    # missing count per row
    out["missing_count"] = out.isna().sum(axis=1)

    # ensure cluster_profile is string
    if "cluster_profile" in out.columns:
        out["cluster_profile"] = out["cluster_profile"].astype(str)

    return out


def _get_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix matching models.py structure."""
    base_feats = [
        "age",
        "screen_time_hours",
        "work_screen_hours",
        "leisure_screen_hours",
        "work_screen_ratio",
        "work_leisure_ratio",
        "mental_wellness_index_0_100",
    ]
    oh_prefixes = ("gender_", "occupation_", "work_mode_")
    oh_feats = [c for c in df.columns if any(c.startswith(p) for p in oh_prefixes)]
    feature_cols = [c for c in base_feats + oh_feats if c in df.columns]
    return df[feature_cols].copy()


def apply_model_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Load saved models and apply predictions without training.
    
    Adds columns:
    - pred_productivity_reg: regression prediction
    - confidence_reg: std deviation across ensemble trees (if available)
    - pred_productivity_class: classification prediction
    - confidence_class: probability of predicted class
    - cluster_profile: updated with clustering model labels
    """
    out = df.copy()
    
    # Regression
    reg_path = MODEL_DIR / "best_regression_model.joblib"
    if reg_path.exists():
        try:
            logging.info(f"Loading regression model: {reg_path}")
            reg_model = joblib.load(reg_path)
            X = _get_model_features(out)
            
            # Match features if model has feature_names_in_
            if hasattr(reg_model, "feature_names_in_"):
                X = X[[c for c in reg_model.feature_names_in_ if c in X.columns]]
            
            y_pred = reg_model.predict(X)
            out["pred_productivity_reg"] = y_pred
            
            # Confidence: std across ensemble trees if available
            if hasattr(reg_model, "estimators_"):
                try:
                    preds = np.array([tree.predict(X) for tree in reg_model.estimators_])
                    out["confidence_reg"] = preds.std(axis=0)
                except Exception:
                    out["confidence_reg"] = np.nan
            else:
                out["confidence_reg"] = np.nan
            
            logging.info("Applied regression model predictions")
        except Exception as e:
            logging.warning(f"Failed to apply regression model: {e}")
    else:
        logging.info("No regression model found, skipping regression predictions")
    
    # Classification
    clf_path = MODEL_DIR / "best_classification_model.joblib"
    if clf_path.exists():
        try:
            logging.info(f"Loading classification model: {clf_path}")
            clf_model = joblib.load(clf_path)
            X = _get_model_features(out)
            
            if hasattr(clf_model, "feature_names_in_"):
                X = X[[c for c in clf_model.feature_names_in_ if c in X.columns]]
            
            y_pred = clf_model.predict(X)
            out["pred_productivity_class"] = y_pred.astype(str)
            
            # Confidence: probability of predicted class
            if hasattr(clf_model, "predict_proba"):
                try:
                    probs = clf_model.predict_proba(X)
                    classes = list(clf_model.classes_)
                    # Get probability of predicted class for each sample
                    idx = [classes.index(lbl) if lbl in classes else 0 for lbl in y_pred]
                    out["confidence_class"] = [probs[i, idx[i]] for i in range(len(idx))]
                except Exception:
                    out["confidence_class"] = np.nan
            else:
                out["confidence_class"] = np.nan
            
            logging.info("Applied classification model predictions")
        except Exception as e:
            logging.warning(f"Failed to apply classification model: {e}")
    else:
        logging.info("No classification model found, skipping classification predictions")
    
    # Clustering
    clust_path = MODEL_DIR / "best_clustering_model.joblib"
    scaler_path = MODEL_DIR / "clustering_scaler.joblib"
    if clust_path.exists() and scaler_path.exists():
        try:
            logging.info(f"Loading clustering model: {clust_path}")
            clust_model = joblib.load(clust_path)
            scaler = joblib.load(scaler_path)
            
            feats = [c for c in ["screen_time_hours", "work_screen_hours", "leisure_screen_hours"] if c in out.columns]
            if feats:
                X_clust = out[feats].copy()
                X_scaled = scaler.transform(X_clust)
                labels = clust_model.predict(X_scaled)
                out["cluster_profile"] = labels.astype(str)
                logging.info("Applied clustering model and updated cluster_profile")
            else:
                logging.warning("Clustering features missing, skipping clustering")
        except Exception as e:
            logging.warning(f"Failed to apply clustering model: {e}")
    else:
        logging.info("No clustering model found, using rule-based cluster_profile")
    
    return out


def export_bi_users(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = [
        "user_id",
        "age",
        "age_group",
        "screen_time_hours",
        "screen_time_bucket",
        "work_screen_hours",
        "leisure_screen_hours",
        "work_screen_ratio",
        "work_leisure_ratio",
        "is_outlier_screen_time",
        "productivity_0_100",
        "productivity_bucket",
        "pred_productivity_reg",
        "confidence_reg",
        "pred_productivity_class",
        "confidence_class",
        "mental_wellness_index_0_100",
        "work_mode",
        "gender",
        "occupation",
        "sleep_hours",
        "stress_level_0_10",
        "cluster_profile",
        "missing_count",
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
    df = enrich_features(df)
    
    # Apply model predictions (regression, classification, clustering)
    df = apply_model_predictions(df)

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
