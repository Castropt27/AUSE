import pandas as pd
import numpy as np
from pathlib import Path

# ===== CONFIG =====
CSV_PATH = Path("Data/ScreenTime vs MentalWellness.csv")

# foco: hábitos digitais + contexto
NUM_COLS = [
    "age",
    "screen_time_hours",
    "work_screen_hours",
    "leisure_screen_hours",
    "sleep_hours",  
    "productivity_0_100",
    "mental_wellness_index_0_100",
]

CAT_COLS = ["gender", "occupation", "work_mode"]

RANGE_CHECKS = {
    "screen_time_hours": (0, 24),
    "work_screen_hours": (0, 24),
    "leisure_screen_hours": (0, 24),
    "productivity_0_100": (0, 100),
    "mental_wellness_index_0_100": (0, 100),
}
IQR_FACTOR = 1.5
# ==================


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]", "", regex=True)
        .str.lower()
    )
    return df

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in [c for c in NUM_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    # imputação: mediana (num.), moda (cat.)
    for c in [c for c in NUM_COLS if c in df.columns]:
        df[c] = df[c].fillna(df[c].median())
    for c in [c for c in CAT_COLS if c in df.columns]:
        mode = df[c].mode()
        fill = mode.iloc[0] if not mode.empty else "desconhecido"
        df[c] = df[c].fillna(fill).astype(str)
    return df

def _apply_range_checks(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    # coerência básica
    if {"work_screen_hours", "screen_time_hours"}.issubset(df.columns):
        df["work_screen_hours"] = np.minimum(df["work_screen_hours"], df["screen_time_hours"])
    if {"leisure_screen_hours", "screen_time_hours"}.issubset(df.columns):
        df["leisure_screen_hours"] = np.minimum(df["leisure_screen_hours"], df["screen_time_hours"])
    return df

def _remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in NUM_COLS if c in df.columns]
    mask = pd.Series(True, index=df.index)
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - IQR_FACTOR * iqr
        high = q3 + IQR_FACTOR * iqr
        mask &= df[col].between(low, high)
    return df[mask].copy()

def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # ratios de equilíbrio digital
    if {"work_screen_hours", "screen_time_hours"}.issubset(df.columns):
        denom = df["screen_time_hours"].replace(0, np.nan)
        df["work_screen_ratio"] = (df["work_screen_hours"] / denom).fillna(0).clip(0, 1)
    if {"work_screen_hours", "leisure_screen_hours"}.issubset(df.columns):
        denom2 = df["leisure_screen_hours"].replace(0, np.nan)
        df["work_leisure_ratio"] = (df["work_screen_hours"] / denom2).replace([np.inf, -np.inf], np.nan).fillna(0)
    if {"work_screen_hours", "screen_time_hours"}.issubset(df.columns):
        denom = df["screen_time_hours"].replace(0, np.nan)
        df["work_screen_ratio"] = (df["work_screen_hours"] / denom).fillna(0).clip(0, 1)
    return df

def _one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    cats = [c for c in CAT_COLS if c in df.columns]
    if not cats:
        return df
    dummies = pd.get_dummies(df[cats], drop_first=False, dtype=int)

    df = pd.concat([df.drop(columns=cats), dummies], axis=1)
    return df

def load_and_clean_data(save_clean_csv: bool = True) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    df = _normalize_columns(df)
    df = df.drop_duplicates()

    # remover colunas que decidiste NÃO usar
    df.drop(columns=[
    "exercise_minutes_per_week",
    "stress_level_0_10",
    "social_hours_per_week",
    "sleep_quality_1_5",
    "user_id",
    "totalscreentime",
    "total_screen_time"
], inplace=True, errors="ignore")
    

    df = _coerce_numeric(df)
    df = _handle_missing(df)
    df = _apply_range_checks(df)
    df = _remove_outliers_iqr(df)
    df = _feature_engineering(df)
    df = _one_hot_encode(df)

    if save_clean_csv:
        out = CSV_PATH.with_name("ScreenTime_clean.csv")
        df.to_csv(out, index=False)
        print(f"✅ CSV limpo guardado em: {out.resolve()}")

    print(f"✅ Dados limpos. Shape: {df.shape}")
    return df

def prepare_features_targets(df: pd.DataFrame):
    """
    Foco: produtividade vs. hábitos digitais e modo de trabalho.
    """
    base_feats = [
        "age",
        "screen_time_hours",
        "work_screen_hours",
        "leisure_screen_hours",
        "work_screen_ratio",
        "work_leisure_ratio",
        "mental_wellness_index_0_100",  # opcional, contexto
    ]
    oh_feats = [c for c in df.columns if any(c.startswith(prefix) for prefix in ["gender_", "occupation_", "work_mode_"])]

    feature_cols = [c for c in base_feats + oh_feats if c in df.columns]
    X = df[feature_cols].copy()
    y = df["productivity_0_100"].copy()
    return X, y, feature_cols
