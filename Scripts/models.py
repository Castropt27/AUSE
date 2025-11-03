from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, f1_score, silhouette_score
import pandas as pd

RANDOM_STATE = 42

def run_models(df):
    print("=== INICIANDO MODELAGEM ===")
    run_regression_productivity(df)
    run_classification_productivity(df)
    run_clustering(df)

def _features_targets(df):
    base_feats = [
        "age",
        "screen_time_hours",
        "work_screen_hours",
        "leisure_screen_hours",
        "work_screen_ratio",
        "work_leisure_ratio",
        "mental_wellness_index_0_100",
    ]
    oh_feats = [c for c in df.columns if any(c.startswith(prefix) for prefix in ["gender_", "occupation_", "work_mode_"])]
    feature_cols = [c for c in base_feats + oh_feats if c in df.columns]
    X = df[feature_cols].copy()
    y = df["productivity_0_100"].copy()
    return X, y, feature_cols

# 1) Regressão — prever produtividade contínua
def run_regression_productivity(df):
    X, y, feats = _features_targets(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5

    print("\n1. REGRESSÃO (Productivity_0_100)")
    print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f}")
    importances = dict(sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]))
    print("Feature importance:", importances)

    # cross-validation
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
    print(f"CV MSE (média ± sd): {(-scores.mean()):.4f} ± {scores.std():.4f}")

# 2) Classificação — produtividade em 3 níveis (quantis)
def run_classification_productivity(df):
    X, y_cont, feats = _features_targets(df)
    y = pd.qcut(y_cont, q=3, labels=["low","medium","high"], duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    clf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n2. CLASSIFICAÇÃO (Productivity: low/medium/high por quantis)")
    print(f"Acurácia: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=["low","medium","high"])
    print("Matriz de confusão (linhas=verdade, colunas=previsto):\n", cm)

# 3) Clustering — perfis de hábitos digitais
def run_clustering(df):
    feats = [c for c in ["screen_time_hours","work_screen_hours","leisure_screen_hours"] if c in df.columns]
    if len(feats) < 2:
        print("\n3. CLUSTERING: poucas features.")
        return

    X = df[feats].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, clusters)
    sizes = pd.Series(clusters).value_counts().to_dict()

    import numpy as np
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feats)
    centroids["cluster"] = range(3)

    print("\n3. CLUSTERING (hábitos digitais)")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Tamanho dos clusters: {sizes}")
    print("\nCentroides (escala original):")
    print(centroids.round(2))
