from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n1. REGRESSÃO (Productivity_0_100)")
    print(f"{'Modelo':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 50)
    print(f"{'RandomForestRegressor':<25} {mae:8.3f} {rmse:8.3f} {r2:8.3f}")

    importances = dict(
        sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda x: -x[1]
        )
    )
    print("\nFeature importance:", importances)

    # cross-validation (MSE -> RMSE)
    cv_scores = cross_val_score(
        model, X, y,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1
    )
    cv_rmse = (-cv_scores) ** 0.5
    print(f"\nCV RMSE (média ± sd): {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

# 2) Classificação — produtividade em 3 níveis (quantis)
def run_classification_productivity(df):
    X, y_cont, feats = _features_targets(df)
    # 3 níveis de produtividade (quantis)
    y = pd.qcut(y_cont, q=3, labels=["low", "medium", "high"], duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\n2. CLASSIFICAÇÃO (Productivity: low/medium/high por quantis)")
    print(f"{'Modelo':<25} {'Acc':>8} {'F1':>8} {'Recall':>8} {'Prec':>8} {'MCC':>8}")
    print("-" * 70)
    print(f"{'RandomForestClassifier':<25} {acc:8.3f} {macro_f1:8.3f} {macro_recall:8.3f} {macro_precision:8.3f} {mcc:8.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=["low","medium","high"])
    print("Matriz de confusão (linhas=verdade, colunas=previsto):\n", cm)

    # cross-validation (opcional – Accuracy)
    cv_acc = cross_val_score(clf, X, y, scoring="accuracy", cv=5, n_jobs=-1)
    print(f"\nCV Accuracy (média ± sd): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

# 3) Clustering — perfis de hábitos digitais
def run_clustering(df):
    feats = [c for c in ["screen_time_hours","work_screen_hours","leisure_screen_hours"] if c in df.columns]
    if len(feats) < 2:
        print("\n3. CLUSTERING: poucas features.")
        return

    X = df[feats].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n3. CLUSTERING (hábitos digitais)")
    print(f"{'k':<3} {'Silhouette':>12} {'Davies-Bouldin':>16} {'CH Index':>12}")
    print("-" * 50)

    results = []
    for k in range(2, 7):   # k = 2,3,4,5,6
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, clusters)
        db = davies_bouldin_score(X_scaled, clusters)
        ch = calinski_harabasz_score(X_scaled, clusters)

        results.append((k, sil, db, ch))
        print(f"{k:<3} {sil:12.4f} {db:16.4f} {ch:12.2f}")

    # se quiseres ainda ver centroides para k=3 (exemplo principal)
    k_best = 3
    kmeans = KMeans(n_clusters=k_best, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feats
    )
    centroids["cluster"] = range(k_best)

    print("\nCentroides para k=3 (escala original):")
    print(centroids.round(2))

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
