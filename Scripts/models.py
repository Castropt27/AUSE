# scripts/models.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def run_models(df):
    """
    Executa todos os modelos no dataset
    """
    print("=== INICIANDO MODELAGEM ===")
    
    # 1. REGRESSÃO - Prever qualidade do sono
    print("\n1. MODELO DE REGRESSÃO (Sleep Quality)")
    run_regression(df, 'sleep_quality_1_5')
    
    # 2. CLASSIFICAÇÃO - Classificar bem-estar mental
    print("\n2. MODELO DE CLASSIFICAÇÃO (Mental Wellness)")
    run_classification(df)
    
    # 3. CLUSTERING - Agrupar usuários
    print("\n3. MODELO DE CLUSTERING (Hábitos Digitais)")
    run_clustering(df)

def run_regression(df, target_col):
    """Modelo de regressão"""
    # Features e target
    X = df[['screen_time_hours', 'sleep_hours', 'exercise_minutes_per_week']]
    y = df[target_col]
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prever e avaliar
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Target: {target_col}")
    print(f"MSE: {mse:.4f}")
    print(f"Feature importance: {dict(zip(X.columns, model.feature_importances_))}")

def run_classification(df):
    """Modelo de classificação"""
    # Criar categoria de bem-estar (baixo/médio/alto)
    df['wellness_category'] = pd.cut(df['mental_wellness_index_0_100'], 
                                    bins=[0, 33, 66, 100], 
                                    labels=['low', 'medium', 'high'])
    
    # Features e target
    X = df[['screen_time_hours', 'sleep_hours', 'stress_level_0_10', 'exercise_minutes_per_week']]
    y = df['wellness_category']
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prever e avaliar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Acurácia: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def run_clustering(df):
    """Modelo de clustering"""
    # Features para clustering
    X = df[['screen_time_hours', 'work_screen_hours', 'leisure_screen_hours', 'sleep_hours']]
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Adicionar clusters ao dataframe
    df['cluster'] = clusters
    
    # Avaliar
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Tamanho dos clusters: {pd.Series(clusters).value_counts().to_dict()}")