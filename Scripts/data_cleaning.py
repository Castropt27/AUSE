# scripts/data_cleaning.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data():
    """
    Carrega e limpa o dataset ScreenTime vs MentalWellness
    """
    # Carregar dados
    df = pd.read_csv('../Data/ScreenTime vs MentalWellness.csv')
    
    # 1. Verificar e tratar valores missing
    print("Valores missing antes da limpeza:")
    print(df.isnull().sum())
    
    # Remover duplicados se existirem
    df = df.drop_duplicates()
    
    # 2. Codificar variáveis categóricas
    categorical_cols = ['gender', 'occupation', 'work_mode']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # 3. Criar novas features (exemplo)
    df['total_screen_time'] = df['work_screen_hours'] + df['leisure_screen_hours']
    df['work_leisure_ratio'] = df['work_screen_hours'] / (df['leisure_screen_hours'] + 0.001)  # evitar divisão por zero
    
    print(f"Dados limpos. Shape: {df.shape}")
    return df

def prepare_features_targets(df):
    """
    Prepara features e targets para modelação
    """
    # Features (X)
    feature_cols = ['age', 'gender', 'occupation', 'work_mode', 
                   'screen_time_hours', 'work_screen_hours', 'leisure_screen_hours',
                   'sleep_hours', 'exercise_minutes_per_week', 'social_hours_per_week']
    
    X = df[feature_cols]
    
    # Targets (y) para diferentes modelos
    targets = {
        'regression_sleep': df['sleep_quality_1_5'],
        'regression_stress': df['stress_level_0_10'],
        'regression_wellness': df['mental_wellness_index_0_100']
    }
    
    return X, targets