# Main.py
from scripts.data_cleaning import load_and_clean_data
from scripts.models import run_models
import pandas as pd

def main():
    print("=== INICIANDO PROJETO AASE ===")
    
    # 1. Carregar e limpar dados
    print("1. Carregando e limpando dados...")
    df = load_and_clean_data()
    
    # 2. Executar modelos
    print("2. Executando modelos...")
    run_models(df)
    
    print("=== PROJETO CONCLUÍDO ===")

if __name__ == "__main__":
    main()