from scripts.data_cleaning import load_and_clean_data
from scripts.models import run_models

def main():
    print("=== INICIANDO PROJETO AASE ===")

    df = load_and_clean_data(save_clean_csv=True)

    print("Verificação final:")
    print(df.isnull().sum())
    print("Shape final:", df.shape)
    print("Resumo estatístico:")
    print(df.describe())

    run_models(df)

    print("=== PROJETO CONCLUÍDO ===")

if __name__ == "__main__":
    main()
