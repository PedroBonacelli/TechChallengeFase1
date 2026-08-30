# Baixa o dataset Breast Cancer Wisconsin (Diagnostic) e salva em data/raw/.
#
# Uso: python src/load_data.py
#
# Peguei a versão do sklearn em vez de baixar do Kaggle direto porque é o
# mesmo dataset (mesma fonte original, UCI) e assim não preciso de
# credenciais do Kaggle pra rodar isso em outra máquina.

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "breast_cancer_wisconsin.csv"


def load_breast_cancer_df() -> pd.DataFrame:
    # 30 features numéricas + target (0=maligno, 1=benigno) + diagnosis
    # (a mesma coisa que target, só que em texto, pra facilitar leitura/plots)
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = load_breast_cancer_df()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset salvo em: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(df["diagnosis"].value_counts())


if __name__ == "__main__":
    main()
