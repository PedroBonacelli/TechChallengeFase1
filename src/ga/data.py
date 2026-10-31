# Carrega os dados já processados pela Fase 1.
#
# A Fase 2 NÃO reprocessa nada: usa exatamente os mesmos CSVs gerados por
# src/preprocessing.py (mesmo split, mesmo random_state=42, mesmas 20
# features após a poda de multicolinearidade). Isso é o que torna o
# comparativo "original vs. otimizado" da Etapa 4 honesto — se o
# pré-processamento mudasse, não daria pra saber se o ganho veio do GA ou
# de um split diferente.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# maligno = 0 no dataset original (sklearn/UCI) e é a classe positiva de
# interesse clínico — mesma convenção da Fase 1 (src/evaluation.py)
POS_LABEL = 0


@dataclass(frozen=True)
class Phase1Data:
    """Os dados da Fase 1, nas duas versões (crua e escalonada)."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def train_for(self, features: str) -> pd.DataFrame:
        return self.X_train_scaled if features == "scaled" else self.X_train

    def test_for(self, features: str) -> pd.DataFrame:
        return self.X_test_scaled if features == "scaled" else self.X_test

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    def describe(self) -> str:
        n_mal_tr = int((self.y_train == POS_LABEL).sum())
        n_mal_te = int((self.y_test == POS_LABEL).sum())
        return (
            f"treino: {len(self.X_train)} amostras "
            f"({n_mal_tr} malignos, {n_mal_tr / len(self.y_train):.1%}) | "
            f"teste: {len(self.X_test)} amostras "
            f"({n_mal_te} malignos, {n_mal_te / len(self.y_test):.1%}) | "
            f"{self.X_train.shape[1]} features"
        )


def load_phase1_data(processed_dir: Path = PROCESSED_DIR) -> Phase1Data:
    faltando = [
        nome
        for nome in (
            "X_train.csv", "X_test.csv", "X_train_scaled.csv",
            "X_test_scaled.csv", "y_train.csv", "y_test.csv",
        )
        if not (processed_dir / nome).exists()
    ]
    if faltando:
        raise FileNotFoundError(
            f"faltam arquivos em {processed_dir}: {faltando}\n"
            "rode primeiro: python src/load_data.py && python src/preprocessing.py"
        )

    read = lambda nome: pd.read_csv(processed_dir / nome)  # noqa: E731
    return Phase1Data(
        X_train=read("X_train.csv"),
        X_test=read("X_test.csv"),
        X_train_scaled=read("X_train_scaled.csv"),
        X_test_scaled=read("X_test_scaled.csv"),
        y_train=read("y_train.csv").squeeze("columns"),
        y_test=read("y_test.csv").squeeze("columns"),
    )


if __name__ == "__main__":
    print(load_phase1_data().describe())
