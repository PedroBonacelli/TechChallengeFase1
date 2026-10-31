# Fixtures compartilhadas da suíte da Fase 2.
#
# A fixture de dados NÃO lê os CSVs de data/processed/. Ela monta um
# Phase1Data a partir do dataset que vem dentro do sklearn, com uma
# amostra pequena. Isso é deliberado: a suíte precisa rodar numa máquina
# limpa, ou no CI, sem que `preprocessing.py` tenha sido executado antes.
# Um teste que depende de artefato gerado por outra etapa não testa o
# código — testa se alguém lembrou de rodar o pipeline na ordem certa.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ga.data import Phase1Data


@pytest.fixture(scope="session")
def data() -> Phase1Data:
    """Amostra reduzida do Breast Cancer Wisconsin, com o mesmo formato
    que a Fase 1 produz (X cru + X escalonado, maligno = 0)."""
    bruto = load_breast_cancer(as_frame=True)
    X_full, y_full = bruto.data, bruto.target

    # 8 features e 180 amostras: suficiente pra um GA rodar de verdade e
    # pra as métricas de maligno fazerem sentido, rápido o bastante pra
    # a suíte inteira terminar em segundos
    X_full = X_full.iloc[:, :8]
    X_full, _, y_full, _ = train_test_split(
        X_full, y_full, train_size=180, random_state=0, stratify=y_full
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.25, random_state=42, stratify=y_full
    )
    scaler = StandardScaler().fit(X_train)
    escalar = lambda X: pd.DataFrame(  # noqa: E731
        scaler.transform(X), columns=X.columns, index=X.index
    )
    return Phase1Data(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        X_train_scaled=escalar(X_train).reset_index(drop=True),
        X_test_scaled=escalar(X_test).reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Generator semeado — todo teste que usa aleatoriedade é determinístico."""
    return np.random.default_rng(1234)
