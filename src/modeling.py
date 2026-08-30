# Modelagem — Breast Cancer Wisconsin (Diagnostic)
#
# Treina 3 classificadores (o mínimo pedido era 2):
#   - Regressão Logística (features escalonadas)
#   - Árvore de Decisão   (features cruas, invariante à escala)
#   - Random Forest       (features cruas)
#
# Assume que src/preprocessing.py já rodou e gerou os CSVs em data/processed/.
#
# python src/modeling.py

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42


def load_processed():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def train_models(X_train, X_train_scaled, y_train):
    # class_weight='balanced' nos três: maligno é a classe minoritária (~37%)
    # e é a que não pode passar batido, então o modelo paga mais caro por
    # errar ela — prioriza recall em vez de acurácia bruta.
    models = {}

    log_reg = LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
    )
    log_reg.fit(X_train_scaled, y_train)
    models["logistic_regression"] = (log_reg, "scaled")

    tree = DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
    )
    tree.fit(X_train, y_train)
    models["decision_tree"] = (tree, "raw")

    forest = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    models["random_forest"] = (forest, "raw")

    return models


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_processed()

    models = train_models(X_train, X_train_scaled, y_train)

    for name, (model, feature_type) in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        X_eval = X_train_scaled if feature_type == "scaled" else X_train
        train_acc = model.score(X_eval, y_train)
        print(f"{name:22s} | features: {feature_type:7s} | acc treino: {train_acc:.4f}")

    print(f"\nmodelos salvos em: {MODELS_DIR}")
    print("avaliação no teste: src/evaluation.py")


if __name__ == "__main__":
    main()
