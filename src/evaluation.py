# Avaliação e explicabilidade — Breast Cancer Wisconsin (Diagnostic)
#
# Avalia os 3 modelos (src/modeling.py) no conjunto de TESTE: accuracy,
# precision, recall, F1, matriz de confusão, ROC/AUC.
#
# Explicabilidade em 3 camadas:
#   - feature importance nativa (coeficientes / feature_importances_)
#   - permutation importance (model-agnostic, medida no teste)
#   - explicação tipo-SHAP pra Regressão Logística: como o modelo é linear,
#     a contribuição de cada feature pra uma predição é
#     coef_i * (x_i - média_treino_i) — a fórmula exata do SHAP pra modelos
#     lineares (Lundberg & Lee, 2017). Calculei isso na mão porque o pacote
#     `shap` não instalou no sandbox usado no desenvolvimento (sem acesso ao
#     índice completo do PyPI); ele fica no requirements.txt pra quem for
#     rodar local/Docker.
#
# python src/evaluation.py

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"

sns.set_theme(style="whitegrid")

POS_LABEL = 0  # maligno é a classe positiva de interesse clínico
CLASS_NAMES = ["malignant (0)", "benign (1)"]

MODEL_SPECS = {
    "logistic_regression": "scaled",
    "decision_tree": "raw",
    "random_forest": "raw",
}


def load_test_data():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    return X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test


def load_models():
    return {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in MODEL_SPECS}


def evaluate_models(models, X_test, X_test_scaled, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        X_eval = X_test_scaled if MODEL_SPECS[name] == "scaled" else X_test
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, POS_LABEL]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=POS_LABEL)
        rec = recall_score(y_test, y_pred, pos_label=POS_LABEL)
        f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL)
        auc = roc_auc_score((y_test == POS_LABEL).astype(int), y_proba)

        rows.append({
            "model": name, "accuracy": acc,
            "precision_malignant": prec, "recall_malignant": rec,
            "f1_malignant": f1, "roc_auc": auc,
        })

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

        cm = confusion_matrix(y_test, y_pred)
        fn = cm[0, 1]  # maligno classificado como benigno, o erro mais grave
        print(f"falsos negativos: {fn}")

    return pd.DataFrame(rows).set_index("model").round(4)


def plot_confusion_matrices(models, X_test, X_test_scaled, y_test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, model) in zip(axes, models.items()):
        X_eval = X_test_scaled if MODEL_SPECS[name] == "scaled" else X_test
        y_pred = model.predict(X_eval)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(models, X_test, X_test_scaled, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    y_true_bin = (y_test == POS_LABEL).astype(int)
    for name, model in models.items():
        X_eval = X_test_scaled if MODEL_SPECS[name] == "scaled" else X_test
        y_proba = model.predict_proba(X_eval)[:, POS_LABEL]
        RocCurveDisplay.from_predictions(y_true_bin, y_proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="aleatório")
    ax.set_title("Curva ROC — classe 'maligno' como positiva")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_roc_curves.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(models, X_train, X_train_scaled) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    log_reg = models["logistic_regression"]
    coefs = pd.Series(log_reg.coef_[0], index=X_train_scaled.columns)
    coefs.reindex(coefs.abs().sort_values(ascending=True).index).tail(15).plot(
        kind="barh", ax=axes[0], color="#4C72B0"
    )
    axes[0].set_title("Regressão Logística\n(coeficientes, top 15)")

    for ax, name in zip(axes[1:], ["decision_tree", "random_forest"]):
        model = models[name]
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        importances.sort_values().tail(15).plot(kind="barh", ax=ax, color="#55A868")
        ax.set_title(f"{name}\n(feature importance, top 15)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_feature_importance.png", dpi=150)
    plt.close(fig)


def plot_permutation_importance(models, X_test, X_test_scaled, y_test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, (name, model) in zip(axes, models.items()):
        X_eval = X_test_scaled if MODEL_SPECS[name] == "scaled" else X_test
        result = permutation_importance(
            model, X_eval, y_test, n_repeats=20, random_state=42, scoring="f1_macro"
        )
        importances = pd.Series(result.importances_mean, index=X_eval.columns)
        importances.sort_values().tail(15).plot(kind="barh", ax=ax, color="#C44E52")
        ax.set_title(f"{name}\n(permutation importance, top 15)")
    fig.suptitle("Permutation importance (model-agnostic, no teste)", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def linear_shap_explanation(model, X_train_scaled, X_test_scaled, y_test, n_examples=4):
    """SHAP-like pra Regressão Logística.

    Modelo linear: f(x) = intercept + soma(coef_i * x_i). O valor SHAP exato
    de cada feature, assumindo independência, é phi_i = coef_i * (x_i - E[x_i]),
    com E[x_i] = média da feature no treino (o "baseline"). Somando todos os
    phi_i com o intercepto, dá exatamente o log-odds que o modelo prevê —
    é a mesma conta que o shap.LinearExplainer faz.

    Sobre o sinal: no sklearn, classes_ = [0, 1] e coef_ é o log-odds da
    classe 1 (benigno). Como aqui tratamos 'maligno' (0) como a classe
    positiva, inverto o sinal — phi_i = -coef_i * (x_i - E[x_i]) — pra
    positivo empurrar pra 'maligno' e negativo pra 'benigno', igual ao
    resto da análise.
    """
    baseline = X_train_scaled.mean()
    coef = -pd.Series(model.coef_[0], index=X_train_scaled.columns)

    # pega os 2 malignos e 2 benignos com maior confiança de predição
    proba = model.predict_proba(X_test_scaled)[:, POS_LABEL]
    idx_malignant = y_test[y_test == POS_LABEL].index
    idx_benign = y_test[y_test == 1].index
    chosen = list(pd.Series(proba, index=y_test.index).loc[idx_malignant].sort_values(ascending=False).head(2).index) + \
             list(pd.Series(proba, index=y_test.index).loc[idx_benign].sort_values(ascending=True).head(2).index)

    explanations = {}
    for i in chosen:
        x = X_test_scaled.loc[i]
        contributions = coef * (x - baseline)
        top_contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(8)
        explanations[i] = {
            "true_label": CLASS_NAMES[0 if y_test.loc[i] == 0 else 1],
            "predicted_proba_malignant": float(proba[list(y_test.index).index(i)]) if i in y_test.index else None,
            "top_feature_contributions": top_contributions,
        }
    return explanations


def print_shap_like_explanations(explanations) -> None:
    print("\nexplicabilidade (SHAP-like) — Regressão Logística")
    print("(contribuição de cada feature pro log-odds de 'maligno';")
    print(" positivo empurra a predição pra 'maligno')")
    for idx, info in explanations.items():
        print(f"\ncaso #{idx} | rótulo real: {info['true_label']}")
        print(info["top_feature_contributions"].round(3).to_string())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test = load_test_data()
    models = load_models()

    metrics_df = evaluate_models(models, X_test, X_test_scaled, y_test)
    print("\nresumo comparativo (teste)")
    print(metrics_df.to_string())
    metrics_df.to_csv(BASE_DIR / "reports" / "model_comparison.csv")

    plot_confusion_matrices(models, X_test, X_test_scaled, y_test)
    plot_roc_curves(models, X_test, X_test_scaled, y_test)
    plot_feature_importance(models, X_train, X_train_scaled)
    plot_permutation_importance(models, X_test, X_test_scaled, y_test)

    explanations = linear_shap_explanation(
        models["logistic_regression"], X_train_scaled, X_test_scaled, y_test
    )
    print_shap_like_explanations(explanations)

    print(f"\nfiguras e métricas em: {FIG_DIR} e reports/model_comparison.csv")


if __name__ == "__main__":
    main()
