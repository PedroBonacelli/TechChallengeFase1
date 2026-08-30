# Decisões de pré-processamento — Breast Cancer Wisconsin
#
# O teste tem só 114 amostras: 2 ou 3 acertos a mais já mexem ~2 pontos
# percentuais nas métricas. Então julgar uma decisão de pré-processamento
# com um único split é basicamente julgar em cima de ruído.
#
# Aqui meço o efeito isolado das duas decisões do preprocessing.py/modeling.py:
#   1. tirar as 8 features redundantes (30 -> 22)
#   2. usar class_weight='balanced'
#
# As 4 combinações passam por validação cruzada estratificada repetida
# (10 folds x 3 repetições = 30 medições cada), com média e desvio entre
# folds. O desvio importa: diferença de média menor que ele não quer dizer
# nada, é só sorteio de amostra.
#
# O scaler entra dentro do Pipeline (não é ajustado uma vez fora) porque
# assim ele é refeito em cada fold só com o treino daquele fold — se
# ajustasse antes, o fold de validação vazaria informação pra métrica.
#
# python src/ablation.py

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_features_target, drop_redundant_features, load_and_clean

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
N_SPLITS = 10
N_REPEATS = 3
POS_LABEL = 0  # maligno

# maligno é a classe 0 (não a 1, que é o default do sklearn), por isso
# precision/recall/f1 precisam de pos_label explícito
SCORING = {
    "recall_mal": make_scorer(recall_score, pos_label=POS_LABEL),
    "precision_mal": make_scorer(precision_score, pos_label=POS_LABEL),
    "f1_mal": make_scorer(f1_score, pos_label=POS_LABEL),
    "accuracy": "accuracy",
}


def montar_modelos(class_weight):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight=class_weight, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, class_weight=class_weight, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight=class_weight,
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def avaliar(X, y, nome_features: str) -> list[dict]:
    # mesmo cv reutilizado em todas as configs, pra comparação ser pareada
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    linhas = []
    for class_weight in [None, "balanced"]:
        for nome_modelo, clf in montar_modelos(class_weight).items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            linhas.append({
                "modelo": nome_modelo,
                "features": nome_features,
                "class_weight": class_weight or "none",
                **{m: round(r[f"test_{m}"].mean(), 4) for m in SCORING},
                "recall_std": round(r["test_recall_mal"].std(), 4),
            })
    return linhas


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    X_completo, y = build_features_target(df)
    df_reduzido, _ = drop_redundant_features(df)
    X_reduzido, _ = build_features_target(df_reduzido)

    print(f"\n{N_SPLITS} folds x {N_REPEATS} repetições = "
          f"{N_SPLITS * N_REPEATS} medições por configuração\n")

    resultados = pd.DataFrame(
        avaliar(X_completo, y, f"{X_completo.shape[1]} features")
        + avaliar(X_reduzido, y, f"{X_reduzido.shape[1]} features")
    )

    pd.set_option("display.width", 200)
    for modelo in resultados["modelo"].unique():
        print(f"--- {modelo} ---")
        print(resultados[resultados["modelo"] == modelo]
              .drop(columns="modelo").to_string(index=False))
        print()

    destino = REPORTS_DIR / "ablacao_preprocessamento.csv"
    resultados.to_csv(destino, index=False)
    print(f"tabela salva em: {destino}")
    print("leitura completa: seção 6.1 do reports/RELATORIO_TECNICO.md")


if __name__ == "__main__":
    main()
