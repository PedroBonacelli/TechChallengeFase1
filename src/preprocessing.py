# Pré-processamento — Breast Cancer Wisconsin (Diagnostic)
#
# 1. carrega os dados brutos e faz limpeza defensiva
# 2. tira features redundantes (multicolinearidade) olhando a correlação
# 3. separa X/y, faz o split treino/teste e escalona
# 4. salva tudo em data/processed/, models/ e reports/
#
# python src/preprocessing.py

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "breast_cancer_wisconsin.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TARGET_COL = "target"          # 0 = maligno, 1 = benigno
DROP_COLS = ["diagnosis"]      # já temos target, isso aqui é só redundante
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 0.92 separou bem os grupos que realmente são a mesma informação
# (radius/perimeter/area e o par concavity/concave points) sem exagerar
# na poda de features que só se parecem um pouco.
CORR_THRESHOLD = 0.92


def load_and_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_before = len(df)

    # o dataset já vem limpo, mas deixo isso aqui pro pipeline aguentar
    # dados "mais sujos" se um dia vier de outra fonte
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include="number").columns
    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # medida física negativa não faz sentido (radius, area etc não podem ser < 0)
    physical_cols = [c for c in numeric_cols if c not in (TARGET_COL,)]
    invalid_mask = (df[physical_cols] < 0).any(axis=1)
    df = df.loc[~invalid_mask].copy()

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        print(f"colunas vazias removidas: {empty_cols}")

    n_after = len(df)
    print(f"linhas removidas na limpeza: {n_before - n_after}")
    print(f"valores ausentes tratados: {n_missing}")

    proporcao = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100
    print(f"maligno: {proporcao.loc[0]:.2f}% | benigno: {proporcao.loc[1]:.2f}%")

    return df


def high_corr_pairs(corr: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """Pares com |correlação| >= threshold, do maior pro menor."""
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                pairs.append((cols[i], cols[j], value))
    return sorted(pairs, key=lambda p: -abs(p[2]))


def drop_redundant_features(
    df: pd.DataFrame, threshold: float = CORR_THRESHOLD
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove features redundantes por multicolinearidade.

    Duas colunas quase perfeitamente correlacionadas carregam a mesma
    informação — manter as duas não ajuda e ainda atrapalha: na Logística
    os coeficientes ficam instáveis (o peso se divide entre elas), e no
    Random Forest a importância de uma variável relevante se dilui entre
    as cópias. radius/perimeter/area, por exemplo, são três jeitos de medir
    o mesmo tamanho do núcleo — a redundância aqui é geométrica.

    O processo é guloso: a cada passo elimina o par mais correlacionado
    que ainda sobrou (mantendo a feature com maior correlação com o alvo),
    recalcula e repete até não sobrar nenhum par acima do threshold.
    """
    features = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    target_corr = features.corrwith(df[TARGET_COL]).abs()

    remaining = list(features.columns)
    decisions = []

    while True:
        corr = features[remaining].corr()
        pairs = high_corr_pairs(corr, threshold)
        if not pairs:
            break

        a, b, value = pairs[0]
        kept, dropped = (a, b) if target_corr[a] >= target_corr[b] else (b, a)
        remaining.remove(dropped)
        decisions.append({
            "feature_removida": dropped,
            "correlacionada_com": kept,
            "corr_entre_features": round(value, 4),
            "corr_removida_vs_alvo": round(target_corr[dropped], 4),
            "corr_mantida_vs_alvo": round(target_corr[kept], 4),
        })

    report = pd.DataFrame(decisions)

    print(f"\nremoção de redundância (|corr| >= {threshold})")
    if report.empty:
        print("nenhum par acima do threshold, nada removido.")
    else:
        for row in decisions:
            print(f"  - '{row['feature_removida']}' saiu (corr "
                  f"{row['corr_entre_features']:+.3f} com '{row['correlacionada_com']}'; "
                  f"alvo: {row['corr_removida_vs_alvo']:.3f} vs "
                  f"{row['corr_mantida_vs_alvo']:.3f} da que ficou)")
    print(f"features: {features.shape[1]} -> {len(remaining)} "
          f"({features.shape[1] - len(remaining)} removidas)")

    kept_cols = [c for c in df.columns if c in remaining or c in DROP_COLS + [TARGET_COL]]
    return df[kept_cols].copy(), report


def build_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    y = df[TARGET_COL]
    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    # scaler só vê o treino, senão vaza informação do teste
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    df, corr_report = drop_redundant_features(df)

    X, y = build_features_target(df)
    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, scaler) = split_and_scale(X, y)

    print(f"\ntreino: {X_train.shape[0]} | teste: {X_test.shape[0]}")
    print(f"proporção maligno — treino: {(y_train == 0).mean():.3f} | "
          f"teste: {(y_test == 0).mean():.3f}")

    # versão crua (árvores) e escalonada (Logística/KNN)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    X_train_scaled.to_csv(PROCESSED_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test_scaled.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    corr_report.to_csv(REPORTS_DIR / "features_removidas.csv", index=False)

    # transposto porque o terminal trunca as 30 colunas
    X.describe().T.to_csv(REPORTS_DIR / "estatisticas_descritivas.csv")

    print(f"\nprocessado salvo em: {PROCESSED_DIR}")
    print(f"scaler em: {MODELS_DIR / 'scaler.joblib'}")
    print(f"features removidas em: {REPORTS_DIR / 'features_removidas.csv'}")


if __name__ == "__main__":
    main()
