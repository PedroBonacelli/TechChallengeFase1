# Etapa 6 (parte 4) — Geração das explicações em linguagem natural
#
# Produz dois tipos de explicação:
#
#   1. POR CASO — pega um paciente do conjunto de teste, roda os 3 modelos
#      otimizados, levanta os fatores que pesaram na decisão e pede à LLM
#      um texto que o médico leia em segundos. É a "explicação do
#      diagnóstico de cada modelo" do enunciado.
#   2. DA OTIMIZAÇÃO — pega a tabela comparativa da Etapa 4 e pede uma
#      interpretação voltada a quem decide se troca o modelo em produção,
#      traduzindo pontos percentuais em número de pacientes. É a parte de
#      "transformar métricas em insights acionáveis".
#
# O PRINCÍPIO que organiza o módulo: a LLM nunca vê os dados brutos nem
# chama o modelo. Ela recebe um PAYLOAD JSON fechado, montado aqui em
# Python, com todos os números já calculados. Três razões:
#
#   - auditabilidade: toda afirmação numérica do texto tem que estar
#     rastreável a um campo do payload, e a Etapa 7 verifica isso
#     programaticamente;
#   - a LLM não é boa em aritmética e não precisa ser — quem calcula é o
#     sklearn;
#   - o payload fica salvo em disco, então uma explicação suspeita pode
#     ser reconferida meses depois contra a entrada exata que a gerou.
#
# SELEÇÃO DOS CASOS. Não são casos aleatórios. O módulo escolhe os que
# exercitam a explicação de verdade: discordância entre modelos, baixa
# confiança, e qualquer falso negativo (maligno que passou batido). São
# exatamente os casos em que uma explicação automática é útil — um caso
# com 99,9% de certeza e consenso dos três modelos não precisa de texto.
#
#   python -m src.llm.explain
#   python -m src.llm.explain --offline       # sem chamar a API
#   python -m src.llm.explain --n-casos 3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..ga.data import POS_LABEL, Phase1Data, load_phase1_data
from ..ga.search_space import SEARCH_SPACES
from ..monitoring import EventLogger, PredictionMonitor, setup_logging
from .client import LLMClient
from .offline import explicacao_caso_offline, explicacao_otimizacao_offline
from .prompts import (
    SYSTEM_PROMPT,
    USER_TEMPLATE_CASE,
    USER_TEMPLATE_OPTIMIZATION,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "fase2"
MODELS_DIR = BASE_DIR / "models"

MODELOS = ("logistic_regression", "decision_tree", "random_forest")
LIMIAR_CONFIANCA = 0.65  # abaixo disso, o caso é marcado pra revisão humana


@dataclass
class ModelBundle:
    """Um modelo pronto pra servir, com as métricas que o qualificam."""

    name: str
    estimator: Any
    features: str
    metrics: dict[str, Any]
    version: str

    def predict_case(self, data: Phase1Data, idx: int) -> dict[str, Any]:
        X = data.test_for(self.features).iloc[[idx]]
        rotulo = int(self.estimator.predict(X)[0])
        proba_mal = float(self.estimator.predict_proba(X)[0, POS_LABEL])
        return {
            "versao": self.version,
            "classe_prevista": "maligno" if rotulo == POS_LABEL else "benigno",
            "rotulo_previsto": rotulo,
            "proba_maligno": round(proba_mal, 4),
            "confianca": round(max(proba_mal, 1 - proba_mal), 4),
            "acuracia_teste_do_modelo": self.metrics["accuracy"],
            "recall_maligno_teste_do_modelo": self.metrics["recall_malignant"],
            "precisao_maligno_teste_do_modelo": self.metrics["precision_malignant"],
        }


def carregar_bundles(versao: str = "otimizado") -> dict[str, ModelBundle]:
    """Carrega os 3 modelos (originais ou otimizados) + métricas do teste."""
    detalhe_path = REPORTS_DIR / "comparison_detail.json"
    if not detalhe_path.exists():
        raise FileNotFoundError(
            f"{detalhe_path} não existe — rode primeiro: python -m src.ga.compare"
        )
    detalhe = json.loads(detalhe_path.read_text(encoding="utf-8"))

    bundles = {}
    for modelo in MODELOS:
        space = SEARCH_SPACES[modelo]
        if versao == "otimizado":
            caminho = MODELS_DIR / f"ga_{modelo}.joblib"
            chave = "teste_otimizado"
        else:
            caminho = MODELS_DIR / f"{modelo}.joblib"
            chave = "teste_original"
        bundles[modelo] = ModelBundle(
            name=modelo,
            estimator=joblib.load(caminho),
            features=space.features,
            metrics=detalhe[modelo][chave],
            version=f"{versao}_ga" if versao == "otimizado" else "fase1_original",
        )
    return bundles


# -----------------------------------------------------------------------
# Fatores que pesaram na decisão
# -----------------------------------------------------------------------


def fatores_do_caso(
    bundle_logistico: ModelBundle,
    data: Phase1Data,
    idx: int,
    top_n: int = 6,
) -> list[dict[str, Any]]:
    """Contribuição de cada feature, via SHAP exato do modelo linear.

    A Regressão Logística é linear, então a contribuição de cada feature
    pra uma predição individual tem forma fechada:

        phi_i = coef_i * (x_i - media_treino_i)

    que é o valor SHAP exato para modelos lineares (Lundberg & Lee, 2017)
    — a mesma conta que a Fase 1 já fazia em src/evaluation.py. Somando
    todos os phi_i com o intercepto, dá exatamente o log-odds previsto.

    Sinal: no sklearn, coef_ é o log-odds da classe 1 (benigno). Como
    aqui a classe de interesse é maligno (0), inverto o sinal, pra
    "positivo = empurra pra maligno" — igual à Fase 1.

    Uso o modelo linear como fonte de atribuição mesmo quando o
    diagnóstico de referência vem da floresta, porque é a única dos três
    cuja atribuição por caso é exata em vez de aproximada. O payload
    registra isso explicitamente (`metodo`) pra não dar a entender que a
    conta vale para os outros modelos.
    """
    modelo = bundle_logistico.estimator
    X_train = data.X_train_scaled
    x = data.X_test_scaled.iloc[idx]
    x_cru = data.X_test.iloc[idx]

    baseline = X_train.mean()
    coef = -pd.Series(modelo.coef_[0], index=X_train.columns)
    contribuicoes = coef * (x - baseline)

    ordem = contribuicoes.abs().sort_values(ascending=False).index[:top_n]
    fatores = []
    for feature in ordem:
        fatores.append({
            "feature": feature,
            "valor_medido": round(float(x_cru[feature]), 4),
            # z-score: o valor escalonado JÁ é (x - média_treino)/desvio_treino,
            # então ele é literalmente o z-score em relação ao treino
            "z_score": round(float(x[feature]), 3),
            "media_referencia_treino": round(float(data.X_train[feature].mean()), 4),
            "contribuicao_log_odds": round(float(contribuicoes[feature]), 4),
            "direcao": "maligno" if contribuicoes[feature] > 0 else "benigno",
        })
    return fatores


def importancias_floresta(
    bundle_floresta: ModelBundle, data: Phase1Data, idx: int, top_n: int = 5
) -> list[dict[str, Any]]:
    """Features mais importantes GLOBALMENTE na floresta, com o valor do caso.

    Importância de floresta é global (quanto a feature reduz impureza no
    conjunto todo), não por caso — por isso aparece no payload como
    contexto separado e nomeado assim, nunca misturada com as
    contribuições individuais do modelo linear.
    """
    est = bundle_floresta.estimator
    if not hasattr(est, "feature_importances_"):
        return []
    X = data.test_for(bundle_floresta.features)
    importancias = pd.Series(est.feature_importances_, index=X.columns)
    x = X.iloc[idx]
    escalonado = data.X_test_scaled.iloc[idx]

    return [
        {
            "feature": f,
            "importancia_global": round(float(importancias[f]), 4),
            "valor_medido": round(float(x[f]), 4),
            "z_score": round(float(escalonado[f]), 3),
        }
        for f in importancias.sort_values(ascending=False).index[:top_n]
    ]


# -----------------------------------------------------------------------
# Payloads
# -----------------------------------------------------------------------


def build_case_payload(
    bundles: dict[str, ModelBundle],
    data: Phase1Data,
    idx: int,
    motivo_selecao: str = "",
) -> dict[str, Any]:
    """Monta o JSON que a LLM recebe para um caso."""
    previsoes = {m: b.predict_case(data, idx) for m, b in bundles.items()}
    rotulos = {m: p["rotulo_previsto"] for m, p in previsoes.items()}
    votos_maligno = sum(1 for r in rotulos.values() if r == POS_LABEL)

    # modelo de referência: o de maior recall do maligno no teste, que é o
    # critério clínico adotado desde a Fase 1
    principal = max(
        bundles, key=lambda m: bundles[m].metrics["recall_malignant"]
    )
    rotulo_real = int(data.y_test.iloc[idx])

    return {
        "caso": {
            "id": int(data.y_test.index[idx]),
            "indice_no_teste": idx,
            "rotulo_real": "maligno" if rotulo_real == POS_LABEL else "benigno",
            "motivo_de_selecao": motivo_selecao,
        },
        "modelo_principal": principal,
        "previsoes": previsoes,
        "consenso": {
            "divergem": len(set(rotulos.values())) > 1,
            "votos_maligno": votos_maligno,
            "votos_benigno": len(rotulos) - votos_maligno,
        },
        "fatores_principais": fatores_do_caso(
            bundles["logistic_regression"], data, idx
        ),
        "metodo_dos_fatores": (
            "SHAP exato do modelo linear (coef_i * (x_i - media_treino_i)), "
            "calculado sobre a Regressão Logística otimizada. z_score é o "
            "desvio em relação à média do conjunto de treino."
        ),
        "features_mais_importantes_na_floresta": importancias_floresta(
            bundles["random_forest"], data, idx
        ),
        "contexto": {
            "dataset": "Breast Cancer Wisconsin (Diagnostic)",
            "n_features": int(data.X_train.shape[1]),
            "n_teste": int(len(data.y_test)),
            "n_malignos_teste": int((data.y_test == POS_LABEL).sum()),
            "prevalencia_maligno_treino": round(
                float((data.y_train == POS_LABEL).mean()), 4
            ),
            "limiar_confianca_para_revisao_humana": LIMIAR_CONFIANCA,
            "natureza_das_features": (
                "medidas morfométricas de núcleos celulares obtidas por "
                "punção aspirativa por agulha fina, em lâmina — não são "
                "sintomas nem achados de imagem"
            ),
        },
    }


def build_optimization_payload() -> dict[str, Any]:
    """Monta o JSON da interpretação da otimização, a partir da Etapa 4."""
    comparativo = pd.read_csv(REPORTS_DIR / "comparison.csv")
    detalhe = json.loads(
        (REPORTS_DIR / "comparison_detail.json").read_text(encoding="utf-8")
    )
    experimentos = json.loads(
        (REPORTS_DIR / "ga_experiments.json").read_text(encoding="utf-8")
    )

    n_teste = int(
        comparativo.iloc[0][[
            "verdadeiros_positivos_mal_original", "falsos_negativos_original",
            "falsos_positivos_original", "verdadeiros_negativos_original",
        ]].sum()
    )
    n_mal = int(
        comparativo.iloc[0]["verdadeiros_positivos_mal_original"]
        + comparativo.iloc[0]["falsos_negativos_original"]
    )

    linhas = []
    for _, r in comparativo.iterrows():
        linhas.append({
            "model": r["model"],
            "modelo": r["modelo"],
            "experimento_vencedor": r["experimento_vencedor"],
            "recall_mal_original": float(r["recall_malignant_original"]),
            "recall_mal_otimizado": float(r["recall_malignant_otimizado"]),
            "precisao_mal_original": float(r["precision_malignant_original"]),
            "precisao_mal_otimizado": float(r["precision_malignant_otimizado"]),
            "acuracia_original": float(r["accuracy_original"]),
            "acuracia_otimizado": float(r["accuracy_otimizado"]),
            "f1_mal_original": float(r["f1_malignant_original"]),
            "f1_mal_otimizado": float(r["f1_malignant_otimizado"]),
            "roc_auc_original": float(r["roc_auc_original"]),
            "roc_auc_otimizado": float(r["roc_auc_otimizado"]),
            "fn_original": int(r["falsos_negativos_original"]),
            "fn_otimizado": int(r["falsos_negativos_otimizado"]),
            "delta_falsos_negativos": int(r["delta_falsos_negativos"]),
            "fp_original": int(r["falsos_positivos_original"]),
            "fp_otimizado": int(r["falsos_positivos_otimizado"]),
            "delta_falsos_positivos": int(r["delta_falsos_positivos"]),
            "fitness_cv_original": float(r["fitness_cv_original"]),
            "fitness_cv_otimizado": float(r["fitness_cv_otimizado"]),
            "hiperparametros_fase1": detalhe[r["model"]]["params_original"],
            "hiperparametros_otimizados": detalhe[r["model"]]["params_otimizado"],
        })

    melhor_recall = max(linhas, key=lambda d: d["recall_mal_otimizado"])
    return {
        "n_teste": n_teste,
        "n_malignos_teste": n_mal,
        "funcao_objetivo_do_ga": (
            "0.60 * recall do maligno + 0.25 * F1 do maligno + "
            "0.15 * acurácia, medida por validação cruzada estratificada "
            "de 5 folds SOBRE O TREINO (o conjunto de teste não foi usado "
            "na otimização)"
        ),
        "orcamento_por_experimento": "600 avaliações de fitness",
        "experimentos": [
            {
                "nome": e["experiment"],
                "modelo": e["model"],
                "populacao": e["config"]["population_size"],
                "geracoes": e["config"]["n_generations"],
                "taxa_cruzamento": e["config"]["crossover_rate"],
                "taxa_mutacao": e["config"]["mutation_rate"],
                "selecao": e["config"]["selection"],
                "fitness_cv_obtido": round(e["best_fitness_cv"], 4),
            }
            for e in experimentos
        ],
        "comparativo": linhas,
        "recomendacao_por_recall": (
            f"{melhor_recall['modelo']} otimizado tem o maior recall do "
            f"maligno ({melhor_recall['recall_mal_otimizado']:.4f}), "
            f"deixando de detectar {melhor_recall['fn_otimizado']} de "
            f"{n_mal} casos malignos"
        ),
    }


# -----------------------------------------------------------------------
# Seleção dos casos
# -----------------------------------------------------------------------


def selecionar_casos(
    bundles: dict[str, ModelBundle], data: Phase1Data, n: int = 5
) -> list[tuple[int, str]]:
    """Escolhe os casos que mais exercitam a explicação.

    Ordem de prioridade (um caso só entra uma vez):
      1. falso negativo do modelo principal — o erro mais grave;
      2. discordância entre os 3 modelos;
      3. menor confiança do modelo principal (zona cinzenta);
      4. um maligno de alta confiança (caso de referência);
      5. um benigno de alta confiança (caso de referência).
    """
    principal = max(bundles, key=lambda m: bundles[m].metrics["recall_malignant"])
    bundle = bundles[principal]
    X = data.test_for(bundle.features)
    proba_mal = bundle.estimator.predict_proba(X)[:, POS_LABEL]
    pred = bundle.estimator.predict(X)
    y = data.y_test.to_numpy()

    rotulos_por_modelo = np.vstack([
        b.estimator.predict(data.test_for(b.features)).astype(int)
        for b in bundles.values()
    ])
    divergem = rotulos_por_modelo.min(axis=0) != rotulos_por_modelo.max(axis=0)
    confianca = np.maximum(proba_mal, 1 - proba_mal)

    escolhidos: list[tuple[int, str]] = []
    usados: set[int] = set()

    def adicionar(indices, motivo) -> None:
        for i in indices:
            i = int(i)
            if i not in usados and len(escolhidos) < n:
                usados.add(i)
                escolhidos.append((i, motivo))

    # 1. falsos negativos: maligno real previsto como benigno
    fn = np.where((y == POS_LABEL) & (pred != POS_LABEL))[0]
    adicionar(
        fn[np.argsort(-confianca[fn])],
        f"falso negativo do modelo {principal} — maligno classificado como benigno",
    )

    # 2. discordância entre modelos
    adicionar(
        np.where(divergem)[0][np.argsort(confianca[divergem])],
        "os 3 modelos divergem neste caso",
    )

    # 3. menor confiança
    adicionar(
        np.argsort(confianca),
        f"caso de menor confiança do modelo {principal} (zona de incerteza)",
    )

    # 4 e 5. referências de alta confiança, um de cada classe
    mal = np.where(y == POS_LABEL)[0]
    adicionar(
        mal[np.argsort(-proba_mal[mal])],
        "maligno de alta confiança — caso de referência",
    )
    ben = np.where(y != POS_LABEL)[0]
    adicionar(
        ben[np.argsort(proba_mal[ben])],
        "benigno de alta confiança — caso de referência",
    )
    return escolhidos


# -----------------------------------------------------------------------
# Geração
# -----------------------------------------------------------------------


def explicar_caso(client: LLMClient, payload: dict[str, Any]):
    return client.complete(
        system=SYSTEM_PROMPT,
        user=USER_TEMPLATE_CASE.format(
            payload=json.dumps(payload, indent=2, ensure_ascii=False)
        ),
        max_tokens=1200,
        tag=f"caso#{payload['caso']['id']}",
        offline_text=explicacao_caso_offline(payload),
    )


def explicar_otimizacao(client: LLMClient, payload: dict[str, Any]):
    return client.complete(
        system=SYSTEM_PROMPT,
        user=USER_TEMPLATE_OPTIMIZATION.format(
            payload=json.dumps(payload, indent=2, ensure_ascii=False)
        ),
        max_tokens=1600,
        tag="otimizacao",
        offline_text=explicacao_otimizacao_offline(payload),
    )


def escrever_markdown(
    registros: list[dict[str, Any]],
    otimizacao: dict[str, Any],
    client: LLMClient,
) -> Path:
    linhas = [
        "# Etapa 6 — Explicações geradas pela LLM",
        "",
        f"**Provedor:** {client.describe()}",
        "",
        "As explicações abaixo foram geradas a partir de payloads JSON "
        "fechados (salvos em `llm_payloads.json`), nunca dos dados brutos. "
        "Todo número citado no texto deve existir no payload "
        "correspondente — a Etapa 7 verifica isso automaticamente.",
        "",
        "## Interpretação da otimização por GA",
        "",
        "> *Público: equipe clínica / responsável técnico decidindo sobre "
        "adoção do modelo otimizado.*",
        "",
        otimizacao["explicacao"],
        "",
        "---",
        "",
        "## Explicações por caso",
        "",
    ]
    for r in registros:
        p = r["payload"]
        caso = p["caso"]
        linhas += [
            f"### Caso #{caso['id']} — rótulo real: {caso['rotulo_real']}",
            "",
            f"**Por que este caso foi selecionado:** {caso['motivo_de_selecao']}",
            "",
            "| Modelo | Sugestão | P(maligno) | Confiança |",
            "|---|---|---|---|",
        ]
        for modelo, prev in p["previsoes"].items():
            linhas.append(
                f"| {modelo} | {prev['classe_prevista']} "
                f"| {prev['proba_maligno']:.4f} | {prev['confianca']:.4f} |"
            )
        linhas += [
            "",
            f"*Divergência entre modelos: "
            f"{'sim' if p['consenso']['divergem'] else 'não'}*",
            "",
            "**Explicação gerada:**",
            "",
            r["explicacao"],
            "",
            "---",
            "",
        ]

    uso = client.usage_summary()
    linhas += [
        "## Consumo da LLM nesta execução",
        "",
        "```json",
        json.dumps(uso, indent=2, ensure_ascii=False),
        "```",
    ]

    caminho = REPORTS_DIR / "llm_explanations.md"
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    return caminho


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 6 — explicações via LLM")
    parser.add_argument("--n-casos", type=int, default=5)
    parser.add_argument(
        "--offline", action="store_true",
        help="usa o gerador por template, sem chamar a API",
    )
    parser.add_argument(
        "--versao", choices=("otimizado", "original"), default="otimizado",
        help="qual versão dos modelos explicar",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("fase2.llm", log_file="llm.log")
    events = EventLogger("llm_events.jsonl")

    client = LLMClient(force_offline=args.offline, events=events, logger=logger)
    logger.info("provedor da LLM: %s", client.describe())

    data = load_phase1_data()
    bundles = carregar_bundles(args.versao)
    principal = max(bundles, key=lambda m: bundles[m].metrics["recall_malignant"])
    monitor = PredictionMonitor(model_name=principal, events=events)

    # --- interpretação da otimização ---
    logger.info("gerando interpretação da otimização...")
    payload_otim = build_optimization_payload()
    resp_otim = explicar_otimizacao(client, payload_otim)

    # --- explicações por caso ---
    casos = selecionar_casos(bundles, data, args.n_casos)
    logger.info("casos selecionados: %s", [i for i, _ in casos])

    registros = []
    for idx, motivo in casos:
        payload = build_case_payload(bundles, data, idx, motivo)
        logger.info(
            "caso #%s (%s) — %s",
            payload["caso"]["id"], payload["caso"]["rotulo_real"], motivo,
        )
        resposta = explicar_caso(client, payload)

        prev_principal = payload["previsoes"][principal]
        monitor.record(
            case_id=payload["caso"]["id"],
            predicted_label=prev_principal["rotulo_previsto"],
            proba_malignant=prev_principal["proba_maligno"],
            true_label=int(data.y_test.iloc[idx]),
        )
        registros.append({
            "payload": payload,
            "explicacao": resposta.text,
            "telemetria": resposta.telemetry(),
        })

    monitor.log_snapshot(logger)

    # --- persistência ---
    (REPORTS_DIR / "llm_payloads.json").write_text(
        json.dumps(
            {
                "provedor": client.describe(),
                "otimizacao": {
                    "payload": payload_otim,
                    "explicacao": resp_otim.text,
                    "telemetria": resp_otim.telemetry(),
                },
                "casos": registros,
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    caminho_md = escrever_markdown(
        registros,
        {"explicacao": resp_otim.text, "telemetria": resp_otim.telemetry()},
        client,
    )

    logger.info("consumo: %s", json.dumps(client.usage_summary(), ensure_ascii=False))
    events.emit("llm_session_end", **client.usage_summary())
    events.close()

    logger.info("explicações em: %s", caminho_md.relative_to(BASE_DIR))
    logger.info("payloads em: %s", (REPORTS_DIR / 'llm_payloads.json').relative_to(BASE_DIR))
    logger.info("próximo passo: python -m src.llm.evaluate_quality")


if __name__ == "__main__":
    main()
