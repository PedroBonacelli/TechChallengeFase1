# Handler da Lambda de inferência (Etapa 11).
#
# Recebe as medidas morfométricas de um caso, roda os 3 modelos
# otimizados pelo GA, e devolve as previsões junto com a explicação em
# linguagem natural gerada pela LLM — o mesmo caminho de
# src/llm/explain.py, mas servido por requisição em vez de em lote.
#
# Decisões que importam num ambiente serverless:
#
# CACHE DE MÓDULO. Os modelos são carregados UMA vez, no escopo do módulo,
# e reaproveitados entre invocações enquanto o container estiver quente.
# Carregar 3 joblib a cada chamada colocaria segundos de latência em toda
# requisição em vez de só no cold start.
#
# DEGRADAÇÃO EM CAMADAS. Se a LLM falhar, a resposta sai com as previsões
# e a explicação por template, com `explicacao_degradada: true`. Se os
# modelos não carregarem, aí sim é erro 503 — sem modelo não há o que
# responder. A distinção importa: a previsão numérica é o produto; a
# explicação é a camada de apresentação.
#
# ENTRADA VALIDADA CONTRA A LISTA DE FEATURES DO MODELO, não contra um
# schema escrito à mão. Se o pré-processamento mudar e o modelo passar a
# esperar 21 features em vez de 22, a validação acompanha sozinha.
#
# Este módulo não é exercitado pela suíte de testes da Etapa 8: ele só
# roda dentro da Lambda, e a infraestrutura não foi aplicada (ver
# infra/README.md). A lógica que ele chama — montagem de payload,
# explicação, auditoria — é a mesma coberta por tests/test_integration.py.

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET = os.getenv("FASE2_S3_BUCKET", "")
SECRET_ID = os.getenv("FASE2_LLM_SECRET_ID", "")
# /tmp é o único diretório gravável na Lambda
CACHE_DIR = Path(os.getenv("FASE2_CACHE_DIR", "/tmp/modelos"))

MODELOS = ("logistic_regression", "decision_tree", "random_forest")
POS_LABEL = 0


def _emitir(event: str, **payload: Any) -> None:
    """Uma linha JSON no stdout.

    Os metric filters do CloudWatch (infra/observability.tf) leem
    exatamente este formato — é o que transforma log em métrica sem
    agente nenhum. O formato tem que casar com src/monitoring.py.
    """
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, default=str))


# -----------------------------------------------------------------------
# Carga (uma vez por container)
# -----------------------------------------------------------------------


def _baixar_do_s3(chave: str, destino: Path) -> Path:
    import boto3

    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        return destino
    boto3.client("s3").download_file(BUCKET, chave, str(destino))
    return destino


def _carregar_modelos() -> dict[str, Any]:
    """Carrega os 3 modelos otimizados e os metadados de avaliação."""
    import joblib

    t0 = time.perf_counter()
    bundles: dict[str, Any] = {}

    metadados = json.loads(
        _baixar_do_s3(
            "reports/fase2/comparison_detail.json",
            CACHE_DIR / "comparison_detail.json",
        ).read_text(encoding="utf-8")
    )

    for nome in MODELOS:
        caminho = _baixar_do_s3(
            f"models/ga_{nome}.joblib", CACHE_DIR / f"ga_{nome}.joblib"
        )
        bundles[nome] = {
            "estimator": joblib.load(caminho),
            "features": "scaled" if nome == "logistic_regression" else "raw",
            "metrics": metadados[nome]["teste_otimizado"],
        }

    scaler = joblib.load(
        _baixar_do_s3("models/scaler.joblib", CACHE_DIR / "scaler.joblib")
    )

    _emitir(
        "models_loaded",
        modelos=list(bundles),
        cold_start_s=round(time.perf_counter() - t0, 3),
    )
    return {"bundles": bundles, "scaler": scaler, "metadados": metadados}


def _carregar_chave_llm() -> str | None:
    """Lê a chave do Secrets Manager e a coloca no ambiente.

    O LLMClient resolve a chave por variável de ambiente; escrever nela
    aqui mantém o cliente ignorante de onde o segredo vive — o mesmo
    código roda local com .env e na Lambda com Secrets Manager.
    """
    if not SECRET_ID:
        return None
    try:
        import boto3

        bruto = boto3.client("secretsmanager").get_secret_value(
            SecretId=SECRET_ID
        )["SecretString"]
        chave = json.loads(bruto).get("ANTHROPIC_API_KEY")
        if chave and not chave.startswith("PREENCHER"):
            os.environ["ANTHROPIC_API_KEY"] = chave
            return chave
    except Exception as exc:  # noqa: BLE001
        logger.warning("nao foi possivel ler o segredo da LLM: %s", exc)
    return None


# Carga no import: acontece no cold start, fora do tempo da requisição.
# Falha aqui NÃO derruba o import — o handler responde 503 com a causa,
# que é muito mais diagnosticável que um erro de inicialização da Lambda.
_ESTADO: dict[str, Any] | None = None
_ERRO_CARGA: str | None = None

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    _ESTADO = _carregar_modelos()
    _carregar_chave_llm()
except Exception as exc:  # noqa: BLE001
    _ERRO_CARGA = f"{type(exc).__name__}: {exc}"
    logger.exception("falha ao carregar os modelos")


# -----------------------------------------------------------------------
# Validação da entrada
# -----------------------------------------------------------------------


def _validar_caso(corpo: dict[str, Any], features_esperadas: list[str]):
    """Confere as features contra o que o modelo treinado espera."""
    medidas = corpo.get("features")
    if not isinstance(medidas, dict):
        raise ValueError("campo 'features' obrigatorio, como objeto JSON")

    faltando = [f for f in features_esperadas if f not in medidas]
    if faltando:
        raise ValueError(f"features ausentes: {faltando}")

    sobrando = [f for f in medidas if f not in features_esperadas]
    if sobrando:
        raise ValueError(
            f"features desconhecidas: {sobrando}. "
            f"esperadas: {features_esperadas}"
        )

    valores = {}
    for f in features_esperadas:
        try:
            v = float(medidas[f])
        except (TypeError, ValueError):
            raise ValueError(f"feature '{f}' nao e numerica: {medidas[f]!r}")
        # medida morfométrica negativa não existe: raio, área, perímetro
        # são todos grandezas positivas
        if v < 0:
            raise ValueError(f"feature '{f}' negativa ({v}); medida invalida")
        valores[f] = v
    return valores


# -----------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------


def _resposta(status: int, corpo: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(corpo, ensure_ascii=False, default=str),
    }


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    if _ESTADO is None:
        _emitir("inference_unavailable", erro=_ERRO_CARGA)
        return _resposta(503, {
            "erro": "modelos indisponiveis",
            "detalhe": _ERRO_CARGA,
        })

    t0 = time.perf_counter()
    try:
        corpo = event.get("body") or "{}"
        corpo = json.loads(corpo) if isinstance(corpo, str) else corpo
    except json.JSONDecodeError as exc:
        return _resposta(400, {"erro": f"JSON invalido: {exc}"})

    import pandas as pd

    bundles = _ESTADO["bundles"]
    scaler = _ESTADO["scaler"]
    features = list(getattr(scaler, "feature_names_in_", []))
    if not features:
        features = list(bundles["decision_tree"]["estimator"].feature_names_in_)

    try:
        valores = _validar_caso(corpo, features)
    except ValueError as exc:
        _emitir("inference_rejected", motivo=str(exc))
        return _resposta(400, {"erro": str(exc)})

    case_id = corpo.get("case_id", "sem-id")
    X_cru = pd.DataFrame([valores], columns=features)
    X_esc = pd.DataFrame(scaler.transform(X_cru), columns=features)

    # --- previsões ---
    previsoes = {}
    for nome, b in bundles.items():
        X = X_esc if b["features"] == "scaled" else X_cru
        rotulo = int(b["estimator"].predict(X)[0])
        proba = float(b["estimator"].predict_proba(X)[0, POS_LABEL])
        previsoes[nome] = {
            "versao": "otimizado_ga",
            "classe_prevista": "maligno" if rotulo == POS_LABEL else "benigno",
            "rotulo_previsto": rotulo,
            "proba_maligno": round(proba, 4),
            "confianca": round(max(proba, 1 - proba), 4),
            "acuracia_teste_do_modelo": b["metrics"]["accuracy"],
            "recall_maligno_teste_do_modelo": b["metrics"]["recall_malignant"],
            "precisao_maligno_teste_do_modelo": b["metrics"]["precision_malignant"],
        }

    principal = max(bundles, key=lambda m: bundles[m]["metrics"]["recall_malignant"])
    rotulos = [p["rotulo_previsto"] for p in previsoes.values()]

    # Um evento `prediction` por requisição, no formato que os metric
    # filters consomem.
    _emitir(
        "prediction",
        model=principal,
        case_id=case_id,
        predicted_label=previsoes[principal]["rotulo_previsto"],
        predicted_class=previsoes[principal]["classe_prevista"],
        proba_malignant=previsoes[principal]["proba_maligno"],
        confidence=previsoes[principal]["confianca"],
        low_confidence=previsoes[principal]["confianca"] < 0.65,
        models_disagree=len(set(rotulos)) > 1,
    )

    # --- explicação (camada que pode degradar) ---
    explicacao, degradada, erro_llm = "", False, None
    if corpo.get("explicar", True):
        try:
            from src.llm.client import LLMClient
            from src.llm.explain import LIMIAR_CONFIANCA
            from src.llm.offline import explicacao_caso_offline
            from src.llm.prompts import SYSTEM_PROMPT, USER_TEMPLATE_CASE

            payload = {
                "caso": {
                    "id": case_id,
                    "rotulo_real": "desconhecido (caso novo)",
                    "motivo_de_selecao": "requisicao de inferencia",
                },
                "modelo_principal": principal,
                "previsoes": previsoes,
                "consenso": {
                    "divergem": len(set(rotulos)) > 1,
                    "votos_maligno": rotulos.count(POS_LABEL),
                    "votos_benigno": len(rotulos) - rotulos.count(POS_LABEL),
                },
                "fatores_principais": _fatores(
                    bundles["logistic_regression"]["estimator"],
                    X_esc, X_cru, features,
                ),
                "metodo_dos_fatores": (
                    "SHAP exato do modelo linear otimizado "
                    "(coef_i * (x_i - media_treino_i))"
                ),
                "contexto": {
                    "n_features": len(features),
                    "limiar_confianca_para_revisao_humana": LIMIAR_CONFIANCA,
                    "natureza_das_features": (
                        "medidas morfometricas de nucleos celulares em lamina"
                    ),
                },
            }
            resposta = LLMClient(logger=logger).complete(
                system=SYSTEM_PROMPT,
                user=USER_TEMPLATE_CASE.format(
                    payload=json.dumps(payload, indent=2, ensure_ascii=False)
                ),
                max_tokens=1200,
                tag=f"inferencia:{case_id}",
                offline_text=explicacao_caso_offline(payload),
            )
            explicacao = resposta.text
            degradada = resposta.provider == "offline"
            erro_llm = resposta.error
            _emitir(
                "llm_call",
                tag=f"inferencia:{case_id}",
                provider=resposta.provider,
                model=resposta.model,
                input_tokens=resposta.input_tokens,
                output_tokens=resposta.output_tokens,
                latency_s=round(resposta.latency_s, 3),
                custo_estimado_usd=round(resposta.custo_estimado_usd, 6),
                error=resposta.error,
            )
        except Exception as exc:  # noqa: BLE001
            degradada, erro_llm = True, f"{type(exc).__name__}: {exc}"
            explicacao = (
                "[explicacao em linguagem natural indisponivel; as "
                "previsoes acima permanecem validas]"
            )
            logger.exception("falha ao gerar a explicacao")

    return _resposta(200, {
        "case_id": case_id,
        "modelo_principal": principal,
        "previsoes": previsoes,
        "modelos_divergem": len(set(rotulos)) > 1,
        "requer_revisao_humana": (
            previsoes[principal]["confianca"] < 0.65 or len(set(rotulos)) > 1
        ),
        "explicacao": explicacao,
        "explicacao_degradada": degradada,
        "erro_llm": erro_llm,
        "aviso": (
            "Ferramenta de apoio a decisao. Nao substitui avaliacao "
            "do profissional de saude."
        ),
        "latencia_ms": round(1000 * (time.perf_counter() - t0), 1),
    })


def _fatores(modelo, X_esc, X_cru, features, top_n: int = 6):
    """Contribuições do modelo linear, como em src/llm/explain.py.

    Reimplementado aqui em vez de importado porque a versão de explain.py
    recebe um Phase1Data (com treino inteiro) e a Lambda só tem um caso.
    O baseline das features escalonadas é zero por construção — a média
    do treino é o que o StandardScaler usou pra centrar.
    """
    import numpy as np

    coef = -np.asarray(modelo.coef_[0])
    x = X_esc.iloc[0].to_numpy()
    contribuicoes = coef * x  # baseline (media do treino escalonada) = 0

    ordem = np.argsort(-np.abs(contribuicoes))[:top_n]
    return [
        {
            "feature": features[i],
            "valor_medido": round(float(X_cru.iloc[0, i]), 4),
            "z_score": round(float(x[i]), 3),
            "contribuicao_log_odds": round(float(contribuicoes[i]), 4),
            "direcao": "maligno" if contribuicoes[i] > 0 else "benigno",
        }
        for i in ordem
    ]
