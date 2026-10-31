# Testes de integração — pipeline completo GA -> modelo -> LLM (Etapa 8).
#
# Os testes anteriores cobrem cada peça isolada. Estes cobrem as COSTURAS,
# que é onde os bugs de verdade moram: o fenótipo que o GA devolve
# alimenta `build_estimator` sem ajuste manual? O modelo treinado produz
# um payload que a camada de explicação aceita? A explicação resultante
# passa na auditoria da Etapa 7?
#
# Sem rede: tudo com `force_offline=True`.

from __future__ import annotations

import numpy as np
import pytest

from src.ga.compare import metricas_no_teste
from src.ga.data import POS_LABEL
from src.ga.engine import GAConfig, run_ga
from src.ga.fitness import build_estimator
from src.ga.search_space import SEARCH_SPACES
from src.llm.client import LLMClient
from src.llm.evaluate_quality import SECOES_CASO, checar
from src.llm.explain import (
    ModelBundle,
    build_case_payload,
    explicar_caso,
    fatores_do_caso,
    importancias_floresta,
    selecionar_casos,
)
from src.monitoring import EventLogger, PredictionMonitor, read_events, setup_logging

CONFIG = GAConfig(name="integracao", population_size=8, n_generations=3, seed=21)


@pytest.fixture(scope="module")
def pipeline(request):
    """Roda o GA nos 3 modelos e devolve os bundles treinados.

    Escopo de módulo: o GA roda uma vez e os testes compartilham o
    resultado. São 3 execuções curtas; repetir por teste multiplicaria o
    tempo da suíte sem cobrir nada novo.
    """
    data = request.getfixturevalue("data")
    bundles = {}
    resultados = {}
    for modelo in SEARCH_SPACES:
        resultado = run_ga(modelo, data, CONFIG)
        space = SEARCH_SPACES[modelo]

        # a costura testada aqui: best_params vai direto pro estimador
        est = build_estimator(modelo, resultado.best_params)
        est.fit(data.train_for(space.features), data.y_train)

        bundles[modelo] = ModelBundle(
            name=modelo,
            estimator=est,
            features=space.features,
            metrics=metricas_no_teste(
                est, data.test_for(space.features), data.y_test
            ),
            version="otimizado_ga",
        )
        resultados[modelo] = resultado
    return {"bundles": bundles, "resultados": resultados, "data": data}


# --- GA -> modelo -----------------------------------------------------


def test_fenotipo_do_ga_alimenta_o_estimador_sem_ajuste(pipeline):
    """A ponte mais importante: decode() -> build_estimator() -> fit()."""
    for modelo, bundle in pipeline["bundles"].items():
        assert hasattr(bundle.estimator, "predict_proba")
        params_ga = pipeline["resultados"][modelo].best_params
        # todo hiperparâmetro que o GA escolheu está no estimador treinado
        for chave, valor in params_ga.items():
            assert bundle.estimator.get_params()[chave] == valor


def test_modelo_otimizado_preve_as_duas_classes(pipeline):
    data = pipeline["data"]
    for bundle in pipeline["bundles"].values():
        pred = bundle.estimator.predict(data.test_for(bundle.features))
        assert set(np.unique(pred)) <= {0, 1}
        assert len(pred) == len(data.y_test)


def test_metricas_no_teste_sao_coerentes(pipeline):
    """A matriz de confusão tem que fechar com o total de amostras."""
    data = pipeline["data"]
    n = len(data.y_test)
    n_mal = int((data.y_test == POS_LABEL).sum())
    for bundle in pipeline["bundles"].values():
        m = bundle.metrics
        assert (
            m["verdadeiros_positivos_mal"] + m["falsos_negativos"]
            + m["falsos_positivos"] + m["verdadeiros_negativos"] == n
        )
        # malignos reais = detectados + perdidos
        assert m["verdadeiros_positivos_mal"] + m["falsos_negativos"] == n_mal
        assert 0.0 <= m["recall_malignant"] <= 1.0
        assert 0.0 <= m["roc_auc"] <= 1.0


def test_recall_do_maligno_bate_com_a_matriz_de_confusao(pipeline):
    for bundle in pipeline["bundles"].values():
        m = bundle.metrics
        tp, fn = m["verdadeiros_positivos_mal"], m["falsos_negativos"]
        assert m["recall_malignant"] == pytest.approx(tp / (tp + fn), abs=1e-4)


# --- modelo -> payload ------------------------------------------------


def test_contribuicoes_lineares_somam_o_log_odds(pipeline):
    """phi_i = coef_i * (x_i - media) tem que reconstruir a predição.

    É a validação de que a atribuição por caso é o SHAP exato do modelo
    linear e não uma heurística: a soma de todas as contribuições mais o
    intercepto é o log-odds que o modelo prevê.
    """
    data = pipeline["data"]
    bundle = pipeline["bundles"]["logistic_regression"]
    modelo = bundle.estimator
    idx = 0

    x = data.X_test_scaled.iloc[idx]
    baseline = data.X_train_scaled.mean()
    coef = -np.asarray(modelo.coef_[0])
    phi_total = float(np.dot(coef, (x - baseline).to_numpy()))

    # log-odds de maligno = -(log-odds de benigno)
    log_odds_benigno = float(
        modelo.decision_function(data.X_test_scaled.iloc[[idx]])[0]
    )
    intercepto_maligno = -float(modelo.intercept_[0])
    baseline_maligno = intercepto_maligno + float(np.dot(coef, baseline.to_numpy()))

    assert phi_total + baseline_maligno == pytest.approx(
        -log_odds_benigno, abs=1e-6
    )


def test_fatores_do_caso_ordena_por_relevancia(pipeline):
    fatores = fatores_do_caso(
        pipeline["bundles"]["logistic_regression"], pipeline["data"], 0, top_n=5
    )
    assert len(fatores) == 5
    magnitudes = [abs(f["contribuicao_log_odds"]) for f in fatores]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for f in fatores:
        assert f["direcao"] in ("maligno", "benigno")
        # o sinal da contribuição e a direção declarada têm que concordar
        assert (f["contribuicao_log_odds"] > 0) == (f["direcao"] == "maligno")


def test_z_score_do_payload_e_a_feature_escalonada(pipeline):
    """O X escalonado da Fase 1 JÁ é (x - média_treino)/desvio_treino."""
    data = pipeline["data"]
    fatores = fatores_do_caso(
        pipeline["bundles"]["logistic_regression"], data, 3, top_n=3
    )
    for f in fatores:
        esperado = float(data.X_test_scaled.iloc[3][f["feature"]])
        assert f["z_score"] == pytest.approx(esperado, abs=1e-3)


def test_importancias_da_floresta_somam_um(pipeline):
    imp = importancias_floresta(
        pipeline["bundles"]["random_forest"], pipeline["data"], 0, top_n=99
    )
    assert imp
    assert sum(i["importancia_global"] for i in imp) == pytest.approx(1.0, abs=1e-3)
    valores = [i["importancia_global"] for i in imp]
    assert valores == sorted(valores, reverse=True)


def test_payload_do_caso_tem_toda_a_estrutura(pipeline):
    payload = build_case_payload(
        pipeline["bundles"], pipeline["data"], 0, "teste de integração"
    )
    for chave in (
        "caso", "modelo_principal", "previsoes", "consenso",
        "fatores_principais", "metodo_dos_fatores", "contexto",
    ):
        assert chave in payload

    assert set(payload["previsoes"]) == set(SEARCH_SPACES)
    assert payload["modelo_principal"] in SEARCH_SPACES
    assert payload["caso"]["motivo_de_selecao"] == "teste de integração"
    assert payload["caso"]["rotulo_real"] in ("maligno", "benigno")


def test_consenso_do_payload_bate_com_as_previsoes(pipeline):
    for idx in range(min(10, len(pipeline["data"].y_test))):
        payload = build_case_payload(pipeline["bundles"], pipeline["data"], idx)
        rotulos = [p["rotulo_previsto"] for p in payload["previsoes"].values()]
        assert payload["consenso"]["divergem"] == (len(set(rotulos)) > 1)
        assert payload["consenso"]["votos_maligno"] == rotulos.count(POS_LABEL)
        assert (
            payload["consenso"]["votos_maligno"]
            + payload["consenso"]["votos_benigno"] == len(rotulos)
        )


def test_probabilidade_e_classe_prevista_sao_consistentes(pipeline):
    """P(maligno) > 0.5 tem que corresponder a classe prevista 'maligno'."""
    for idx in range(min(8, len(pipeline["data"].y_test))):
        payload = build_case_payload(pipeline["bundles"], pipeline["data"], idx)
        for prev in payload["previsoes"].values():
            assert (prev["proba_maligno"] > 0.5) == (
                prev["classe_prevista"] == "maligno"
            )
            assert prev["confianca"] == pytest.approx(
                max(prev["proba_maligno"], 1 - prev["proba_maligno"]), abs=1e-4
            )


def test_selecao_de_casos_nao_repete_e_prioriza_os_dificeis(pipeline):
    casos = selecionar_casos(pipeline["bundles"], pipeline["data"], n=5)
    indices = [i for i, _ in casos]
    assert len(indices) == len(set(indices)), "o mesmo caso foi escolhido 2x"
    assert len(indices) <= 5
    for _, motivo in casos:
        assert motivo, "todo caso selecionado tem que declarar o motivo"


def test_selecao_respeita_o_limite_pedido(pipeline):
    assert len(selecionar_casos(pipeline["bundles"], pipeline["data"], n=2)) == 2
    assert len(selecionar_casos(pipeline["bundles"], pipeline["data"], n=1)) == 1


# --- payload -> LLM -> auditoria --------------------------------------


def test_pipeline_completo_gera_explicacao_auditavel(pipeline):
    """GA -> modelo -> payload -> explicação -> auditoria, ponta a ponta."""
    client = LLMClient(force_offline=True)
    casos = selecionar_casos(pipeline["bundles"], pipeline["data"], n=3)

    for idx, motivo in casos:
        payload = build_case_payload(
            pipeline["bundles"], pipeline["data"], idx, motivo
        )
        resposta = explicar_caso(client, payload)

        assert resposta.text
        assert resposta.provider == "offline"
        for secao in SECOES_CASO:
            assert secao in resposta.text

        checagem = checar(resposta.text, payload)
        assert checagem.numeros["fidelidade_numerica_ok"], (
            f"caso {idx}: números sem respaldo "
            f"{checagem.numeros['numeros_inventados']}"
        )
        assert checagem.linguagem["n_linguagem_conclusiva"] == 0
        assert checagem.formato["formato_ok"]
        assert checagem.nota >= 4.5


def test_explicacao_cita_a_probabilidade_do_caso(pipeline):
    """A explicação tem que conter o número que veio do modelo."""
    client = LLMClient(force_offline=True)
    payload = build_case_payload(pipeline["bundles"], pipeline["data"], 0)
    texto = explicar_caso(client, payload).text

    principal = payload["modelo_principal"]
    proba = payload["previsoes"][principal]["proba_maligno"]
    assert f"{100 * proba:.1f}".replace(".", ",") in texto


# --- monitoramento ----------------------------------------------------


def test_monitor_de_previsoes_registra_e_resume(pipeline, tmp_path, monkeypatch):
    """O log estruturado da Etapa 5 tem que ser relido corretamente."""
    import src.monitoring as monitoring

    monkeypatch.setattr(monitoring, "LOGS_DIR", tmp_path)
    logger = setup_logging("teste.monitor", log_file=None)

    events = EventLogger("teste_previsoes.jsonl", run_id="run-teste")
    monitor = PredictionMonitor(model_name="random_forest", events=events)

    data = pipeline["data"]
    bundle = pipeline["bundles"]["random_forest"]
    X = data.test_for(bundle.features)
    proba = bundle.estimator.predict_proba(X)[:, POS_LABEL]
    pred = bundle.estimator.predict(X)

    for i in range(len(X)):
        monitor.record(
            case_id=i, predicted_label=int(pred[i]),
            proba_malignant=float(proba[i]), true_label=int(data.y_test.iloc[i]),
        )

    snap = monitor.snapshot()
    assert snap["total_previsoes"] == len(X)
    assert 0.0 <= snap["taxa_maligno"] <= 1.0
    assert 0.5 <= snap["confianca_media"] <= 1.0
    monitor.log_snapshot(logger)
    events.close()

    registros = read_events("teste_previsoes.jsonl", event="prediction")
    assert len(registros) == len(X)
    assert all(r["run_id"] == "run-teste" for r in registros)
    assert all("proba_malignant" in r for r in registros)

    # a taxa de maligno prevista tem que bater com os registros gravados
    taxa_log = sum(1 for r in registros if r["predicted_label"] == 0) / len(registros)
    assert taxa_log == pytest.approx(snap["taxa_maligno"], abs=0.01)


def test_log_estruturado_ignora_linha_corrompida(tmp_path, monkeypatch):
    """Execução morta no meio de uma escrita não pode invalidar o histórico."""
    import src.monitoring as monitoring

    monkeypatch.setattr(monitoring, "LOGS_DIR", tmp_path)
    events = EventLogger("corrompido.jsonl")
    events.emit("generation", generation=1, best_fitness=0.9)
    events.close()

    with (tmp_path / "corrompido.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"event": "generation", "best_fit\n')  # linha truncada

    registros = read_events("corrompido.jsonl", event="generation")
    assert len(registros) == 1
    assert registros[0]["best_fitness"] == 0.9


def test_read_events_de_arquivo_inexistente_devolve_vazio(tmp_path, monkeypatch):
    import src.monitoring as monitoring

    monkeypatch.setattr(monitoring, "LOGS_DIR", tmp_path)
    assert read_events("nao_existe.jsonl") == []


def test_setup_logging_nao_duplica_handlers():
    """Chamar duas vezes faria cada linha aparecer duplicada no console."""
    a = setup_logging("teste.idempotente", log_file=None)
    n = len(a.handlers)
    b = setup_logging("teste.idempotente", log_file=None)
    assert a is b
    assert len(b.handlers) == n
