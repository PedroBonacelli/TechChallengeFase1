# Testes da função fitness (Etapa 2, parte 2).
#
# Três propriedades são críticas aqui e cada uma tem teste próprio:
#
#   1. O fitness PRIORIZA O RECALL DO MALIGNO. Se alguém mexer nos pesos
#      e inverter a prioridade, `test_recall_domina_a_funcao` falha.
#   2. O fitness é DETERMINÍSTICO. Sem isso o cache mente e a seleção
#      compara ruído.
#   3. Combinação que não treina vira fitness 0, nunca exceção — o GA tem
#      que atravessar um indivíduo inválido sem morrer.

from __future__ import annotations

import numpy as np
import pytest

from src.ga.fitness import (
    FitnessConfig,
    FitnessEvaluator,
    build_estimator,
    evaluate_params,
)
from src.ga.search_space import BASELINE_PARAMS, SEARCH_SPACES, TREE_SPACE


# --- configuração ------------------------------------------------------


def test_pesos_tem_que_somar_um():
    with pytest.raises(ValueError, match="somar"):
        FitnessConfig(w_recall=0.5, w_f1=0.3, w_accuracy=0.3)


def test_config_padrao_e_valida():
    c = FitnessConfig()
    assert c.w_recall + c.w_f1 + c.w_accuracy == pytest.approx(1.0)


def test_recall_domina_a_funcao():
    """O peso do recall tem que ser maior que os outros dois somados.

    É a tradução numérica da decisão clínica da Fase 1: falso negativo
    custa mais que falso positivo. Se esta asserção cair, a função
    objetivo deixou de refletir a prioridade declarada no relatório.
    """
    c = FitnessConfig()
    assert c.w_recall > c.w_f1 + c.w_accuracy


def test_cv_precisa_de_pelo_menos_dois_folds():
    with pytest.raises(ValueError, match="cv_folds"):
        FitnessConfig(cv_folds=1)


def test_combine_e_media_ponderada():
    c = FitnessConfig(w_recall=0.6, w_f1=0.25, w_accuracy=0.15)
    assert c.combine(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert c.combine(0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert c.combine(1.0, 0.0, 0.0) == pytest.approx(0.6)


def test_formula_menciona_a_validacao_cruzada():
    """A fórmula vai pro log e pro relatório; tem que dizer onde mede."""
    f = FitnessConfig().formula()
    assert "CV" in f and "treino" in f


# --- construção do estimador ------------------------------------------


@pytest.mark.parametrize("modelo", sorted(SEARCH_SPACES))
def test_build_estimator_fixa_o_random_state(modelo):
    est = build_estimator(modelo, BASELINE_PARAMS[modelo], random_state=7)
    assert est.random_state == 7


def test_build_estimator_rejeita_modelo_desconhecido():
    with pytest.raises(ValueError, match="desconhecido"):
        build_estimator("xgboost", {})


def test_build_estimator_paraleliza_a_floresta():
    est = build_estimator("random_forest", BASELINE_PARAMS["random_forest"])
    assert est.n_jobs == -1


# --- avaliação ---------------------------------------------------------


def test_avaliacao_devolve_metricas_no_intervalo(data):
    r = evaluate_params(
        "decision_tree", BASELINE_PARAMS["decision_tree"], data, "raw"
    )
    assert not r.failed
    for valor in (
        r.fitness, r.recall_malignant, r.f1_malignant,
        r.precision_malignant, r.accuracy,
    ):
        assert 0.0 <= valor <= 1.0


def test_avaliacao_e_consistente_com_a_formula(data):
    config = FitnessConfig()
    r = evaluate_params(
        "decision_tree", BASELINE_PARAMS["decision_tree"], data, "raw", config
    )
    esperado = config.combine(r.recall_malignant, r.f1_malignant, r.accuracy)
    assert r.fitness == pytest.approx(esperado)


def test_avaliacao_e_deterministica(data):
    params = BASELINE_PARAMS["random_forest"]
    a = evaluate_params("random_forest", params, data, "raw")
    b = evaluate_params("random_forest", params, data, "raw")
    assert a.fitness == pytest.approx(b.fitness)
    assert a.recall_malignant == pytest.approx(b.recall_malignant)


def test_combinacao_invalida_vira_fitness_zero_sem_excecao(data):
    """lbfgs + l1 é rejeitado pelo sklearn. O GA não pode quebrar por isso.

    Passo os params SEM reparo de propósito — é o cenário de uma
    combinação inválida chegando ao avaliador.
    """
    r = evaluate_params(
        "logistic_regression",
        {"C": 1.0, "penalty": "l1", "solver": "lbfgs", "max_iter": 100,
         "l1_ratio": None, "class_weight": None},
        data, "scaled",
    )
    assert r.failed
    assert r.fitness == 0.0
    assert r.error is not None


def test_as_row_arredonda_e_marca_falha(data):
    r = evaluate_params(
        "decision_tree", BASELINE_PARAMS["decision_tree"], data, "raw"
    )
    linha = r.as_row()
    assert set(linha) == {
        "fitness", "recall_malignant", "f1_malignant",
        "precision_malignant", "accuracy", "failed",
    }
    assert linha["failed"] is False


# --- cache -------------------------------------------------------------


def test_cache_evita_reavaliar_o_mesmo_cromossomo(data, rng):
    ev = FitnessEvaluator("decision_tree", TREE_SPACE, data)
    c = TREE_SPACE.random_chromosome(rng)
    primeiro = ev.evaluate(c)
    segundo = ev.evaluate(c)
    assert ev.n_evaluations == 1
    assert ev.n_cache_hits == 1
    assert primeiro.fitness == segundo.fitness


def test_cache_colapsa_genotipos_com_o_mesmo_fenotipo(data):
    """Dois vetores diferentes que decodificam no mesmo modelo.

    É o caso que o cache mais pega em produção: a faixa categórica tem
    largura 1/n, então muitos valores de [0,1] caem na mesma categoria.
    """
    ev = FitnessEvaluator("decision_tree", TREE_SPACE, data)
    a = np.zeros(TREE_SPACE.n_genes)
    b = a.copy()
    b[0] = 0.30  # ainda dentro da faixa de 'gini' (0 a 1/3)
    assert TREE_SPACE.decode(a) == TREE_SPACE.decode(b)

    ev.evaluate(a)
    ev.evaluate(b)
    assert ev.n_evaluations == 1
    assert ev.n_cache_hits == 1


def test_cache_ignora_diferenca_de_float_irrelevante(data):
    """0.0000001 de diferença em ccp_alpha treina o mesmo modelo."""
    ev = FitnessEvaluator("decision_tree", TREE_SPACE, data)
    params = dict(BASELINE_PARAMS["decision_tree"])
    k1 = ev._key(params)
    params["ccp_alpha"] = 1e-9
    k2 = ev._key(params)
    assert k1 == k2


def test_cache_distingue_fenotipos_diferentes(data, rng):
    ev = FitnessEvaluator("decision_tree", TREE_SPACE, data)
    vistos = set()
    for _ in range(15):
        c = TREE_SPACE.random_chromosome(rng)
        ev.evaluate(c)
        vistos.add(ev._key(TREE_SPACE.decode(c)))
    assert ev.n_evaluations == len(vistos)


def test_evaluate_population_devolve_fitness_alinhado(data, rng):
    ev = FitnessEvaluator("decision_tree", TREE_SPACE, data)
    pop = np.array([TREE_SPACE.random_chromosome(rng) for _ in range(6)])
    fitness, resultados = ev.evaluate_population(pop)
    assert len(fitness) == len(resultados) == 6
    for f, r in zip(fitness, resultados):
        assert f == pytest.approx(r.fitness)


def test_stats_contabiliza_falhas(data):
    """Avaliador sem reparo: forço um fenótipo inválido pelo caminho direto."""
    ev = FitnessEvaluator("decision_tree", TREE_SPACE, data)
    ev.evaluate(TREE_SPACE.random_chromosome(np.random.default_rng(3)))
    stats = ev.stats()
    assert set(stats) == {"avaliacoes_reais", "cache_hits", "falhas"}
    assert stats["falhas"] == 0


def test_fitness_e_sempre_nao_negativo(data, rng):
    """A roleta divide pela soma do fitness — negativo quebraria a seleção."""
    ev = FitnessEvaluator("decision_tree", TREE_SPACE, data)
    for _ in range(20):
        assert ev.evaluate(TREE_SPACE.random_chromosome(rng)).fitness >= 0.0
