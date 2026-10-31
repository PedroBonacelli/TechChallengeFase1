# Testes do motor do GA.
#
# A propriedade mais importante é a MONOTONICIDADE DO ELITE: com
# elitismo >= 1, o melhor fitness nunca pode cair de uma geração pra
# outra. Se cair, o elitismo está furado — e o gráfico de convergência da
# Etapa 4 estaria mostrando uma busca que às vezes perde o que achou.

from __future__ import annotations

import numpy as np
import pytest

from src.ga.engine import ExperimentSuite, GAConfig, run_ga
from src.ga.fitness import FitnessConfig
from src.ga.search_space import SEARCH_SPACES

CONFIG_RAPIDA = dict(population_size=8, n_generations=4, seed=7)


# --- validação da configuração ----------------------------------------


def test_populacao_minima():
    with pytest.raises(ValueError, match="population_size"):
        GAConfig(name="x", population_size=3)


def test_elitismo_nao_pode_tomar_a_populacao_inteira():
    """elitismo == população faria o GA parar de gerar filhos."""
    with pytest.raises(ValueError, match="elitism"):
        GAConfig(name="x", population_size=10, elitism=10)


@pytest.mark.parametrize("taxa", [-0.1, 1.5])
def test_taxas_fora_do_intervalo_sao_rejeitadas(taxa):
    with pytest.raises(ValueError, match="crossover_rate"):
        GAConfig(name="x", crossover_rate=taxa)
    with pytest.raises(ValueError, match="mutation_rate"):
        GAConfig(name="x", mutation_rate=taxa)


def test_max_evaluations_e_o_orcamento():
    c = GAConfig(name="x", population_size=30, n_generations=20)
    assert c.max_evaluations == 600


def test_summary_expoe_os_parametros_variados():
    """O resumo vai pro log; precisa conter o que distingue experimentos."""
    s = GAConfig(name="x", population_size=25, n_generations=11,
                 selection="tournament", tournament_size=4).summary()
    for esperado in ("pop=25", "ger=11", "tournament", "k=4", "seed="):
        assert esperado in s


# --- execução ----------------------------------------------------------


@pytest.mark.parametrize("modelo", sorted(SEARCH_SPACES))
def test_ga_roda_nos_tres_modelos(modelo, data):
    r = run_ga(modelo, data, GAConfig(name="t", **CONFIG_RAPIDA))
    assert r.model == modelo
    assert r.generations_run == 4
    assert 0.0 <= r.best_fitness <= 1.0
    # o fenótipo devolvido tem que ser decodificável e completo
    assert set(r.best_params) == {g.name for g in SEARCH_SPACES[modelo].genes}


def test_elitismo_garante_monotonicidade(data):
    """O melhor fitness nunca piora de uma geração pra outra."""
    r = run_ga(
        "decision_tree", data,
        GAConfig(name="t", population_size=12, n_generations=10,
                 elitism=2, mutation_rate=0.4, seed=3),
    )
    melhores = [h.best_fitness for h in r.history]
    assert melhores == sorted(melhores), (
        f"o elite foi perdido em alguma geração: {melhores}"
    )


def test_sem_elitismo_a_monotonicidade_pode_quebrar(data):
    """Teste de contraste: mostra que o elitismo é o que garante a propriedade.

    Não asserto que VAI quebrar (depende do sorteio), só que o GA roda
    sem elitismo — se este teste começasse a exigir queda, ficaria
    instável.
    """
    r = run_ga(
        "decision_tree", data,
        GAConfig(name="t", population_size=10, n_generations=6,
                 elitism=0, mutation_rate=0.5, seed=11),
    )
    assert r.generations_run == 6


def test_mesma_seed_produz_resultado_identico(data):
    config = GAConfig(name="t", population_size=10, n_generations=5, seed=99)
    a = run_ga("decision_tree", data, config)
    b = run_ga("decision_tree", data, config)
    assert a.best_fitness == pytest.approx(b.best_fitness)
    assert a.best_params == b.best_params
    assert np.allclose(a.best_chromosome, b.best_chromosome)


def test_seeds_diferentes_produzem_buscas_diferentes(data):
    """Se duas seeds derem o mesmo caminho, o rng não está sendo usado."""
    a = run_ga("random_forest", data,
               GAConfig(name="t", population_size=8, n_generations=3, seed=1))
    b = run_ga("random_forest", data,
               GAConfig(name="t", population_size=8, n_generations=3, seed=2))
    assert not np.allclose(a.best_chromosome, b.best_chromosome)


def test_populacao_mantem_o_tamanho_configurado(data):
    """O laço de reprodução tem um +1 pra par ímpar; não pode inflar a população."""
    for tamanho in (5, 8, 9, 13):
        r = run_ga(
            "decision_tree", data,
            GAConfig(name="t", population_size=tamanho, n_generations=3,
                     elitism=1, seed=5),
        )
        # se a população tivesse crescido, o teto de avaliações estouraria
        assert r.evaluator_stats["avaliacoes_reais"] <= tamanho * 3


def test_parada_antecipada_dispara_com_patience(data):
    """patience=1 em espaço pequeno: estagna e para antes do fim."""
    r = run_ga(
        "decision_tree", data,
        GAConfig(name="t", population_size=8, n_generations=40,
                 mutation_rate=0.0, crossover_rate=0.0, patience=2, seed=4),
    )
    assert r.stopped_early
    assert r.generations_run < 40


def test_sem_patience_roda_todas_as_geracoes(data):
    r = run_ga(
        "decision_tree", data,
        GAConfig(name="t", population_size=8, n_generations=6,
                 mutation_rate=0.0, crossover_rate=0.0, patience=None, seed=4),
    )
    assert not r.stopped_early
    assert r.generations_run == 6


@pytest.mark.parametrize("selecao", ["tournament", "roulette"])
@pytest.mark.parametrize("cruzamento", ["uniform", "one_point", "blx"])
def test_todas_as_combinacoes_de_operador_rodam(selecao, cruzamento, data):
    r = run_ga(
        "decision_tree", data,
        GAConfig(name="t", population_size=8, n_generations=3,
                 selection=selecao, crossover_op=cruzamento, seed=8),
    )
    assert r.generations_run == 3
    assert r.best_fitness > 0


# --- histórico ---------------------------------------------------------


def test_historico_registra_uma_linha_por_geracao(data):
    r = run_ga("decision_tree", data, GAConfig(name="t", **CONFIG_RAPIDA))
    assert [h.generation for h in r.history] == [1, 2, 3, 4]


def test_historico_tem_os_campos_do_grafico_de_convergencia(data):
    """A Etapa 4 lê exatamente estas colunas do CSV."""
    r = run_ga("decision_tree", data, GAConfig(name="t", **CONFIG_RAPIDA))
    linha = r.history_rows()[0]
    for campo in (
        "model", "experiment", "generation", "best_fitness", "mean_fitness",
        "std_fitness", "diversity", "best_recall_malignant",
    ):
        assert campo in linha


def test_melhor_nunca_e_menor_que_o_medio(data):
    r = run_ga("decision_tree", data,
               GAConfig(name="t", population_size=12, n_generations=5, seed=6))
    for h in r.history:
        assert h.best_fitness >= h.mean_fitness >= h.worst_fitness


def test_fitness_final_bate_com_a_ultima_geracao(data):
    """Com elitismo, o melhor da última população é o melhor global."""
    r = run_ga("decision_tree", data,
               GAConfig(name="t", population_size=10, n_generations=5,
                        elitism=2, seed=2))
    assert r.best_fitness == pytest.approx(r.history[-1].best_fitness)


def test_summary_relata_o_ganho(data):
    r = run_ga("decision_tree", data, GAConfig(name="t", **CONFIG_RAPIDA))
    s = r.summary()
    assert "fitness" in s and "gerações" in s and "cache" in s


def test_fitness_config_customizada_e_respeitada(data):
    """Peso 100% em accuracy muda o que o GA busca."""
    config_fitness = FitnessConfig(w_recall=0.0, w_f1=0.0, w_accuracy=1.0)
    r = run_ga(
        "decision_tree", data,
        GAConfig(name="t", **CONFIG_RAPIDA),
        fitness_config=config_fitness,
    )
    assert r.best_fitness == pytest.approx(r.best_result.accuracy)


# --- suíte -------------------------------------------------------------


def test_suite_tabela_lista_as_configuracoes():
    suite = ExperimentSuite([
        GAConfig(name="a", description="primeira"),
        GAConfig(name="b", description="segunda"),
    ])
    tabela = suite.table()
    assert "a" in tabela and "b" in tabela
    assert "primeira" in tabela and "segunda" in tabela
