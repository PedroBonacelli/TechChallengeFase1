# Testes dos operadores genéticos (Etapa 2, parte 1).
#
# Cada operador é testado pela PROPRIEDADE que ele tem que garantir, não
# por saída fixa: "o torneio devolve o melhor quando o torneio é a
# população inteira", "a mutação nunca sai de [0,1]", "o cruzamento
# uniforme produz filhos cujos genes vêm de um dos pais em cada posição".
# Propriedade sobrevive a refatoração; valor fixo não.

from __future__ import annotations

import numpy as np
import pytest

from src.ga.operators import (
    blx_alpha_crossover,
    crossover,
    elite_indices,
    mutate,
    one_point_crossover,
    population_diversity,
    roulette_selection,
    select,
    tournament_selection,
    uniform_crossover,
)
from src.ga.search_space import FOREST_SPACE, LOGISTIC_SPACE, TREE_SPACE


@pytest.fixture
def populacao(rng):
    return rng.random((10, 6))


@pytest.fixture
def fitness():
    return np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05])


# --- seleção -----------------------------------------------------------


def test_torneio_do_tamanho_da_populacao_sempre_devolve_o_melhor(
    populacao, fitness, rng
):
    """Com torneio = população, não há aleatoriedade: ganha o melhor."""
    selecionados = tournament_selection(
        populacao, fitness, rng, n_select=5, tournament_size=len(populacao)
    )
    melhor = populacao[int(np.argmax(fitness))]
    for individuo in selecionados:
        assert np.allclose(individuo, melhor)


def test_torneio_devolve_a_quantidade_pedida(populacao, fitness, rng):
    assert len(tournament_selection(populacao, fitness, rng, n_select=7)) == 7


def test_torneio_tem_pressao_seletiva(populacao, fitness, rng):
    """O fitness médio dos selecionados tem que superar o da população."""
    selecionados = tournament_selection(
        populacao, fitness, rng, n_select=500, tournament_size=3
    )
    # recupera o fitness de cada selecionado casando com a população
    fit_sel = [
        fitness[int(np.argmin(np.abs(populacao - s).sum(axis=1)))]
        for s in selecionados
    ]
    assert np.mean(fit_sel) > fitness.mean()


def test_torneio_maior_aumenta_a_pressao(populacao, fitness, rng):
    def media_selecionada(k, seed):
        g = np.random.default_rng(seed)
        sel = tournament_selection(populacao, fitness, g, 400, tournament_size=k)
        return np.mean([
            fitness[int(np.argmin(np.abs(populacao - s).sum(axis=1)))] for s in sel
        ])

    assert media_selecionada(5, 7) > media_selecionada(2, 7)


def test_torneio_rejeita_tamanho_invalido(populacao, fitness, rng):
    with pytest.raises(ValueError, match="tournament_size"):
        tournament_selection(populacao, fitness, rng, 5, tournament_size=0)


def test_roleta_devolve_individuos_da_populacao(populacao, fitness, rng):
    selecionados = roulette_selection(populacao, fitness, rng, n_select=20)
    assert len(selecionados) == 20
    for s in selecionados:
        assert any(np.allclose(s, individuo) for individuo in populacao)


def test_roleta_com_fitness_todo_zero_nao_divide_por_zero(populacao, rng):
    """População inteira falhando ao treinar é um estado possível."""
    zeros = np.zeros(len(populacao))
    selecionados = roulette_selection(populacao, zeros, rng, n_select=5)
    assert len(selecionados) == 5


def test_roleta_favorece_fitness_alto(populacao, rng):
    """Fitness muito desigual: o dominante tem que levar quase tudo."""
    fit = np.zeros(len(populacao))
    fit[3] = 1.0
    selecionados = roulette_selection(populacao, fit, rng, n_select=50)
    assert all(np.allclose(s, populacao[3]) for s in selecionados)


def test_select_despacha_as_duas_estrategias(populacao, fitness, rng):
    assert len(select("tournament", populacao, fitness, rng, 4)) == 4
    assert len(select("roulette", populacao, fitness, rng, 4)) == 4


def test_select_rejeita_estrategia_desconhecida(populacao, fitness, rng):
    with pytest.raises(ValueError, match="desconhecida"):
        select("torneio-maluco", populacao, fitness, rng, 4)


# --- cruzamento --------------------------------------------------------


def test_cruzamento_uniforme_herda_de_um_dos_pais_em_cada_posicao(rng):
    a = np.zeros(8)
    b = np.ones(8)
    f1, f2 = uniform_crossover(a, b, rng)
    assert np.all((f1 == 0) | (f1 == 1))
    # o que um filho não herdou, o outro herdou — são complementares
    assert np.allclose(f1 + f2, np.ones(8))


def test_cruzamento_uniforme_com_prob_zero_nao_troca_nada(rng):
    a, b = np.zeros(6), np.ones(6)
    f1, f2 = uniform_crossover(a, b, rng, swap_prob=0.0)
    assert np.allclose(f1, a)
    assert np.allclose(f2, b)


def test_cruzamento_um_ponto_preserva_prefixo_e_sufixo(rng):
    a = np.zeros(8)
    b = np.ones(8)
    f1, f2 = one_point_crossover(a, b, rng)
    # prefixo de zeros seguido de sufixo de uns (nunca alternado)
    transicoes = np.sum(np.abs(np.diff(f1)))
    assert transicoes == 1
    assert f1[0] == 0 and f1[-1] == 1
    assert f2[0] == 1 and f2[-1] == 0


def test_cruzamento_um_ponto_nunca_copia_os_pais(rng):
    """O ponto de corte é sorteado em 1..n-1, nunca em 0 nem em n."""
    a, b = np.zeros(5), np.ones(5)
    for _ in range(30):
        f1, _ = one_point_crossover(a, b, rng)
        assert not np.allclose(f1, a)
        assert not np.allclose(f1, b)


def test_blx_fica_no_intervalo_unitario(rng):
    """Com alpha=0.5 a extensão sai de [0,1]; o clip tem que segurar."""
    for _ in range(100):
        a, b = rng.random(6), rng.random(6)
        f1, f2 = blx_alpha_crossover(a, b, rng, alpha=0.5)
        assert np.all((f1 >= 0) & (f1 <= 1))
        assert np.all((f2 >= 0) & (f2 <= 1))


def test_blx_cria_valores_intermediarios(rng):
    """É o que o uniforme não faz: valores que não existiam nos pais."""
    a = np.full(6, 0.3)
    b = np.full(6, 0.7)
    novos = 0
    for _ in range(50):
        f1, _ = blx_alpha_crossover(a, b, rng, alpha=0.2)
        novos += int(np.any((f1 != 0.3) & (f1 != 0.7)))
    assert novos > 45


def test_blx_com_alpha_zero_fica_entre_os_pais(rng):
    a = np.full(6, 0.3)
    b = np.full(6, 0.7)
    f1, _ = blx_alpha_crossover(a, b, rng, alpha=0.0)
    assert np.all((f1 >= 0.3) & (f1 <= 0.7))


def test_taxa_de_cruzamento_zero_copia_os_pais(rng):
    a, b = np.zeros(6), np.ones(6)
    f1, f2 = crossover("uniform", a, b, rng, crossover_rate=0.0)
    assert np.allclose(f1, a)
    assert np.allclose(f2, b)


def test_taxa_de_cruzamento_um_sempre_cruza(rng):
    a, b = np.zeros(6), np.ones(6)
    trocou = 0
    for _ in range(40):
        f1, _ = crossover("uniform", a, b, rng, crossover_rate=1.0)
        trocou += int(not np.allclose(f1, a))
    assert trocou > 35  # com 6 genes, ~1.6% de chance de não trocar nenhum


def test_crossover_rejeita_estrategia_desconhecida(rng):
    with pytest.raises(ValueError, match="desconhecido"):
        crossover("ponto-duplo", np.zeros(4), np.ones(4), rng, crossover_rate=1.0)


# --- mutação -----------------------------------------------------------


def test_mutacao_com_taxa_zero_nao_altera_nada(rng):
    c = rng.random(TREE_SPACE.n_genes)
    assert np.allclose(mutate(c, TREE_SPACE, rng, mutation_rate=0.0), c)


def test_mutacao_com_taxa_um_altera_tudo(rng):
    """Cada gene é visitado; os contínuos recebem ruído gaussiano."""
    c = np.full(TREE_SPACE.n_genes, 0.5)
    mutado = mutate(c, TREE_SPACE, rng, mutation_rate=1.0, sigma=0.2)
    assert not np.allclose(mutado, c)


def test_mutacao_nao_muta_a_entrada(rng):
    c = rng.random(FOREST_SPACE.n_genes)
    original = c.copy()
    mutate(c, FOREST_SPACE, rng, mutation_rate=1.0)
    assert np.allclose(c, original)


def test_mutacao_fica_no_intervalo_unitario(rng):
    """sigma grande + valor na borda é o caso que quebraria sem o clip."""
    for valor in (0.0, 1.0, 0.5):
        c = np.full(FOREST_SPACE.n_genes, valor)
        for _ in range(50):
            m = mutate(c, FOREST_SPACE, rng, mutation_rate=1.0, sigma=0.8)
            assert np.all((m >= 0.0) & (m <= 1.0))


def test_mutacao_de_gene_categorico_troca_de_categoria(rng):
    """Re-sorteio uniforme: com 3 categorias, ~2/3 das vezes muda.

    Perturbação gaussiana com sigma=0.15 quase nunca mudaria (as faixas
    têm 0.33 de largura) — é a razão de a mutação ser sensível ao tipo.
    """
    # gene 0 do espaço da árvore é 'criterion', categórico com 3 opções
    c = np.zeros(TREE_SPACE.n_genes)  # criterion = 'gini'
    categorias = set()
    for _ in range(200):
        m = mutate(c, TREE_SPACE, rng, mutation_rate=1.0, sigma=0.15)
        categorias.add(TREE_SPACE.decode(m)["criterion"])
    assert categorias == {"gini", "entropy", "log_loss"}


def test_mutacao_de_gene_continuo_e_busca_local(rng):
    """Gaussiana com sigma pequeno: o valor anda pouco."""
    # gene 5 do espaço da árvore é ccp_alpha, contínuo
    c = np.full(TREE_SPACE.n_genes, 0.5)
    desvios = []
    for _ in range(200):
        m = mutate(c, TREE_SPACE, rng, mutation_rate=1.0, sigma=0.1)
        desvios.append(abs(m[5] - 0.5))
    assert np.mean(desvios) < 0.12  # ~0.08 esperado para |N(0, 0.1)|


def test_mutacao_rejeita_taxa_invalida(rng):
    with pytest.raises(ValueError, match="mutation_rate"):
        mutate(np.zeros(6), LOGISTIC_SPACE, rng, mutation_rate=1.5)


# --- elitismo e diversidade -------------------------------------------


def test_elite_indices_ordena_do_melhor_pro_pior(fitness):
    idx = elite_indices(fitness, 3)
    assert list(idx) == [1, 6, 4]  # fitness 0.9, 0.8, 0.7
    assert list(fitness[idx]) == sorted(fitness[idx], reverse=True)


def test_elite_indices_com_zero_devolve_vazio(fitness):
    assert len(elite_indices(fitness, 0)) == 0
    assert len(elite_indices(fitness, -1)) == 0


def test_elite_indices_limita_ao_tamanho_da_populacao(fitness):
    assert len(elite_indices(fitness, 999)) == len(fitness)


def test_diversidade_e_zero_quando_a_populacao_e_clone():
    clones = np.tile(np.array([0.2, 0.5, 0.8]), (6, 1))
    assert population_diversity(clones) == pytest.approx(0.0)


def test_diversidade_cai_quando_a_populacao_converge(rng):
    espalhada = rng.random((20, 6))
    concentrada = 0.5 + 0.01 * rng.random((20, 6))
    assert population_diversity(espalhada) > population_diversity(concentrada)


def test_diversidade_de_individuo_unico_e_zero():
    assert population_diversity(np.array([[0.1, 0.2]])) == 0.0
