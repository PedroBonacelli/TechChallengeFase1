# Etapa 2 (parte 1) — Operadores genéticos
#
# Seleção, cruzamento e mutação para a codificação real normalizada de
# search_space.py. Todos os operadores recebem um `numpy.random.Generator`
# explícito em vez de usar o estado global do numpy: é o que torna um
# experimento reproduzível e os testes determinísticos.
#
# Convenções:
#   - população = array (n_individuos, n_genes) com valores em [0, 1]
#   - fitness   = array (n_individuos,), MAIOR É MELHOR, sempre >= 0
#     (a função fitness em fitness.py é uma média ponderada de métricas
#      em [0,1], então o pressuposto de não-negatividade vale — a roleta
#      depende dele)

from __future__ import annotations

import numpy as np

from .search_space import CategoricalGene, SearchSpace

# -----------------------------------------------------------------------
# Seleção
# -----------------------------------------------------------------------


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    n_select: int,
    tournament_size: int = 3,
) -> np.ndarray:
    """Seleção por torneio: sorteia k indivíduos, o melhor vence.

    A pressão seletiva é controlada por `tournament_size` e não pela
    diferença absoluta de fitness. Isso importa muito aqui: as métricas
    dos modelos ficam todas apertadas entre 0.90 e 0.99, então na roleta
    um indivíduo ruim tem quase a mesma chance de um ótimo. O torneio
    olha só a ORDEM, então continua separando bem mesmo com fitness
    comprimido — é por isso que ele é o padrão nos experimentos.

    Amostra com reposição (um bom indivíduo pode vencer vários torneios).
    """
    if tournament_size < 1:
        raise ValueError("tournament_size deve ser >= 1")
    n_pop = len(population)
    size = min(tournament_size, n_pop)

    selected = np.empty((n_select, population.shape[1]))
    for i in range(n_select):
        competidores = rng.choice(n_pop, size=size, replace=False)
        vencedor = competidores[np.argmax(fitness[competidores])]
        selected[i] = population[vencedor]
    return selected


def roulette_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    n_select: int,
) -> np.ndarray:
    """Seleção proporcional ao fitness (roleta viciada).

    Probabilidade de cada indivíduo = fitness_i / soma(fitness).
    Implementada pra comparação com o torneio no Experimento 2 — e o
    resultado esperado é que convirja mais devagar, justamente pela
    compressão de fitness descrita acima.

    Quando a soma do fitness é zero (toda a população falhou em treinar),
    cai pra sorteio uniforme em vez de dividir por zero.
    """
    total = float(fitness.sum())
    if total <= 0 or not np.isfinite(total):
        idx = rng.choice(len(population), size=n_select, replace=True)
    else:
        probs = fitness / total
        idx = rng.choice(len(population), size=n_select, replace=True, p=probs)
    return population[idx].copy()


SELECTION_OPERATORS = {
    "tournament": tournament_selection,
    "roulette": roulette_selection,
}


def select(
    strategy: str,
    population: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    n_select: int,
    tournament_size: int = 3,
) -> np.ndarray:
    """Despacha pra estratégia de seleção escolhida pelo experimento."""
    if strategy == "tournament":
        return tournament_selection(
            population, fitness, rng, n_select, tournament_size
        )
    if strategy == "roulette":
        return roulette_selection(population, fitness, rng, n_select)
    raise ValueError(
        f"seleção '{strategy}' desconhecida; use {sorted(SELECTION_OPERATORS)}"
    )


# -----------------------------------------------------------------------
# Cruzamento
# -----------------------------------------------------------------------


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    swap_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruzamento uniforme: cada gene é sorteado independentemente.

    Diferente do corte em 1 ponto, não tem viés posicional — genes
    vizinhos no cromossomo não têm mais chance de viajar juntos. Como a
    ordem dos genes aqui é arbitrária (`max_depth` ao lado de
    `min_samples_split` por acaso, não por afinidade), não existe
    "bloco construtivo" contíguo pra preservar, e o uniforme é a escolha
    natural.
    """
    mask = rng.random(parent_a.shape) < swap_prob
    child_a = np.where(mask, parent_b, parent_a)
    child_b = np.where(mask, parent_a, parent_b)
    return child_a, child_b


def one_point_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruzamento de 1 ponto: corta em uma posição e troca as caudas."""
    n = len(parent_a)
    if n < 2:
        return parent_a.copy(), parent_b.copy()
    ponto = int(rng.integers(1, n))  # 1..n-1, nunca degenera em cópia
    child_a = np.concatenate([parent_a[:ponto], parent_b[ponto:]])
    child_b = np.concatenate([parent_b[:ponto], parent_a[ponto:]])
    return child_a, child_b


def blx_alpha_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """BLX-alpha: cruzamento aritmético para codificação real.

    Para cada gene, sorteia um valor no intervalo entre os dois pais
    ESTENDIDO por alpha * distância pra cada lado. O uniforme só
    recombina valores que já existem na população; o BLX cria valores
    intermediários e um pouco além — é o que permite refinar o C da
    Logística continuamente em vez de ficar preso nos valores sorteados
    na população inicial.

    Resultado é recortado em [0,1] pra manter o genótipo válido.
    """
    lo = np.minimum(parent_a, parent_b)
    hi = np.maximum(parent_a, parent_b)
    d = hi - lo
    low, high = lo - alpha * d, hi + alpha * d
    child_a = rng.uniform(low, high)
    child_b = rng.uniform(low, high)
    return np.clip(child_a, 0.0, 1.0), np.clip(child_b, 0.0, 1.0)


CROSSOVER_OPERATORS = {
    "uniform": uniform_crossover,
    "one_point": one_point_crossover,
    "blx": blx_alpha_crossover,
}


def crossover(
    strategy: str,
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    crossover_rate: float = 0.9,
    blx_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica o cruzamento com probabilidade `crossover_rate`.

    Se o sorteio não passar, os filhos são cópias dos pais — é assim que
    uma taxa de cruzamento < 1 preserva indivíduos inteiros entre
    gerações, independente do elitismo.
    """
    if rng.random() >= crossover_rate:
        return parent_a.copy(), parent_b.copy()

    if strategy == "uniform":
        return uniform_crossover(parent_a, parent_b, rng)
    if strategy == "one_point":
        return one_point_crossover(parent_a, parent_b, rng)
    if strategy == "blx":
        return blx_alpha_crossover(parent_a, parent_b, rng, alpha=blx_alpha)
    raise ValueError(
        f"cruzamento '{strategy}' desconhecido; use {sorted(CROSSOVER_OPERATORS)}"
    )


# -----------------------------------------------------------------------
# Mutação
# -----------------------------------------------------------------------


def mutate(
    chromosome: np.ndarray,
    space: SearchSpace,
    rng: np.random.Generator,
    mutation_rate: float = 0.1,
    sigma: float = 0.15,
) -> np.ndarray:
    """Mutação por gene, sensível ao TIPO do gene.

    `mutation_rate` é a probabilidade de cada gene (não do indivíduo) ser
    mutado. O tipo decide COMO:

      - gene contínuo/inteiro: perturbação gaussiana de desvio `sigma`,
        recortada em [0,1]. É uma busca local — anda pouco em torno do
        valor atual, o que refina um bom indivíduo sem destruí-lo.
      - gene categórico: re-sorteio uniforme em [0,1]. Perturbar
        gaussianamente um categórico quase nunca muda a categoria (com
        3 opções, cada faixa tem largura 0.33 e sigma=0.15 raramente
        atravessa a fronteira), então a mutação viraria um no-op
        silencioso justamente nos genes mais discretos do espaço.

    Retorna um array novo — não altera a entrada.
    """
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate deve estar em [0, 1]")

    mutated = chromosome.copy()
    for i, gene in enumerate(space.genes):
        if rng.random() >= mutation_rate:
            continue
        if isinstance(gene, CategoricalGene):
            mutated[i] = rng.random()
        else:
            mutated[i] = np.clip(mutated[i] + rng.normal(0.0, sigma), 0.0, 1.0)
    return mutated


# -----------------------------------------------------------------------
# Elitismo e diversidade
# -----------------------------------------------------------------------


def elite_indices(fitness: np.ndarray, n_elite: int) -> np.ndarray:
    """Índices dos `n_elite` melhores, do melhor pro pior.

    Elitismo garante monotonicidade: o melhor fitness nunca piora de uma
    geração pra outra. Sem isso, cruzamento e mutação podem perder o
    melhor indivíduo e a curva de convergência oscila pra baixo.
    """
    if n_elite <= 0:
        return np.array([], dtype=int)
    n_elite = min(n_elite, len(fitness))
    return np.argsort(-fitness, kind="stable")[:n_elite]


def population_diversity(population: np.ndarray) -> float:
    """Diversidade = desvio padrão médio por gene.

    Monitorada a cada geração: cair pra perto de zero sinaliza
    convergência prematura (a população virou clone do elite), e é o
    sinal pra aumentar mutação ou população no próximo experimento.
    """
    if len(population) < 2:
        return 0.0
    return float(population.std(axis=0).mean())
