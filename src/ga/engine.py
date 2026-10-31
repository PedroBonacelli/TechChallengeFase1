# Motor do algoritmo genético — junta codificação, operadores e fitness.
#
# Laço clássico de GA geracional com elitismo:
#
#   1. população inicial aleatória em [0,1]^n
#   2. avalia o fitness de todo mundo (CV no treino)
#   3. copia os N melhores direto pra próxima geração (elitismo)
#   4. preenche o resto: seleciona pais -> cruza -> muta
#   5. volta pro passo 2 até esgotar as gerações (ou estagnar)
#
# Cada geração registra melhor fitness, fitness médio, desvio e
# diversidade — é esse histórico que vira o gráfico de convergência da
# Etapa 4 e o log estruturado da Etapa 5.

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..monitoring import EventLogger
from .data import Phase1Data
from .fitness import FitnessConfig, FitnessEvaluator, FitnessResult
from .operators import (
    crossover,
    elite_indices,
    mutate,
    population_diversity,
    select,
)
from .search_space import SEARCH_SPACES


@dataclass(frozen=True)
class GAConfig:
    """Configuração de um experimento do GA.

    São esses campos que variam entre os 3 experimentos da Etapa 3.
    """

    name: str
    population_size: int = 30
    n_generations: int = 25
    crossover_rate: float = 0.9
    mutation_rate: float = 0.10
    mutation_sigma: float = 0.15
    selection: str = "tournament"
    tournament_size: int = 3
    crossover_op: str = "uniform"
    blx_alpha: float = 0.5
    elitism: int = 2
    # None = roda todas as gerações. Com um número, para quando o melhor
    # fitness não melhora por essa quantidade de gerações seguidas.
    patience: int | None = None
    seed: int = 42
    description: str = ""

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size deve ser >= 4")
        if self.elitism >= self.population_size:
            raise ValueError("elitism tem que ser menor que population_size")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate deve estar em [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate deve estar em [0, 1]")

    @property
    def max_evaluations(self) -> int:
        """Teto de avaliações de fitness (sem contar o cache)."""
        return self.population_size * self.n_generations

    def summary(self) -> str:
        return (
            f"pop={self.population_size} ger={self.n_generations} "
            f"cx={self.crossover_op}@{self.crossover_rate:.2f} "
            f"mut={self.mutation_rate:.2f}/sigma={self.mutation_sigma:.2f} "
            f"sel={self.selection}"
            + (f"(k={self.tournament_size})" if self.selection == "tournament" else "")
            + f" elite={self.elitism} seed={self.seed}"
        )


@dataclass
class GenerationRecord:
    """Uma linha do histórico de convergência."""

    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    worst_fitness: float
    diversity: float
    best_recall_malignant: float
    best_f1_malignant: float
    best_accuracy: float
    n_failures: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GAResult:
    """Resultado de uma execução completa do GA."""

    model: str
    config: GAConfig
    best_chromosome: np.ndarray
    best_params: dict[str, Any]
    best_result: FitnessResult
    history: list[GenerationRecord]
    evaluator_stats: dict[str, int]
    elapsed_s: float
    stopped_early: bool = False
    run_id: str = ""

    @property
    def best_fitness(self) -> float:
        return self.best_result.fitness

    @property
    def generations_run(self) -> int:
        return len(self.history)

    def history_rows(self) -> list[dict[str, Any]]:
        return [
            {"model": self.model, "experiment": self.config.name, **h.as_dict()}
            for h in self.history
        ]

    def summary(self) -> str:
        inicial = self.history[0].best_fitness if self.history else float("nan")
        return (
            f"{self.model} / {self.config.name}: fitness "
            f"{inicial:.4f} -> {self.best_fitness:.4f} "
            f"(+{100 * (self.best_fitness - inicial):.2f} p.p.) em "
            f"{self.generations_run} gerações, {self.elapsed_s:.1f}s, "
            f"{self.evaluator_stats['avaliacoes_reais']} avaliações reais "
            f"(+{self.evaluator_stats['cache_hits']} do cache)"
        )


def run_ga(
    model: str,
    data: Phase1Data,
    config: GAConfig,
    fitness_config: FitnessConfig | None = None,
    logger: logging.Logger | None = None,
    events: EventLogger | None = None,
) -> GAResult:
    """Roda o GA para um modelo e devolve o melhor indivíduo encontrado."""
    logger = logger or logging.getLogger("fase2.ga")
    space = SEARCH_SPACES[model]
    fitness_config = fitness_config or FitnessConfig()
    evaluator = FitnessEvaluator(model, space, data, fitness_config)

    # Generator próprio por execução, semeado pela config: dois
    # experimentos com a mesma seed e mesma config dão exatamente o mesmo
    # resultado, e seeds diferentes dão execuções independentes de verdade
    rng = np.random.default_rng(config.seed)

    t0 = time.perf_counter()
    logger.info("[%s / %s] %s", model, config.name, config.summary())
    logger.info("[%s] espaço de busca: %d genes | fitness: %s",
                model, space.n_genes, fitness_config.formula())

    if events:
        events.emit(
            "experiment_start",
            model=model,
            experiment=config.name,
            config=asdict(config),
            fitness_formula=fitness_config.formula(),
            n_genes=space.n_genes,
            search_space=[g.name for g in space.genes],
        )

    population = np.array(
        [space.random_chromosome(rng) for _ in range(config.population_size)]
    )
    fitness, results = evaluator.evaluate_population(population)

    history: list[GenerationRecord] = []
    melhor_global = float(fitness.max())
    geracoes_sem_melhora = 0
    stopped_early = False

    for geracao in range(1, config.n_generations + 1):
        t_ger = time.perf_counter()

        ordem = elite_indices(fitness, max(config.elitism, 1))
        idx_melhor = int(ordem[0])
        melhor_resultado = results[idx_melhor]

        registro = GenerationRecord(
            generation=geracao,
            best_fitness=float(fitness[idx_melhor]),
            mean_fitness=float(fitness.mean()),
            std_fitness=float(fitness.std()),
            worst_fitness=float(fitness.min()),
            diversity=population_diversity(population),
            best_recall_malignant=melhor_resultado.recall_malignant,
            best_f1_malignant=melhor_resultado.f1_malignant,
            best_accuracy=melhor_resultado.accuracy,
            n_failures=evaluator.n_failures,
            elapsed_s=time.perf_counter() - t0,
        )
        history.append(registro)

        logger.info(
            "  ger %2d/%d | melhor %.4f | médio %.4f (±%.4f) | "
            "divers. %.4f | recall_mal %.4f",
            geracao, config.n_generations, registro.best_fitness,
            registro.mean_fitness, registro.std_fitness,
            registro.diversity, registro.best_recall_malignant,
        )
        if events:
            events.emit(
                "generation", model=model, experiment=config.name,
                **registro.as_dict(),
                best_params=melhor_resultado.params,
            )

        # --- critério de parada por estagnação ---
        if registro.best_fitness > melhor_global + 1e-9:
            melhor_global = registro.best_fitness
            geracoes_sem_melhora = 0
        else:
            geracoes_sem_melhora += 1
        if config.patience is not None and geracoes_sem_melhora >= config.patience:
            logger.info(
                "  parada antecipada: %d gerações sem melhora (patience=%d)",
                geracoes_sem_melhora, config.patience,
            )
            stopped_early = True
            break
        if geracao == config.n_generations:
            break

        # --- nova geração ---
        nova = [population[i].copy() for i in elite_indices(fitness, config.elitism)]
        n_faltam = config.population_size - len(nova)

        # seleciona em pares; +1 ímpar evita ficar sem par no último filho
        pais = select(
            config.selection, population, fitness, rng,
            n_select=n_faltam + (n_faltam % 2),
            tournament_size=config.tournament_size,
        )
        for i in range(0, len(pais), 2):
            f1, f2 = crossover(
                config.crossover_op, pais[i], pais[i + 1], rng,
                crossover_rate=config.crossover_rate, blx_alpha=config.blx_alpha,
            )
            nova.append(mutate(f1, space, rng, config.mutation_rate, config.mutation_sigma))
            if len(nova) < config.population_size:
                nova.append(
                    mutate(f2, space, rng, config.mutation_rate, config.mutation_sigma)
                )

        population = np.array(nova[: config.population_size])
        fitness, results = evaluator.evaluate_population(population)

    # melhor indivíduo da ÚLTIMA população avaliada; com elitismo >= 1 ele
    # é também o melhor de toda a execução
    idx_final = int(np.argmax(fitness))
    elapsed = time.perf_counter() - t0

    result = GAResult(
        model=model,
        config=config,
        best_chromosome=population[idx_final].copy(),
        best_params=results[idx_final].params,
        best_result=results[idx_final],
        history=history,
        evaluator_stats=evaluator.stats(),
        elapsed_s=elapsed,
        stopped_early=stopped_early,
        run_id=events.run_id if events else "",
    )

    logger.info("[%s / %s] %s", model, config.name, result.summary())
    logger.info("  melhores hiperparâmetros: %s", result.best_params)
    if evaluator.n_failures:
        logger.warning(
            "  %d combinações falharam ao treinar (fitness 0)", evaluator.n_failures
        )

    if events:
        events.emit(
            "experiment_end", model=model, experiment=config.name,
            best_fitness=result.best_fitness,
            best_params=result.best_params,
            best_metrics_cv=result.best_result.as_row(),
            generations_run=result.generations_run,
            stopped_early=stopped_early,
            elapsed_s=round(elapsed, 3),
            **evaluator.stats(),
        )
    return result


@dataclass
class ExperimentSuite:
    """Conjunto de configurações do GA a rodar em todos os modelos."""

    configs: list[GAConfig] = field(default_factory=list)

    def table(self) -> str:
        linhas = ["configurações dos experimentos:"]
        for c in self.configs:
            linhas.append(f"  {c.name:12s} {c.summary()}")
            if c.description:
                linhas.append(f"               -> {c.description}")
        return "\n".join(linhas)
