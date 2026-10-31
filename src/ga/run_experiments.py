# Etapa 3 — Rodar os experimentos
#
# Roda o GA para os 3 modelos da Fase 1 em 3 configurações diferentes
# (9 execuções no total), registra o log de cada geração e salva o melhor
# indivíduo de cada combinação.
#
# ORÇAMENTO FIXO: as três configurações têm o mesmo teto de avaliações de
# fitness (população x gerações = 600). Isso é deliberado — se o
# Experimento 2 tivesse o dobro do orçamento, um resultado melhor não
# diria nada sobre a ESTRATÉGIA, apenas que ele treinou mais modelos. Com
# orçamento igual, a comparação mede o que se quer medir: dado o mesmo
# custo computacional, população grande e muita mutação (exploração)
# rende mais ou menos que população pequena e pressão seletiva alta
# (intensificação)?
#
#   python -m src.ga.run_experiments              # tudo (~35 min)
#   python -m src.ga.run_experiments --quick      # versão curta, pra testar
#   python -m src.ga.run_experiments --models decision_tree

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from ..monitoring import EventLogger, setup_logging, timed
from .data import load_phase1_data
from .engine import ExperimentSuite, GAConfig, GAResult, run_ga
from .fitness import FitnessConfig, build_estimator
from .search_space import SEARCH_SPACES, search_space_table

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "fase2"
MODELS_DIR = BASE_DIR / "models"

ORCAMENTO = 600  # avaliações de fitness por experimento

# -----------------------------------------------------------------------
# Os 3 experimentos (Etapa 3)
# -----------------------------------------------------------------------

EXPERIMENTOS = [
    GAConfig(
        name="equilibrado",
        population_size=30,
        n_generations=20,
        crossover_rate=0.90,
        crossover_op="uniform",
        mutation_rate=0.10,
        mutation_sigma=0.15,
        selection="tournament",
        tournament_size=3,
        elitism=2,
        seed=42,
        description=(
            "Referência. Parâmetros de meio de tabela, os que a literatura "
            "usa como ponto de partida: cruzamento alto, mutação baixa, "
            "torneio de 3. Serve de régua pros outros dois."
        ),
    ),
    GAConfig(
        name="exploratorio",
        population_size=50,
        n_generations=12,
        crossover_rate=0.70,
        crossover_op="blx",
        blx_alpha=0.5,
        mutation_rate=0.30,
        mutation_sigma=0.25,
        selection="roulette",
        tournament_size=3,
        elitism=1,
        seed=43,
        description=(
            "Exploração máxima: população grande, mutação 3x maior, roleta "
            "(pressão seletiva fraca) e BLX-alpha, que inventa valores "
            "intermediários. Elitismo mínimo. Aposta em cobrir o espaço; "
            "o risco é não ter gerações suficientes pra refinar."
        ),
    ),
    GAConfig(
        name="intensivo",
        population_size=15,
        n_generations=40,
        crossover_rate=0.95,
        crossover_op="blx",
        blx_alpha=0.3,
        mutation_rate=0.05,
        mutation_sigma=0.08,
        selection="tournament",
        tournament_size=5,
        elitism=4,
        seed=44,
        description=(
            "Intensificação: população pequena com o dobro de gerações, "
            "torneio de 5 (pressão alta), mutação baixa e elitismo forte. "
            "Converge rápido; o risco é travar num ótimo local."
        ),
    ),
]

for _c in EXPERIMENTOS:
    assert _c.max_evaluations == ORCAMENTO, (
        f"{_c.name} tem orçamento {_c.max_evaluations}, esperado {ORCAMENTO}"
    )

QUICK = [
    GAConfig(name="equilibrado", population_size=10, n_generations=4, seed=42),
    GAConfig(
        name="exploratorio", population_size=12, n_generations=3, seed=43,
        selection="roulette", crossover_op="blx", mutation_rate=0.30,
    ),
    GAConfig(
        name="intensivo", population_size=8, n_generations=5, seed=44,
        tournament_size=5, elitism=2, mutation_rate=0.05,
    ),
]


# -----------------------------------------------------------------------
# Execução
# -----------------------------------------------------------------------


def salvar_resultados(resultados: list[GAResult], sufixo: str = "") -> None:
    """Grava histórico, melhores indivíduos e os modelos otimizados."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    historico = pd.DataFrame(
        [linha for r in resultados for linha in r.history_rows()]
    )
    historico.to_csv(REPORTS_DIR / f"ga_history{sufixo}.csv", index=False)

    melhores = pd.DataFrame([
        {
            "model": r.model,
            "experiment": r.config.name,
            "fitness_cv": round(r.best_fitness, 6),
            **{f"cv_{k}": v for k, v in r.best_result.as_row().items()
               if k != "fitness"},
            "generations_run": r.generations_run,
            "stopped_early": r.stopped_early,
            "elapsed_s": round(r.elapsed_s, 2),
            **r.evaluator_stats,
            "best_params": json.dumps(r.best_params, default=str),
        }
        for r in resultados
    ])
    melhores.to_csv(REPORTS_DIR / f"ga_best{sufixo}.csv", index=False)

    # JSON com tudo, incluindo a config completa de cada experimento —
    # é o arquivo que permite reproduzir uma execução exatamente
    detalhes = [
        {
            "model": r.model,
            "experiment": r.config.name,
            "run_id": r.run_id,
            "config": asdict(r.config),
            "best_fitness_cv": r.best_fitness,
            "best_params": r.best_params,
            "best_metrics_cv": r.best_result.as_row(),
            "evaluator_stats": r.evaluator_stats,
            "generations_run": r.generations_run,
            "stopped_early": r.stopped_early,
            "elapsed_s": r.elapsed_s,
            "chromosome": r.best_chromosome.tolist(),
        }
        for r in resultados
    ]
    (REPORTS_DIR / f"ga_experiments{sufixo}.json").write_text(
        json.dumps(detalhes, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def treinar_modelos_vencedores(
    resultados: list[GAResult], data, logger, sufixo: str = ""
) -> dict[str, dict]:
    """Retreina, no treino COMPLETO, o melhor indivíduo de cada modelo.

    Dois pontos importantes aqui:

    1. O vencedor de cada modelo é escolhido pelo fitness de CV, nunca
       pelo desempenho no teste — escolher pelo teste seria selecionar
       contra o conjunto de avaliação, exatamente o vazamento que a CV
       existe pra evitar.
    2. O refit usa as 455 amostras de treino inteiras. Durante o GA cada
       indivíduo foi avaliado em 5 folds de ~364 amostras; o modelo que
       vai pra produção não tem motivo pra desperdiçar os 20% restantes.
    """
    vencedores: dict[str, dict] = {}
    for modelo in sorted({r.model for r in resultados}):
        do_modelo = [r for r in resultados if r.model == modelo]
        melhor = max(do_modelo, key=lambda r: r.best_fitness)

        space = SEARCH_SPACES[modelo]
        est = build_estimator(modelo, melhor.best_params)
        est.fit(data.train_for(space.features), data.y_train)

        caminho = MODELS_DIR / f"ga_{modelo}{sufixo}.joblib"
        joblib.dump(est, caminho)

        vencedores[modelo] = {
            "experiment": melhor.config.name,
            "fitness_cv": melhor.best_fitness,
            "params": melhor.best_params,
            "model_path": str(caminho.relative_to(BASE_DIR)),
            "features": space.features,
        }
        logger.info(
            "vencedor %s: experimento '%s' (fitness CV %.4f) -> %s",
            modelo, melhor.config.name, melhor.best_fitness, caminho.name,
        )

    (REPORTS_DIR / f"ga_winners{sufixo}.json").write_text(
        json.dumps(vencedores, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return vencedores


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 3 — experimentos do GA")
    parser.add_argument(
        "--models", nargs="+", default=sorted(SEARCH_SPACES),
        choices=sorted(SEARCH_SPACES),
        help="modelos a otimizar (padrão: os 3)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="configurações reduzidas, só pra validar o pipeline",
    )
    parser.add_argument(
        "--fitness-robusto", action="store_true",
        help=(
            "roda só a config 'equilibrado' com CV repetida 5x3 em vez de "
            "5-fold simples. É o experimento corretivo do sobreajuste de "
            "fold diagnosticado na Etapa 4 — ver o relatório técnico."
        ),
    )
    args = parser.parse_args()

    if args.quick and args.fitness_robusto:
        parser.error("--quick e --fitness-robusto são mutuamente exclusivos")

    if args.fitness_robusto:
        sufixo = "_robusto"
        # Só a 'equilibrado': o objetivo aqui é isolar o efeito do
        # PROTOCOLO de avaliação, não comparar estratégias de GA de novo.
        # Rodar as três triplicaria o custo (15 fits por avaliação em vez
        # de 5) sem responder nada que a primeira rodada já não respondeu.
        suite = ExperimentSuite([EXPERIMENTOS[0]])
    elif args.quick:
        sufixo = "_quick"
        suite = ExperimentSuite(QUICK)
    else:
        sufixo = ""
        suite = ExperimentSuite(EXPERIMENTOS)

    logger = setup_logging("fase2.ga", log_file=f"ga_experiments{sufixo}.log")
    events = EventLogger(f"ga_events{sufixo}.jsonl")

    data = load_phase1_data()
    logger.info("dados da Fase 1 — %s", data.describe())
    logger.info("\n%s", search_space_table())
    logger.info("\n%s", suite.table())
    if not args.quick:
        logger.info(
            "orçamento por experimento: %d avaliações de fitness (igual pros 3)",
            ORCAMENTO,
        )

    fitness_config = FitnessConfig(cv_repeats=3 if args.fitness_robusto else 1)
    logger.info("fitness: %s", fitness_config.formula())
    if args.fitness_robusto:
        logger.info(
            "modo robusto: %d treinos por avaliação (era %d) — "
            "a média sobre 3 particionamentos diferentes reduz a chance de "
            "o GA explorar ruído de um fold específico",
            fitness_config.n_fits_por_avaliacao, FitnessConfig().n_fits_por_avaliacao,
        )

    resultados: list[GAResult] = []

    with timed(logger, f"{len(args.models)} modelos x {len(suite.configs)} experimentos"):
        for modelo in args.models:
            for config in suite.configs:
                resultados.append(
                    run_ga(modelo, data, config, fitness_config, logger, events)
                )

    salvar_resultados(resultados, sufixo)
    vencedores = treinar_modelos_vencedores(resultados, data, logger, sufixo)

    logger.info("\n--- resumo (fitness de CV no treino) ---")
    tabela = pd.DataFrame([
        {
            "modelo": r.model,
            "experimento": r.config.name,
            "fitness_cv": round(r.best_fitness, 4),
            "recall_mal_cv": round(r.best_result.recall_malignant, 4),
            "gerações": r.generations_run,
            "segundos": round(r.elapsed_s, 1),
            "aval. reais": r.evaluator_stats["avaliacoes_reais"],
            "cache": r.evaluator_stats["cache_hits"],
        }
        for r in resultados
    ])
    logger.info("\n%s", tabela.to_string(index=False))

    events.emit(
        "suite_end", n_runs=len(resultados), winners=vencedores,
        models=args.models, experiments=[c.name for c in suite.configs],
    )
    events.close()

    logger.info("\nresultados em: %s", REPORTS_DIR)
    logger.info("modelos otimizados em: %s/ga_*.joblib", MODELS_DIR)
    logger.info("próximo passo: python -m src.ga.compare")


if __name__ == "__main__":
    main()
