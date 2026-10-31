# Etapa 2 (parte 2) — Função fitness
#
# O fitness de um indivíduo é a qualidade do modelo que o cromossomo
# descreve. Duas decisões definem essa função:
#
# 1) QUAIS MÉTRICAS. As mesmas da Fase 1 (accuracy, recall do maligno,
#    F1 do maligno), combinadas numa média ponderada com o RECALL DO
#    MALIGNO dominante:
#
#        fitness = 0.60 * recall_mal + 0.25 * f1_mal + 0.15 * accuracy
#
#    O peso vem do custo clínico assimétrico que o relatório da Fase 1 já
#    tinha estabelecido: um falso negativo é um câncer que vai embora sem
#    tratamento; um falso positivo é uma biópsia desnecessária. Os dois
#    erros não são iguais, então a função objetivo não pode tratá-los
#    igual. O F1 entra pra impedir a solução degenerada — um modelo que
#    chamasse TUDO de maligno teria recall 1.0 e fitness alto se o recall
#    fosse o único termo; com F1 no meio, a precisão despencando puxa o
#    fitness pra baixo. A accuracy entra com peso pequeno, só pra desempatar.
#
# 2) ONDE MEDIR. Validação cruzada estratificada em 5 folds SOBRE O TREINO.
#    Esse é o ponto metodológico central da fase: se o fitness fosse
#    medido no conjunto de teste, o GA otimizaria direto contra ele e as
#    métricas finais seriam fantasia — o teste teria virado, de fato,
#    conjunto de treino, e qualquer ganho reportado na Etapa 4 seria
#    vazamento, não desempenho. Aqui o teste fica intocado até a
#    avaliação final.
#    Estratificado porque a proporção maligno/benigno (~37/63) tem que se
#    manter em cada fold — num fold desbalanceado o recall do maligno
#    ficaria instável e o fitness mediria ruído de amostragem.

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .data import POS_LABEL, Phase1Data
from .search_space import SearchSpace

RANDOM_STATE = 42

ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}


def build_estimator(model: str, params: dict[str, Any], random_state: int = RANDOM_STATE):
    """Instancia o estimador do sklearn a partir do fenótipo decodificado.

    `random_state` fixo em todos: sem isso, duas avaliações do MESMO
    cromossomo dariam fitness diferente (a Random Forest é estocástica),
    o cache de fitness seria inválido e a seleção estaria comparando
    ruído junto com sinal.
    """
    if model not in ESTIMATORS:
        raise ValueError(f"modelo '{model}' desconhecido; use {sorted(ESTIMATORS)}")

    kwargs = dict(params)
    kwargs["random_state"] = random_state
    if model == "random_forest":
        kwargs["n_jobs"] = -1
    return ESTIMATORS[model](**kwargs)


@dataclass(frozen=True)
class FitnessConfig:
    """Pesos e protocolo de avaliação do fitness."""

    w_recall: float = 0.60
    w_f1: float = 0.25
    w_accuracy: float = 0.15
    cv_folds: int = 5
    # Repetições da CV, cada uma com um particionamento diferente.
    # 1 = StratifiedKFold simples (o padrão da primeira rodada).
    # > 1 = RepeatedStratifiedKFold. Ver a nota sobre sobreajuste de fold
    # no cabeçalho do módulo e em reports/fase2/RELATORIO_TECNICO_FASE2.md.
    cv_repeats: int = 1
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        soma = self.w_recall + self.w_f1 + self.w_accuracy
        if abs(soma - 1.0) > 1e-9:
            raise ValueError(f"os pesos devem somar 1.0 (somaram {soma})")
        if self.cv_folds < 2:
            raise ValueError("cv_folds deve ser >= 2")
        if self.cv_repeats < 1:
            raise ValueError("cv_repeats deve ser >= 1")

    @property
    def n_fits_por_avaliacao(self) -> int:
        return self.cv_folds * self.cv_repeats

    def splitter(self):
        """O particionador da CV, conforme o número de repetições."""
        if self.cv_repeats == 1:
            return StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True,
                random_state=self.random_state,
            )
        return RepeatedStratifiedKFold(
            n_splits=self.cv_folds, n_repeats=self.cv_repeats,
            random_state=self.random_state,
        )

    def combine(self, recall: float, f1: float, accuracy: float) -> float:
        return (
            self.w_recall * recall
            + self.w_f1 * f1
            + self.w_accuracy * accuracy
        )

    def formula(self) -> str:
        protocolo = (
            f"CV estratificada {self.cv_folds}-fold"
            if self.cv_repeats == 1
            else f"CV estratificada {self.cv_folds}-fold x {self.cv_repeats} repetições"
        )
        return (
            f"{self.w_recall:.2f}*recall_mal + {self.w_f1:.2f}*f1_mal "
            f"+ {self.w_accuracy:.2f}*accuracy  "
            f"({protocolo} no treino)"
        )


@dataclass
class FitnessResult:
    """Fitness de um indivíduo + as métricas que o compõem."""

    fitness: float
    recall_malignant: float
    f1_malignant: float
    precision_malignant: float
    accuracy: float
    params: dict[str, Any]
    failed: bool = False
    error: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "fitness": round(self.fitness, 6),
            "recall_malignant": round(self.recall_malignant, 6),
            "f1_malignant": round(self.f1_malignant, 6),
            "precision_malignant": round(self.precision_malignant, 6),
            "accuracy": round(self.accuracy, 6),
            "failed": self.failed,
        }


def _metrics(y_true, y_pred) -> tuple[float, float, float, float]:
    # zero_division=0: uma árvore muito podada pode não prever nenhum
    # maligno; precisão aí é indefinida e o certo é tratar como 0, não
    # como 1 (que premiaria o modelo por não arriscar)
    return (
        recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        accuracy_score(y_true, y_pred),
    )


def evaluate_params(
    model: str,
    params: dict[str, Any],
    data: Phase1Data,
    features: str,
    config: FitnessConfig | None = None,
) -> FitnessResult:
    """Avalia um fenótipo por CV estratificada no treino.

    Um conjunto de hiperparâmetros que não treina (combinação que o
    sklearn rejeita, solver que não converge e estoura) recebe fitness 0
    em vez de propagar a exceção: o GA tem que continuar rodando, e
    fitness 0 já basta pra seleção descartar o indivíduo naturalmente.
    """
    config = config or FitnessConfig()
    X = data.train_for(features)
    y = data.y_train

    cv = config.splitter()

    recalls, f1s, precisions, accuracies = [], [], [], []
    try:
        with warnings.catch_warnings():
            # saga/liblinear com C alto e max_iter baixo avisa que não
            # convergiu; isso é informação pro GA (o fitness cai), não um
            # erro — silencio o aviso pra não poluir 50 gerações de log
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)

            for idx_tr, idx_va in cv.split(X, y):
                est = build_estimator(model, params, config.random_state)
                est.fit(X.iloc[idx_tr], y.iloc[idx_tr])
                y_pred = est.predict(X.iloc[idx_va])
                rec, f1, prec, acc = _metrics(y.iloc[idx_va], y_pred)
                recalls.append(rec)
                f1s.append(f1)
                precisions.append(prec)
                accuracies.append(acc)
    except Exception as exc:  # noqa: BLE001 — qualquer falha vira fitness 0
        return FitnessResult(
            fitness=0.0, recall_malignant=0.0, f1_malignant=0.0,
            precision_malignant=0.0, accuracy=0.0, params=params,
            failed=True, error=f"{type(exc).__name__}: {exc}",
        )

    recall, f1, precision, accuracy = (
        float(np.mean(recalls)), float(np.mean(f1s)),
        float(np.mean(precisions)), float(np.mean(accuracies)),
    )
    return FitnessResult(
        fitness=config.combine(recall, f1, accuracy),
        recall_malignant=recall, f1_malignant=f1,
        precision_malignant=precision, accuracy=accuracy, params=params,
    )


@dataclass
class FitnessEvaluator:
    """Avalia cromossomos, com cache por fenótipo.

    O cache não é só otimização: elitismo mais convergência fazem o MESMO
    indivíduo reaparecer em muitas gerações, e o decodificador mapeia
    genótipos distintos no mesmo fenótipo (dois valores diferentes de
    [0,1] caem na mesma faixa categórica). Sem cache, uma rodada de
    população 30 x 25 gerações treinaria 3.750 modelos x 5 folds; com
    cache, costuma cair pra menos de um terço disso. `n_evaluations`
    contra `n_cache_hits` é reportado no log de cada experimento.
    """

    model: str
    space: SearchSpace
    data: Phase1Data
    config: FitnessConfig = field(default_factory=FitnessConfig)
    _cache: dict[str, FitnessResult] = field(default_factory=dict, repr=False)
    n_evaluations: int = 0
    n_cache_hits: int = 0
    n_failures: int = 0

    @staticmethod
    def _key(params: dict[str, Any]) -> str:
        # repr dos pares ordenados: duas chaves iguais <=> mesmo modelo.
        # Floats entram com precisão reduzida de propósito — dois C que
        # diferem na 7a casa decimal treinam o mesmo modelo.
        #
        # A chave NÃO inclui o protocolo de CV porque o cache vive dentro
        # de um FitnessEvaluator, e cada avaliador tem um único
        # FitnessConfig. Dois protocolos diferentes são dois avaliadores,
        # com caches separados — não há como um contaminar o outro.
        partes = []
        for k in sorted(params):
            v = params[k]
            partes.append(f"{k}={round(v, 6) if isinstance(v, float) else v!r}")
        return "|".join(partes)

    def evaluate(self, chromosome) -> FitnessResult:
        params = self.space.decode(chromosome)
        key = self._key(params)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]

        result = evaluate_params(
            self.model, params, self.data, self.space.features, self.config
        )
        self.n_evaluations += 1
        if result.failed:
            self.n_failures += 1
        self._cache[key] = result
        return result

    def evaluate_population(self, population) -> tuple[np.ndarray, list[FitnessResult]]:
        results = [self.evaluate(ind) for ind in population]
        return np.array([r.fitness for r in results]), results

    def stats(self) -> dict[str, int]:
        return {
            "avaliacoes_reais": self.n_evaluations,
            "cache_hits": self.n_cache_hits,
            "falhas": self.n_failures,
        }
