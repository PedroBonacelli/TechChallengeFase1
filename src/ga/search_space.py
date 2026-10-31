# Etapa 1 — Codificação do algoritmo genético (representação dos genes)
#
# CODIFICAÇÃO ESCOLHIDA: real normalizada ("random-key") com decodificação
# por tipo de gene.
#
# Cada cromossomo é um vetor de floats em [0, 1] — um por hiperparâmetro.
# O decodificador traduz cada posição pro domínio real do hiperparâmetro
# conforme o tipo do gene (categórico, inteiro ou contínuo). Ou seja: o
# espaço de busca é misto (há hiperparâmetro categórico, inteiro e
# contínuo), mas o GENÓTIPO é homogêneo.
#
# Por que assim e não codificação mista nativa:
#   - cruzamento fica genérico: trocar posições de dois vetores [0,1] é
#     sempre válido, sem precisar de um operador por tipo de gene;
#   - escala logarítmica sai de graça — o C da Logística varia de 1e-3 a
#     1e2, e amostrar uniformemente em [0,1] no expoente dá peso igual a
#     cada ordem de magnitude (amostrar linear em [0.001, 100] colocaria
#     99% da massa acima de 1);
#   - limites nunca são violados: basta manter o vetor em [0, 1], o que o
#     operador de mutação garante por clipping.
# A mutação, mesmo com genótipo homogêneo, É sensível ao tipo (ver
# operators.py): perturbação gaussiana não muda categoria na prática, então
# gene categórico é re-sorteado uniformemente.
#
# RESTRIÇÕES: alguns pares de hiperparâmetros são inválidos juntos
# (solver='lbfgs' não aceita penalty='l1'). Em vez de descartar o indivíduo
# — o que jogaria fora material genético e distorceria a seleção — uso um
# operador de REPARO (repair) que conserta o fenótipo depois de decodificar.
# É a forma padrão de lidar com restrição em GA quando o conserto é barato
# e óbvio.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

# -----------------------------------------------------------------------
# Tipos de gene
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class CategoricalGene:
    """Gene categórico: [0,1] é fatiado em len(choices) faixas iguais."""

    name: str
    choices: Sequence[Any]
    kind: Literal["categorical"] = "categorical"

    def decode(self, value: float) -> Any:
        n = len(self.choices)
        # min(...) protege o caso value == 1.0, que cairia no índice n
        idx = min(int(value * n), n - 1)
        return self.choices[idx]

    def describe(self) -> str:
        return f"categórico{list(self.choices)}"


@dataclass(frozen=True)
class IntGene:
    """Gene inteiro em [low, high], opcionalmente em passos de `step`."""

    name: str
    low: int
    high: int
    step: int = 1
    kind: Literal["int"] = "int"

    def decode(self, value: float) -> int:
        n_steps = (self.high - self.low) // self.step
        # round em vez de int(): arredondar pra baixo daria ao último valor
        # metade da probabilidade dos outros
        k = int(round(value * n_steps))
        return self.low + k * self.step

    def describe(self) -> str:
        passo = "" if self.step == 1 else f" passo {self.step}"
        return f"inteiro [{self.low}, {self.high}]{passo}"


@dataclass(frozen=True)
class FloatGene:
    """Gene contínuo em [low, high]; `log=True` amostra em escala log10."""

    name: str
    low: float
    high: float
    log: bool = False
    kind: Literal["float"] = "float"

    def decode(self, value: float) -> float:
        if self.log:
            lo, hi = np.log10(self.low), np.log10(self.high)
            return float(10 ** (lo + value * (hi - lo)))
        return float(self.low + value * (self.high - self.low))

    def describe(self) -> str:
        escala = " (log)" if self.log else ""
        return f"contínuo [{self.low}, {self.high}]{escala}"


Gene = CategoricalGene | IntGene | FloatGene


# -----------------------------------------------------------------------
# Espaço de busca por modelo
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class SearchSpace:
    """Espaço de busca de um modelo: lista ordenada de genes + reparo.

    A ordem da tupla É o layout do cromossomo — gene i ocupa a posição i.
    """

    model: str
    features: Literal["raw", "scaled"]
    genes: tuple[Gene, ...]

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    def decode(self, chromosome: Sequence[float]) -> dict[str, Any]:
        """Genótipo (vetor [0,1]) -> fenótipo (kwargs do estimador)."""
        if len(chromosome) != self.n_genes:
            raise ValueError(
                f"cromossomo com {len(chromosome)} genes, "
                f"esperado {self.n_genes} para {self.model}"
            )
        params = {g.name: g.decode(float(v)) for g, v in zip(self.genes, chromosome)}
        return self.repair(params)

    def repair(self, params: dict[str, Any]) -> dict[str, Any]:
        """Conserta combinações inválidas. Sobrescrito por modelo."""
        return params

    def random_chromosome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(self.n_genes)

    def summary(self) -> str:
        linhas = [f"{self.model} ({self.n_genes} genes, features={self.features})"]
        for i, g in enumerate(self.genes):
            linhas.append(f"  [{i}] {g.name:20s} {g.describe()}")
        return "\n".join(linhas)


class LogisticRegressionSpace(SearchSpace):
    def repair(self, params: dict[str, Any]) -> dict[str, Any]:
        """Compatibilidade solver x penalty da LogisticRegression do sklearn.

        - 'lbfgs' só aceita 'l2' (ou None);
        - 'elasticnet' só existe no 'saga';
        - 'liblinear' aceita l1/l2 mas não elasticnet;
        - l1_ratio só é lido quando penalty='elasticnet' — fora disso tem
          que ir como None, senão o sklearn avisa que o parâmetro é inútil.
        """
        p = dict(params)
        if p["penalty"] == "elasticnet":
            p["solver"] = "saga"
        elif p["solver"] == "lbfgs" and p["penalty"] != "l2":
            p["penalty"] = "l2"
        if p["penalty"] != "elasticnet":
            p["l1_ratio"] = None
        return p


class RandomForestSpace(SearchSpace):
    def repair(self, params: dict[str, Any]) -> dict[str, Any]:
        """class_weight='balanced_subsample' só faz sentido com bootstrap.

        Sem bootstrap toda árvore vê a amostra inteira, então o peso
        "por subamostra" vira o 'balanced' comum — o sklearn não reclama,
        mas deixar dois nomes pro mesmo comportamento polui o log de
        convergência com indivíduos que parecem diferentes e não são.
        """
        p = dict(params)
        if not p["bootstrap"] and p["class_weight"] == "balanced_subsample":
            p["class_weight"] = "balanced"
        return p


# Logística opera nas features escalonadas (é sensível à escala);
# árvore e floresta nas cruas (invariantes a transformação monotônica) —
# mesma divisão da Fase 1.
LOGISTIC_SPACE = LogisticRegressionSpace(
    model="logistic_regression",
    features="scaled",
    genes=(
        FloatGene("C", 1e-3, 1e2, log=True),
        CategoricalGene("penalty", ("l1", "l2", "elasticnet")),
        CategoricalGene("solver", ("liblinear", "lbfgs", "saga")),
        IntGene("max_iter", 200, 5000, step=200),
        FloatGene("l1_ratio", 0.0, 1.0),
        CategoricalGene("class_weight", ("balanced", None)),
    ),
)

TREE_SPACE = SearchSpace(
    model="decision_tree",
    features="raw",
    genes=(
        CategoricalGene("criterion", ("gini", "entropy", "log_loss")),
        IntGene("max_depth", 2, 20),
        IntGene("min_samples_split", 2, 30),
        IntGene("min_samples_leaf", 1, 20),
        CategoricalGene("max_features", (None, "sqrt", "log2")),
        # poda por complexidade: o regularizador mais eficaz numa árvore só,
        # e a Fase 1 nem tocou nele (usou apenas max_depth=5)
        FloatGene("ccp_alpha", 0.0, 0.05),
        CategoricalGene("class_weight", ("balanced", None)),
    ),
)

FOREST_SPACE = RandomForestSpace(
    model="random_forest",
    features="raw",
    genes=(
        IntGene("n_estimators", 50, 400, step=10),
        IntGene("max_depth", 2, 20),
        IntGene("min_samples_split", 2, 30),
        IntGene("min_samples_leaf", 1, 10),
        CategoricalGene("max_features", ("sqrt", "log2", 0.3, 0.5, 0.7)),
        CategoricalGene("criterion", ("gini", "entropy")),
        CategoricalGene("class_weight", ("balanced", "balanced_subsample", None)),
        CategoricalGene("bootstrap", (True, False)),
    ),
)

SEARCH_SPACES: dict[str, SearchSpace] = {
    "logistic_regression": LOGISTIC_SPACE,
    "decision_tree": TREE_SPACE,
    "random_forest": FOREST_SPACE,
}

# Hiperparâmetros exatos da Fase 1 (src/modeling.py) — o ponto de partida
# que o GA tem que bater. Usados no comparativo da Etapa 4 e como
# referência no relatório.
BASELINE_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "C": 1.0, "penalty": "l2", "solver": "lbfgs",
        "max_iter": 5000, "l1_ratio": None, "class_weight": "balanced",
    },
    "decision_tree": {
        "criterion": "gini", "max_depth": 5, "min_samples_split": 2,
        "min_samples_leaf": 1, "max_features": None, "ccp_alpha": 0.0,
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 300, "max_depth": 8, "min_samples_split": 2,
        "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini",
        "class_weight": "balanced", "bootstrap": True,
    },
}


def search_space_size(space: SearchSpace) -> str:
    """Tamanho (ou cardinalidade) do espaço, pro relatório.

    Genes contínuos tornam o espaço infinito; nesse caso reporto a
    combinação discreta e marco quantos eixos são contínuos.
    """
    discreto = 1
    n_continuos = 0
    for g in space.genes:
        if isinstance(g, CategoricalGene):
            discreto *= len(g.choices)
        elif isinstance(g, IntGene):
            discreto *= (g.high - g.low) // g.step + 1
        else:
            n_continuos += 1
    if n_continuos:
        return f"{discreto:,} combinações discretas x {n_continuos} eixo(s) contínuo(s)"
    return f"{discreto:,} combinações"


def search_space_table() -> str:
    """Resumo dos 3 espaços de busca, pro relatório."""
    blocos = []
    for space in SEARCH_SPACES.values():
        blocos.append(f"{space.summary()}\n  -> {search_space_size(space)}")
    return "\n\n".join(blocos)


if __name__ == "__main__":
    print(search_space_table())
    print("\n--- exemplo de decodificação (vetor aleatório) ---")
    rng = np.random.default_rng(42)
    for name, space in SEARCH_SPACES.items():
        chrom = space.random_chromosome(rng)
        print(f"\n{name}")
        print(f"  genótipo:  {np.round(chrom, 3).tolist()}")
        print(f"  fenótipo:  {space.decode(chrom)}")
