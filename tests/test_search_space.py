# Testes da codificação (Etapa 1).
#
# O teste mais importante do arquivo é `test_todo_fenotipo_treina`: ele
# sorteia cromossomos aleatórios, decodifica e TREINA DE VERDADE cada
# fenótipo no sklearn. É o único jeito de garantir que o decodificador e
# o operador de reparo cobrem todas as combinações inválidas — um teste
# que só confira limites passaria mesmo com `solver='lbfgs'` colado em
# `penalty='l1'`, que o sklearn rejeita em tempo de execução.

from __future__ import annotations

import numpy as np
import pytest

from src.ga.fitness import build_estimator
from src.ga.search_space import (
    SEARCH_SPACES,
    CategoricalGene,
    FloatGene,
    IntGene,
    LOGISTIC_SPACE,
    FOREST_SPACE,
    search_space_size,
)

TODOS = list(SEARCH_SPACES.values())


# --- genes isolados ----------------------------------------------------


def test_gene_categorico_cobre_todas_as_escolhas():
    gene = CategoricalGene("x", ("a", "b", "c"))
    # amostra densa em [0,1] tem que produzir as três categorias
    decodificados = {gene.decode(v) for v in np.linspace(0, 1, 100)}
    assert decodificados == {"a", "b", "c"}


def test_gene_categorico_nao_estoura_no_limite_superior():
    """value == 1.0 cairia no índice len(choices) sem o min()."""
    gene = CategoricalGene("x", ("a", "b", "c"))
    assert gene.decode(1.0) == "c"
    assert gene.decode(0.0) == "a"


def test_gene_inteiro_atinge_os_extremos():
    gene = IntGene("x", 2, 20)
    assert gene.decode(0.0) == 2
    assert gene.decode(1.0) == 20
    assert gene.decode(0.5) == 11


def test_gene_inteiro_respeita_o_passo():
    gene = IntGene("n", 50, 400, step=10)
    valores = {gene.decode(v) for v in np.linspace(0, 1, 200)}
    assert min(valores) == 50
    assert max(valores) == 400
    assert all(v % 10 == 0 for v in valores)


def test_gene_float_log_distribui_por_ordem_de_magnitude():
    """Em escala log, o meio do intervalo [1e-3, 1e2] é 10^-0.5."""
    gene = FloatGene("C", 1e-3, 1e2, log=True)
    assert gene.decode(0.0) == pytest.approx(1e-3)
    assert gene.decode(1.0) == pytest.approx(1e2)
    assert gene.decode(0.5) == pytest.approx(10 ** -0.5)

    # metade dos sorteios tem que cair abaixo de 10^-0.5 — é exatamente o
    # efeito que a escala log existe pra produzir
    amostras = [gene.decode(v) for v in np.linspace(0, 1, 1000)]
    abaixo = sum(1 for a in amostras if a < 10 ** -0.5)
    assert 450 < abaixo < 550


def test_gene_float_linear_nao_usa_log():
    gene = FloatGene("ccp_alpha", 0.0, 0.05)
    assert gene.decode(0.0) == pytest.approx(0.0)
    assert gene.decode(0.5) == pytest.approx(0.025)
    assert gene.decode(1.0) == pytest.approx(0.05)


# --- espaços de busca --------------------------------------------------


@pytest.mark.parametrize("space", TODOS, ids=lambda s: s.model)
def test_decode_rejeita_cromossomo_de_tamanho_errado(space):
    with pytest.raises(ValueError, match="cromossomo"):
        space.decode([0.5] * (space.n_genes + 1))


@pytest.mark.parametrize("space", TODOS, ids=lambda s: s.model)
def test_random_chromosome_fica_no_intervalo_unitario(space, rng):
    for _ in range(50):
        c = space.random_chromosome(rng)
        assert len(c) == space.n_genes
        assert np.all((c >= 0.0) & (c <= 1.0))


@pytest.mark.parametrize("space", TODOS, ids=lambda s: s.model)
def test_decode_e_deterministico(space, rng):
    c = space.random_chromosome(rng)
    assert space.decode(c) == space.decode(c)


@pytest.mark.parametrize("space", TODOS, ids=lambda s: s.model)
def test_decode_devolve_todos_os_genes(space, rng):
    params = space.decode(space.random_chromosome(rng))
    assert set(params) == {g.name for g in space.genes}


@pytest.mark.parametrize("space", TODOS, ids=lambda s: s.model)
def test_valores_decodificados_respeitam_os_limites(space, rng):
    for _ in range(200):
        params = space.decode(space.random_chromosome(rng))
        for gene in space.genes:
            valor = params[gene.name]
            if isinstance(gene, CategoricalGene):
                # reparo pode trocar uma categoria por outra do mesmo gene,
                # mas nunca por algo fora da lista
                assert valor in gene.choices
            elif isinstance(gene, IntGene):
                assert isinstance(valor, int)
                assert gene.low <= valor <= gene.high
            elif valor is not None:  # l1_ratio vira None no reparo
                assert gene.low <= valor <= gene.high


# --- operador de reparo ------------------------------------------------


def test_reparo_lbfgs_forca_l2():
    """lbfgs não aceita l1 nem elasticnet no sklearn."""
    params = LOGISTIC_SPACE.repair({
        "C": 1.0, "penalty": "l1", "solver": "lbfgs",
        "max_iter": 1000, "l1_ratio": 0.5, "class_weight": None,
    })
    assert params["penalty"] == "l2"
    assert params["solver"] == "lbfgs"


def test_reparo_elasticnet_forca_saga():
    """elasticnet só existe no saga."""
    params = LOGISTIC_SPACE.repair({
        "C": 1.0, "penalty": "elasticnet", "solver": "liblinear",
        "max_iter": 1000, "l1_ratio": 0.3, "class_weight": None,
    })
    assert params["solver"] == "saga"
    assert params["penalty"] == "elasticnet"
    assert params["l1_ratio"] == 0.3  # aqui o l1_ratio É usado, mantém


def test_reparo_anula_l1_ratio_fora_do_elasticnet():
    params = LOGISTIC_SPACE.repair({
        "C": 1.0, "penalty": "l2", "solver": "liblinear",
        "max_iter": 1000, "l1_ratio": 0.7, "class_weight": None,
    })
    assert params["l1_ratio"] is None


def test_reparo_nao_altera_combinacao_valida():
    original = {
        "C": 1.0, "penalty": "l1", "solver": "liblinear",
        "max_iter": 1000, "l1_ratio": None, "class_weight": "balanced",
    }
    assert LOGISTIC_SPACE.repair(original) == original


def test_reparo_nao_muta_a_entrada():
    original = {
        "C": 1.0, "penalty": "l1", "solver": "lbfgs",
        "max_iter": 1000, "l1_ratio": 0.5, "class_weight": None,
    }
    copia = dict(original)
    LOGISTIC_SPACE.repair(original)
    assert original == copia


def test_reparo_floresta_normaliza_balanced_subsample_sem_bootstrap():
    params = FOREST_SPACE.repair({
        "n_estimators": 100, "max_depth": 5, "min_samples_split": 2,
        "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini",
        "class_weight": "balanced_subsample", "bootstrap": False,
    })
    assert params["class_weight"] == "balanced"


def test_reparo_floresta_mantem_balanced_subsample_com_bootstrap():
    params = FOREST_SPACE.repair({
        "n_estimators": 100, "max_depth": 5, "min_samples_split": 2,
        "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini",
        "class_weight": "balanced_subsample", "bootstrap": True,
    })
    assert params["class_weight"] == "balanced_subsample"


# --- o teste que importa de verdade ------------------------------------


@pytest.mark.parametrize("space", TODOS, ids=lambda s: s.model)
def test_todo_fenotipo_treina(space, data, rng):
    """Sorteia 25 cromossomos e treina cada fenótipo no sklearn.

    Se o decodificador ou o reparo deixarem passar uma combinação
    inválida, isto levanta exceção. É a rede de segurança que impede o
    GA de desperdiçar orçamento com indivíduos que não podem ser
    avaliados.
    """
    X = data.train_for(space.features)
    for _ in range(25):
        params = space.decode(space.random_chromosome(rng))
        est = build_estimator(space.model, params)
        est.fit(X, data.y_train)
        assert len(est.predict(X)) == len(data.y_train)


def test_search_space_size_reporta_eixos_continuos():
    assert "contínuo" in search_space_size(LOGISTIC_SPACE)
    # a floresta é 100% discreta
    assert "contínuo" not in search_space_size(FOREST_SPACE)


def test_espaco_da_floresta_e_grande_o_suficiente_para_justificar_ga():
    """Busca exaustiva está fora de questão — é o que motiva o GA."""
    texto = search_space_size(FOREST_SPACE)
    combinacoes = int(texto.split()[0].replace(",", ""))
    assert combinacoes > 1_000_000
