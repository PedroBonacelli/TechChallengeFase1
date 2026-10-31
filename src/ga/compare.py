# Etapa 4 — Comparar otimizado vs. original
#
# Avalia os modelos da Fase 1 e os otimizados pelo GA no MESMO conjunto de
# teste (as 114 amostras separadas por src/preprocessing.py, que o GA
# nunca viu) e gera a tabela comparativa e os gráficos.
#
# Duas comparações, porque elas respondem perguntas diferentes:
#
#   - FITNESS DE CV (no treino): foi o que o GA otimizou. É a comparação
#     justa entre as configurações de hiperparâmetros, porque é a métrica
#     que a busca de fato viu. Avalio os hiperparâmetros da Fase 1 com a
#     mesma função fitness pra ter o ponto de partida na mesma régua.
#   - MÉTRICAS NO TESTE: é o que importa clinicamente. Serve pra responder
#     se o ganho do GA sobreviveu fora do treino ou se foi só a busca se
#     ajustando ao ruído dos folds.
#
# O segundo número pode ser PIOR que o primeiro sugeria, e isso não é um
# bug — é o resultado honesto de um conjunto de teste de 114 amostras, em
# que cada caso vale ~0,9 ponto percentual.
#
#   python -m src.ga.compare

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..monitoring import setup_logging
from .data import POS_LABEL, load_phase1_data
from .fitness import FitnessConfig, build_estimator, evaluate_params
from .search_space import BASELINE_PARAMS, SEARCH_SPACES

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "fase2"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

MODELOS = ("logistic_regression", "decision_tree", "random_forest")
NOMES_CURTOS = {
    "logistic_regression": "Reg. Logística",
    "decision_tree": "Árvore de Decisão",
    "random_forest": "Random Forest",
}

# ---------------------------------------------------------------------
# Paleta — slots 1-3 da paleta categórica validada (passa os testes de
# separação para daltonismo em todos os pares). Cor segue a ENTIDADE:
# cada experimento tem sempre a mesma cor em todos os gráficos.
# ---------------------------------------------------------------------
COR_EXPERIMENTO = {
    "equilibrado": "#2a78d6",   # azul
    "exploratorio": "#eb6834",  # laranja
    "intensivo": "#1baf7a",     # aqua
}
COR_ORIGINAL = "#8a8a85"    # cinza: a linha de base é recessiva
COR_OTIMIZADO = "#2a78d6"   # azul: o resultado novo
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRADE = "#e3e2de"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRADE,
    "axes.labelcolor": TINTA_SECUNDARIA,
    "axes.titlecolor": TINTA_PRIMARIA,
    "text.color": TINTA_PRIMARIA,
    "xtick.color": TINTA_SECUNDARIA,
    "ytick.color": TINTA_SECUNDARIA,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------
# Avaliação no teste
# ---------------------------------------------------------------------


def metricas_no_teste(est, X_test, y_test) -> dict[str, Any]:
    y_pred = est.predict(X_test)
    y_proba = est.predict_proba(X_test)[:, POS_LABEL]

    # labels=[0,1] fixa a ordem da matriz mesmo que um modelo degenerado
    # preveja só uma classe — sem isso cm viraria 1x1 e a indexação quebraria
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # linha 0 = maligno real, linha 1 = benigno real
    tp_mal, fn_mal = int(cm[0, 0]), int(cm[0, 1])
    fp_mal, tn_mal = int(cm[1, 0]), int(cm[1, 1])

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_malignant": precision_score(
            y_test, y_pred, pos_label=POS_LABEL, zero_division=0
        ),
        "recall_malignant": recall_score(
            y_test, y_pred, pos_label=POS_LABEL, zero_division=0
        ),
        "f1_malignant": f1_score(
            y_test, y_pred, pos_label=POS_LABEL, zero_division=0
        ),
        "roc_auc": roc_auc_score((y_test == POS_LABEL).astype(int), y_proba),
        # nomes em termos clínicos: "falso negativo" = maligno dito benigno
        "verdadeiros_positivos_mal": tp_mal,
        "falsos_negativos": fn_mal,
        "falsos_positivos": fp_mal,
        "verdadeiros_negativos": tn_mal,
        "matriz_confusao": cm.tolist(),
    }


def baseline_refeito(modelo: str, data, logger):
    """Retreina o modelo da Fase 1 a partir dos hiperparâmetros dela.

    Por que refazer em vez de carregar `models/{modelo}.joblib`:

    Os .joblib da Fase 1 foram serializados com scikit-learn 1.8.0 e o
    ambiente da Fase 2 roda 1.4.2 — carregar levanta
    `InconsistentVersionWarning` e, na Regressão Logística,
    `AttributeError` em `predict_proba` (o atributo `multi_class` deixou
    de existir). É literalmente o problema que `requirements-lock.txt`
    existe pra prevenir, aparecendo dentro do próprio projeto.

    Refazer é também metodologicamente melhor, independente da versão:
    original e otimizado passam a ser treinados pela MESMA biblioteca, no
    MESMO caminho de código, sobre os MESMOS dados. Se o baseline viesse
    de um pickle de outra versão, uma diferença nas métricas poderia ser
    da versão do sklearn e não do GA — e não haveria como separar as duas
    coisas.

    Os hiperparâmetros vêm de `BASELINE_PARAMS`, transcritos de
    `src/modeling.py` da Fase 1. A função compara o resultado com as
    métricas publicadas em `reports/model_comparison.csv` e avisa se
    divergirem.
    """
    space = SEARCH_SPACES[modelo]
    est = build_estimator(modelo, BASELINE_PARAMS[modelo])
    est.fit(data.train_for(space.features), data.y_train)
    return est


def conferir_contra_fase1(comparativo: pd.DataFrame, logger) -> pd.DataFrame:
    """Compara o baseline refeito com as métricas publicadas na Fase 1.

    Divergência pequena é esperada (versão diferente do sklearn pode
    mudar desempate de split de árvore, por exemplo). Divergência grande
    significaria que `BASELINE_PARAMS` não corresponde ao que a Fase 1
    realmente treinou, e aí o comparativo inteiro estaria medindo a coisa
    errada.
    """
    caminho = BASE_DIR / "reports" / "model_comparison.csv"
    if not caminho.exists():
        logger.warning("reports/model_comparison.csv não encontrado; "
                       "não foi possível conferir contra a Fase 1")
        return pd.DataFrame()

    fase1 = pd.read_csv(caminho).set_index("model")
    linhas = []
    for _, r in comparativo.iterrows():
        if r["model"] not in fase1.index:
            continue
        pub = fase1.loc[r["model"]]
        for metrica in ("accuracy", "recall_malignant", "f1_malignant"):
            publicado = float(pub[metrica])
            refeito = float(r[f"{metrica}_original"])
            linhas.append({
                "model": r["model"],
                "metrica": metrica,
                "fase1_publicado": round(publicado, 4),
                "refeito_nesta_versao": round(refeito, 4),
                "diferenca": round(refeito - publicado, 4),
            })

    conferencia = pd.DataFrame(linhas)
    divergentes = conferencia[conferencia["diferenca"].abs() > 0.02]
    if len(divergentes):
        logger.warning(
            "o baseline refeito divergiu da Fase 1 em %d métrica(s) "
            "acima de 2 p.p.:\n%s",
            len(divergentes), divergentes.to_string(index=False),
        )
    else:
        logger.info(
            "baseline refeito reproduz a Fase 1 (maior diferença: %.4f)",
            conferencia["diferenca"].abs().max(),
        )
    return conferencia


def construir_comparativo(data, logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tabela original vs. otimizado, no teste e em fitness de CV."""
    vencedores_path = REPORTS_DIR / "ga_winners.json"
    if not vencedores_path.exists():
        raise FileNotFoundError(
            f"{vencedores_path} não existe — rode primeiro: "
            "python -m src.ga.run_experiments"
        )
    vencedores = json.loads(vencedores_path.read_text(encoding="utf-8"))
    fitness_config = FitnessConfig()

    linhas, detalhe = [], {}
    for modelo in MODELOS:
        space = SEARCH_SPACES[modelo]
        X_test = data.test_for(space.features)

        # --- original (Fase 1), retreinado nesta versão do sklearn ---
        params_orig = BASELINE_PARAMS[modelo]
        est_orig = baseline_refeito(modelo, data, logger)
        m_orig = metricas_no_teste(est_orig, X_test, data.y_test)
        cv_orig = evaluate_params(
            modelo, params_orig, data, space.features, fitness_config
        )

        # --- otimizado (GA) ---
        info = vencedores[modelo]
        est_ga = joblib.load(BASE_DIR / info["model_path"])
        m_ga = metricas_no_teste(est_ga, X_test, data.y_test)
        cv_ga = evaluate_params(
            modelo, info["params"], data, space.features, fitness_config
        )

        linhas.append({
            "model": modelo,
            "modelo": NOMES_CURTOS[modelo],
            "experimento_vencedor": info["experiment"],
            "fitness_cv_original": round(cv_orig.fitness, 4),
            "fitness_cv_otimizado": round(cv_ga.fitness, 4),
            "delta_fitness_cv": round(cv_ga.fitness - cv_orig.fitness, 4),
            **{f"{k}_original": round(v, 4) if isinstance(v, float) else v
               for k, v in m_orig.items() if k != "matriz_confusao"},
            **{f"{k}_otimizado": round(v, 4) if isinstance(v, float) else v
               for k, v in m_ga.items() if k != "matriz_confusao"},
            "delta_recall_malignant": round(
                m_ga["recall_malignant"] - m_orig["recall_malignant"], 4
            ),
            "delta_accuracy": round(m_ga["accuracy"] - m_orig["accuracy"], 4),
            "delta_f1_malignant": round(
                m_ga["f1_malignant"] - m_orig["f1_malignant"], 4
            ),
            "delta_roc_auc": round(m_ga["roc_auc"] - m_orig["roc_auc"], 4),
            "delta_falsos_negativos": m_ga["falsos_negativos"] - m_orig["falsos_negativos"],
            "delta_falsos_positivos": m_ga["falsos_positivos"] - m_orig["falsos_positivos"],
        })

        detalhe[modelo] = {
            "experimento_vencedor": info["experiment"],
            "params_original": params_orig,
            "params_otimizado": info["params"],
            "teste_original": m_orig,
            "teste_otimizado": m_ga,
            "cv_original": cv_orig.as_row(),
            "cv_otimizado": cv_ga.as_row(),
        }

        logger.info(
            "%-20s | recall_mal %.4f -> %.4f | FN %d -> %d | acc %.4f -> %.4f",
            modelo, m_orig["recall_malignant"], m_ga["recall_malignant"],
            m_orig["falsos_negativos"], m_ga["falsos_negativos"],
            m_orig["accuracy"], m_ga["accuracy"],
        )

    return pd.DataFrame(linhas), detalhe


# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------


def _rotular_fim_da_linha(ax, x, y, texto, cor, desvio_y: float = 0.0) -> None:
    """Rótulo direto no fim da série — evita depender só da legenda.

    `desvio_y` (em pontos) escalona rótulos que cairiam um sobre o outro.
    Duas configurações do GA que convergem pro MESMO fitness terminam na
    mesma altura, e aí os dois rótulos se sobrepõem — foi o que aconteceu
    com 'equilibrado' e 'exploratorio' na Random Forest, ambos em 0.9533.
    """
    ax.annotate(
        texto, xy=(x, y), xytext=(5, desvio_y), textcoords="offset points",
        color=cor, fontsize=8.5, fontweight="medium",
        va="center", ha="left",
    )


def _desvios_sem_colisao(valores_finais: list[float], span: float) -> list[float]:
    """Desvio vertical de cada rótulo pra que nenhum cubra o outro.

    Séries cujo valor final fica a menos de 4% da altura do eixo umas das
    outras são consideradas colidentes e recebem desvios alternados.
    """
    if span <= 0:
        return [0.0] * len(valores_finais)

    ordem = sorted(range(len(valores_finais)), key=lambda i: valores_finais[i])
    desvios = [0.0] * len(valores_finais)
    grupo: list[int] = []

    def distribuir(indices: list[int]) -> None:
        if len(indices) < 2:
            return
        # espalha simetricamente em torno de zero: -9, 0, +9 para três
        passo = 9.0
        inicio = -passo * (len(indices) - 1) / 2
        for pos, i in enumerate(indices):
            desvios[i] = inicio + pos * passo

    for i in ordem:
        if grupo and abs(valores_finais[i] - valores_finais[grupo[-1]]) / span > 0.04:
            distribuir(grupo)
            grupo = []
        grupo.append(i)
    distribuir(grupo)
    return desvios


def grafico_convergencia(historico: pd.DataFrame) -> Path:
    """Fitness do melhor indivíduo por geração — um painel por modelo.

    Pequenos múltiplos em vez de um gráfico só: as três curvas de um
    modelo vivem numa faixa estreita de fitness (0.95-0.98), e sobrepor
    os 3 modelos no mesmo eixo achataria tudo. Cada painel tem seu
    próprio zoom no eixo y.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, modelo in zip(axes, MODELOS):
        do_modelo = historico[historico["model"] == modelo]
        series = []
        for experimento, grupo in do_modelo.groupby("experiment", sort=False):
            grupo = grupo.sort_values("generation")
            cor = COR_EXPERIMENTO[experimento]
            ax.plot(
                grupo["generation"], grupo["best_fitness"],
                color=cor, linewidth=2, label=experimento, solid_capstyle="round",
            )
            series.append((
                grupo["generation"].iloc[-1], grupo["best_fitness"].iloc[-1],
                experimento[:4], cor,
            ))

        finais = [s[1] for s in series]
        span = max(finais) - min(finais)
        for (x, y, rotulo, cor), desvio in zip(
            series, _desvios_sem_colisao(finais, span)
        ):
            _rotular_fim_da_linha(ax, x, y, rotulo, cor, desvio)

        ax.set_title(NOMES_CURTOS[modelo], fontsize=11, pad=10)
        ax.set_xlabel("geração")
        ax.grid(True, axis="y", alpha=0.7)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.margins(x=0.16)
    axes[0].set_ylabel("melhor fitness na população")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.04), fontsize=9,
    )
    fig.suptitle(
        "Convergência do GA — melhor fitness por geração "
        "(orçamento igual de 600 avaliações nos 3 experimentos)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    caminho = FIG_DIR / "30_ga_convergencia.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def grafico_melhor_vs_medio(historico: pd.DataFrame, vencedores: dict) -> Path:
    """Melhor vs. médio da população, no experimento vencedor de cada modelo.

    A distância entre as duas linhas é a leitura importante: ela encolher
    significa que a população inteira está migrando pro ótimo, não só o
    elite sendo copiado. Se o médio ficasse parado enquanto o melhor
    sobe, o GA estaria funcionando como busca aleatória com memória.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, modelo in zip(axes, MODELOS):
        experimento = vencedores[modelo]["experiment"]
        g = historico[
            (historico["model"] == modelo) & (historico["experiment"] == experimento)
        ].sort_values("generation")

        ax.fill_between(
            g["generation"],
            g["mean_fitness"] - g["std_fitness"],
            g["mean_fitness"] + g["std_fitness"],
            color=COR_OTIMIZADO, alpha=0.12, linewidth=0,
        )
        ax.plot(g["generation"], g["best_fitness"], color=COR_OTIMIZADO,
                linewidth=2, label="melhor", solid_capstyle="round")
        ax.plot(g["generation"], g["mean_fitness"], color=TINTA_SECUNDARIA,
                linewidth=2, linestyle="--", label="médio (±1 desvio)")

        # Quando a população alcança o elite, as duas linhas terminam
        # praticamente na mesma altura e os rótulos se sobrepõem — é o
        # caso da Logística, onde melhor e médio convergem.
        x_fim = g["generation"].iloc[-1]
        finais = [g["best_fitness"].iloc[-1], g["mean_fitness"].iloc[-1]]
        eixo = ax.get_ylim()
        desvios = _desvios_sem_colisao(finais, eixo[1] - eixo[0])
        _rotular_fim_da_linha(ax, x_fim, finais[0], "melhor",
                              COR_OTIMIZADO, desvios[0])
        _rotular_fim_da_linha(ax, x_fim, finais[1], "médio",
                              TINTA_SECUNDARIA, desvios[1])

        ax.set_title(f"{NOMES_CURTOS[modelo]}\nexperimento '{experimento}'",
                     fontsize=11, pad=10)
        ax.set_xlabel("geração")
        ax.grid(True, axis="y", alpha=0.7)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.margins(x=0.16)
    axes[0].set_ylabel("fitness")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(
        "A população acompanha o elite? Melhor vs. fitness médio",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    caminho = FIG_DIR / "31_ga_melhor_vs_medio.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def grafico_diversidade(historico: pd.DataFrame) -> Path:
    """Diversidade genética por geração (desvio médio por gene).

    Em painel separado, nunca num segundo eixo y sobre a convergência:
    fitness e diversidade têm escalas e significados distintos, e
    empilhá-los no mesmo gráfico com dois eixos convida à leitura errada
    de que um causa o outro.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, modelo in zip(axes, MODELOS):
        do_modelo = historico[historico["model"] == modelo]
        series = []
        for experimento, grupo in do_modelo.groupby("experiment", sort=False):
            grupo = grupo.sort_values("generation")
            cor = COR_EXPERIMENTO[experimento]
            ax.plot(grupo["generation"], grupo["diversity"], color=cor,
                    linewidth=2, label=experimento, solid_capstyle="round")
            series.append((
                grupo["generation"].iloc[-1], grupo["diversity"].iloc[-1],
                experimento[:4], cor,
            ))

        finais = [s[1] for s in series]
        span = max(finais) - min(finais)
        for (x, y, rotulo, cor), desvio in zip(
            series, _desvios_sem_colisao(finais, span)
        ):
            _rotular_fim_da_linha(ax, x, y, rotulo, cor, desvio)

        ax.set_title(NOMES_CURTOS[modelo], fontsize=11, pad=10)
        ax.set_xlabel("geração")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.7)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.margins(x=0.16)
    axes[0].set_ylabel("diversidade (desvio médio por gene)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle(
        "Diversidade da população — queda rápida indica convergência prematura",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    caminho = FIG_DIR / "32_ga_diversidade.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def grafico_comparativo_metricas(comparativo: pd.DataFrame) -> Path:
    """Original vs. otimizado — um painel por métrica.

    Barras com rótulo direto em cada uma: a diferença entre 0.976 e 0.952
    é invisível a olho nu numa barra, então o número tem que estar
    escrito. O eixo começa em 0.80 (não em 0) de propósito, e o rótulo
    direto é o que impede a leitura exagerada que um eixo truncado
    normalmente causa.
    """
    metricas = [
        ("recall_malignant", "Recall do maligno\n(casos malignos detectados)"),
        ("f1_malignant", "F1 do maligno"),
        ("accuracy", "Acurácia"),
        ("roc_auc", "ROC AUC"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    x = np.arange(len(MODELOS))
    largura = 0.36

    for ax, (chave, titulo) in zip(axes, metricas):
        orig = comparativo[f"{chave}_original"].to_numpy()
        otim = comparativo[f"{chave}_otimizado"].to_numpy()

        # gap de 2px entre as barras adjacentes (largura - 0.02)
        b1 = ax.bar(x - largura / 2 - 0.01, orig, largura, color=COR_ORIGINAL,
                    label="original (Fase 1)")
        b2 = ax.bar(x + largura / 2 + 0.01, otim, largura, color=COR_OTIMIZADO,
                    label="otimizado (GA)")
        for barras in (b1, b2):
            ax.bar_label(barras, fmt="%.3f", fontsize=8,
                         color=TINTA_SECUNDARIA, padding=2)

        piso = min(orig.min(), otim.min())
        ax.set_ylim(max(0.0, piso - 0.08), 1.035)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [NOMES_CURTOS[m].replace(" ", "\n") for m in MODELOS], fontsize=9
        )
        ax.set_title(titulo, fontsize=10.5, pad=10)
        ax.grid(True, axis="y", alpha=0.7)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle(
        "Métricas no conjunto de teste (114 amostras, 42 malignos) — "
        "antes e depois do GA",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    caminho = FIG_DIR / "33_comparativo_metricas.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def grafico_matrizes_confusao(detalhe: dict) -> Path:
    """Matrizes de confusão: original (linha de cima) vs. otimizado.

    Rampa sequencial de um hue só (azul claro -> escuro), nunca arco-íris:
    a quantidade de casos é uma magnitude, e magnitude se lê por
    intensidade. Os números estão escritos em todas as células, então a
    cor é reforço, não a única informação.
    """
    rotulos = ["maligno", "benigno"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "azul_seq", ["#eef5fd", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"]
    )

    for coluna, modelo in enumerate(MODELOS):
        for linha, versao in enumerate(("teste_original", "teste_otimizado")):
            ax = axes[linha, coluna]
            cm = np.array(detalhe[modelo][versao]["matriz_confusao"])
            ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() * 1.05)

            for i in range(2):
                for j in range(2):
                    # texto claro sobre célula escura, escuro sobre clara
                    escura = cm[i, j] > cm.max() * 0.55
                    ax.text(
                        j, i, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=17, fontweight="semibold",
                        color="white" if escura else TINTA_PRIMARIA,
                    )
            # destaca a célula do erro grave: maligno previsto como benigno
            ax.add_patch(plt.Rectangle(
                (0.5, -0.5), 1, 1, fill=False, edgecolor="#e34948",
                linewidth=2.5, zorder=5,
            ))

            ax.set_xticks([0, 1], [f"previsto\n{r}" for r in rotulos], fontsize=9)
            ax.set_yticks([0, 1], [f"real\n{r}" for r in rotulos], fontsize=9)
            nome_versao = "original" if linha == 0 else "otimizado (GA)"
            fn = cm[0, 1]
            ax.set_title(
                f"{NOMES_CURTOS[modelo]} — {nome_versao}\n"
                f"{fn} maligno(s) não detectado(s)",
                fontsize=10.5, pad=10,
            )
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)

    fig.suptitle(
        "Matrizes de confusão no teste — a moldura vermelha marca o falso "
        "negativo (maligno classificado como benigno)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    caminho = FIG_DIR / "34_matrizes_confusao.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------
# Saída em markdown
# ---------------------------------------------------------------------


def analisar_generalizacao(data, logger) -> pd.DataFrame:
    """O fitness de CV prevê o desempenho no teste?

    Esta é a análise que explica o resultado da Etapa 4. Avalia no teste
    TODOS os 9 indivíduos vencedores (3 modelos x 3 experimentos), não só
    o melhor de cada modelo, e põe lado a lado o fitness de CV e as
    métricas de teste.

    A pergunta que ela responde: quando o GA melhora o fitness de CV, o
    desempenho no teste acompanha? Se a correlação for fraca ou negativa,
    o ganho de CV é exploração de ruído dos folds — e qualquer conclusão
    baseada só no fitness estaria errada.

    IMPORTANTE: isto é diagnóstico, não critério de seleção. O vencedor
    de cada modelo continua escolhido pelo fitness de CV (ver
    run_experiments.treinar_modelos_vencedores). Escolher pelo teste
    seria vazamento — é justamente o erro que esta análise serve pra
    tornar visível, não pra cometer.
    """
    caminho = REPORTS_DIR / "ga_experiments.json"
    if not caminho.exists():
        return pd.DataFrame()
    experimentos = json.loads(caminho.read_text(encoding="utf-8"))

    linhas = []
    for modelo in MODELOS:
        space = SEARCH_SPACES[modelo]
        X_train = data.train_for(space.features)
        X_test = data.test_for(space.features)

        # o baseline da Fase 1 entra como ponto de referência na mesma escala
        est_base = build_estimator(modelo, BASELINE_PARAMS[modelo])
        est_base.fit(X_train, data.y_train)
        m_base = metricas_no_teste(est_base, X_test, data.y_test)
        cv_base = evaluate_params(
            modelo, BASELINE_PARAMS[modelo], data, space.features
        )
        linhas.append({
            "model": modelo, "origem": "baseline Fase 1",
            "fitness_cv": round(cv_base.fitness, 4),
            "teste_recall_mal": round(m_base["recall_malignant"], 4),
            "teste_f1_mal": round(m_base["f1_malignant"], 4),
            "teste_accuracy": round(m_base["accuracy"], 4),
            "teste_falsos_negativos": m_base["falsos_negativos"],
            "teste_falsos_positivos": m_base["falsos_positivos"],
        })

        for exp in (e for e in experimentos if e["model"] == modelo):
            est = build_estimator(modelo, exp["best_params"])
            est.fit(X_train, data.y_train)
            m = metricas_no_teste(est, X_test, data.y_test)
            linhas.append({
                "model": modelo, "origem": f"GA/{exp['experiment']}",
                "fitness_cv": round(exp["best_fitness_cv"], 4),
                "teste_recall_mal": round(m["recall_malignant"], 4),
                "teste_f1_mal": round(m["f1_malignant"], 4),
                "teste_accuracy": round(m["accuracy"], 4),
                "teste_falsos_negativos": m["falsos_negativos"],
                "teste_falsos_positivos": m["falsos_positivos"],
            })

    tabela = pd.DataFrame(linhas)

    # Correlação de Spearman (ordem, não valor): a pergunta é se o
    # RANQUEAMENTO por fitness de CV reproduz o ranqueamento por
    # desempenho no teste. Pearson mediria força linear, que não é o que
    # interessa pra uma decisão de seleção.
    correlacoes = []
    for modelo, grupo in tabela.groupby("model", sort=False):
        if len(grupo) < 3:
            continue
        for metrica in ("teste_f1_mal", "teste_accuracy", "teste_recall_mal"):
            if grupo[metrica].nunique() < 2:
                rho = float("nan")  # métrica constante: correlação indefinida
            else:
                rho = grupo["fitness_cv"].corr(grupo[metrica], method="spearman")
            correlacoes.append({
                "model": modelo, "metrica_de_teste": metrica,
                "spearman_vs_fitness_cv": None if pd.isna(rho) else round(rho, 3),
                "n_pontos": len(grupo),
                "metrica_constante_no_teste": bool(grupo[metrica].nunique() < 2),
            })

    corr_df = pd.DataFrame(correlacoes)
    tabela.to_csv(REPORTS_DIR / "generalizacao.csv", index=False)
    if not corr_df.empty:
        corr_df.to_csv(REPORTS_DIR / "generalizacao_correlacao.csv", index=False)

    logger.info("--- CV vs. teste (todos os 9 vencedores + baseline) ---")
    logger.info("\n%s", tabela.to_string(index=False))
    if not corr_df.empty:
        logger.info("\ncorrelação de Spearman (fitness de CV x métrica de teste)")
        logger.info("\n%s", corr_df.to_string(index=False))

    return tabela


def analisar_experimento_robusto(data, logger) -> pd.DataFrame:
    """Compara CV 5-fold simples contra CV 5x3 repetida (Etapa 6 do relatório).

    Só roda se `ga_winners_robusto.json` existir (gerado por
    `run_experiments --fitness-robusto`). Põe três colunas lado a lado
    para cada modelo — baseline da Fase 1, GA com CV simples, GA com CV
    repetida — e avalia as três no mesmo conjunto de teste.

    A pergunta: a CV repetida reduz o sobreajuste de fold diagnosticado
    em `analisar_generalizacao`?
    """
    caminho = REPORTS_DIR / "ga_winners_robusto.json"
    if not caminho.exists():
        return pd.DataFrame()

    robustos = json.loads(caminho.read_text(encoding="utf-8"))
    simples = json.loads(
        (REPORTS_DIR / "ga_winners.json").read_text(encoding="utf-8")
    )
    # Os dois conjuntos de vencedores são medidos nas DUAS réguas:
    # CV 5-fold simples e CV 5x3 repetida. Essa tabela cruzada é a
    # evidência direta de sobreajuste de seleção — se cada vencedor for
    # melhor APENAS no estimador contra o qual foi otimizado, então o que
    # o GA encontrou foi uma particularidade do estimador, não um modelo
    # melhor.
    regua_simples = FitnessConfig()
    regua_repetida = FitnessConfig(cv_repeats=3)

    linhas = []
    for modelo in MODELOS:
        if modelo not in robustos:
            continue
        space = SEARCH_SPACES[modelo]
        X_train, X_test = data.train_for(space.features), data.test_for(space.features)

        versoes = {
            "baseline Fase 1": BASELINE_PARAMS[modelo],
            "GA / CV 5-fold": simples[modelo]["params"],
            "GA / CV 5x3": robustos[modelo]["params"],
        }
        for nome, params in versoes.items():
            est = build_estimator(modelo, params)
            est.fit(X_train, data.y_train)
            m = metricas_no_teste(est, X_test, data.y_test)
            cv_s = evaluate_params(modelo, params, data, space.features, regua_simples)
            cv_r = evaluate_params(modelo, params, data, space.features, regua_repetida)
            linhas.append({
                "model": modelo,
                "modelo": NOMES_CURTOS[modelo],
                "versao": nome,
                "fitness_cv_5fold": round(cv_s.fitness, 4),
                "fitness_cv_5x3": round(cv_r.fitness, 4),
                "teste_recall_mal": round(m["recall_malignant"], 4),
                "teste_precisao_mal": round(m["precision_malignant"], 4),
                "teste_f1_mal": round(m["f1_malignant"], 4),
                "teste_accuracy": round(m["accuracy"], 4),
                "teste_falsos_negativos": m["falsos_negativos"],
                "teste_falsos_positivos": m["falsos_positivos"],
            })

    tabela = pd.DataFrame(linhas)
    if tabela.empty:
        return tabela

    tabela.to_csv(REPORTS_DIR / "comparison_robusto.csv", index=False)
    logger.info("--- experimento corretivo: CV 5-fold vs. CV 5x3 ---")
    logger.info("\n%s", tabela.to_string(index=False))
    return tabela


def grafico_experimento_robusto(robusto: pd.DataFrame) -> Path:
    """A evidência cruzada de sobreajuste de seleção.

    Três painéis, todos com as mesmas três barras por modelo (baseline,
    vencedor da CV 5-fold, vencedor da CV 5x3). O que importa é a
    comparação ENTRE os painéis:

      painel 1 — fitness medido na régua 5-fold: o vencedor laranja ganha
      painel 2 — fitness medido na régua 5x3:    o vencedor azul ganha
      painel 3 — F1 no teste:                    nenhum dos dois ganha

    Cada configuração é superior apenas no estimador contra o qual foi
    otimizada. É a definição operacional de sobreajuste do critério de
    seleção, e o painel 3 mostra que a vantagem não existe fora dele.

    Três painéis separados em vez de eixo duplo: são grandezas distintas
    e empilhá-las convidaria à leitura de que uma causa a outra.
    """
    versoes = ["baseline Fase 1", "GA / CV 5-fold", "GA / CV 5x3"]
    cores = {
        "baseline Fase 1": COR_ORIGINAL,
        "GA / CV 5-fold": "#eb6834",
        "GA / CV 5x3": COR_OTIMIZADO,
    }

    paineis = [
        ("fitness_cv_5fold",
         "1. Fitness na régua CV 5-fold\n(o que o GA laranja otimizou)"),
        ("fitness_cv_5x3",
         "2. Fitness na régua CV 5x3\n(o que o GA azul otimizou)"),
        ("teste_f1_mal",
         "3. F1 do maligno NO TESTE\n(o que de fato interessa)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(MODELOS))
    largura = 0.26

    for ax, (coluna, titulo) in zip(axes, paineis):
        todos = []
        for i, versao in enumerate(versoes):
            valores = [
                robusto[
                    (robusto["model"] == m) & (robusto["versao"] == versao)
                ][coluna].iloc[0]
                for m in MODELOS
            ]
            todos.extend(valores)
            barras = ax.bar(
                x + (i - 1) * (largura + 0.015), valores, largura,
                color=cores[versao], label=versao,
            )
            ax.bar_label(barras, fmt="%.4f", fontsize=7.5,
                         color=TINTA_SECUNDARIA, padding=2, rotation=90)

        # eixo apertado em torno dos valores: as diferenças são de
        # milésimos e sumiriam numa escala de 0 a 1
        piso, teto = min(todos), max(todos)
        folga = max((teto - piso) * 0.55, 0.01)
        ax.set_ylim(piso - folga * 0.6, teto + folga)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [NOMES_CURTOS[m].replace(" ", "\n") for m in MODELOS], fontsize=9
        )
        ax.set_title(titulo, fontsize=10.5, pad=10)
        ax.grid(True, axis="y", alpha=0.7)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(
        "Cada vencedor ganha só na régua contra a qual foi otimizado — "
        "e nenhum ganha no teste",
        fontsize=12.5, y=1.0,
    )
    fig.tight_layout()
    caminho = FIG_DIR / "36_experimento_robusto.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def grafico_cv_vs_teste(generalizacao: pd.DataFrame) -> Path:
    """Fitness de CV (x) contra F1 do maligno no teste (y).

    Se o fitness de CV previsse o teste, os pontos subiriam da esquerda
    pra direita. Um painel por modelo; o baseline da Fase 1 marcado com
    losango pra separar o ponto de partida dos resultados do GA.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    for ax, modelo in zip(axes, MODELOS):
        g = generalizacao[generalizacao["model"] == modelo]
        base = g[g["origem"] == "baseline Fase 1"]
        ga = g[g["origem"] != "baseline Fase 1"].sort_values("fitness_cv")

        pontos = [
            (r["fitness_cv"], r["teste_f1_mal"], r["origem"].split("/")[-1])
            for _, r in ga.iterrows()
        ]
        span = (
            max(p[1] for p in pontos) - min(p[1] for p in pontos)
            if pontos else 0.0
        )
        # Quando várias configurações dão o MESMO F1 de teste (acontece na
        # Logística: as três coincidem), os rótulos cairiam um sobre o
        # outro. Aqui eles são empilhados para baixo em degraus de 13 pt.
        empilhado = 0
        for i, (x, y, experimento) in enumerate(pontos):
            cor = COR_EXPERIMENTO.get(experimento, COR_OTIMIZADO)
            ax.scatter(
                x, y, s=130, color=cor, label=experimento,
                edgecolor="white", linewidth=2, zorder=3,
            )
            colide = i > 0 and (
                span <= 0 or abs(y - pontos[i - 1][1]) / span < 0.05
            )
            empilhado = empilhado + 1 if colide else 0
            ax.annotate(
                experimento, xy=(x, y),
                xytext=(0, -17 - 13 * empilhado), textcoords="offset points",
                ha="center", fontsize=8.5, color=cor,
            )

        if len(base):
            b = base.iloc[0]
            ax.scatter(
                b["fitness_cv"], b["teste_f1_mal"], s=170, marker="D",
                color=COR_ORIGINAL, label="baseline Fase 1",
                edgecolor="white", linewidth=2, zorder=3,
            )
            ax.annotate(
                "Fase 1", xy=(b["fitness_cv"], b["teste_f1_mal"]),
                xytext=(0, 15), textcoords="offset points",
                ha="center", fontsize=8.5, color=COR_ORIGINAL,
            )

        ax.set_title(NOMES_CURTOS[modelo], fontsize=11, pad=10)
        ax.set_xlabel("fitness de CV (treino) — o que o GA otimizou")
        ax.grid(True, alpha=0.7)
        ax.set_axisbelow(True)
        ax.margins(0.26)
    axes[0].set_ylabel("F1 do maligno no teste")

    # Legenda deduplicada: cada experimento aparece nos 3 painéis, mas na
    # legenda tem que constar uma vez só.
    vistos: dict[str, Any] = {}
    for ax in axes:
        for h, rotulo in zip(*ax.get_legend_handles_labels()):
            vistos.setdefault(rotulo, h)
    fig.legend(
        vistos.values(), vistos.keys(), loc="lower center", ncol=4,
        frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=9,
    )

    fig.suptitle(
        "O fitness de CV prevê o desempenho no teste? "
        "(subida da esquerda pra direita = sim)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    caminho = FIG_DIR / "35_cv_vs_teste.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def tabela_markdown(comparativo: pd.DataFrame, detalhe: dict) -> str:
    def seta(d: float, casas: int = 4) -> str:
        if abs(d) < 10 ** (-casas):
            return "="
        return f"{d:+.{casas}f}"

    linhas = [
        "## Comparativo — modelos originais (Fase 1) vs. otimizados por GA",
        "",
        "### Métricas no conjunto de teste (114 amostras, 42 malignos)",
        "",
        "| Modelo | Experimento vencedor | Recall maligno | F1 maligno | Acurácia | ROC AUC | Falsos negativos |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in comparativo.iterrows():
        linhas.append(
            f"| {r['modelo']} | `{r['experimento_vencedor']}` "
            f"| {r['recall_malignant_original']:.4f} -> "
            f"**{r['recall_malignant_otimizado']:.4f}** "
            f"({seta(r['delta_recall_malignant'])}) "
            f"| {r['f1_malignant_original']:.4f} -> "
            f"{r['f1_malignant_otimizado']:.4f} ({seta(r['delta_f1_malignant'])}) "
            f"| {r['accuracy_original']:.4f} -> {r['accuracy_otimizado']:.4f} "
            f"({seta(r['delta_accuracy'])}) "
            f"| {r['roc_auc_original']:.4f} -> {r['roc_auc_otimizado']:.4f} "
            f"({seta(r['delta_roc_auc'])}) "
            f"| {int(r['falsos_negativos_original'])} -> "
            f"{int(r['falsos_negativos_otimizado'])} "
            f"({int(r['delta_falsos_negativos']):+d}) |"
        )

    linhas += [
        "",
        "### Fitness de validação cruzada no treino (a métrica que o GA otimizou)",
        "",
        "| Modelo | Fitness original | Fitness otimizado | Ganho |",
        "|---|---|---|---|",
    ]
    for _, r in comparativo.iterrows():
        linhas.append(
            f"| {r['modelo']} | {r['fitness_cv_original']:.4f} "
            f"| **{r['fitness_cv_otimizado']:.4f}** "
            f"| {seta(r['delta_fitness_cv'])} |"
        )

    linhas += ["", "### Hiperparâmetros encontrados", ""]
    for modelo in MODELOS:
        d = detalhe[modelo]
        linhas += [
            f"**{NOMES_CURTOS[modelo]}** (experimento `{d['experimento_vencedor']}`)",
            "",
            "| Hiperparâmetro | Fase 1 | Otimizado pelo GA |",
            "|---|---|---|",
        ]
        for chave in d["params_otimizado"]:
            antes = d["params_original"].get(chave, "—")
            depois = d["params_otimizado"][chave]
            fmt = lambda v: (  # noqa: E731
                f"{v:.4g}" if isinstance(v, float) else f"`{v}`"
            )
            mudou = "**" if str(antes) != str(depois) else ""
            linhas.append(
                f"| `{chave}` | {fmt(antes)} | {mudou}{fmt(depois)}{mudou} |"
            )
        linhas.append("")

    return "\n".join(linhas)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("fase2.compare", log_file="compare.log")

    data = load_phase1_data()
    logger.info("dados — %s", data.describe())

    historico = pd.read_csv(REPORTS_DIR / "ga_history.csv")
    vencedores = json.loads(
        (REPORTS_DIR / "ga_winners.json").read_text(encoding="utf-8")
    )

    logger.info("--- avaliação no conjunto de teste ---")
    comparativo, detalhe = construir_comparativo(data, logger)

    conferencia = conferir_contra_fase1(comparativo, logger)
    if not conferencia.empty:
        conferencia.to_csv(REPORTS_DIR / "baseline_vs_fase1.csv", index=False)

    comparativo.to_csv(REPORTS_DIR / "comparison.csv", index=False)
    (REPORTS_DIR / "comparison_detail.json").write_text(
        json.dumps(detalhe, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (REPORTS_DIR / "comparison.md").write_text(
        tabela_markdown(comparativo, detalhe), encoding="utf-8"
    )

    generalizacao = analisar_generalizacao(data, logger)
    robusto = analisar_experimento_robusto(data, logger)

    logger.info("--- gerando figuras ---")
    figuras = [
        (grafico_convergencia, (historico,)),
        (grafico_melhor_vs_medio, (historico, vencedores)),
        (grafico_diversidade, (historico,)),
        (grafico_comparativo_metricas, (comparativo,)),
        (grafico_matrizes_confusao, (detalhe,)),
    ]
    if not generalizacao.empty:
        figuras.append((grafico_cv_vs_teste, (generalizacao,)))
    if not robusto.empty:
        figuras.append((grafico_experimento_robusto, (robusto,)))

    for fn, args in figuras:
        caminho = fn(*args)
        logger.info("  %s", caminho.relative_to(BASE_DIR))

    logger.info("\n%s", comparativo[[
        "modelo", "experimento_vencedor",
        "recall_malignant_original", "recall_malignant_otimizado",
        "falsos_negativos_original", "falsos_negativos_otimizado",
        "accuracy_original", "accuracy_otimizado",
    ]].to_string(index=False))

    logger.info("\ntabela em: %s", REPORTS_DIR / "comparison.md")
    logger.info("próximo passo: python -m src.llm.explain")


if __name__ == "__main__":
    main()
