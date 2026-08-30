# [EXTRA] Avaliação dos modelos de CNN no conjunto de teste + explicabilidade Grad-CAM.
#
# Gera as figuras 18-21 (reports/figures/) e reports/cnn_model_comparison.csv.
#
# Além do limiar padrão de 0,5, reporta o limiar que atinge um recall-alvo na classe
# maligna. Em rastreio de câncer o custo dos erros é assimétrico: um falso negativo
# manda uma paciente com câncer pra casa, um falso positivo gera um exame a mais.
# É a mesma lógica de métrica usada no modelo de dados estruturados (etapa 6).
#
# python src/cnn_evaluate.py

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "processed" / "cnn_cache"
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "cbis_manifest.csv"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"
REPORTS_DIR = BASE_DIR / "reports"

sns.set_theme(style="whitegrid")
CLASS_NAMES = ["benigno", "maligno"]
RECALL_ALVO = 0.90
BATCH_SIZE = 32

# mesma configuração de src/cnn_mammography.py
ARCH_CONFIG = {"scratch": {"size": 128, "channels": 1},
               "transfer": {"size": 160, "channels": 3}}
COMBINACOES = [(v, a) for v in ("patch", "full") for a in ("scratch", "transfer")]
VARIANT_LABEL = {"patch": "recorte", "full": "inteira"}


def carregar_teste(variant: str, arch: str):
    cfg = ARCH_CONFIG[arch]
    imagens = np.load(CACHE_DIR / f"{variant}_test_{cfg['size']}.npy")
    rotulos = np.load(CACHE_DIR / f"{variant}_test_labels.npy").astype(int)
    chaves = np.load(CACHE_DIR / f"{variant}_test_keys.npy", allow_pickle=True)

    x = imagens[..., None].astype("float32")
    if cfg["channels"] == 3:
        x = np.repeat(x, 3, axis=-1)
    return imagens, x, rotulos, chaves


def limiar_para_recall(y_true, y_proba, alvo: float) -> float:
    # entre os limiares que atingem o recall-alvo, escolhe o de maior precisão
    precisao, revocacao, limiares = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve devolve um ponto a mais que limiares
    validos = np.flatnonzero(revocacao[:-1] >= alvo)
    if len(validos) == 0:
        return 0.0
    melhor = validos[np.argmax(precisao[:-1][validos])]
    return float(limiares[melhor])


def metricas(y_true, y_pred, y_proba, limiar: float) -> dict:
    precisao, revocacao, _ = precision_recall_curve(y_true, y_proba)
    return {
        "limiar": round(limiar, 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(auc(revocacao, precisao), 4),
    }


def avaliar_todos() -> tuple:
    resultados, predicoes = [], {}

    for variant, arch in COMBINACOES:
        caminho = MODELS_DIR / f"cnn_{variant}_{arch}.keras"
        if not caminho.exists():
            print(f"[aviso] modelo não encontrado, pulando: {caminho.name}")
            continue

        nome = f"{variant}_{arch}"
        print(f"\n{'=' * 70}\n{nome}\n{'=' * 70}")
        model = tf.keras.models.load_model(caminho)
        imagens, x, y_true, chaves = carregar_teste(variant, arch)
        y_proba = model.predict(x, batch_size=BATCH_SIZE, verbose=0).flatten()

        # limiar padrão
        y_pred = (y_proba >= 0.5).astype(int)
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                    zero_division=0))
        base = metricas(y_true, y_pred, y_proba, 0.5)

        # limiar ajustado para o recall-alvo na classe maligna
        limiar = limiar_para_recall(y_true, y_proba, RECALL_ALVO)
        y_pred_alvo = (y_proba >= limiar).astype(int)
        ajustado = metricas(y_true, y_pred_alvo, y_proba, limiar)
        print(f"limiar para recall >= {RECALL_ALVO:.0%}: {limiar:.4f} -> "
              f"recall {ajustado['recall']:.3f}, precisão {ajustado['precision']:.3f}")

        resultados.append({"modelo": nome, "variant": variant, "arch": arch,
                           "n_teste": len(y_true), "criterio": "limiar 0,5", **base})
        resultados.append({"modelo": nome, "variant": variant, "arch": arch,
                           "n_teste": len(y_true),
                           "criterio": f"recall >= {RECALL_ALVO:.0%}", **ajustado})
        predicoes[nome] = {"y_true": y_true, "y_proba": y_proba, "y_pred": y_pred,
                           "limiar": limiar, "imagens": imagens, "x": x,
                           "chaves": chaves, "model": model}

    return pd.DataFrame(resultados), predicoes


# -------------------------------------------------------------------------- Grad-CAM

def separar_extrator_e_cabeca(model):
    """Divide o modelo no último mapa de ativação 4D: camadas antes e depois.

    Devolve duas listas de camadas, aplicadas em sequência na hora de usar. As duas
    arquiteturas são cadeias lineares, então isso basta — e evita montar um
    tf.keras.Model(inputs, camada.output) intermediário, que no Keras 3 não funciona
    quando a camada é um submodelo aninhado (o backbone MobileNetV2 entra no modelo
    como uma única camada Functional, e o tensor de saída dela não pertence ao grafo
    externo).

    Sem nome de camada fixo: na CNN do zero o corte cai no último MaxPooling2D; na de
    transfer learning, na saída do backbone. Era isso que a versão anterior, com
    'conv2d_3' escrito na mão, não conseguia fazer.
    """
    camadas = [c for c in model.layers if not isinstance(c, tf.keras.layers.InputLayer)]
    corte = max(i for i, camada in enumerate(camadas)
                if len(camada.output.shape) == 4)
    return camadas[:corte + 1], camadas[corte + 1:]


def _aplicar(camadas, tensor):
    for camada in camadas:
        tensor = camada(tensor, training=False)
    return tensor


def grad_cam(model, imagens_x: np.ndarray) -> np.ndarray:
    # mapa de calor das regiões que mais empurraram a predição para "maligno"
    corpo, cabeca = separar_extrator_e_cabeca(model)
    entrada = tf.convert_to_tensor(imagens_x)

    with tf.GradientTape() as tape:
        mapa = _aplicar(corpo, entrada)
        tape.watch(mapa)
        prob = _aplicar(cabeca, mapa)[:, 0]
        # deriva o logit, não a probabilidade: a sigmoide satura nos extremos e
        # zeraria o gradiente exatamente nos casos em que o modelo está confiante
        prob = tf.clip_by_value(prob, 1e-6, 1 - 1e-6)
        score = tf.math.log(prob) - tf.math.log(1 - prob)

    grads = tape.gradient(score, mapa)
    pesos = tf.reduce_mean(grads, axis=(1, 2))              # importância por canal
    cam = tf.nn.relu(tf.einsum("bhwc,bc->bhw", mapa, pesos))
    maximo = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
    return (cam / (maximo + 1e-8)).numpy()


def plot_gradcam(predicoes: dict, nome: str, arquivo: str) -> None:
    # mostra os quatro desfechos possíveis — inclusive os erros, que é onde o
    # Grad-CAM mais informa: revela se o modelo olhou pra lesão ou pra um artefato
    if nome not in predicoes:
        print(f"[aviso] Grad-CAM pulado, modelo ausente: {nome}")
        return
    p = predicoes[nome]
    y_true, y_pred, y_proba = p["y_true"], p["y_pred"], p["y_proba"]

    desfechos = [
        ("VP — maligno detectado", (y_true == 1) & (y_pred == 1)),
        ("VN — benigno correto", (y_true == 0) & (y_pred == 0)),
        ("FP — alarme falso", (y_true == 0) & (y_pred == 1)),
        ("FN — maligno perdido", (y_true == 1) & (y_pred == 0)),
    ]
    rng = np.random.default_rng(42)
    n_col = 4
    fig, axes = plt.subplots(4, n_col, figsize=(13, 13.5))

    for linha, (titulo, mascara) in enumerate(desfechos):
        idx_disponivel = np.flatnonzero(mascara)
        escolhidos = (rng.choice(idx_disponivel, min(n_col, len(idx_disponivel)),
                                 replace=False) if len(idx_disponivel) else [])
        mapas = grad_cam(p["model"], p["x"][escolhidos]) if len(escolhidos) else []

        for col in range(n_col):
            ax = axes[linha, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col >= len(escolhidos):
                ax.axis("off")
                continue
            i = escolhidos[col]
            imagem = p["imagens"][i]
            calor = np.array(tf.image.resize(mapas[col][..., None], imagem.shape)[..., 0])
            ax.imshow(imagem, cmap="gray")
            ax.imshow(calor, cmap="jet", alpha=0.4)
            ax.set_title(f"p(maligno)={y_proba[i]:.2f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(titulo, fontsize=10, fontweight="bold")

    fig.suptitle(f"Grad-CAM — {nome}: onde a CNN olhou, em acertos e erros",
                 y=0.997, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / arquivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM gerado para {nome} -> {arquivo}")


# --------------------------------------------------------------------------- figuras

def plot_training_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    cores = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (variant, arch) in enumerate(COMBINACOES):
        caminho = REPORTS_DIR / f"cnn_history_{variant}_{arch}.csv"
        if not caminho.exists():
            continue
        hist = pd.read_csv(caminho)
        rotulo = f"{variant}_{arch}"
        for ax, metrica in zip(axes, ["loss", "auc", "recall"]):
            ax.plot(hist.index + 1, hist[f"val_{metrica}"], label=rotulo, color=cores[i])
            ax.set_title(f"val_{metrica}")
            ax.set_xlabel("época (estágios concatenados)")

    axes[0].legend(fontsize=9)
    fig.suptitle("Curvas de validação por modelo", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "18_cnn_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(predicoes: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))

    for col, (nome, p) in enumerate(predicoes.items()):
        for linha, (criterio, y_pred) in enumerate([
                ("limiar 0,5", p["y_pred"]),
                (f"recall >= {RECALL_ALVO:.0%}", (p["y_proba"] >= p["limiar"]).astype(int))]):
            ax = axes[linha, col]
            cm = confusion_matrix(p["y_true"], y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
            ax.set_title(f"{nome}\n{criterio}", fontsize=10)
            ax.set_xlabel("predito")
            ax.set_ylabel("real" if col == 0 else "")

    fig.suptitle("Matrizes de confusão no teste", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "19_cnn_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(predicoes: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cores = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (nome, p) in enumerate(predicoes.items()):
        fpr, tpr, _ = roc_curve(p["y_true"], p["y_proba"])
        axes[0].plot(fpr, tpr, color=cores[i],
                     label=f"{nome} (AUC={roc_auc_score(p['y_true'], p['y_proba']):.3f})")

        precisao, revocacao, _ = precision_recall_curve(p["y_true"], p["y_proba"])
        axes[1].plot(revocacao, precisao, color=cores[i],
                     label=f"{nome} (AP={auc(revocacao, precisao):.3f})")
        # a linha de base da curva PR é a prevalência da classe positiva
        axes[1].axhline(p["y_true"].mean(), color=cores[i], ls=":", lw=0.8, alpha=0.5)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("taxa de falso positivo")
    axes[0].set_ylabel("taxa de verdadeiro positivo (recall)")
    axes[0].set_title("Curva ROC")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precisão")
    axes[1].set_title("Curva Precisão-Recall (pontilhado = prevalência)")
    for ax in axes:
        ax.legend(fontsize=9)

    fig.suptitle("Desempenho no teste — ROC e Precisão-Recall", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "20_cnn_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sonda_de_atalho() -> pd.DataFrame:
    """Linha de base honesta: quanto dá pra acertar SEM olhar a lesão.

    Treina uma regressão logística só sobre estatísticas globais da imagem
    (brilho médio, contraste, fração ocupada por tecido). Se a CNN não superar
    isso com folga, ela provavelmente aprendeu o mesmo atalho — tamanho e
    densidade da mama — em vez do achado radiológico em si.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def estatisticas(variant: str, split: str):
        imagens = np.load(CACHE_DIR / f"{variant}_{split}_128.npy").astype("float32")
        rotulos = np.load(CACHE_DIR / f"{variant}_{split}_labels.npy").astype(int)
        chapado = imagens.reshape(len(imagens), -1)
        tecido = (imagens > 25).mean(axis=(1, 2))   # separa a mama do fundo preto
        x = np.c_[chapado.mean(1), chapado.std(1), tecido,
                  np.percentile(chapado, 90, axis=1), chapado.max(1)]
        return x, rotulos

    linhas = []
    for variant in ("patch", "full"):
        x_treino, y_treino = estatisticas(variant, "train")
        x_teste, y_teste = estatisticas(variant, "test")
        escala = StandardScaler().fit(x_treino)
        modelo = LogisticRegression(max_iter=1000, class_weight="balanced")
        modelo.fit(escala.transform(x_treino), y_treino)
        proba = modelo.predict_proba(escala.transform(x_teste))[:, 1]
        linhas.append({"variant": variant,
                       "roc_auc_estatisticas_globais": round(roc_auc_score(y_teste, proba), 4)})

    sonda = pd.DataFrame(linhas)
    print(f"\n{'=' * 70}\nSONDA DE ATALHO (sem olhar a lesão)\n{'=' * 70}")
    print(sonda.to_string(index=False))
    return sonda


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    comparacao, predicoes = avaliar_todos()

    if not predicoes:
        raise FileNotFoundError(
            "Nenhum modelo encontrado em models/. Rode antes: "
            "python src/cnn_mammography.py --all")

    comparacao.to_csv(REPORTS_DIR / "cnn_model_comparison.csv", index=False)
    plot_training_curves()
    plot_confusion_matrices(predicoes)
    plot_roc_pr(predicoes)
    # o melhor modelo de cada entrada, pra poder comparar onde cada um olha
    plot_gradcam(predicoes, "patch_transfer", "21_cnn_gradcam_patch.png")
    plot_gradcam(predicoes, "full_scratch", "22_cnn_gradcam_full.png")

    sonda = sonda_de_atalho()
    sonda.to_csv(REPORTS_DIR / "cnn_shortcut_probe.csv", index=False)

    print(f"\n{'=' * 70}\nCOMPARAÇÃO DOS MODELOS\n{'=' * 70}")
    print(comparacao.to_string(index=False))
    print(f"\nCSV salvo em: {REPORTS_DIR / 'cnn_model_comparison.csv'}")
    print(f"figuras salvas em: {FIG_DIR}")


if __name__ == "__main__":
    main()
