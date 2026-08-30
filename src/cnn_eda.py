# [EXTRA] EDA das imagens do CBIS-DDSM — o equivalente do src/eda.py para a etapa de
# visão computacional. Gera as figuras 13-17 (reports/figures/) e um CSV de estatísticas.
#
# python src/cnn_eda.py

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "cbis_manifest.csv"
CACHE_DIR = BASE_DIR / "data" / "processed" / "cnn_cache"
FIG_DIR = BASE_DIR / "reports" / "figures"
STATS_PATH = BASE_DIR / "reports" / "cnn_eda_stats.csv"

sns.set_theme(style="whitegrid")
PALETTE = {"benign": "#4C9A8E", "malignant": "#C0546B"}
VARIANT_LABEL = {"patch": "recorte da lesão", "full": "mamografia inteira"}


def load_cache(variant: str, split: str, size: int = 128):
    imagens = np.load(CACHE_DIR / f"{variant}_{split}_{size}.npy")
    rotulos = np.load(CACHE_DIR / f"{variant}_{split}_labels.npy")
    return imagens, rotulos


def print_overview(manifest: pd.DataFrame) -> None:
    print("VISÃO GERAL DAS IMAGENS")
    for variant, grupo in manifest.groupby("variant"):
        print(f"\n{variant} ({VARIANT_LABEL[variant]}): {len(grupo)} imagens")
        print(pd.crosstab(grupo["split"], grupo["label"]))
        print("proporção de malignos por split:")
        taxa = grupo.groupby("split")["label"].apply(lambda s: (s == "malignant").mean())
        print(taxa.round(3).to_string())

    print(f"\npacientes distintos: {manifest['patient_id'].nunique()}")
    realocadas = int(manifest["split_fixed"].sum())
    print(f"imagens de pacientes realocados pela correção de vazamento: {realocadas}")


def plot_class_distribution(manifest: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, variant in zip(axes[:2], ["patch", "full"]):
        grupo = manifest[manifest.variant == variant]
        sns.countplot(data=grupo, x="split", hue="label", order=["train", "test"],
                      palette=PALETTE, ax=ax)
        ax.set_title(f"{variant} — {VARIANT_LABEL[variant]}")
        ax.set_xlabel("conjunto")
        ax.set_ylabel("número de imagens")
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{int(p.get_height())}",
                            (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha="center", va="bottom", fontsize=9)

    # massa e calcificação são achados radiológicos bem diferentes; vale ver o balanço
    patches = manifest[manifest.variant == "patch"]
    sns.countplot(data=patches, x="abnormality_type", hue="label", palette=PALETTE,
                  ax=axes[2])
    axes[2].set_title("Tipo de anormalidade (recortes)")
    axes[2].set_xlabel("tipo")
    axes[2].set_ylabel("")

    fig.suptitle("Distribuição de classes no CBIS-DDSM", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_cnn_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clinical_factors(manifest: pd.DataFrame) -> pd.DataFrame:
    # usa os recortes: cada linha é uma anormalidade, então os fatores clínicos não
    # ficam misturados pela agregação que a imagem inteira exige
    patches = manifest[manifest.variant == "patch"].copy()
    fatores = [
        ("breast_density", "Densidade mamária (BI-RADS 1-4)"),
        ("assessment", "Avaliação BI-RADS (0-5)"),
        ("subtlety", "Sutileza do achado (1=sutil, 5=óbvio)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    linhas = []
    for ax, (col, titulo) in zip(axes, fatores):
        taxa = patches.groupby(col)["label"].agg(
            n="size", taxa_maligno=lambda s: (s == "malignant").mean())
        sns.barplot(x=taxa.index, y=taxa["taxa_maligno"], ax=ax, color="#C0546B")
        for i, (n, t) in enumerate(zip(taxa["n"], taxa["taxa_maligno"])):
            ax.annotate(f"{t:.0%}\n(n={n})", (i, t), ha="center", va="bottom", fontsize=8)
        ax.set_title(titulo)
        ax.set_xlabel("")
        ax.set_ylabel("proporção de malignos")
        ax.set_ylim(0, 1.15)
        for valor, n, t in zip(taxa.index, taxa["n"], taxa["taxa_maligno"]):
            linhas.append({"fator": col, "valor": valor, "n": n,
                           "taxa_maligno": round(t, 4)})

    fig.suptitle("Fatores clínicos vs. malignidade (recortes de lesão)", y=1.04,
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_cnn_clinical_factors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = pd.DataFrame(linhas)
    print("\nFATORES CLÍNICOS vs MALIGNIDADE (recortes)")
    print(stats.to_string(index=False))
    return stats


def plot_sample_grid() -> None:
    # o ponto desta figura é visual: mostrar quanto sinal sobra em cada entrada
    fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
    rng = np.random.default_rng(42)

    combinacoes = [("patch", "benign"), ("patch", "malignant"),
                   ("full", "benign"), ("full", "malignant")]
    for linha, (variant, label) in enumerate(combinacoes):
        imagens, rotulos = load_cache(variant, "train", 128)
        alvo = 0 if label == "benign" else 1
        idx = rng.choice(np.flatnonzero(rotulos == alvo), 6, replace=False)
        for col, i in enumerate(idx):
            ax = axes[linha, col]
            ax.imshow(imagens[i], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{variant}\n{label}", fontsize=10,
                              color=PALETTE[label], fontweight="bold")

    fig.suptitle("Exemplos de treino: recorte da lesão vs. mamografia inteira (128x128)",
                 y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_cnn_sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_intensity_profile() -> pd.DataFrame:
    # checagem de atalho espúrio: se benigno e maligno tivessem brilho médio bem
    # diferente, a CNN poderia acertar sem olhar a lesão
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    linhas = []
    for ax, variant in zip(axes, ["patch", "full"]):
        imagens, rotulos = load_cache(variant, "train", 128)
        medias = imagens.reshape(len(imagens), -1).mean(axis=1)
        for label, alvo in [("benign", 0), ("malignant", 1)]:
            valores = medias[rotulos == alvo]
            sns.histplot(valores, ax=ax, color=PALETTE[label], label=label,
                         element="step", stat="density", bins=40, alpha=0.45)
            linhas.append({"variant": variant, "label": label,
                           "intensidade_media": round(float(valores.mean()), 2),
                           "desvio": round(float(valores.std()), 2)})
        ax.set_title(f"{variant} — {VARIANT_LABEL[variant]}")
        ax.set_xlabel("intensidade média do pixel (0-255)")
        ax.legend()

    fig.suptitle("Intensidade média por classe — teste de atalho espúrio", y=1.03,
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "16_cnn_intensity_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = pd.DataFrame(linhas)
    print("\nINTENSIDADE MÉDIA POR CLASSE")
    print(stats.to_string(index=False))
    return stats


def plot_original_sizes(manifest: pd.DataFrame, cbis_root: Path) -> None:
    # tamanho original das imagens, justificando o resize escolhido
    dicom = pd.read_csv(cbis_root / "csv" / "dicom_info.csv", low_memory=False)
    dicom["jpeg_path"] = dicom["image_path"].str.replace(r"^CBIS-DDSM/", "", regex=True)
    tamanhos = dicom[["jpeg_path", "Rows", "Columns"]].drop_duplicates("jpeg_path")
    df = manifest.merge(tamanhos, on="jpeg_path", how="left").dropna(subset=["Rows"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, variant in zip(axes, ["patch", "full"]):
        grupo = df[df.variant == variant]
        ax.scatter(grupo["Columns"], grupo["Rows"], s=6, alpha=0.25, color="#4C6A9A")
        ax.axhline(128, ls="--", color="#C0546B", lw=1)
        ax.axvline(128, ls="--", color="#C0546B", lw=1)
        ax.set_title(f"{variant} — mediana {int(grupo['Rows'].median())}x"
                     f"{int(grupo['Columns'].median())} px")
        ax.set_xlabel("largura (px)")
        ax.set_ylabel("altura (px)")

    fig.suptitle("Tamanho original das imagens (tracejado = 128px do resize)",
                 y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "17_cnn_image_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nTAMANHO ORIGINAL (mediana)")
    print(df.groupby("variant")[["Rows", "Columns"]].median().to_string())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    cbis_root = BASE_DIR.parent / "cnn"

    print_overview(manifest)
    plot_class_distribution(manifest)
    clinicos = plot_clinical_factors(manifest)
    plot_sample_grid()
    intensidade = plot_intensity_profile()
    plot_original_sizes(manifest, cbis_root)

    # um CSV só, com as duas tabelas empilhadas e uma coluna dizendo qual é qual
    stats = pd.concat([
        clinicos.assign(tabela="fatores_clinicos"),
        intensidade.assign(tabela="intensidade_media"),
    ], ignore_index=True)
    stats.to_csv(STATS_PATH, index=False)

    print(f"\nfiguras salvas em: {FIG_DIR}")
    print(f"estatísticas salvas em: {STATS_PATH}")


if __name__ == "__main__":
    main()
