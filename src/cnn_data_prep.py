# [EXTRA] Monta o manifesto do CBIS-DDSM a partir do download bruto do kagglehub.
#
# O kagglehub baixa duas pastas: csv/ (metadados) e jpeg/ (as imagens, uma pasta por
# SeriesInstanceUID). Os nomes de pasta em jpeg/ não batem com nada nos CSVs de caso —
# o de-para tem que passar por dicom_info.csv:
#
#   - mass/calc_case_description_*.csv: cada linha é uma ANORMALIDADE (não a imagem
#     toda), com 'pathology' (BENIGN/BENIGN_WITHOUT_CALLBACK/MALIGNANT) e
#     'image file path', cujo primeiro pedaço ('Mass-Training_P_00001_LEFT_CC')
#     identifica a mamografia completa.
#   - dicom_info.csv: 'PatientName' bate com esse identificador e 'image_path' tem o
#     caminho real do jpeg. O 'SeriesDescription' separa os três tipos de imagem:
#     'full mammogram images' (2857), 'cropped images' (3567), 'ROI mask images' (3247).
#
# Duas entradas possíveis, e o script gera as duas:
#
#   full  — a mamografia inteira (~5236x3016). Uma imagem pode ter várias anormalidades;
#           regra conservadora: se qualquer uma é maligna, a imagem é maligna.
#   patch — o recorte da lesão (~330x330). 'PatientName' é '<image_id>_<abnormality id>',
#           então o rótulo vem direto da linha, sem agregação.
#
# Em vez de copiar as imagens (~3 GB duplicados), escreve um manifesto CSV apontando
# para os arquivos originais e carregando junto os metadados clínicos que a EDA usa.
#
# python src/cnn_data_prep.py --cbis-root ../cnn

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "cbis_manifest.csv"

CASE_DESCRIPTION_FILES = {
    "train": ["mass_case_description_train_set.csv", "calc_case_description_train_set.csv"],
    "test": ["mass_case_description_test_set.csv", "calc_case_description_test_set.csv"],
}

# colunas que existem só no mass ou só no calc — viram um descritor único
TYPE_SPECIFIC = {
    "mass": ["mass_shape", "mass_margins"],
    "calc": ["calc_type", "calc_distribution"],
}


def load_dicom_lookup(csv_dir: Path, series_description: str) -> pd.DataFrame:
    # de-para 'PatientName' -> caminho do jpeg, para um tipo de imagem específico
    dicom_info = pd.read_csv(csv_dir / "dicom_info.csv", low_memory=False)
    subset = dicom_info[dicom_info["SeriesDescription"] == series_description].copy()
    # vem como 'CBIS-DDSM/jpeg/<uid>/<file>.jpg'; guardo relativo à raiz do download
    subset["jpeg_path"] = subset["image_path"].str.replace(r"^CBIS-DDSM/", "", regex=True)
    lookup = subset[["PatientName", "jpeg_path"]].drop_duplicates(subset=["PatientName"])
    return lookup.rename(columns={"PatientName": "key"})


def binarize_pathology(pathology: str) -> str:
    # BENIGN_WITHOUT_CALLBACK conta como benigno — convenção padrão do CBIS-DDSM.
    # São achados considerados benignos que não exigiram retorno da paciente.
    return "malignant" if str(pathology).strip().upper() == "MALIGNANT" else "benign"


def read_case_descriptions(csv_dir: Path) -> pd.DataFrame:
    # junta os 4 CSVs de caso (mass/calc x train/test) num quadro só, com colunas
    # padronizadas entre os dois tipos de anormalidade
    frames = []
    for split, filenames in CASE_DESCRIPTION_FILES.items():
        for filename in filenames:
            path = csv_dir / filename
            if not path.exists():
                print(f"  [aviso] arquivo não encontrado, pulando: {path}")
                continue

            kind = "mass" if filename.startswith("mass") else "calc"
            df = pd.read_csv(path)
            # 'breast_density' no mass, 'breast density' no calc — underscore unifica
            df.columns = [c.strip().replace(" ", "_") for c in df.columns]

            df["split_oficial"] = split
            df["abnormality_type"] = kind
            df["image_id"] = df["image_file_path"].str.split("/").str[0]
            df["abnormality_id"] = df["abnormality_id"].astype(int)
            df["label"] = df["pathology"].apply(binarize_pathology)

            # descritor clínico da anormalidade, unificado entre mass e calc
            cols = TYPE_SPECIFIC[kind]
            df["descriptor"] = df[cols[0]].astype(str) + " | " + df[cols[1]].astype(str)

            frames.append(df[[
                "patient_id", "image_id", "abnormality_id", "abnormality_type",
                "split_oficial", "pathology", "label", "descriptor",
                "breast_density", "left_or_right_breast", "image_view",
                "assessment", "subtlety",
            ]].rename(columns={
                "left_or_right_breast": "side",
                "image_view": "view",
            }))

    combined = pd.concat(frames, ignore_index=True)
    print(f"  {len(combined)} anormalidades lidas dos 4 CSVs de caso.")
    return combined


def fix_split_leakage(cases: pd.DataFrame) -> pd.DataFrame:
    # Os splits oficiais de mass e calc foram feitos de forma independente, então
    # existem pacientes com um caso de massa no treino e um de calcificação no teste.
    # Isso é vazamento: a mesma mama, o mesmo tecido, dos dois lados da avaliação.
    # Correção: quem aparece nos dois vai inteiro pro treino (mantém o teste limpo
    # sem descartar dado).
    por_paciente = cases.groupby("patient_id")["split_oficial"].nunique()
    contaminados = set(por_paciente[por_paciente > 1].index)

    cases = cases.copy()
    cases["split"] = cases["split_oficial"]
    cases.loc[cases["patient_id"].isin(contaminados), "split"] = "train"
    cases["split_fixed"] = cases["patient_id"].isin(contaminados)

    if contaminados:
        print(f"  [vazamento] {len(contaminados)} pacientes apareciam em treino E teste "
              f"nos splits oficiais; realocados inteiramente para treino.")
        print(f"              ex.: {sorted(contaminados)[:5]}")
    return cases


def build_patch_manifest(cases: pd.DataFrame, csv_dir: Path) -> pd.DataFrame:
    # recorte da lesão: uma linha do CSV de caso = uma imagem, rótulo direto
    lookup = load_dicom_lookup(csv_dir, "cropped images")
    df = cases.copy()
    df["key"] = df["image_id"] + "_" + df["abnormality_id"].astype(str)
    df["variant"] = "patch"
    return df.merge(lookup, on="key", how="left")


def build_full_manifest(cases: pd.DataFrame, csv_dir: Path) -> pd.DataFrame:
    # mamografia inteira: várias anormalidades podem cair na mesma imagem, então
    # agrega por image_id antes de casar com o jpeg
    lookup = load_dicom_lookup(csv_dir, "full mammogram images")

    agg = cases.groupby("image_id").agg(
        patient_id=("patient_id", "first"),
        abnormality_type=("abnormality_type", lambda s: "+".join(sorted(set(s)))),
        split=("split", "first"),
        split_oficial=("split_oficial", "first"),
        split_fixed=("split_fixed", "any"),
        # regra conservadora: qualquer anormalidade maligna torna a imagem maligna
        label=("label", lambda s: "malignant" if (s == "malignant").any() else "benign"),
        pathology=("pathology", lambda s: "|".join(sorted(set(s)))),
        descriptor=("descriptor", lambda s: " ; ".join(sorted(set(s)))),
        breast_density=("breast_density", "max"),
        side=("side", "first"),
        view=("view", "first"),
        assessment=("assessment", "max"),   # pior BI-RADS da imagem
        subtlety=("subtlety", "min"),       # a anormalidade mais sutil manda
        n_abnormalities=("abnormality_id", "count"),
    ).reset_index()

    agg["abnormality_id"] = 0  # não se aplica: a imagem inteira agrega várias
    agg["key"] = agg["image_id"]
    agg["variant"] = "full"
    return agg.merge(lookup, on="key", how="left")


def finalize(df: pd.DataFrame, cbis_root: Path, variant: str) -> pd.DataFrame:
    # descarta o que não casou no dicom_info e o que não está no disco
    n_total = len(df)
    sem_lookup = df["jpeg_path"].isna().sum()
    df = df.dropna(subset=["jpeg_path"]).copy()

    existe = df["jpeg_path"].map(lambda p: (cbis_root / p).exists())
    sem_arquivo = int((~existe).sum())
    df = df[existe].copy()

    print(f"  {variant}: {len(df)} de {n_total} imagens utilizáveis "
          f"({sem_lookup} sem match no dicom_info, {sem_arquivo} ausentes no disco)")
    for split, grupo in df.groupby("split"):
        dist = grupo["label"].value_counts()
        print(f"    {split}: {len(grupo)} imagens — "
              f"{dist.get('benign', 0)} benignas, {dist.get('malignant', 0)} malignas")
    return df


COLUMNS = [
    "key", "image_id", "patient_id", "abnormality_id", "abnormality_type", "variant",
    "split", "split_oficial", "split_fixed", "pathology", "label", "descriptor",
    "breast_density", "side", "view", "assessment", "subtlety", "jpeg_path",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cbis-root", default=str(BASE_DIR.parent / "cnn"),
        help="Pasta do download do kagglehub (contém as subpastas 'csv' e 'jpeg'). "
             "Default: ../cnn",
    )
    parser.add_argument("--variant", choices=["patch", "full", "both"], default="both")
    args = parser.parse_args()

    cbis_root = Path(args.cbis_root).resolve()
    csv_dir = cbis_root / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"Pasta 'csv' não encontrada em {cbis_root}")

    print(f"Lendo metadados de {csv_dir}")
    cases = read_case_descriptions(csv_dir)
    cases = fix_split_leakage(cases)

    variants = ["patch", "full"] if args.variant == "both" else [args.variant]
    builders = {"patch": build_patch_manifest, "full": build_full_manifest}

    frames = []
    for variant in variants:
        print(f"\nMontando manifesto '{variant}'...")
        df = builders[variant](cases, csv_dir)
        df = finalize(df, cbis_root, variant)
        frames.append(df.reindex(columns=COLUMNS))

    manifest = pd.concat(frames, ignore_index=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    # checagem que vale imprimir: depois da correção, treino e teste não podem
    # compartilhar paciente nenhum
    treino = set(manifest.loc[manifest.split == "train", "patient_id"])
    teste = set(manifest.loc[manifest.split == "test", "patient_id"])
    print(f"\nPacientes: {len(treino)} treino, {len(teste)} teste, "
          f"interseção: {len(treino & teste)}")

    print(f"Manifesto salvo em: {MANIFEST_PATH} ({len(manifest)} linhas)")
    print("Próximo passo: python src/cnn_cache.py")


if __name__ == "__main__":
    main()
