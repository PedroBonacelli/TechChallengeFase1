# [EXTRA] Decodifica e redimensiona as imagens do manifesto uma única vez, salvando
# arrays uint8 em disco.
#
# Sem isso cada época re-decodifica JPEGs de 5236x3016 e o treino em CPU vira I/O puro.
# Com o cache, uma época passa a ser só aritmética sobre um array já em memória.
#
# Gera, por (variante, split, tamanho):
#   data/processed/cnn_cache/{variant}_{split}_{size}.npy   imagens uint8 (N, size, size)
#   data/processed/cnn_cache/{variant}_{split}_labels.npy   0=benigno, 1=maligno
#   data/processed/cnn_cache/{variant}_{split}_groups.npy   patient_id (split agrupado)
#   data/processed/cnn_cache/{variant}_{split}_keys.npy     id da imagem (rastreabilidade)
#
# Guarda em escala de cinza (mamografia não tem cor relevante); o canal RGB que a
# MobileNetV2 espera é replicado na hora do treino, o que economiza 2/3 do disco.
#
# python src/cnn_cache.py [--force]

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "cbis_manifest.csv"
CACHE_DIR = BASE_DIR / "data" / "processed" / "cnn_cache"

SIZES = [128, 160]  # 128 para a CNN do zero, 160 para a MobileNetV2
LABEL_TO_INT = {"benign": 0, "malignant": 1}


def load_one(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        # draft() deixa o decoder do JPEG já devolver numa escala reduzida — numa
        # mamografia de 5236x3016 isso corta a maior parte do custo de decodificação
        im.draft("L", (size, size))
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def build_split(df: pd.DataFrame, cbis_root: Path, variant: str, split: str,
                force: bool) -> None:
    prefix = CACHE_DIR / f"{variant}_{split}"
    pendentes = [s for s in SIZES if force or not (CACHE_DIR / f"{variant}_{split}_{s}.npy").exists()]
    if not pendentes:
        print(f"  {variant}/{split}: cache já existe, pulando (use --force para regerar)")
        return

    paths = [cbis_root / p for p in df["jpeg_path"]]
    print(f"  {variant}/{split}: {len(paths)} imagens, tamanhos {pendentes}...", flush=True)

    for size in pendentes:
        inicio = time.time()
        with ThreadPoolExecutor(max_workers=8) as pool:
            # PIL solta o GIL na decodificação, então thread pool já paraleliza bem
            arrays = list(pool.map(lambda p: load_one(p, size), paths))
        imagens = np.stack(arrays)
        np.save(CACHE_DIR / f"{variant}_{split}_{size}.npy", imagens)
        mb = imagens.nbytes / 1024 ** 2
        print(f"    {size}px: {imagens.shape} ({mb:.0f} MB) em {time.time() - inicio:.0f}s")

    np.save(f"{prefix}_labels.npy", df["label"].map(LABEL_TO_INT).to_numpy(np.int8))
    np.save(f"{prefix}_groups.npy", df["patient_id"].to_numpy(dtype=object))
    np.save(f"{prefix}_keys.npy", df["key"].to_numpy(dtype=object))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbis-root", default=str(BASE_DIR.parent / "cnn"))
    parser.add_argument("--force", action="store_true", help="regera o cache já existente")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado em {MANIFEST_PATH}. Rode antes: "
            f"python src/cnn_data_prep.py"
        )

    cbis_root = Path(args.cbis_root).resolve()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)

    for variant in manifest["variant"].unique():
        for split in ("train", "test"):
            df = manifest[(manifest.variant == variant) & (manifest.split == split)]
            df = df.reset_index(drop=True)
            build_split(df, cbis_root, variant, split, args.force)

    print(f"\nCache salvo em: {CACHE_DIR}")
    print("Próximo passo: python src/cnn_eda.py")


if __name__ == "__main__":
    main()
