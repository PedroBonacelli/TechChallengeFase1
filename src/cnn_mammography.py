# [EXTRA] CNN para diagnóstico via mamografia (CBIS-DDSM).
#
# Treina 4 combinações: 2 entradas x 2 arquiteturas.
#
#   entrada  patch    recorte da lesão (~330px no original) — a lesão ocupa a imagem
#            full     mamografia inteira (~5236x3016) — a lesão é uma fração do quadro
#
#   arch     scratch  CNN de 4 blocos treinada do zero, 128x128 em escala de cinza
#            transfer MobileNetV2 (ImageNet) a 160x160, em dois estágios
#
# Roda sobre o cache de arrays gerado por src/cnn_cache.py (sem isso, cada época
# re-decodificaria JPEGs enormes e o treino em CPU viraria I/O puro).
#
# O split de validação é AGRUPADO POR PACIENTE. Uma paciente costuma ter várias
# anormalidades e a mesma lesão aparece nas incidências CC e MLO; um split aleatório
# colocaria imagens da mesma mama nos dois lados e inflaria a métrica de validação.
#
# python src/cnn_mammography.py --all
# python src/cnn_mammography.py --variant patch --arch transfer
# python src/cnn_mammography.py --all --quick     # valida o pipeline em poucos minutos

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "processed" / "cnn_cache"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

SEED = 42
BATCH_SIZE = 32
VAL_FRACTION = 0.15

# cada arquitetura lê o cache no tamanho que ela espera
ARCH_CONFIG = {
    "scratch": {"size": 128, "channels": 1, "epochs": 40},
    "transfer": {"size": 160, "channels": 3, "epochs_frozen": 12, "epochs_finetune": 8},
}


# --------------------------------------------------------------------------- dados

def load_split(variant: str, split: str, size: int):
    imagens = np.load(CACHE_DIR / f"{variant}_{split}_{size}.npy")
    rotulos = np.load(CACHE_DIR / f"{variant}_{split}_labels.npy").astype("float32")
    grupos = np.load(CACHE_DIR / f"{variant}_{split}_groups.npy", allow_pickle=True)
    return imagens, rotulos, grupos


def split_por_paciente(imagens, rotulos, grupos):
    # GroupShuffleSplit garante que nenhuma paciente apareça em treino e validação
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    idx_treino, idx_val = next(splitter.split(imagens, rotulos, groups=grupos))

    vazamento = set(grupos[idx_treino]) & set(grupos[idx_val])
    assert not vazamento, f"vazamento treino/validação: {vazamento}"

    print(f"  treino {len(idx_treino)} imagens / {len(set(grupos[idx_treino]))} pacientes")
    print(f"  valid. {len(idx_val)} imagens / {len(set(grupos[idx_val]))} pacientes")
    return idx_treino, idx_val


def make_dataset(imagens, rotulos, channels: int, shuffle: bool):
    ds = tf.data.Dataset.from_tensor_slices((imagens[..., None], rotulos[:, None]))
    if shuffle:
        ds = ds.shuffle(len(imagens), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    if channels == 3:
        # mamografia é monocromática; a MobileNetV2 espera 3 canais, então replico.
        # Feito no pipeline (e não numa camada Lambda) pra não complicar o save/load.
        ds = ds.map(lambda x, y: (tf.repeat(x, 3, axis=-1), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def pesos_de_classe(rotulos) -> dict:
    # CBIS-DDSM tem mais benignos que malignos; compensar evita que o modelo
    # aprenda a chutar a classe majoritária
    classes = np.unique(rotulos)
    pesos = compute_class_weight("balanced", classes=classes, y=rotulos)
    return dict(zip(classes.astype(int), pesos))


# ------------------------------------------------------------------------ modelos

def build_augmentation():
    # Augmentation leve: mamografia não deve ser espelhada verticalmente nem girada
    # demais, senão a anatomia deixa de fazer sentido. Só flip horizontal (equivale a
    # trocar mama esquerda/direita), rotações e zoom pequenos, e leve variação de
    # contraste (imita diferença de equipamento/protocolo).
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_scratch(input_shape=(128, 128, 1)) -> tf.keras.Model:
    # 4 blocos conv-batchnorm-pool com filtros crescentes, GlobalAveragePooling
    # (menos parâmetros e menos overfitting que Flatten + Dense grande) e cabeça
    # densa com dropout.
    inputs = tf.keras.Input(shape=input_shape)
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filtros in (32, 64, 128, 256):
        x = layers.Conv2D(filtros, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="cnn_scratch")


def build_transfer(input_shape=(160, 160, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = build_augmentation()(inputs)
    # preprocessamento da MobileNetV2: [0,255] -> [-1,1]. Dentro do modelo, para
    # que treino e inferência usem exatamente a mesma escala.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet")
    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="cnn_transfer"), backbone


def compilar(model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall")],
    )


# ------------------------------------------------------------------------- treino

def callbacks_padrao(checkpoint: Path, patience: int):
    # monitora val_auc (e não accuracy): com classe desbalanceada, accuracy engana
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                             min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_auc",
                                           mode="max", save_best_only=True),
    ]


def historico_para_df(historicos: list) -> pd.DataFrame:
    # o Keras renomeia métricas repetidas ('auc', 'auc_1', ...) quando mais de um
    # modelo é construído no mesmo processo; normaliza os nomes antes de salvar
    frames = []
    for estagio, hist in historicos:
        df = pd.DataFrame(hist.history)
        df.columns = [_normaliza_metrica(c) for c in df.columns]
        df.insert(0, "epoca", range(1, len(df) + 1))
        df.insert(0, "estagio", estagio)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _normaliza_metrica(nome: str) -> str:
    for base in ("auc", "recall"):
        if nome == base or nome.startswith(f"{base}_"):
            return base
        if nome == f"val_{base}" or nome.startswith(f"val_{base}_"):
            return f"val_{base}"
    return nome


def treinar(variant: str, arch: str, quick: bool) -> dict:
    cfg = ARCH_CONFIG[arch]
    size, channels = cfg["size"], cfg["channels"]
    nome = f"{variant}_{arch}"
    print(f"\n{'=' * 70}\n{nome}  (entrada={variant}, arquitetura={arch}, {size}px)\n{'=' * 70}")

    imagens, rotulos, grupos = load_split(variant, "train", size)
    idx_treino, idx_val = split_por_paciente(imagens, rotulos, grupos)

    if quick:
        # só para validar o pipeline: subamostra e poucas épocas
        idx_treino = idx_treino[:256]
        idx_val = idx_val[:128]
        print("  [quick] subamostrando para 256 treino / 128 validação")

    train_ds = make_dataset(imagens[idx_treino], rotulos[idx_treino], channels, shuffle=True)
    val_ds = make_dataset(imagens[idx_val], rotulos[idx_val], channels, shuffle=False)

    pesos = pesos_de_classe(rotulos[idx_treino])
    print(f"  pesos de classe: {({k: round(v, 3) for k, v in pesos.items()})}")

    checkpoint = MODELS_DIR / f"cnn_{nome}.keras"
    inicio = time.time()
    historicos = []

    if arch == "scratch":
        model = build_scratch((size, size, 1))
        compilar(model, lr=1e-4)
        print(f"  parâmetros treináveis: {model.count_params():,}")
        epocas = 2 if quick else cfg["epochs"]
        h = model.fit(train_ds, validation_data=val_ds, epochs=epocas,
                      class_weight=pesos, verbose=2, callbacks=callbacks_padrao(checkpoint, 8))
        historicos.append(("scratch", h))
    else:
        model, backbone = build_transfer((size, size, 3))
        # Estágio 1 — backbone congelado, treina só a cabeça densa. Barato e evita
        # que gradientes grandes de uma cabeça aleatória destruam os pesos ImageNet.
        compilar(model, lr=1e-3)
        print(f"  [estágio 1] backbone congelado — {model.count_params():,} params "
              f"({sum(np.prod(w.shape) for w in model.trainable_weights):,.0f} treináveis)")
        epocas = 2 if quick else cfg["epochs_frozen"]
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=epocas,
                       class_weight=pesos, verbose=2, callbacks=callbacks_padrao(checkpoint, 5))
        historicos.append(("congelado", h1))

        # Estágio 2 — descongela o último bloco convolucional com learning rate baixo.
        # Fine-tune completo seria inviável em CPU (medido: ~7,7 s/batch).
        backbone.trainable = True  # libera tudo, e então congela o que não deve treinar
        for layer in backbone.layers[:-24]:
            layer.trainable = False
        for layer in backbone.layers[-24:]:
            # BatchNorm fica congelada mesmo no bloco liberado: com batch pequeno,
            # atualizar as estatísticas de um backbone pré-treinado costuma piorar
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        compilar(model, lr=1e-5)
        print(f"  [estágio 2] fine-tune do último bloco — "
              f"{sum(np.prod(w.shape) for w in model.trainable_weights):,.0f} params treináveis")
        epocas = 1 if quick else cfg["epochs_finetune"]
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=epocas,
                       class_weight=pesos, verbose=2, callbacks=callbacks_padrao(checkpoint, 4))
        historicos.append(("fine-tune", h2))

    duracao = time.time() - inicio
    hist_df = historico_para_df(historicos)
    hist_df.to_csv(REPORTS_DIR / f"cnn_history_{nome}.csv", index=False)

    melhor_auc = float(hist_df["val_auc"].max())
    print(f"  concluído em {duracao / 60:.1f} min — melhor val_auc: {melhor_auc:.4f}")
    print(f"  modelo salvo em {checkpoint}")

    return {"modelo": nome, "variant": variant, "arch": arch,
            "epocas": int(len(hist_df)), "minutos": round(duracao / 60, 1),
            "melhor_val_auc": round(melhor_auc, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["patch", "full"], default="patch")
    parser.add_argument("--arch", choices=["scratch", "transfer"], default="scratch")
    parser.add_argument("--all", action="store_true",
                        help="roda as 4 combinações de entrada x arquitetura")
    parser.add_argument("--quick", action="store_true",
                        help="subamostra e poucas épocas, só para validar o pipeline")
    args = parser.parse_args()

    if not CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Cache não encontrado em {CACHE_DIR}. Rode antes:\n"
            f"    python src/cnn_data_prep.py\n    python src/cnn_cache.py")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    combinacoes = ([(v, a) for v in ("patch", "full") for a in ("scratch", "transfer")]
                   if args.all else [(args.variant, args.arch)])

    resumo = [treinar(variant, arch, args.quick) for variant, arch in combinacoes]

    print(f"\n{'=' * 70}\nRESUMO DO TREINO\n{'=' * 70}")
    print(pd.DataFrame(resumo).to_string(index=False))
    if not args.quick:
        (REPORTS_DIR / "cnn_training_summary.json").write_text(
            json.dumps(resumo, indent=2), encoding="utf-8")
    print("\nPróximo passo: python src/cnn_evaluate.py")


if __name__ == "__main__":
    main()
