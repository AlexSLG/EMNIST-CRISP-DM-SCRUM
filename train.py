"""Pipeline CRISP-DM sobre EMNIST Balanced (47 clases) con una CNN.

Fases que se ejecutan aqui (la 1 y la 6 estan en FASES_CRISP-DM.txt y app.py):
  2. Comprension de los datos  -> reports/*.png + reports/data_understanding.json
  3. Preparacion de los datos  -> normalizacion, correccion de orientacion, particion
  4. Modelado                  -> baseline MLP y CNN
  5. Evaluacion                -> reports/metrics.json, matriz de confusion, errores
  6. Despliegue (artefactos)   -> model/emnist_cnn.keras, model/labels.json, model/samples.npz

Uso:  python train.py [--epochs 15]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).parent
DATA = ROOT / "archive"
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "model"
SEED = 42
NUM_CLASSES = 47

REPORTS.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------------------------------------------------------
# FASE 2 - Comprension de los datos
# ----------------------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].to_numpy(np.int64), df.iloc[:, 1:].to_numpy(np.uint8)


def load_mapping():
    """Cada linea: <indice de clase> <codigo ASCII del caracter>."""
    mapping = {}
    for line in (DATA / "emnist-balanced-mapping.txt").read_text().split("\n"):
        if line.strip():
            k, v = line.split()
            mapping[int(k)] = chr(int(v))
    return mapping


def phase2_understanding(y_tr, X_tr, y_te, X_te, mapping):
    print("\n=== FASE 2: Comprension de los datos ===")
    info = {
        "train_muestras": int(len(y_tr)),
        "test_muestras": int(len(y_te)),
        "pixeles_por_imagen": int(X_tr.shape[1]),
        "clases": int(len(np.unique(y_tr))),
        "valores_nulos": int(np.isnan(X_tr.astype(float)).sum()),
        "pixel_min": int(X_tr.min()),
        "pixel_max": int(X_tr.max()),
        "pixel_media": round(float(X_tr.mean()), 3),
        "pixel_std": round(float(X_tr.std()), 3),
        "porcentaje_pixeles_cero": round(float((X_tr == 0).mean() * 100), 2),
        "duplicados_exactos_en_muestra_20k": int(
            pd.DataFrame(X_tr[:20000]).duplicated().sum()
        ),
    }
    counts = np.bincount(y_tr, minlength=NUM_CLASSES)
    info["muestras_por_clase_min"] = int(counts.min())
    info["muestras_por_clase_max"] = int(counts.max())
    info["balanceado"] = bool(counts.min() == counts.max())
    (REPORTS / "data_understanding.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    # Distribucion de clases
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar([mapping[i] for i in range(NUM_CLASSES)], counts, color="#3b82f6")
    ax.set_title("Muestras por clase en entrenamiento (EMNIST Balanced)")
    ax.set_ylabel("imagenes")
    fig.tight_layout()
    fig.savefig(REPORTS / "01_distribucion_clases.png", dpi=130)
    plt.close(fig)

    # Imagenes crudas (sin corregir) vs corregidas -> hallazgo de la transposicion
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
    for j in range(8):
        raw = X_tr[j].reshape(28, 28)
        axes[0, j].imshow(raw, cmap="gray")
        axes[0, j].set_title("crudo", fontsize=8)
        axes[1, j].imshow(raw.T, cmap="gray")
        axes[1, j].set_title(f"corregido: {mapping[y_tr[j]]}", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    fig.suptitle("Hallazgo: las imagenes de EMNIST vienen transpuestas")
    fig.tight_layout()
    fig.savefig(REPORTS / "02_orientacion.png", dpi=130)
    plt.close(fig)

    # Una muestra por clase
    fig, axes = plt.subplots(5, 10, figsize=(11, 6))
    for ax in axes.ravel():
        ax.axis("off")
    for c in range(NUM_CLASSES):
        idx = np.where(y_tr == c)[0][0]
        axes.ravel()[c].imshow(X_tr[idx].reshape(28, 28).T, cmap="gray")
        axes.ravel()[c].set_title(mapping[c], fontsize=10)
    fig.suptitle("Un ejemplo por cada una de las 47 clases")
    fig.tight_layout()
    fig.savefig(REPORTS / "03_ejemplos_clases.png", dpi=130)
    plt.close(fig)

    # Imagen media (donde se concentra la informacion)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.imshow(X_tr.mean(0).reshape(28, 28).T, cmap="magma")
    ax.set_title("Imagen promedio (todas las clases)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(REPORTS / "04_imagen_promedio.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# FASE 3 - Preparacion de los datos
# ----------------------------------------------------------------------------
def phase3_preparation(X_tr, y_tr, X_te, y_te):
    print("\n=== FASE 3: Preparacion de los datos ===")

    def prep(X):
        # (N,784) -> (N,28,28,1), transponer (orientacion correcta), escalar 0-1
        X = X.reshape(-1, 28, 28).transpose(0, 2, 1)
        return (X.astype("float32") / 255.0)[..., None]

    X_tr, X_te = prep(X_tr), prep(X_te)
    # 10% de entrenamiento se aparta como validacion (estratificado)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=SEED
    )
    print(f"train={X_train.shape} val={X_val.shape} test={X_te.shape}")
    return X_train, y_train, X_val, y_val, X_te, y_te


# ----------------------------------------------------------------------------
# FASE 4 - Modelado
# ----------------------------------------------------------------------------
def build_baseline():
    return keras.Sequential(
        [
            layers.Input((28, 28, 1)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="baseline_mlp",
    )


def conv_block(x, filters, drop):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(drop)(x)


def build_cnn():
    inp = layers.Input((28, 28, 1))
    # Aumento de datos ligero: solo activo durante entrenamiento
    x = layers.RandomRotation(0.06)(inp)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomZoom(0.08)(x)
    x = conv_block(x, 32, 0.20)
    x = conv_block(x, 64, 0.25)
    x = conv_block(x, 128, 0.30)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name="emnist_cnn")


def fit(model, data, epochs, lr, tag):
    X_train, y_train, X_val, y_val = data
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    t0 = time.time()
    hist = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=128, callbacks=cbs, verbose=2,
    )
    print(f"[{tag}] entrenado en {time.time() - t0:.0f}s")
    return hist.history


def phase4_modeling(data, epochs):
    print("\n=== FASE 4: Modelado ===")
    base = build_baseline()
    h_base = fit(base, data, epochs=6, lr=1e-3, tag="baseline MLP")
    cnn = build_cnn()
    cnn.summary()
    h_cnn = fit(cnn, data, epochs=epochs, lr=1e-3, tag="CNN")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, ax, title in (("accuracy", axes[0], "Exactitud"), ("loss", axes[1], "Perdida")):
        ax.plot(h_cnn[key], label="CNN train")
        ax.plot(h_cnn["val_" + key], label="CNN val")
        ax.plot(h_base["val_" + key], "--", label="MLP val")
        ax.set_title(title)
        ax.set_xlabel("epoca")
        ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "05_curvas_entrenamiento.png", dpi=130)
    plt.close(fig)
    return base, cnn, h_cnn


# ----------------------------------------------------------------------------
# FASE 5 - Evaluacion
# ----------------------------------------------------------------------------
def phase5_evaluation(base, cnn, X_te, y_te, mapping, X_val, y_val, hist):
    print("\n=== FASE 5: Evaluacion ===")
    names = [mapping[i] for i in range(NUM_CLASSES)]
    base_acc = float(base.evaluate(X_te, y_te, verbose=0)[1])
    probs = cnn.predict(X_te, batch_size=512, verbose=0)
    pred = probs.argmax(1)
    cnn_acc = float((pred == y_te).mean())
    val_acc = float(cnn.evaluate(X_val, y_val, verbose=0)[1])
    top3 = float(np.mean([y in np.argsort(p)[-3:] for y, p in zip(y_te, probs)]))
    rep = classification_report(y_te, pred, target_names=names, output_dict=True, zero_division=0)

    per_class = {n: round(rep[n]["f1-score"], 4) for n in names}
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:8]

    cm = confusion_matrix(y_te, pred)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for flat in np.argsort(off, axis=None)[::-1][:10]:
        i, j = divmod(int(flat), NUM_CLASSES)
        pairs.append({"real": names[i], "predicho": names[j], "veces": int(off[i, j])})

    metrics = {
        "baseline_mlp_test_accuracy": round(base_acc, 4),
        "cnn_val_accuracy": round(val_acc, 4),
        "cnn_test_accuracy": round(cnn_acc, 4),
        "cnn_test_top3_accuracy": round(top3, 4),
        "cnn_macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "cnn_weighted_f1": round(rep["weighted avg"]["f1-score"], 4),
        "epocas_entrenadas": len(hist["loss"]),
        "peores_clases_f1": worst,
        "confusiones_mas_frecuentes": pairs,
        "f1_por_clase": per_class,
        "parametros_cnn": int(cnn.count_params()),
        "azar": round(1 / NUM_CLASSES, 4),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in metrics.items() if k != "f1_por_clase"}, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", ax=ax, xticklabels=names, yticklabels=names, cbar=False)
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_title(f"Matriz de confusion - CNN (test acc = {cnn_acc:.3f})")
    fig.tight_layout()
    fig.savefig(REPORTS / "06_matriz_confusion.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(names, [per_class[n] for n in names], color="#10b981")
    ax.set_ylim(0.4, 1)
    ax.set_title("F1-score por clase (test)")
    fig.tight_layout()
    fig.savefig(REPORTS / "07_f1_por_clase.png", dpi=130)
    plt.close(fig)

    wrong = np.where(pred != y_te)[0][:24]
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), wrong):
        ax.imshow(X_te[k, ..., 0], cmap="gray")
        ax.set_title(f"real {names[y_te[k]]} / pred {names[pred[k]]}", fontsize=8)
    fig.suptitle("Ejemplos mal clasificados (muchos son ambiguos incluso para humanos)")
    fig.tight_layout()
    fig.savefig(REPORTS / "08_errores.png", dpi=130)
    plt.close(fig)
    return metrics


# ----------------------------------------------------------------------------
# FASE 6 - Despliegue (artefactos que consume app.py)
# ----------------------------------------------------------------------------
def phase6_export(cnn, X_te, y_te, mapping):
    print("\n=== FASE 6: Exportando artefactos ===")
    cnn.save(MODEL_DIR / "emnist_cnn.keras")
    (MODEL_DIR / "labels.json").write_text(
        json.dumps({str(k): v for k, v in mapping.items()}, ensure_ascii=False)
    )
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_te), 300, replace=False)
    np.savez_compressed(
        MODEL_DIR / "samples.npz",
        X=(X_te[idx, ..., 0] * 255).astype(np.uint8), y=y_te[idx],
    )
    size = (MODEL_DIR / "emnist_cnn.keras").stat().st_size / 1e6
    print(f"modelo guardado ({size:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    mapping = load_mapping()
    y_tr, X_tr = load_csv(DATA / "emnist-balanced-train.csv")
    y_te, X_te = load_csv(DATA / "emnist-balanced-test.csv")

    phase2_understanding(y_tr, X_tr, y_te, X_te, mapping)
    X_train, y_train, X_val, y_val, X_te, y_te = phase3_preparation(X_tr, y_tr, X_te, y_te)
    base, cnn, hist = phase4_modeling((X_train, y_train, X_val, y_val), args.epochs)
    phase5_evaluation(base, cnn, X_te, y_te, mapping, X_val, y_val, hist)
    phase6_export(cnn, X_te, y_te, mapping)


if __name__ == "__main__":
    main()
