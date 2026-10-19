"""Agente CNN para reconocer digitos (MNIST) con Keras + MaxPooling.

Ciclo agentico: PERCIBE (carga dataset) -> PLANEA (elige configuracion)
-> ACTUA (entrena) -> EVALUA (test) -> DECIDE (para si alcanza la meta,
si no prueba una configuracion mas potente). Exporta el mejor modelo a
docs/ para GitHub Pages.
"""
import json
import os
import struct

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
from keras import layers

BASE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(BASE, "archive")
OUT = os.path.join(BASE, "docs")
META_ACC = 0.99


def leer_imagenes(ruta):
    with open(ruta, "rb") as f:
        _, n, h, w = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), np.uint8).reshape(n, h, w)


def leer_etiquetas(ruta):
    with open(ruta, "rb") as f:
        f.read(8)
        return np.frombuffer(f.read(), np.uint8)


def cargar():
    x_tr = leer_imagenes(os.path.join(ARCHIVE, "train-images.idx3-ubyte"))
    y_tr = leer_etiquetas(os.path.join(ARCHIVE, "train-labels.idx1-ubyte"))
    x_te = leer_imagenes(os.path.join(ARCHIVE, "t10k-images.idx3-ubyte"))
    y_te = leer_etiquetas(os.path.join(ARCHIVE, "t10k-labels.idx1-ubyte"))
    f = lambda x: (x.astype("float32") / 255.0)[..., None]
    return f(x_tr), y_tr, f(x_te), y_te


def construir(c1, c2, dropout):
    # Arquitectura fija en el front-end (JS): Conv-MaxPool-Conv-MaxPool-Dense
    return keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(c1, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(c2, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax"),
    ])


PLANES = [
    dict(c1=16, c2=32, dropout=0.3, epochs=4),
    dict(c1=32, c2=64, dropout=0.4, epochs=6),
    dict(c1=32, c2=64, dropout=0.4, epochs=12),
]


def main():
    keras.utils.set_random_seed(42)
    x_tr, y_tr, x_te, y_te = cargar()
    print(f"[percibe] train={x_tr.shape} test={x_te.shape}")
    bitacora, mejor, mejor_acc = [], None, 0.0

    for i, plan in enumerate(PLANES, 1):
        print(f"[planea] intento {i}: {plan}")
        m = construir(plan["c1"], plan["c2"], plan["dropout"])
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        h = m.fit(x_tr, y_tr, epochs=plan["epochs"], batch_size=128,
                  validation_split=0.1, verbose=2)
        loss, acc = m.evaluate(x_te, y_te, verbose=0)
        print(f"[evalua] accuracy test = {acc:.4f}")
        bitacora.append(dict(intento=i, **plan, test_acc=round(float(acc), 4),
                             val_acc=[round(float(v), 4) for v in h.history["val_accuracy"]]))
        if acc > mejor_acc:
            mejor, mejor_acc = m, acc
        if acc >= META_ACC:
            print(f"[decide] meta {META_ACC} alcanzada, el agente se detiene")
            break
        print("[decide] meta no alcanzada, probando configuracion mas potente")

    os.makedirs(OUT, exist_ok=True)
    mejor.save(os.path.join(BASE, "modelo_cnn.keras"))
    pesos = [w.tolist() for w in mejor.get_weights()]
    with open(os.path.join(OUT, "pesos.json"), "w") as f:
        json.dump(dict(filtros=[mejor.layers[0].filters, mejor.layers[2].filters],
                       pesos=pesos, test_acc=round(float(mejor_acc), 4),
                       bitacora=bitacora), f, separators=(",", ":"))
    # muestras reales del test para probar en la web
    idx = [int(np.where(y_te == d)[0][0]) for d in range(10)]
    with open(os.path.join(OUT, "muestras.json"), "w") as f:
        json.dump([dict(label=int(y_te[k]),
                        px=(x_te[k, :, :, 0] * 255).astype(int).flatten().tolist())
                   for k in idx], f, separators=(",", ":"))
    print(f"[listo] accuracy final {mejor_acc:.4f}")


if __name__ == "__main__":
    main()
