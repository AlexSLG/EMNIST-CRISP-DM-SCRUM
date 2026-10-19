"""App Streamlit: recorrido por las 6 fases de CRISP-DM + demo del modelo CNN EMNIST."""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="EMNIST CNN - CRISP-DM", page_icon="🔤", layout="wide")


@st.cache_resource
def load_model():
    from tensorflow import keras

    return keras.models.load_model(MODEL_DIR / "emnist_cnn.keras", compile=False)


@st.cache_data
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_samples():
    d = np.load(MODEL_DIR / "samples.npz")
    return d["X"], d["y"]


def preprocess(img: Image.Image) -> np.ndarray:
    """Imagen libre -> tensor 28x28 con el estilo de EMNIST (trazo blanco sobre negro,
    caracter de ~20 px centrado, leve desenfoque)."""
    img = ImageOps.exif_transpose(img).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    if arr.mean() > 127:  # fondo claro -> invertir
        arr = 255 - arr
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    mask = arr > 40
    if not mask.any():
        return np.zeros((28, 28), np.float32)
    ys, xs = np.where(mask)
    crop = Image.fromarray(arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1].astype(np.uint8))
    w, h = crop.size
    scale = 20 / max(w, h)
    crop = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, ((28 - crop.size[0]) // 2, (28 - crop.size[1]) // 2))
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.6))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def predict(x28: np.ndarray, labels: dict):
    probs = load_model().predict(x28[None, ..., None], verbose=0)[0]
    top = np.argsort(probs)[::-1][:5]
    return pd.DataFrame(
        {"caracter": [labels[str(i)] for i in top], "probabilidad": probs[top]}
    )


def show_image(name, caption=None):
    p = REPORTS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


st.title("🔤 Reconocimiento de caracteres manuscritos con CNN")
st.caption("Dataset EMNIST Balanced (47 clases) · Metodologia CRISP-DM · TensorFlow/Keras")

ready = (MODEL_DIR / "emnist_cnn.keras").exists() and (REPORTS / "metrics.json").exists()
if not ready:
    st.error("Faltan los artefactos. Ejecuta primero `python train.py`.")
    st.stop()

metrics = load_json(REPORTS / "metrics.json")
info = load_json(REPORTS / "data_understanding.json")
labels = load_json(MODEL_DIR / "labels.json")

tabs = st.tabs([
    "1 · Negocio", "2 · Datos", "3 · Preparación",
    "4 · Modelado", "5 · Evaluación", "6 · Despliegue y demo",
])

# ------------------------------------------------------------------ 1
with tabs[0]:
    st.header("1. Comprensión del negocio")
    st.markdown("""
**Problema.** Digitalizar texto escrito a mano (formularios, cheques, examenes, notas) obliga a
reconocer cada caracter de forma automatica. Hacerlo manualmente es lento y propenso a errores.

**Objetivo de negocio.** Reducir la captura manual de caracteres manuscritos con un
clasificador automatico y confiable.

**Objetivo de minería de datos.** Entrenar una red neuronal convolucional (CNN) que, dada una
imagen de 28×28 px en escala de grises, prediga a cual de las **47 clases** pertenece
(10 digitos + 26 mayusculas + 11 minusculas distintivas).

| Criterio de exito | Meta |
|---|---|
| Exactitud en test | ≥ 88 % (la literatura reporta ~90 % para EMNIST Balanced) |
| Mejora sobre un modelo simple | CNN > baseline MLP |
| Top-3 accuracy | ≥ 97 % |
| Entrega | App web publica donde cualquiera pruebe el modelo |

**Supuestos y riesgos.** Hay letras casi identicas (O/0, I/l, S/5, Z/2, g/q), por lo que
se espera un techo de exactitud por debajo del 100 %. El modelo solo ve caracteres aislados,
no palabras completas.
""")

# ------------------------------------------------------------------ 2
with tabs[1]:
    st.header("2. Comprensión de los datos")
    c = st.columns(4)
    c[0].metric("Imagenes de entrenamiento", f"{info['train_muestras']:,}")
    c[1].metric("Imagenes de test", f"{info['test_muestras']:,}")
    c[2].metric("Clases", info["clases"])
    c[3].metric("Pixeles por imagen", "28 × 28")
    st.markdown(f"""
- **Fuente:** EMNIST (Extended MNIST) de Kaggle, derivado del NIST Special Database 19. Se usa la variante **Balanced**.
- **Calidad:** valores nulos = **{info['valores_nulos']}**; pixeles en [{info['pixel_min']}, {info['pixel_max']}];
  {info['porcentaje_pixeles_cero']} % de los pixeles son 0 (fondo negro).
- **Balance:** entre {info['muestras_por_clase_min']} y {info['muestras_por_clase_max']} imagenes por clase → perfectamente balanceado, no hace falta re-muestreo.
- **Hallazgo clave:** las imagenes vienen **transpuestas** (rotadas/espejadas); hay que corregirlo.
""")
    a, b = st.columns(2)
    with a:
        show_image("01_distribucion_clases.png")
        show_image("02_orientacion.png")
    with b:
        show_image("03_ejemplos_clases.png")
        show_image("04_imagen_promedio.png")

# ------------------------------------------------------------------ 3
with tabs[2]:
    st.header("3. Preparación de los datos")
    st.markdown("""
| Paso | Que se hizo | Por que |
|---|---|---|
| Lectura | CSV → `label` + 784 pixeles | formato de Kaggle |
| Reestructurar | `(N,784)` → `(N,28,28,1)` | la CNN espera canales |
| Orientacion | transponer cada imagen | EMNIST viene rotado |
| Normalizacion | pixeles / 255 → `[0,1]` | entrenamiento mas estable |
| Validacion | 10 % del train, **estratificado** | afinar sin tocar el test |
| Aumento de datos | rotacion ±6 %, traslacion 8 %, zoom 8 % (solo en train) | robustez a trazos reales |
| Test | `emnist-balanced-test.csv` intacto | evaluacion honesta |

**Particion final:** 101 520 train · 11 280 validación · 18 800 test.
""")

# ------------------------------------------------------------------ 4
with tabs[3]:
    st.header("4. Modelado")
    st.markdown(f"""
**Baseline:** MLP de una capa oculta (256 neuronas). Sirve de referencia: si la CNN no lo supera, no vale la pena.

**Modelo final (CNN, {metrics['parametros_cnn']:,} parametros):**
```
Entrada 28×28×1
→ Aumento de datos (rotacion, traslacion, zoom)
→ [Conv3×3(32)  + BN + ReLU] ×2 → MaxPool → Dropout 0.20
→ [Conv3×3(64)  + BN + ReLU] ×2 → MaxPool → Dropout 0.25
→ [Conv3×3(128) + BN + ReLU] ×2 → MaxPool → Dropout 0.30
→ GlobalAveragePooling → Dense 256 ReLU → Dropout 0.40
→ Dense 47 Softmax
```
**Entrenamiento:** Adam (lr 1e-3), perdida *sparse categorical crossentropy*, batch 128,
`ReduceLROnPlateau` y `EarlyStopping` (restaura los mejores pesos). Epocas usadas: {metrics['epocas_entrenadas']}.
""")
    show_image("05_curvas_entrenamiento.png")

# ------------------------------------------------------------------ 5
with tabs[4]:
    st.header("5. Evaluación")
    c = st.columns(5)
    c[0].metric("CNN · exactitud test", f"{metrics['cnn_test_accuracy']:.2%}")
    c[1].metric("Top-3", f"{metrics['cnn_test_top3_accuracy']:.2%}")
    c[2].metric("F1 macro", f"{metrics['cnn_macro_f1']:.3f}")
    c[3].metric("Baseline MLP", f"{metrics['baseline_mlp_test_accuracy']:.2%}")
    c[4].metric("Azar", f"{metrics['azar']:.2%}")
    ok_acc = metrics["cnn_test_accuracy"] >= 0.88
    ok_base = metrics["cnn_test_accuracy"] > metrics["baseline_mlp_test_accuracy"]
    ok_top3 = metrics["cnn_test_top3_accuracy"] >= 0.97
    st.markdown(
        f"**Criterios del negocio:** exactitud ≥ 88 % {'✅' if ok_acc else '❌'} · "
        f"supera al baseline {'✅' if ok_base else '❌'} · top-3 ≥ 97 % {'✅' if ok_top3 else '❌'}"
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Confusiones mas frecuentes")
        st.dataframe(pd.DataFrame(metrics["confusiones_mas_frecuentes"]), hide_index=True)
        st.subheader("Clases con peor F1")
        st.dataframe(
            pd.DataFrame(metrics["peores_clases_f1"], columns=["clase", "F1"]), hide_index=True
        )
        st.info(
            "Los errores se concentran en pares visualmente casi identicos (por ejemplo "
            "O/0, F/f, L/1, I/1, q/9). Es una limitacion del dato, no solo del modelo."
        )
    with right:
        show_image("07_f1_por_clase.png")
        show_image("08_errores.png")
    show_image("06_matriz_confusion.png")

# ------------------------------------------------------------------ 6
with tabs[5]:
    st.header("6. Despliegue y demo")
    st.markdown(
        "El modelo (`model/emnist_cnn.keras`) se sirve con **Streamlit** y se publica en "
        "Streamlit Community Cloud desde un repositorio de GitHub. Prueba el modelo:"
    )
    mode = st.radio(
        "Entrada", ["Muestras del test set", "Subir una imagen", "Usar la camara"], horizontal=True
    )
    x28, truth = None, None

    if mode == "Muestras del test set":
        X, y = load_samples()
        k = st.slider("Muestra", 0, len(y) - 1, 0)
        x28, truth = X[k] / 255.0, labels[str(int(y[k]))]
    elif mode == "Subir una imagen":
        up = st.file_uploader("PNG/JPG con un caracter (negro sobre blanco o al reves)", ["png", "jpg", "jpeg"])
        if up:
            x28 = preprocess(Image.open(up))
    else:
        shot = st.camera_input("Toma una foto de un caracter (tinta oscura sobre papel claro)")
        if shot:
            x28 = preprocess(Image.open(shot))

    if x28 is not None:
        df = predict(x28, labels)
        a, b, c = st.columns([1, 1, 2])
        a.image((x28 * 255).astype(np.uint8), caption="Entrada 28×28", width=160, clamp=True)
        b.metric("Prediccion", df.iloc[0]["caracter"], f"{df.iloc[0]['probabilidad']:.1%}")
        if truth is not None:
            b.caption(f"Etiqueta real: **{truth}**")
        c.bar_chart(df.set_index("caracter"))

    with st.expander("Como se desplego"):
        st.markdown("""
1. `python train.py` genera `model/` y `reports/`.
2. Se sube el repo a GitHub (sin `archive/`, ver `.gitignore`).
3. En share.streamlit.io: *New app* → repo → `app.py` → Deploy.
4. Dependencias en `requirements.txt` (TensorFlow CPU 2.21; elegir Python 3.11).

**Monitoreo/mantenimiento sugerido:** registrar predicciones de baja confianza, reentrenar con
escritura real de los usuarios y vigilar la caida de exactitud (data drift).
""")
