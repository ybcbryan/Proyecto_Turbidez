"""Genera las figuras del informe y la presentación.

Entradas : src/resultados_caracteristicas/caracteristicas_v2.csv
           src/resultados_modelos/*.csv
Salida   : src/resultados_figuras/*.png
Uso      : python src/generar_figuras.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_zona_clara import detectar_roi  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DATASET = RAIZ / "WaTur-Water-Turbidity-Dataset-main" / "Formazine"
CARAC = RAIZ / "src" / "resultados_caracteristicas" / "caracteristicas_v2.csv"
MODELOS = RAIZ / "src" / "resultados_modelos"
SALIDA = RAIZ / "src" / "resultados_figuras"

# Paleta de referencia (modo claro)
AZUL, NARANJA = "#2a78d6", "#eb6834"
TEXTO, TEXTO2, REJILLA = "#0b0b0b", "#52514e", "#e6e5e1"
SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul", ["#cfe2f8", "#8ab8ec", "#2a78d6", "#123f77"]
)
NIVELES = [0, 0.5, 1, 2.5, 4, 5, 7.5, 10, 40]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": TEXTO, "axes.labelcolor": TEXTO2,
    "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.edgecolor": REJILLA, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def nombre_nivel(n):
    return str(int(n)) if float(n) == int(n) else str(n)


def guardar(fig, nombre):
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / nombre, facecolor="white")
    plt.close(fig)
    print("OK", nombre)


def fig_roi_ejemplos():
    casos = [("0", "1137.jpg"), ("0", "2.jpg"), ("2.5", "195.jpg"), ("5", "1166.jpg"),
             ("10", "1893.jpg"), ("40", "353.jpg"),
             ("1", "1127.jpg"), ("1", "946.jpg")]
    fig, ejes = plt.subplots(2, len(casos), figsize=(2.1 * len(casos), 4.7))
    for k, (n, f) in enumerate(casos):
        im = cv2.imread(str(DATASET / n / f))
        r = detectar_roi(im)
        ov = im.copy()
        m = r.mascara > 0
        ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
        for fila, img in enumerate((im, ov)):
            ax = ejes[fila, k]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis("off")
        estado = "ROI válida" if r.ok else f"Rechazada\n({r.motivo.lower()})"
        ejes[0, k].set_title(f"{n} NTU\n{estado}", fontsize=9.5,
                             color=TEXTO if r.ok else NARANJA)
    ejes[0, 0].text(-0.08, 0.5, "Original", transform=ejes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", color=TEXTO2)
    ejes[1, 0].text(-0.08, 0.5, "ROI (verde)", transform=ejes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", color=TEXTO2)
    guardar(fig, "01_roi_ejemplos.png")


def fig_estados(df):
    d = df[df.ok]
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = [(i, nombre_nivel(n)) for i, n in enumerate(NIVELES)]
    for desp, (p, color) in zip((-0.19, 0.19), (("ARRIBA", AZUL), ("ABAJO", NARANJA))):
        datos = [d[(d.nivel_ntu == n) & (d.posicion == p)].V_P50 for n in NIVELES]
        bp = ax.boxplot(datos, positions=np.arange(len(NIVELES)) + desp, widths=0.32,
                        patch_artist=True, showfliers=False)
        for caja in bp["boxes"]:
            caja.set(facecolor=color, edgecolor=color, alpha=0.85)
        for el in bp["medians"]:
            el.set(color="white", linewidth=1.6)
        for el in bp["whiskers"] + bp["caps"]:
            el.set(color=color)
        ax.plot([], [], "s", color=color, label=f"Zona clara {p.lower()}")
    ax.set_xticks(range(len(NIVELES)), [nombre_nivel(n) for n in NIVELES])
    ax.set_xlabel("Turbidez real (NTU)")
    ax.set_ylabel("Brillo mediano de la ROI (V, P50)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("El estado de la escena domina el brillo", loc="left", color=TEXTO)
    guardar(fig, "02_brillo_por_estado.png")


def fig_boxplot_ntu(df):
    d = df[df.ok]
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (p, color) in zip(ejes, (("ARRIBA", AZUL), ("ABAJO", NARANJA))):
        datos = [d[(d.nivel_ntu == n) & (d.posicion == p)].Lab_L_media for n in NIVELES]
        bp = ax.boxplot(datos, patch_artist=True, showfliers=False, widths=0.6)
        for caja in bp["boxes"]:
            caja.set(facecolor=color, edgecolor=color, alpha=0.85)
        for el in bp["medians"]:
            el.set(color="white", linewidth=1.6)
        for el in bp["whiskers"] + bp["caps"]:
            el.set(color=color)
        ax.set_xticks(range(1, len(NIVELES) + 1), [nombre_nivel(n) for n in NIVELES])
        ax.set_xlabel("Turbidez real (NTU)")
        ax.set_title(f"Zona clara {p.lower()}", loc="left", color=TEXTO)
    ejes[0].set_ylabel("Luminosidad media L* de la ROI")
    guardar(fig, "03_boxplot_L_por_ntu.png")


def fig_correlaciones(df):
    d = df[df.ok]
    excluir = {"nivel_ntu", "frame", "ok", "area_pct", "contraste_clara_oscura", "roi_mad"}
    cols = [c for c in d.select_dtypes("number").columns if c not in excluir]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True)
    for ax, (p, color) in zip(ejes, (("ARRIBA", AZUL), ("ABAJO", NARANJA))):
        sub = d[d.posicion == p]
        rho = sub[cols].apply(lambda c: c.corr(sub.nivel_ntu, method="spearman"))
        top = rho.reindex(rho.abs().sort_values(ascending=False).index).head(12)[::-1]
        ax.barh(top.index, top.values, color=color, height=0.62)
        ax.axvline(0, color=TEXTO2, linewidth=0.8)
        ax.set_title(f"Zona clara {p.lower()}", loc="left", color=TEXTO)
        ax.set_xlabel("Correlación de Spearman con NTU")
        ax.grid(axis="y", visible=False)
    guardar(fig, "04_correlaciones.png")
    return


def fig_pca(df):
    d = df[df.ok & (df.posicion == "ARRIBA")]
    excluir = {"nivel_ntu", "frame", "ok", "area_pct", "contraste_clara_oscura", "roi_mad"}
    cols = [c for c in d.select_dtypes("number").columns if c not in excluir]
    Z = StandardScaler().fit_transform(d[cols])
    pca = PCA(n_components=2).fit(Z)
    P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sc = ax.scatter(P[:, 0], P[:, 1], c=np.log1p(d.nivel_ntu), cmap=SECUENCIAL,
                    s=7, alpha=0.7, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, ticks=np.log1p([0, 1, 5, 10, 40]))
    cb.ax.set_yticklabels(["0", "1", "5", "10", "40"])
    cb.set_label("Turbidez (NTU)")
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f} % de la varianza)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f} % de la varianza)")
    ax.set_title("PCA de las características (zona clara arriba)", loc="left", color=TEXTO)
    guardar(fig, "05_pca.png")


def fig_modelos():
    m = pd.read_csv(MODELOS / "metricas_modelos.csv")
    cfg = [("sin_posicion", "Un modelo, sin indicar estado", "#8ab8ec"),
           ("todas", "Un modelo, con estado", AZUL),
           ("por_posicion", "Un modelo por estado", "#123f77")]
    orden = ["Ridge", "SVR", "RandomForest", "GradientBoosting"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ancho = 0.26
    for k, (c, etiqueta, color) in enumerate(cfg):
        v = [m[(m.config == c) & (m.modelo == o)].RMSE.iloc[0] for o in orden]
        b = ax.bar(np.arange(len(orden)) + (k - 1) * (ancho + 0.02), v, ancho,
                   color=color, label=etiqueta)
        ax.bar_label(b, fmt="%.1f", fontsize=8.5, color=TEXTO2, padding=2)
    ax.set_xticks(range(len(orden)), ["Ridge", "SVR", "Random Forest", "Gradient Boosting"])
    ax.set_ylabel("RMSE (NTU) — menor es mejor")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Comparación de modelos (validación por bloques)", loc="left", color=TEXTO)
    guardar(fig, "06_comparacion_modelos.png")


def fig_real_vs_pred():
    p = pd.read_csv(MODELOS / "predicciones_mejor_modelo.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    datos = [p[p.nivel_ntu == n].pred_ntu for n in NIVELES]
    bp = ax.boxplot(datos, positions=range(len(NIVELES)), widths=0.6,
                    patch_artist=True, showfliers=False)
    for caja in bp["boxes"]:
        caja.set(facecolor="#8ab8ec", edgecolor=AZUL)
    for el in bp["medians"]:
        el.set(color=TEXTO, linewidth=1.4)
    for el in bp["whiskers"] + bp["caps"]:
        el.set(color=AZUL)
    ax.plot(range(len(NIVELES)), NIVELES, "--", color=NARANJA, linewidth=1.8,
            label="Predicción perfecta")
    ax.set_xticks(range(len(NIVELES)), [nombre_nivel(n) for n in NIVELES])
    ax.set_xlabel("Turbidez real (NTU)")
    ax.set_ylabel("Turbidez estimada (NTU)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Estimado vs. real (mejor modelo)", loc="left", color=TEXTO)
    guardar(fig, "07_real_vs_estimado.png")


def fig_confusion():
    cm = pd.read_csv(MODELOS / "matriz_confusion.csv", index_col=0)
    norm = cm.div(cm.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(norm.values, cmap=SECUENCIAL, vmin=0, vmax=1)
    etiquetas = [nombre_nivel(float(c)) for c in cm.columns]
    ax.set_xticks(range(len(etiquetas)), etiquetas)
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm.values[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.5 else TEXTO)
    ax.grid(False)
    ax.set_xlabel("Nivel estimado (NTU)")
    ax.set_ylabel("Nivel real (NTU)")
    fig.colorbar(im, ax=ax, label="Fracción de la fila")
    ax.set_title("Matriz de confusión (% por fila)", loc="left", color=TEXTO)
    guardar(fig, "08_matriz_confusion.png")


def fig_error_por_nivel():
    p = pd.read_csv(MODELOS / "predicciones_mejor_modelo.csv")
    p["err"] = (p.pred_ntu - p.nivel_ntu).abs()
    e = p.groupby("nivel_ntu").err.mean().reindex(NIVELES)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    b = ax.bar(range(len(NIVELES)), e.values, color=AZUL, width=0.6)
    ax.bar_label(b, fmt="%.2f", fontsize=9, color=TEXTO2, padding=2)
    ax.set_xticks(range(len(NIVELES)), [nombre_nivel(n) for n in NIVELES])
    ax.set_xlabel("Turbidez real (NTU)")
    ax.set_ylabel("Error absoluto medio (NTU)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Error por nivel de turbidez", loc="left", color=TEXTO)
    guardar(fig, "09_error_por_nivel.png")


def main():
    df = pd.read_csv(CARAC)
    fig_roi_ejemplos()
    fig_estados(df)
    fig_boxplot_ntu(df)
    fig_correlaciones(df)
    fig_pca(df)
    fig_modelos()
    fig_real_vs_pred()
    fig_confusion()
    fig_error_por_nivel()


if __name__ == "__main__":
    main()
