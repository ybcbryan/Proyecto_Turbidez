"""Experimentos de refuerzo para el informe final.

1. Ablación de grupos de características (validación por bloques, un modelo por
   estado de la escena, Gradient Boosting).
2. Importancia de variables por permutación sobre el tramo final (holdout) con
   el modelo final guardado.
3. Figuras 10-12 en src/resultados_figuras y tablas en src/resultados_modelos.

Requiere haber ejecutado antes entrenar_modelo_final.py.
Uso:  python src/experimentos_refuerzo.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_modelo_final import (NO_FEATURES, PARAMS_BASE, TAM_BLOQUE,  # noqa: E402,E501
                                   particion_temporal)

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "src" / "resultados_caracteristicas" / "caracteristicas_v2.csv"
RES = RAIZ / "src" / "resultados_modelos"
FIG = RAIZ / "src" / "resultados_figuras"
MODELO = RAIZ / "src" / "modelo" / "modelo_turbidez.joblib"

AZUL, NARANJA, TEXTO, TEXTO2, REJILLA = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": TEXTO,
    "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.edgecolor": REJILLA, "axes.grid": True, "grid.color": REJILLA,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def es_color(c):
    """Color e intensidad: medias, cuantiles, cromaticidad y razones (el gradiente es textura)."""
    return (c.endswith("_media") and c != "gradiente_media") or c.startswith(("V_P", "L_P", "chroma", "razon"))


def grupos_de_variables(cols):
    color = [c for c in cols if es_color(c)]
    variab = [c for c in cols if c not in color]  # desviaciones, GLCM, Laplaciano, gradiente
    return {"Color e intensidad (media, cuantiles, cromaticidad)": color,
            "Variabilidad y textura (desviaciones, GLCM, gradiente)": variab,
            "Todas las variables": cols}


def cv_por_estado(df, cols):
    """RMSE/MAE/R2 con validación por bloques, un modelo por estado."""
    pred = pd.Series(np.nan, index=df.index)
    for estado in ("ARRIBA", "ABAJO"):
        sub = df[df.posicion == estado]
        X, y = sub[cols].values, sub.nivel_ntu.values
        p = np.zeros(len(sub))
        for tr, te in GroupKFold(5).split(X, y, sub.grupo):
            p[te] = HistGradientBoostingRegressor(**PARAMS_BASE).fit(X[tr], y[tr]).predict(X[te])
        pred[sub.index] = p
    y = df.nivel_ntu.values
    return {"MAE": mean_absolute_error(y, pred), "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": r2_score(y, pred)}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    df = df[df.ok].reset_index(drop=True)
    cols = [c for c in df.columns if c not in NO_FEATURES]
    df["grupo"] = df.nivel_ntu.astype(str) + "_" + (df.frame // TAM_BLOQUE).astype(str)

    # ---------- 1. Ablación ----------
    filas = []
    for nombre, c in grupos_de_variables(cols).items():
        m = cv_por_estado(df, c)
        m.update(grupo=nombre, n_variables=len(c))
        filas.append(m)
        print(f"{nombre:60s} n={len(c):2d}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}")
    abl = pd.DataFrame(filas)[["grupo", "n_variables", "MAE", "RMSE", "R2"]]
    abl.to_csv(RES / "ablacion.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    etiquetas = ["Color e\nintensidad", "Variabilidad\ny textura", "Todas"]
    colores = ["#8ab8ec", "#8ab8ec", "#123f77"]
    b = ax.bar(etiquetas, abl.RMSE, color=colores, width=0.55)
    ax.bar_label(b, labels=[f"{v:.2f}" for v in abl.RMSE], color=TEXTO2, padding=2)
    for rect, n in zip(b, abl.n_variables):
        ax.text(rect.get_x() + rect.get_width() / 2, 0.15, f"{n} variables", ha="center", color="white", fontsize=10)
    ax.set_ylabel("RMSE (NTU), menor es mejor")
    ax.set_ylim(0, abl.RMSE.max() * 1.25)
    ax.grid(axis="x", visible=False)
    ax.set_title("Aporte de cada grupo de características", loc="left")
    fig.savefig(FIG / "10_ablacion.png", facecolor="white")
    plt.close(fig)

    # ---------- 2. Importancia por permutación (holdout) ----------
    paquete = joblib.load(MODELO)
    _, es_pru = particion_temporal(df)
    imp_total = pd.Series(0.0, index=cols)
    for estado, modelo in paquete["modelos"].items():
        sub = df[es_pru & (df.posicion == estado).values]
        r = permutation_importance(modelo, sub[cols].values, sub.nivel_ntu.values, n_repeats=5,
                                   random_state=0, scoring="neg_root_mean_squared_error")
        imp_total += pd.Series(r.importances_mean, index=cols) * len(sub)
    imp = (imp_total / es_pru.sum()).sort_values(ascending=False)
    imp.rename("aumento_RMSE").to_csv(RES / "importancia_variables.csv")
    print("Top 10 importancia (aumento de RMSE al permutar):")
    print(imp.head(10).round(3).to_string())

    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    es_textura = [not es_color(c) for c in top.index]
    ax.barh(top.index, top.values, color=[NARANJA if t else AZUL for t in es_textura], height=0.62)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AZUL, label="Color e intensidad"), Patch(color=NARANJA, label="Variabilidad y textura")],
              frameon=False, loc="lower right")
    ax.set_xlabel("Aumento del RMSE (NTU) al permutar la variable")
    ax.grid(axis="y", visible=False)
    ax.set_title("Variables más importantes (tramo final)", loc="left")
    fig.savefig(FIG / "11_importancia.png", facecolor="white")
    plt.close(fig)

    # ---------- 3. Holdout: real vs estimado ----------
    p = pd.read_csv(RES / "predicciones_holdout.csv")
    niveles = [0, 0.5, 1, 2.5, 4, 5, 7.5, 10, 40]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bp = ax.boxplot([p[p.nivel_ntu == n].pred_ntu for n in niveles], positions=range(len(niveles)),
                    widths=0.6, patch_artist=True, showfliers=False)
    for c in bp["boxes"]:
        c.set(facecolor="#8ab8ec", edgecolor=AZUL)
    for el in bp["medians"]:
        el.set(color=TEXTO, linewidth=1.4)
    for el in bp["whiskers"] + bp["caps"]:
        el.set(color=AZUL)
    ax.plot(range(len(niveles)), niveles, "--", color=NARANJA, linewidth=1.8, label="Predicción perfecta")
    ax.set_xticks(range(len(niveles)), [str(int(n)) if n == int(n) else str(n) for n in niveles])
    ax.set_xlabel("Turbidez real (NTU)")
    ax.set_ylabel("Turbidez estimada (NTU)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Tramo final de cada grabación (no usado para entrenar)", loc="left", fontsize=11)
    fig.savefig(FIG / "12_holdout_real_vs_estimado.png", facecolor="white")
    plt.close(fig)
    print("Figuras 10, 11 y 12 guardadas en", FIG)


if __name__ == "__main__":
    main()
