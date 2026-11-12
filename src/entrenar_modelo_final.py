"""Entrena el modelo final (un Gradient Boosting por estado de la escena).

Evaluación estricta con partición temporal:
  - Cada nivel de NTU se divide en el tramo inicial (80 %) y el final (20 %).
  - Se deja un margen (EMBARGO) de frames entre ambos para que ningún frame de
    prueba tenga vecinos casi idénticos en el entrenamiento.
  - Los hiperparámetros se eligen con validación cruzada por bloques usando
    SOLO el tramo de entrenamiento; el tramo final no se usa para decidir nada.

Salidas (src/modelo/ y src/resultados_modelos/):
  modelo_turbidez.joblib, metricas_holdout.json, predicciones_holdout.csv,
  demo_imagenes.csv

Uso:  python src/entrenar_modelo_final.py
"""

import itertools
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "src" / "resultados_caracteristicas" / "caracteristicas_v2.csv"
CARPETA_MODELO = RAIZ / "src" / "modelo"
CARPETA_RES = RAIZ / "src" / "resultados_modelos"

FRACCION_PRUEBA = 0.20
EMBARGO = 100          # frames descartados entre entrenamiento y prueba
TAM_BLOQUE = 300
NIVELES = np.array([0, 0.5, 1, 2.5, 4, 5, 7.5, 10, 40])
ESTADOS = ("ARRIBA", "ABAJO")

NO_FEATURES = {
    "nivel_ntu", "archivo", "frame", "ok", "motivo", "posicion",
    "area_pct", "contraste_clara_oscura", "roi_mad", "grupo",
}
PARAMS_BASE = dict(max_iter=300, learning_rate=0.05, random_state=0)
REJILLA = {
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
    "l2_regularization": [0.0, 1.0],
}


def nivel_mas_cercano(v):
    v = np.asarray(v, dtype=float)
    return NIVELES[np.abs(v[:, None] - NIVELES[None, :]).argmin(axis=1)]


def metricas(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "acc_nivel": float(np.mean(nivel_mas_cercano(p) == nivel_mas_cercano(y))),
    }


def particion_temporal(df):
    """Devuelve máscaras booleanas (entrenamiento, prueba) con embargo."""
    entrena = np.zeros(len(df), bool)
    prueba = np.zeros(len(df), bool)
    for _, g in df.groupby("nivel_ntu"):
        corte = g.frame.quantile(1 - FRACCION_PRUEBA)
        entrena[g.index[g.frame < corte - EMBARGO]] = True
        prueba[g.index[g.frame >= corte]] = True
    return entrena, prueba


def elegir_parametros(X, y, grupos):
    mejor, mejor_rmse = None, np.inf
    gkf = GroupKFold(n_splits=5)
    claves = list(REJILLA)
    for valores in itertools.product(*REJILLA.values()):
        p = {**PARAMS_BASE, **dict(zip(claves, valores))}
        pred = np.zeros(len(y))
        for tr, te in gkf.split(X, y, grupos):
            pred[te] = HistGradientBoostingRegressor(**p).fit(X[tr], y[tr]).predict(X[te])
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        if rmse < mejor_rmse:
            mejor, mejor_rmse = p, rmse
    return mejor, mejor_rmse


def main():
    CARPETA_MODELO.mkdir(parents=True, exist_ok=True)
    CARPETA_RES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV)
    df = df[df.ok].reset_index(drop=True)
    cols = [c for c in df.columns if c not in NO_FEATURES]
    df["grupo"] = df.nivel_ntu.astype(str) + "_" + (df.frame // TAM_BLOQUE).astype(str)

    es_ent, es_pru = particion_temporal(df)
    print(f"Entrenamiento: {es_ent.sum()}  |  Prueba (tramo final): {es_pru.sum()}  "
          f"|  descartadas por embargo: {(~es_ent & ~es_pru).sum()}")

    modelos, params_elegidos = {}, {}
    pred = pd.Series(np.nan, index=df.index)
    pred_base = pd.Series(np.nan, index=df.index)

    for estado in ESTADOS:
        m_ent = es_ent & (df.posicion == estado).values
        m_pru = es_pru & (df.posicion == estado).values
        X_ent, y_ent = df.loc[m_ent, cols].values, df.loc[m_ent, "nivel_ntu"].values
        p, rmse_cv = elegir_parametros(X_ent, y_ent, df.loc[m_ent, "grupo"].values)
        print(f"[{estado}] mejores parámetros: "
              f"lr={p['learning_rate']}, hojas={p['max_leaf_nodes']}, "
              f"l2={p['l2_regularization']}  (RMSE CV en entrenamiento = {rmse_cv:.3f})")
        modelo = HistGradientBoostingRegressor(**p).fit(X_ent, y_ent)
        base = HistGradientBoostingRegressor(**PARAMS_BASE).fit(X_ent, y_ent)
        modelos[estado], params_elegidos[estado] = modelo, p
        pred[m_pru] = modelo.predict(df.loc[m_pru, cols].values)
        pred_base[m_pru] = base.predict(df.loc[m_pru, cols].values)

    pru = df[es_pru].copy()
    pru["pred_ntu"] = pred[es_pru].values
    pru["pred_ntu_base"] = pred_base[es_pru].values
    y = pru.nivel_ntu.values

    resultado = {
        "n_entrenamiento": int(es_ent.sum()),
        "n_prueba": int(es_pru.sum()),
        "embargo_frames": EMBARGO,
        "fraccion_prueba": FRACCION_PRUEBA,
        "parametros": params_elegidos,
        "holdout_ajustado": metricas(y, pru.pred_ntu.values),
        "holdout_parametros_base": metricas(y, pru.pred_ntu_base.values),
        "por_estado": {
            e: metricas(pru[pru.posicion == e].nivel_ntu.values, pru[pru.posicion == e].pred_ntu.values)
            for e in ESTADOS
        },
        "por_nivel_MAE": {
            str(n): float((g.pred_ntu - g.nivel_ntu).abs().mean())
            for n, g in pru.groupby("nivel_ntu")
        },
        "por_nivel_acc": {
            str(n): float(np.mean(nivel_mas_cercano(g.pred_ntu.values) == nivel_mas_cercano(g.nivel_ntu.values)))
            for n, g in pru.groupby("nivel_ntu")
        },
    }
    print(json.dumps({k: resultado[k] for k in ("holdout_ajustado", "holdout_parametros_base", "por_estado")}, indent=1))
    print("MAE por nivel:", {k: round(v, 2) for k, v in resultado["por_nivel_MAE"].items()})
    print("Exactitud por nivel:", {k: round(v, 2) for k, v in resultado["por_nivel_acc"].items()})

    joblib.dump({"modelos": modelos, "features": cols, "parametros": params_elegidos},
                CARPETA_MODELO / "modelo_turbidez.joblib")
    json.dump(resultado, open(CARPETA_RES / "metricas_holdout.json", "w"), indent=2)
    pru[["nivel_ntu", "archivo", "frame", "posicion", "pred_ntu", "pred_ntu_base"]].to_csv(
        CARPETA_RES / "predicciones_holdout.csv", index=False)

    # Imágenes de demostración: del tramo de prueba, una por estado y nivel
    demo = []
    rng = np.random.default_rng(7)
    for (n, e), g in pru.groupby(["nivel_ntu", "posicion"]):
        f = g.iloc[rng.integers(len(g))]
        nombre = str(int(n)) if float(n) == int(n) else str(n)
        demo.append({"nivel_ntu": n, "estado": e, "archivo": f.archivo,
                     "ruta": f"WaTur-Water-Turbidity-Dataset-main/Formazine/{nombre}/{f.archivo}"})
    pd.DataFrame(demo).to_csv(CARPETA_RES / "demo_imagenes.csv", index=False)
    print("Modelo guardado en", CARPETA_MODELO / "modelo_turbidez.joblib")


if __name__ == "__main__":
    main()
