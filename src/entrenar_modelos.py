"""Entrenamiento y evaluación de modelos para estimar la turbidez (NTU).

Validación: bloques temporales contiguos por nivel (GroupKFold). Frames
consecutivos son casi idénticos, por lo que un split aleatorio inflaría los
resultados. Cada bloque (frame // TAM_BLOQUE dentro de cada nivel) es un grupo.

Se comparan tres configuraciones de variables:
  - todas:           todas las características de la ROI
  - sin_posicion:    sin indicar si la zona clara está arriba/abajo
  - por_posicion:    un modelo independiente para cada estado (ARRIBA / ABAJO)

Uso:  python src/entrenar_modelos.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "src" / "resultados_caracteristicas" / "caracteristicas_v2.csv"
SALIDA = RAIZ / "src" / "resultados_modelos"
TAM_BLOQUE = 300
N_PLIEGUES = 5
NIVELES = [0, 0.5, 1, 2.5, 4, 5, 7.5, 10, 40]

NO_FEATURES = {
    "nivel_ntu", "archivo", "frame", "ok", "motivo", "posicion",
    "area_pct", "contraste_clara_oscura", "roi_mad", "grupo",
}


def modelos():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "SVR": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1)),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0
        ),
        "GradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=0
        ),
    }


def metricas(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
        "acc_nivel": float(
            np.mean(_nivel_mas_cercano(p) == _nivel_mas_cercano(y))
        ),
    }


def _nivel_mas_cercano(valores):
    n = np.asarray(NIVELES)
    return n[np.abs(np.asarray(valores)[:, None] - n[None, :]).argmin(axis=1)]


def validar(df, cols, nombre_cfg, resultados, predicciones):
    X, y, g = df[cols].values, df.nivel_ntu.values, df.grupo.values
    gkf = GroupKFold(n_splits=N_PLIEGUES)
    for nombre, _ in modelos().items():
        pred = np.zeros(len(df))
        for tr, te in gkf.split(X, y, g):
            m = modelos()[nombre].fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        r = metricas(y, pred)
        r.update(modelo=nombre, config=nombre_cfg, n=len(df))
        resultados.append(r)
        predicciones[(nombre_cfg, nombre)] = pred
        print(f"{nombre_cfg:14s} {nombre:17s} MAE={r['MAE']:.3f} "
              f"RMSE={r['RMSE']:.3f} R2={r['R2']:.3f} acc={r['acc_nivel']:.3f}")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    df = df[df.ok].reset_index(drop=True)
    df["grupo"] = df.nivel_ntu.astype(str) + "_" + (df.frame // TAM_BLOQUE).astype(str)
    cols = [c for c in df.columns if c not in NO_FEATURES]
    df["pos_arriba"] = (df.posicion == "ARRIBA").astype(int)

    resultados, predicciones = [], {}
    print(f"Imágenes: {len(df)}  | variables: {len(cols)}  | grupos: {df.grupo.nunique()}")

    validar(df, cols, "sin_posicion", resultados, predicciones)
    validar(df, cols + ["pos_arriba"], "todas", resultados, predicciones)

    # Un modelo por estado de la escena
    for nombre in modelos():
        pred = np.zeros(len(df))
        for pos in ("ARRIBA", "ABAJO"):
            idx = np.flatnonzero(df.posicion.values == pos)
            sub = df.iloc[idx]
            gkf = GroupKFold(n_splits=N_PLIEGUES)
            for tr, te in gkf.split(sub[cols], sub.nivel_ntu, sub.grupo):
                m = modelos()[nombre].fit(sub.iloc[tr][cols].values, sub.nivel_ntu.values[tr])
                pred[idx[te]] = m.predict(sub.iloc[te][cols].values)
        r = metricas(df.nivel_ntu.values, pred)
        r.update(modelo=nombre, config="por_posicion", n=len(df))
        resultados.append(r)
        predicciones[("por_posicion", nombre)] = pred
        print(f"{'por_posicion':14s} {nombre:17s} MAE={r['MAE']:.3f} "
              f"RMSE={r['RMSE']:.3f} R2={r['R2']:.3f} acc={r['acc_nivel']:.3f}")

    res = pd.DataFrame(resultados)[["config", "modelo", "MAE", "RMSE", "R2", "acc_nivel", "n"]]
    res.to_csv(SALIDA / "metricas_modelos.csv", index=False)

    mejor = res.sort_values("RMSE").iloc[0]
    print("\nMejor:", mejor.config, mejor.modelo)
    pred = predicciones[(mejor.config, mejor.modelo)]
    out = df[["nivel_ntu", "archivo", "frame", "posicion", "grupo"]].copy()
    out["pred_ntu"] = pred
    out.to_csv(SALIDA / "predicciones_mejor_modelo.csv", index=False)

    # Error por nivel de NTU y matriz de confusión
    out["err_abs"] = (out.pred_ntu - out.nivel_ntu).abs()
    print(out.groupby("nivel_ntu").err_abs.mean().round(3).to_string())
    cm = pd.crosstab(
        pd.Series(_nivel_mas_cercano(out.nivel_ntu.values), name="real"),
        pd.Series(_nivel_mas_cercano(out.pred_ntu.values), name="pred"),
    )
    cm.to_csv(SALIDA / "matriz_confusion.csv")
    json.dump({"config": mejor.config, "modelo": mejor.modelo},
              open(SALIDA / "mejor_modelo.json", "w"))


if __name__ == "__main__":
    main()
