"""Entrena el modelo multifuente y el detector de imágenes fuera de dominio.

Usa las tres fuentes con el mismo montaje (Formazine, Kaolin, Clear Plastic) y las
dos familias de variables: absolutas (zona clara) y relativas (contraste y nitidez
del borde disco/fondo). Un Gradient Boosting por estado de la escena.

  - Partición temporal por (fuente, nivel): 80 % inicial entrena, 20 % final prueba,
    con un margen de frames entre ambos. Esta prueba es de la MISMA grabación; la
    medida de generalización real está en evaluar_generalizacion*.py.
  - Detector de dominio: Isolation Forest por estado sobre las variables; el umbral
    es el percentil 1 de las puntuaciones del entrenamiento.

Uso:  python src/entrenar_modelo_multifuente.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_modelo_final import NO_FEATURES, PARAMS_BASE, metricas  # noqa: E402
from evaluar_generalizacion import CARAC_FORM, CARAC_OTRAS  # noqa: E402
from evaluar_generalizacion_relativas import CACHE as CACHE_REL, _ruta  # noqa: E402

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "src" / "modelo" / "modelo_turbidez_multifuente.joblib"
RES = RAIZ / "src" / "resultados_modelos"
ESTADOS = ("ARRIBA", "ABAJO")
FRACCION_PRUEBA = 0.20
PERCENTIL_DOMINIO = 1.0


def particion(df):
    ent = np.zeros(len(df), bool)
    pru = np.zeros(len(df), bool)
    for _, g in df.groupby(["fuente", "nivel_ntu"]):
        corte = g.frame.quantile(1 - FRACCION_PRUEBA)
        margen = 0.10 * (g.frame.max() - g.frame.min())
        ent[g.index[g.frame < corte - margen]] = True
        pru[g.index[g.frame >= corte]] = True
    return ent, pru


def main():
    form = pd.read_csv(CARAC_FORM)
    form["fuente"] = "Formazine"
    todo = pd.concat([form, pd.read_csv(CARAC_OTRAS)], ignore_index=True)
    todo = todo[todo.ok & todo.posicion.isin(ESTADOS)]
    rel = pd.read_csv(CACHE_REL)
    cols_rel = [c for c in rel.columns if c not in ("fuente", "nivel_ntu", "archivo")]
    todo = todo.merge(rel, on=["fuente", "nivel_ntu", "archivo"], how="left").dropna(subset=cols_rel)
    todo = todo.reset_index(drop=True)
    cols_abs = [c for c in todo.columns if c not in NO_FEATURES and c not in cols_rel and c != "fuente"]
    cols = cols_abs + cols_rel

    ent, pru = particion(todo)
    print(f"Entrenamiento: {ent.sum()} | prueba (tramo final): {pru.sum()} | variables: {len(cols)}")

    modelos, dominio = {}, {}
    pred = pd.Series(np.nan, index=todo.index)
    for estado in ESTADOS:
        m_ent = ent & (todo.posicion == estado).values
        m_pru = pru & (todo.posicion == estado).values
        X, y = todo.loc[m_ent, cols].values, todo.loc[m_ent, "nivel_ntu"].values
        modelos[estado] = HistGradientBoostingRegressor(**PARAMS_BASE).fit(X, y)
        pred[m_pru] = modelos[estado].predict(todo.loc[m_pru, cols].values)
        iso = IsolationForest(n_estimators=300, random_state=0).fit(X)
        umbral = float(np.percentile(iso.score_samples(X), PERCENTIL_DOMINIO))
        dominio[estado] = {"modelo": iso, "umbral": umbral}

    t = todo[pru].copy()
    t["pred_ntu"] = pred[pru].values
    res = {"n_entrenamiento": int(ent.sum()), "n_prueba": int(pru.sum()), "global": metricas(t.nivel_ntu.values, t.pred_ntu.values)}
    res["por_fuente"] = {f: metricas(g.nivel_ntu.values, g.pred_ntu.values) for f, g in t.groupby("fuente")}
    for u in (1, 4):
        res[f"acc_turbia>={u}"] = float(np.mean((t.nivel_ntu >= u) == (t.pred_ntu >= u)))
    print(json.dumps({k: res[k] for k in ("global", "acc_turbia>=1", "acc_turbia>=4")}, indent=1))

    joblib.dump({"modelos": modelos, "dominio": dominio, "features": cols, "features_abs": cols_abs,
                 "features_rel": cols_rel}, MODELO)
    json.dump(res, open(RES / "metricas_multifuente.json", "w"), indent=2)
    t[["fuente", "nivel_ntu", "archivo", "frame", "posicion", "pred_ntu"]].to_csv(
        RES / "predicciones_holdout_multifuente.csv", index=False)

    # Imágenes de demostración (tramo final, no usadas para entrenar): una por fuente, estado y nivel
    rng = np.random.default_rng(7)
    demo = []
    for (fuente, n, e), g in t.groupby(["fuente", "nivel_ntu", "posicion"]):
        f = g.iloc[rng.integers(len(g))]
        demo.append({"fuente": fuente, "nivel_ntu": n, "estado": e, "archivo": f.archivo,
                     "ruta": str(_ruta(fuente, n, f.archivo).relative_to(RAIZ)).replace("\\", "/")})
    pd.DataFrame(demo).to_csv(RES / "demo_imagenes_multifuente.csv", index=False)
    print("Modelo guardado:", MODELO, "| imágenes de demostración:", len(demo))


if __name__ == "__main__":
    main()
