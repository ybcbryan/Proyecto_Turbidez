"""Evalúa si el modelo generaliza a grabaciones INDEPENDIENTES.

Fuentes (subconjuntos de WaTur con el mismo montaje: disco oscuro + zona clara):
  Formazine (patrón de formazina), Kaolin (arcilla de caolín), Clear Plastic
  (otro recipiente). El subconjunto Text usa otro protocolo y no se incluye.

Experimentos (entrenar con unas fuentes, probar con otra distinta):
  A) Solo Formazine -> Kaolin y Clear Plastic.
  B) Leave-one-source-out: entrenar con dos fuentes, probar con la tercera.

Las características de Kaolin y Clear Plastic se calculan una vez y se guardan en
src/resultados_caracteristicas/caracteristicas_otras_fuentes.csv.

Uso:  python src/evaluar_generalizacion.py
"""

import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_modelo_final import NO_FEATURES, PARAMS_BASE  # noqa: E402
from extraer_caracteristicas_v2 import caracteristicas_roi  # noqa: E402
from roi_zona_clara import detectar_roi  # noqa: E402

import cv2  # noqa: E402

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "WaTur-Water-Turbidity-Dataset-main"
CARAC_FORM = RAIZ / "src" / "resultados_caracteristicas" / "caracteristicas_v2.csv"
CARAC_OTRAS = RAIZ / "src" / "resultados_caracteristicas" / "caracteristicas_otras_fuentes.csv"
SALIDA = RAIZ / "src" / "resultados_modelos" / "generalizacion.csv"
OTRAS = ["Kaolin", "Clear Plastic"]
UMBRALES = (1, 2.5, 4, 5)


def _procesar(args):
    fuente, nivel, ruta = args
    img = cv2.imread(str(ruta))
    base = {"fuente": fuente, "nivel_ntu": float(nivel), "archivo": ruta.name, "frame": int(ruta.stem)}
    if img is None:
        return {**base, "ok": False, "motivo": "NO_SE_PUDO_LEER"}
    if img.shape[:2] != (244, 244):
        img = cv2.resize(img, (244, 244), interpolation=cv2.INTER_AREA)
    r = detectar_roi(img)
    base.update(ok=bool(r.ok), motivo=r.motivo, posicion=r.posicion, area_pct=round(r.area_pct, 3),
                contraste_clara_oscura=round(r.contraste, 3), roi_mad=round(r.mad, 3))
    if r.ok:
        base.update(caracteristicas_roi(img, r.mascara))
    return base


def extraer_otras():
    trabajos = []
    for fuente in OTRAS:
        for carpeta in sorted((DATOS / fuente).iterdir()):
            sub = carpeta / carpeta.name
            if not sub.is_dir():
                continue
            for f in sub.glob("*.jpg"):
                if f.stem.isdigit():
                    trabajos.append((fuente, carpeta.name, f))
    print(f"Imágenes de otras fuentes: {len(trabajos)}")
    with Pool(os.cpu_count() or 2) as p:
        filas = p.map(_procesar, trabajos, chunksize=50)
    df = pd.DataFrame(filas)
    df.to_csv(CARAC_OTRAS, index=False)
    return df


def metricas(y, p):
    m = {"n": int(len(y)), "MAE": float(mean_absolute_error(y, p)),
         "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": float(r2_score(y, p))}
    for u in UMBRALES:
        m[f"acc_turbia>={u}"] = float(np.mean((np.asarray(y) >= u) == (np.asarray(p) >= u)))
    return m


def entrenar_predecir(train, test, cols):
    """Un modelo por estado de la escena; devuelve (test_valido, predicciones)."""
    salida_idx, salida_pred = [], []
    for estado in ("ARRIBA", "ABAJO"):
        tr = train[train.posicion == estado]
        te = test[test.posicion == estado]
        if len(tr) == 0 or len(te) == 0:
            continue
        m = HistGradientBoostingRegressor(**PARAMS_BASE).fit(tr[cols].values, tr.nivel_ntu.values)
        salida_idx.append(te.index)
        salida_pred.append(m.predict(te[cols].values))
    idx = np.concatenate([i.values for i in salida_idx])
    return test.loc[idx], np.concatenate(salida_pred)


def main():
    otras = pd.read_csv(CARAC_OTRAS) if CARAC_OTRAS.exists() else extraer_otras()
    form = pd.read_csv(CARAC_FORM)
    form["fuente"] = "Formazine"
    todo = pd.concat([form, otras], ignore_index=True)

    print("\nROI válidas por fuente:")
    print(todo.groupby("fuente").ok.agg(["sum", "size", "mean"]).round(3).to_string())
    todo = todo[todo.ok & todo.posicion.isin(["ARRIBA", "ABAJO"])].reset_index(drop=True)
    cols = [c for c in todo.columns if c not in NO_FEATURES and c != "fuente"]
    fuentes = ["Formazine"] + OTRAS

    filas = []

    def registrar(nombre, train_f, test_f):
        train = todo[todo.fuente.isin(train_f)]
        test = todo[todo.fuente == test_f]
        t, p = entrenar_predecir(train, test, cols)
        m = metricas(t.nivel_ntu.values, p)
        m.update(experimento=nombre, entrenamiento="+".join(train_f), prueba=test_f)
        filas.append(m)
        print(f"{nombre:28s} {'+'.join(train_f):28s} -> {test_f:14s} n={m['n']:5d} "
              f"MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R2={m['R2']:.3f} "
              f"acc>=5:{m['acc_turbia>=5']:.2f} acc>=4:{m['acc_turbia>=4']:.2f}")

    print("\nA) Una sola fuente de entrenamiento")
    for f in fuentes:
        for g in fuentes:
            if f != g:
                registrar("A) fuente única", [f], g)
    print("\nB) Dos fuentes de entrenamiento -> tercera fuente")
    for g in fuentes:
        registrar("B) leave-one-source-out", [f for f in fuentes if f != g], g)

    pd.DataFrame(filas)[["experimento", "entrenamiento", "prueba", "n", "MAE", "RMSE", "R2",
                         *[f"acc_turbia>={u}" for u in UMBRALES]]].to_csv(SALIDA, index=False)
    print("\nGuardado:", SALIDA)


if __name__ == "__main__":
    main()
