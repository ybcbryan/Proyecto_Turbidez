"""Compara tres conjuntos de variables en pruebas con grabaciones independientes.

  absolutas : las 38 características de la zona clara (versión 2)
  relativas : medidas de contraste y nitidez del borde disco/fondo (invariantes a la iluminación)
  ambas     : unión de los dos conjuntos

Para cada conjunto: (A) tramo final de Formazine (misma grabación) y
(B) leave-one-source-out entre Formazine, Kaolin y Clear Plastic.

Uso:  python src/evaluar_generalizacion_relativas.py
"""

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caracteristicas_relativas import caracteristicas_relativas  # noqa: E402
from entrenar_modelo_final import NO_FEATURES, particion_temporal  # noqa: E402
from evaluar_generalizacion import (CARAC_FORM, CARAC_OTRAS, DATOS, OTRAS, UMBRALES,  # noqa: E402
                                    entrenar_predecir, metricas)

RAIZ = Path(__file__).resolve().parents[1]
CACHE = RAIZ / "src" / "resultados_caracteristicas" / "caracteristicas_relativas.csv"
SALIDA = RAIZ / "src" / "resultados_modelos" / "generalizacion_relativas.csv"


def _ruta(fuente, nivel, archivo):
    n = str(int(nivel)) if float(nivel) == int(nivel) else str(nivel)
    return DATOS / fuente / n / archivo if fuente == "Formazine" else DATOS / fuente / n / n / archivo


def _calc(args):
    fuente, nivel, archivo = args
    img = cv2.imread(str(_ruta(fuente, nivel, archivo)))
    r = caracteristicas_relativas(img) if img is not None else None
    return {"fuente": fuente, "nivel_ntu": nivel, "archivo": archivo, **(r or {})}


def cargar_relativas(todo):
    if CACHE.exists():
        return pd.read_csv(CACHE)
    trabajos = list(todo[["fuente", "nivel_ntu", "archivo"]].itertuples(index=False, name=None))
    print(f"Calculando características relativas de {len(trabajos)} imágenes...")
    with Pool(os.cpu_count() or 2) as p:
        df = pd.DataFrame(p.map(_calc, trabajos, chunksize=100))
    df.to_csv(CACHE, index=False)
    return df


def main():
    form = pd.read_csv(CARAC_FORM)
    form["fuente"] = "Formazine"
    todo = pd.concat([form, pd.read_csv(CARAC_OTRAS)], ignore_index=True)
    todo = todo[todo.ok & todo.posicion.isin(["ARRIBA", "ABAJO"])].reset_index(drop=True)

    rel = cargar_relativas(todo)
    cols_rel = [c for c in rel.columns if c not in ("fuente", "nivel_ntu", "archivo")]
    todo = todo.merge(rel, on=["fuente", "nivel_ntu", "archivo"], how="left")
    antes = len(todo)
    todo = todo.dropna(subset=cols_rel).reset_index(drop=True)
    print(f"Imágenes con características relativas: {len(todo)} de {antes}")

    cols_abs = [c for c in todo.columns if c not in NO_FEATURES and c not in cols_rel and c != "fuente"]
    conjuntos = {"absolutas": cols_abs, "relativas": cols_rel, "ambas": cols_abs + cols_rel}
    fuentes = ["Formazine"] + OTRAS
    filas = []

    for nombre, cols in conjuntos.items():
        print(f"\n=== Variables {nombre} ({len(cols)}) ===")
        # A) misma grabación: tramo final de Formazine
        f = todo[todo.fuente == "Formazine"].reset_index(drop=True)
        es_ent, es_pru = particion_temporal(f)
        t, p = entrenar_predecir(f[es_ent], f[es_pru], cols)
        m = metricas(t.nivel_ntu.values, p)
        m.update(variables=nombre, prueba="Formazine (tramo final, misma grabación)", entrenamiento="Formazine (80 % inicial)")
        filas.append(m)
        print(f"{'misma grabación':26s} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R2={m['R2']:.3f} acc>=5:{m['acc_turbia>=5']:.2f}")
        # B) otra grabación independiente
        for g in fuentes:
            train = todo[todo.fuente != g]
            t, p = entrenar_predecir(train, todo[todo.fuente == g], cols)
            m = metricas(t.nivel_ntu.values, p)
            m.update(variables=nombre, prueba=g, entrenamiento="+".join(x for x in fuentes if x != g))
            filas.append(m)
            print(f"{'prueba en ' + g:26s} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R2={m['R2']:.3f} "
                  f"acc>=5:{m['acc_turbia>=5']:.2f} acc>=4:{m['acc_turbia>=4']:.2f}")

    pd.DataFrame(filas)[["variables", "entrenamiento", "prueba", "n", "MAE", "RMSE", "R2",
                         *[f"acc_turbia>={u}" for u in UMBRALES]]].to_csv(SALIDA, index=False)
    print("Guardado:", SALIDA)


if __name__ == "__main__":
    main()
