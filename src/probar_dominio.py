"""Comprueba si el sistema rechaza imágenes que NO pertenecen al montaje de entrenamiento.

Grupos: (1) tramo final de las grabaciones de entrenamiento (deberían aceptarse),
(2) subconjunto Text de WaTur (otro protocolo), (3) imagenes/agua.jpg del proyecto y
(4) imágenes sintéticas (gris uniforme, degradado, ruido, mitad oscura/mitad clara).

Uso:  python src/probar_dominio.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimar_turbidez import cargar_modelo, estimar  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
PRED = RAIZ / "src" / "resultados_modelos" / "predicciones_holdout_multifuente.csv"
N = 200


def ruta(fuente, nivel, archivo):
    n = str(int(nivel)) if float(nivel) == int(nivel) else str(nivel)
    base = RAIZ / "WaTur-Water-Turbidity-Dataset-main" / fuente
    return base / n / archivo if fuente == "Formazine" else base / n / n / archivo


def tasa(paquete, imagenes):
    r = [estimar(i, paquete) for i in imagenes]
    acept = sum(x["ok"] for x in r)
    motivos = pd.Series([x["motivo"].split(":")[0] for x in r if not x["ok"]]).value_counts().to_dict()
    return acept, len(r), motivos


def main():
    paquete = cargar_modelo()
    rng = np.random.default_rng(0)
    grupos = {}

    p = pd.read_csv(PRED).sample(N, random_state=0)
    grupos["Mismo montaje (tramo final)"] = [cv2.imread(str(ruta(f.fuente, f.nivel_ntu, f.archivo))) for f in p.itertuples()]

    texto = sorted((RAIZ / "WaTur-Water-Turbidity-Dataset-main" / "Text").glob("*/*/*.jpg"))
    texto = [texto[i] for i in rng.choice(len(texto), N, replace=False)]
    grupos["Subconjunto Text (otro protocolo)"] = [cv2.imread(str(t)) for t in texto]

    agua = RAIZ / "imagenes" / "agua.jpg"
    if agua.exists():
        grupos["imagenes/agua.jpg"] = [cv2.imread(str(agua))]

    sint = []
    sint.append(np.full((244, 244, 3), 128, np.uint8))
    g = np.tile(np.linspace(0, 255, 244, dtype=np.uint8)[:, None, None], (1, 244, 3))
    sint.append(g)
    sint += [rng.integers(0, 255, (244, 244, 3), dtype=np.uint8) for _ in range(5)]
    mitad = np.full((244, 244, 3), 40, np.uint8)
    mitad[122:] = 170
    sint.append(mitad)                                   # imita dos zonas, sin ser agua
    sint.append(cv2.GaussianBlur(mitad, (0, 0), 6))
    grupos["Sintéticas (gris, degradado, ruido, 2 zonas)"] = sint

    filas = []
    for nombre, imgs in grupos.items():
        a, n, mot = tasa(paquete, imgs)
        filas.append({"grupo": nombre, "aceptadas": a, "total": n, "% aceptadas": round(100 * a / n, 1), "motivos_rechazo": str(mot)})
        print(f"{nombre:48s} aceptadas {a:3d}/{n:3d} ({100 * a / n:5.1f} %)  rechazos: {mot}")
    pd.DataFrame(filas).to_csv(RAIZ / "src" / "resultados_modelos" / "prueba_dominio.csv", index=False)


if __name__ == "__main__":
    main()
