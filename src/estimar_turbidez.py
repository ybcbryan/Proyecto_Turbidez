"""Demostración: estima la turbidez (NTU) de una o varias imágenes.

Pasos: detecta la zona clara (ROI) -> extrae las 38 características ->
aplica el modelo del estado correspondiente (zona clara arriba/abajo).
Si la ROI no es fiable, la imagen se rechaza y no se da estimación.

Ejemplos:
  python src/estimar_turbidez.py ruta/imagen.jpg
  python src/estimar_turbidez.py img1.jpg img2.jpg --salida resultados_demo
  python src/estimar_turbidez.py --demo          # usa imágenes de prueba (no vistas en el entrenamiento)

Con --salida se guarda, por cada imagen, una figura con la ROI y la estimación.
"""

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extraer_caracteristicas_v2 import caracteristicas_roi  # noqa: E402
from roi_zona_clara import detectar_roi  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "src" / "modelo" / "modelo_turbidez.joblib"
DEMO_CSV = RAIZ / "src" / "resultados_modelos" / "demo_imagenes.csv"
NIVELES = np.array([0, 0.5, 1, 2.5, 4, 5, 7.5, 10, 40])


def cargar_modelo(ruta=MODELO):
    if not Path(ruta).exists():
        sys.exit(f"No se encontró el modelo en {ruta}. Ejecuta antes: python src/entrenar_modelo_final.py")
    return joblib.load(ruta)


def estimar(imagen_bgr, paquete):
    """Devuelve un diccionario con la estimación o el motivo de rechazo."""
    roi = detectar_roi(imagen_bgr)
    if not roi.ok:
        return {"ok": False, "motivo": roi.motivo, "mascara": roi.mascara, "estado": roi.posicion}
    if roi.posicion not in paquete["modelos"]:
        return {"ok": False, "motivo": f"ESTADO_NO_SOPORTADO_{roi.posicion}", "mascara": roi.mascara,
                "estado": roi.posicion}
    carac = caracteristicas_roi(imagen_bgr, roi.mascara)
    x = np.array([[carac[c] for c in paquete["features"]]])
    ntu = float(paquete["modelos"][roi.posicion].predict(x)[0])
    ntu = max(ntu, 0.0)
    nivel = float(NIVELES[np.abs(NIVELES - ntu).argmin()])
    return {"ok": True, "ntu": ntu, "nivel_cercano": nivel, "estado": roi.posicion,
            "mascara": roi.mascara, "area_pct": roi.area_pct}


def guardar_figura(imagen_bgr, res, ruta, titulo, real=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ov = imagen_bgr.copy()
    m = res["mascara"] > 0
    ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.5))
    for a, im, t in zip(ax, (imagen_bgr, ov), ("Imagen", "ROI (verde)")):
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        a.set_title(t, fontsize=10)
        a.axis("off")
    if res["ok"]:
        txt = f"Turbidez estimada: {res['ntu']:.1f} NTU"
        if real is not None:
            txt += f"  (real: {real} NTU)"
    else:
        txt = f"Imagen rechazada: {res['motivo']}"
    fig.suptitle(f"{titulo}\n{txt}", fontsize=10)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagenes", nargs="*", help="rutas de imágenes")
    ap.add_argument("--demo", action="store_true", help="usar las imágenes de prueba del tramo final")
    ap.add_argument("--salida", help="carpeta donde guardar las figuras")
    args = ap.parse_args()

    paquete = cargar_modelo()
    trabajos = []  # (ruta, real)
    if args.demo:
        for _, f in pd.read_csv(DEMO_CSV).iterrows():
            trabajos.append((RAIZ / f.ruta, f.nivel_ntu))
    trabajos += [(Path(r), None) for r in args.imagenes]
    if not trabajos:
        ap.error("indica al menos una imagen o usa --demo")

    filas = []
    for ruta, real in trabajos:
        img = cv2.imread(str(ruta))
        if img is None:
            print(f"No se pudo leer: {ruta}")
            continue
        res = estimar(img, paquete)
        if res["ok"]:
            extra = f" (real: {real} NTU, error {res['ntu'] - real:+.2f})" if real is not None else ""
            print(f"{ruta.name:>10s}  estado={res['estado']:6s}  turbidez estimada = {res['ntu']:6.2f} NTU"
                  f"  (nivel más cercano: {res['nivel_cercano']:g}){extra}")
            filas.append({"archivo": ruta.name, "real": real, "estimado": res["ntu"]})
        else:
            print(f"{ruta.name:>10s}  RECHAZADA ({res['motivo']})")
        if args.salida:
            guardar_figura(img, res, Path(args.salida) / f"{ruta.parent.name}_{ruta.stem}.png",
                           f"{ruta.parent.name} / {ruta.name}", real)

    con_real = [f for f in filas if f["real"] is not None]
    if con_real:
        err = np.array([f["estimado"] - f["real"] for f in con_real])
        print(f"\nImágenes evaluadas: {len(con_real)}  |  MAE = {np.abs(err).mean():.2f} NTU  |  "
              f"RMSE = {np.sqrt((err ** 2).mean()):.2f} NTU")


if __name__ == "__main__":
    main()
