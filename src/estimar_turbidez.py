"""Estima la turbidez (NTU) del agua y la clasifica como clara o turbia.

Pasos: reescala a 244x244 -> detecta la zona clara (ROI) -> calcula variables absolutas
y relativas (contraste y borde del disco) -> modelo del estado de la escena (zona clara
arriba/abajo) -> NTU y categoría. Antes de estimar, comprueba que la imagen se parezca al
montaje de entrenamiento (disco oscuro + fondo claro); si no, la rechaza en lugar de
inventar un número.

Ejemplos (desde cualquier carpeta):
  python src/estimar_turbidez.py                     # sin argumentos: corre la demostración
  python src/estimar_turbidez.py foto.jpg            # una imagen
  python src/estimar_turbidez.py a.jpg b.png --salida resultados_demo
  python src/estimar_turbidez.py carpeta_con_imagenes
  python src/estimar_turbidez.py foto.jpg --forzar   # da el valor aunque la imagen sea dudosa

Categorías (referencia OMS: ideal < 1 NTU para desinfección; el agua se ve turbia desde ~4 NTU):
  CLARA < 1 NTU  |  LIGERAMENTE TURBIA 1-4 NTU  |  TURBIA >= 4 NTU

Limitación: el modelo se entrenó con el montaje de WaTur; en grabaciones independientes
la precisión baja de forma importante (ver el informe). Con --salida se guarda una figura
por imagen con la ROI y la estimación.
"""

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caracteristicas_relativas import caracteristicas_relativas  # noqa: E402
from extraer_caracteristicas_v2 import caracteristicas_roi  # noqa: E402
from roi_zona_clara import detectar_roi  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "src" / "modelo" / "modelo_turbidez_multifuente.joblib"
DEMO_CSV = RAIZ / "src" / "resultados_modelos" / "demo_imagenes_multifuente.csv"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

UMBRAL_CLARA = 1.0    # NTU
UMBRAL_TURBIA = 4.0   # NTU


def categoria(ntu):
    if ntu < UMBRAL_CLARA:
        return "CLARA"
    if ntu < UMBRAL_TURBIA:
        return "LIGERAMENTE TURBIA"
    return "TURBIA"


def cargar_modelo(ruta=MODELO):
    if not Path(ruta).exists():
        sys.exit(f"No se encontró el modelo en {ruta}.\n"
                 "Ejecuta antes: python src/entrenar_modelo_multifuente.py")
    return joblib.load(ruta)


def estimar(imagen_bgr, paquete, forzar=False):
    """Devuelve un diccionario: ok, motivo, ntu, categoria, estado, mascara, dominio_ok."""
    original = imagen_bgr.shape[:2]
    if original != (244, 244):
        imagen_bgr = cv2.resize(imagen_bgr, (244, 244), interpolation=cv2.INTER_AREA)
    roi = detectar_roi(imagen_bgr)
    res = {"ok": False, "motivo": roi.motivo, "mascara": roi.mascara, "estado": roi.posicion,
           "imagen": imagen_bgr, "original": original}
    if not roi.ok:
        return res
    if roi.posicion not in paquete["modelos"]:
        res["motivo"] = f"ESTADO_NO_SOPORTADO_{roi.posicion}"
        return res
    rel = caracteristicas_relativas(imagen_bgr)
    if rel is None:
        res["motivo"] = "NO_SE_MIDE_EL_CONTRASTE_DISCO_FONDO"
        return res
    carac = {**caracteristicas_roi(imagen_bgr, roi.mascara), **rel}
    x = np.array([[carac[c] for c in paquete["features"]]])

    dom = paquete["dominio"][roi.posicion]
    puntaje = float(dom["modelo"].score_samples(x)[0])
    res["dominio_ok"] = puntaje >= dom["umbral"]
    ntu = max(float(paquete["modelos"][roi.posicion].predict(x)[0]), 0.0)
    res.update(ntu=ntu, categoria=categoria(ntu), area_pct=roi.area_pct)
    if res["dominio_ok"] or forzar:
        res.update(ok=True, motivo="OK" if res["dominio_ok"] else "FUERA_DE_DOMINIO (valor no fiable)")
    else:
        res["motivo"] = "FUERA_DE_DOMINIO: la imagen no se parece al montaje de entrenamiento"
    return res


def guardar_figura(res, ruta, titulo, real=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = res["imagen"]
    ov = img.copy()
    m = res["mascara"] > 0
    ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.6))
    for a, im, t in zip(ax, (img, ov), ("Imagen", "ROI (verde)")):
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        a.set_title(t, fontsize=10)
        a.axis("off")
    if "ntu" in res and res["ok"]:
        txt = f"{res['categoria']}  ·  {res['ntu']:.1f} NTU"
        if real is not None:
            txt += f"  (real: {real:g} NTU)"
        if res["motivo"] != "OK":
            txt += "\n(valor no fiable: imagen fuera de dominio)"
    else:
        txt = f"Imagen rechazada: {res['motivo']}"
    fig.suptitle(f"{titulo}\n{txt}", fontsize=10)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, facecolor="white")
    plt.close(fig)


def listar_imagenes(rutas):
    salida = []
    for r in rutas:
        p = Path(r)
        if p.is_dir():
            salida += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTENSIONES)
        else:
            salida.append(p)
    return salida


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagenes", nargs="*", help="imágenes o carpetas (sin argumentos: demostración)")
    ap.add_argument("--demo", action="store_true", help="correr también la demostración con imágenes de prueba")
    ap.add_argument("--salida", help="carpeta donde guardar las figuras")
    ap.add_argument("--forzar", action="store_true", help="mostrar el valor aunque la imagen esté fuera de dominio")
    args = ap.parse_args()

    paquete = cargar_modelo()
    trabajos = []  # (ruta, real, nombre)
    if args.demo or not args.imagenes:
        if not DEMO_CSV.exists():
            sys.exit(f"Falta {DEMO_CSV}. Ejecuta: python src/entrenar_modelo_multifuente.py")
        print("Demostración con imágenes del tramo final de cada grabación (no usadas para entrenar):\n")
        trabajos += [(RAIZ / f.ruta, float(f.nivel_ntu), f"{f.fuente} {f.nivel_ntu:g} NTU · {f.archivo}")
                     for _, f in pd.read_csv(DEMO_CSV).iterrows()]
    trabajos += [(p, None, p.name) for p in listar_imagenes(args.imagenes)]

    errores, aciertos_cat, n_cat = [], 0, 0
    for ruta, real, nombre in trabajos:
        img = cv2.imread(str(ruta))
        if img is None:
            print(f"{ruta.name:>12s}  no se pudo leer")
            continue
        res = estimar(img, paquete, forzar=args.forzar)
        if res["ok"]:
            fiable = "" if res["motivo"] == "OK" else "  [NO FIABLE]"
            extra = ""
            if real is not None:
                extra = f"  (real {real:g} NTU, error {res['ntu'] - real:+.2f})"
                errores.append(res["ntu"] - real)
                n_cat += 1
                aciertos_cat += (res["ntu"] >= UMBRAL_TURBIA) == (real >= UMBRAL_TURBIA)
            print(f"{nombre:>34s}  {res['categoria']:<19s} {res['ntu']:6.2f} NTU{extra}{fiable}")
        else:
            print(f"{nombre:>34s}  RECHAZADA: {res['motivo']}")
        if args.salida:
            guardar_figura(res, Path(args.salida) / f"{ruta.parent.name}_{ruta.stem}.png", nombre, real)

    if errores:
        e = np.array(errores)
        print(f"\nImágenes con valor real conocido: {len(e)}  |  MAE = {np.abs(e).mean():.2f} NTU  |  "
              f"RMSE = {np.sqrt((e ** 2).mean()):.2f} NTU  |  "
              f"acierto turbia/no turbia (>= {UMBRAL_TURBIA:g} NTU) = {100 * aciertos_cat / n_cat:.0f} %")
        print("Nota: estas imágenes provienen de las mismas grabaciones del entrenamiento (tramo final); "
              "con grabaciones nuevas el error es mayor.")


if __name__ == "__main__":
    main()
