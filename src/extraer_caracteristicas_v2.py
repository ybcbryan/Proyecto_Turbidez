"""Extracción de características (v2) usando la ROI robusta de la zona clara.

Por cada imagen del subconjunto Formazine de WaTur:
  1. Detecta la ROI con roi_zona_clara.detectar_roi.
  2. Si la ROI es válida, calcula características dentro de la ROI:
       - media y desviación estándar en RGB, HSV y Lab
       - cuantiles (P10, P50, P90) de V y L
       - cromaticidad normalizada (r, g, b) y razones entre canales
       - textura GLCM (contraste, disimilitud, homogeneidad, energía,
         correlación, entropía) restringida a pares de píxeles dentro de la ROI
       - varianza del Laplaciano y magnitud media del gradiente
  3. Guarda todo en un CSV (las imágenes rechazadas quedan con ok=False).

Uso:
    python src/extraer_caracteristicas_v2.py
"""

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_zona_clara import detectar_roi  # noqa: E402

BASE_PROYECTO = Path(__file__).resolve().parents[1]
BASE_DATASET = BASE_PROYECTO / "WaTur-Water-Turbidity-Dataset-main" / "Formazine"
CARPETA_SALIDA = Path(__file__).resolve().parent / "resultados_caracteristicas"
CSV_SALIDA = CARPETA_SALIDA / "caracteristicas_v2.csv"

NIVELES = ["0", "0.5", "1", "2.5", "4", "5", "7.5", "10", "40"]
NIVELES_GLCM = 64  # cuantización fija: gris // 4


# ----------------------------------------------------------------------
# Textura GLCM restringida a la ROI
# ----------------------------------------------------------------------
def _glcm(gris_q, mascara, dy, dx, niveles=NIVELES_GLCM):
    h, w = gris_q.shape
    a = gris_q[: h - dy, : w - dx]
    b = gris_q[dy:, dx:]
    valido = mascara[: h - dy, : w - dx] & mascara[dy:, dx:]
    if valido.sum() == 0:
        return None
    idx = a[valido].astype(np.int64) * niveles + b[valido].astype(np.int64)
    c = np.bincount(idx, minlength=niveles * niveles).reshape(niveles, niveles)
    c = c + c.T  # simétrica
    return c / c.sum()


def _props_glcm(p):
    i, j = np.indices(p.shape)
    dif = (i - j).astype(np.float64)
    mi, mj = (i * p).sum(), (j * p).sum()
    si = np.sqrt(((i - mi) ** 2 * p).sum())
    sj = np.sqrt(((j - mj) ** 2 * p).sum())
    corr = ((i - mi) * (j - mj) * p).sum() / (si * sj) if si * sj > 1e-12 else 1.0
    nz = p[p > 0]
    return {
        "contraste": float((p * dif ** 2).sum()),
        "disimilitud": float((p * np.abs(dif)).sum()),
        "homogeneidad": float((p / (1.0 + dif ** 2)).sum()),
        "energia": float(np.sqrt((p ** 2).sum())),
        "correlacion": float(corr),
        "entropia": float(-(nz * np.log2(nz)).sum()),
    }


def textura_glcm(gris, mascara):
    gris_q = (gris // (256 // NIVELES_GLCM)).astype(np.uint8)
    m = mascara > 0
    acumulado = None
    n = 0
    for dy, dx in ((0, 1), (1, 0)):  # horizontal y vertical, distancia 1
        p = _glcm(gris_q, m, dy, dx)
        if p is None:
            continue
        props = _props_glcm(p)
        acumulado = props if acumulado is None else {
            k: acumulado[k] + props[k] for k in props
        }
        n += 1
    if n == 0:
        return {}
    return {f"GLCM_{k}": v / n for k, v in acumulado.items()}


# ----------------------------------------------------------------------
# Características de color / intensidad
# ----------------------------------------------------------------------
def caracteristicas_roi(imagen_bgr, mascara):
    m = mascara > 0
    f = {}

    espacios = [
        ("RGB", cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB), ["R", "G", "B"]),
        ("HSV", cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV), ["H", "S", "V"]),
        ("Lab", cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB), ["L", "a", "b"]),
    ]
    for nombre, img, canales in espacios:
        pix = img[m].astype(np.float64)
        for k, canal in enumerate(canales):
            f[f"{nombre}_{canal}_media"] = pix[:, k].mean()
            f[f"{nombre}_{canal}_std"] = pix[:, k].std()

    # Cuantiles de brillo
    v = espacios[1][1][:, :, 2][m].astype(np.float64)
    L = espacios[2][1][:, :, 0][m].astype(np.float64)
    for q in (10, 50, 90):
        f[f"V_P{q}"] = np.percentile(v, q)
        f[f"L_P{q}"] = np.percentile(L, q)

    # Cromaticidad normalizada y razones (menos sensibles a la iluminación)
    r, g, b = (f["RGB_R_media"], f["RGB_G_media"], f["RGB_B_media"])
    s = r + g + b + 1e-9
    f["chroma_r"], f["chroma_g"], f["chroma_b"] = r / s, g / s, b / s
    f["razon_R_G"] = r / (g + 1e-9)
    f["razon_B_G"] = b / (g + 1e-9)
    f["razon_B_R"] = b / (r + 1e-9)

    # Textura
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    f.update(textura_glcm(gris, mascara))
    lap = cv2.Laplacian(gris, cv2.CV_64F)
    f["laplaciano_var"] = lap[m].var()
    gx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    f["gradiente_media"] = np.hypot(gx, gy)[m].mean()

    return {k: float(v) for k, v in f.items()}


# ----------------------------------------------------------------------
# Procesamiento por imagen y principal
# ----------------------------------------------------------------------
def procesar(args):
    nivel, archivo = args
    imagen = cv2.imread(str(BASE_DATASET / nivel / archivo))
    base = {
        "nivel_ntu": float(nivel),
        "archivo": archivo,
        "frame": int(Path(archivo).stem),
    }
    if imagen is None:
        base.update(ok=False, motivo="NO_SE_PUDO_LEER")
        return base
    r = detectar_roi(imagen)
    base.update(
        ok=bool(r.ok),
        motivo=r.motivo,
        posicion=r.posicion,
        area_pct=round(r.area_pct, 3),
        contraste_clara_oscura=round(r.contraste, 3),
        roi_mad=round(r.mad, 3),
    )
    if r.ok:
        base.update(caracteristicas_roi(imagen, r.mascara))
    return base


def main():
    if not BASE_DATASET.exists():
        print("ERROR: no se encontró el dataset:", BASE_DATASET)
        return
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    trabajos = []
    for nivel in NIVELES:
        carpeta = BASE_DATASET / nivel
        archivos = sorted(
            (a for a in os.listdir(carpeta) if a.lower().endswith(".jpg")),
            key=lambda a: int(Path(a).stem),
        )
        trabajos += [(nivel, a) for a in archivos]
    print(f"Imágenes a procesar: {len(trabajos)}")

    with Pool(max(1, (os.cpu_count() or 2) - 0)) as pool:
        filas = pool.map(procesar, trabajos, chunksize=100)

    df = pd.DataFrame(filas)
    df.to_csv(CSV_SALIDA, index=False)
    print(f"Guardado: {CSV_SALIDA}")
    print(f"Imágenes válidas: {int(df.ok.sum())} de {len(df)} "
          f"({100 * df.ok.mean():.2f} %)")
    print(df.groupby("nivel_ntu").ok.agg(["sum", "size"]))


if __name__ == "__main__":
    main()
