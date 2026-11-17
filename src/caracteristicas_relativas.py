"""Características RELATIVAS (invariantes a la iluminación) del montaje disco oscuro + fondo claro.

La turbidez dispersa luz: el disco oscuro se aclara respecto al fondo (menos contraste)
y su borde se desenfoca (transición más ancha). Estas medidas comparan la zona
oscura con la clara DENTRO de la misma imagen, por lo que dependen menos del brillo
absoluto de cada grabación que las características de la zona clara por sí sola.
"""

import cv2
import numpy as np

DIST_BORDE = 10      # px de margen respecto a la frontera claro/oscuro
BANDA = 25           # px alrededor de la frontera donde se mide la transición
MIN_PIXELES = 500
CONTRASTE_MIN = 5.0


def _mayor_componente(binaria):
    n, etq, stats, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8), 8)
    if n < 2:
        return None
    return (etq == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def caracteristicas_relativas(imagen_bgr, sigma=3.0):
    """Devuelve un diccionario de características relativas o None si no se pueden medir."""
    if imagen_bgr.shape[:2] != (244, 244):
        imagen_bgr = cv2.resize(imagen_bgr, (244, 244), interpolation=cv2.INTER_AREA)
    v = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float64)
    vs = cv2.GaussianBlur(v, (0, 0), sigma)
    umbral, _ = cv2.threshold(vs.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    claro = _mayor_componente(vs > umbral)
    oscuro = _mayor_componente(vs <= umbral)
    if claro is None or oscuro is None:
        return None
    zc = cv2.distanceTransform(claro, cv2.DIST_L2, 5) >= DIST_BORDE
    zo = cv2.distanceTransform(oscuro, cv2.DIST_L2, 5) >= DIST_BORDE
    if zc.sum() < MIN_PIXELES or zo.sum() < MIN_PIXELES:
        return None

    vl, vd = v[zc].mean(), v[zo].mean()
    contraste = vl - vd
    if contraste < CONTRASTE_MIN:
        return None

    bgr = imagen_bgr.astype(np.float64)
    ml, md = bgr[zc].mean(axis=0), bgr[zo].mean(axis=0)          # B, G, R
    cl, cd = ml / ml.sum(), md / max(md.sum(), 1e-9)              # cromaticidad
    f = {
        "rel_V_osc_claro": vd / vl,
        "rel_michelson": (vl - vd) / (vl + vd),
        "rel_B": md[0] / ml[0], "rel_G": md[1] / ml[1], "rel_R": md[2] / ml[2],
        "dif_chroma_b": cd[0] - cl[0], "dif_chroma_g": cd[1] - cl[1], "dif_chroma_r": cd[2] - cl[2],
        "rel_std_osc": v[zo].std() / vl, "rel_std_claro": v[zc].std() / vl,
    }

    # Nitidez del borde: gradiente normalizado por el contraste y ancho de transición
    gx = cv2.Sobel(cv2.GaussianBlur(v, (0, 0), 1.0), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(cv2.GaussianBlur(v, (0, 0), 1.0), cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    k = np.ones((3, 3), np.uint8)
    borde = (cv2.dilate(claro, k) - cv2.erode(claro, k)) > 0
    if borde.sum() < 20:
        return None
    f["grad_borde_media"] = mag[borde].mean() / contraste
    f["grad_borde_p90"] = np.percentile(mag[borde], 90) / contraste
    dist_borde = cv2.distanceTransform((~borde).astype(np.uint8), cv2.DIST_L2, 5)
    banda = dist_borde <= BANDA
    transicion = banda & (vs > vd + 0.2 * contraste) & (vs < vd + 0.8 * contraste)
    f["ancho_transicion"] = transicion.sum() / borde.sum()
    return {k_: float(x) for k_, x in f.items()}
