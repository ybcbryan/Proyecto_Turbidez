"""Detección robusta de la zona clara (agua) en imágenes del dataset WaTur.

Idea: en lugar de buscar UNA transición en el perfil de filas (frágil ante
reflejos y sombras), se segmenta la imagen en dos clases (clara/oscura) con
Otsu sobre el canal V suavizado y se trabaja en 2D:

1. Otsu sobre V suavizado -> mascara de la zona clara.
2. Se conserva el componente conexo más grande (descarta manchas aisladas).
3. Se aleja la ROI del borde claro/oscuro (distancia en píxeles) para evitar
   la transición borrosa y el menisco.
4. Se excluyen píxeles saturados y valores atípicos respecto a un ajuste
   plano de V (reflejos, partículas, objetos pequeños).
5. Se calculan banderas de calidad: contraste entre clases, área, posición.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ResultadoROI:
    mascara: np.ndarray      # uint8, 255 = píxel de la ROI
    ok: bool                 # True si pasa los criterios de calidad
    motivo: str              # "OK" o razón del rechazo
    area_pct: float          # % de la imagen cubierto por la ROI final
    contraste: float         # media V (clara) - media V (oscura)
    posicion: str            # ARRIBA | ABAJO | CENTRO | NINGUNA
    umbral: float            # umbral de Otsu usado
    mad: float = 0.0         # dispersión robusta de V en la ROI (heterogeneidad)
    frac_excluida: float = 0.0  # fracción de píxeles descartados como atípicos


def _vacio(forma, motivo, contraste=0.0, umbral=0.0):
    return ResultadoROI(
        np.zeros(forma[:2], np.uint8), False, motivo, 0.0,
        contraste, "NINGUNA", umbral,
    )


def detectar_roi(
    imagen_bgr,
    sigma=3.0,
    dist_borde=10,
    margen_imagen=4,
    saturacion_v=250,
    k_mad=3.5,
    contraste_min=15.0,
    area_min_pct=20.0,   # una zona clara típica ocupa ~32 % (P1-P99: 25-35 %)
    area_max_pct=45.0,   # áreas ~60 % corresponden a frames de transición
    mad_max=15.0,        # ROI heterogénea: vetas, espuma, burbujas
    pixeles_min=500,
):
    """Devuelve un ResultadoROI con la máscara de la zona clara."""
    alto, ancho = imagen_bgr.shape[:2]
    v = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]

    # 1) Segmentación clara/oscura (Otsu sobre V suavizado)
    v_suave = cv2.GaussianBlur(v, (0, 0), sigma)
    umbral, _ = cv2.threshold(
        v_suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    claro = (v_suave > umbral).astype(np.uint8)

    if claro.sum() == 0 or claro.sum() == claro.size:
        return _vacio(imagen_bgr.shape, "SIN_DOS_CLASES", umbral=umbral)

    contraste = float(v[claro > 0].mean() - v[claro == 0].mean())
    if contraste < contraste_min:
        return _vacio(
            imagen_bgr.shape, "CONTRASTE_BAJO", contraste, umbral
        )

    # 2) Componente conexo más grande de la zona clara
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(claro, 8)
    if n < 2:
        return _vacio(imagen_bgr.shape, "SIN_COMPONENTE", contraste, umbral)
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (etiquetas == mayor).astype(np.uint8)

    # 3) Alejarse del borde claro/oscuro y de los bordes de la imagen
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    roi = dist >= dist_borde
    if margen_imagen > 0:
        roi[:margen_imagen, :] = False
        roi[-margen_imagen:, :] = False
        roi[:, :margen_imagen] = False
        roi[:, -margen_imagen:] = False

    if roi.sum() < pixeles_min:
        return _vacio(imagen_bgr.shape, "ROI_MUY_PEQUENA", contraste, umbral)

    # 4) Excluir saturados y atípicos respecto a un plano ajustado de V
    roi &= v < saturacion_v
    ys, xs = np.nonzero(roi)
    if len(ys) < pixeles_min:
        return _vacio(imagen_bgr.shape, "ROI_MUY_PEQUENA", contraste, umbral)

    vv = v[ys, xs].astype(np.float64)
    A = np.column_stack([np.ones_like(xs), xs, ys]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
    residuo = vv - A @ coef
    mad_real = 1.4826 * np.median(np.abs(residuo - np.median(residuo)))
    mad = max(mad_real, 1.5)  # piso para zonas muy uniformes
    buenos = np.abs(residuo) <= k_mad * mad
    frac_excluida = 1.0 - float(buenos.mean())

    final = np.zeros((alto, ancho), np.uint8)
    final[ys[buenos], xs[buenos]] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, k)

    area_pct = 100.0 * cv2.countNonZero(final) / (alto * ancho)
    if cv2.countNonZero(final) < pixeles_min:
        return _vacio(imagen_bgr.shape, "ROI_MUY_PEQUENA", contraste, umbral)

    # Posición de la zona clara
    cy = np.nonzero(final)[0].mean()
    if cy < alto * 0.40:
        posicion = "ARRIBA"
    elif cy > alto * 0.60:
        posicion = "ABAJO"
    else:
        posicion = "CENTRO"

    motivo = "OK"
    ok = True
    if area_pct < area_min_pct:
        ok, motivo = False, "AREA_PEQUENA"
    elif area_pct > area_max_pct:
        ok, motivo = False, "AREA_GRANDE"
    elif mad_real > mad_max:
        ok, motivo = False, "HETEROGENEA"

    return ResultadoROI(
        final, ok, motivo, area_pct, contraste, posicion, float(umbral),
        float(mad_real), frac_excluida,
    )
