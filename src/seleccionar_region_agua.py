import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Cambia esta ruta por la carpeta Formazine de tu computadora
carpeta_formazine = Path(
    r"C:\proyecto"
    r"\\WaTur-Water-Turbidity-Dataset-main\Formazine"
)

# Imagen que queremos probar
nivel = "0.5"
nombre_imagen = "326.jpg"

ruta_imagen = carpeta_formazine / nivel / nombre_imagen


# Porcentaje mínimo de la imagen que debe ocupar
# una región para considerarla candidata.
area_minima = 0.05


# ============================================================
# 1. LEER IMAGEN
# ============================================================

imagen = cv2.imread(str(ruta_imagen))

if imagen is None:
    print("ERROR: no se pudo cargar la imagen.")
    print("Ruta:", ruta_imagen)
    raise SystemExit


alto, ancho = imagen.shape[:2]

print("Imagen:", nombre_imagen)
print("Nivel:", nivel)
print("Tamaño:", ancho, "x", alto)


# ============================================================
# 2. CONVERTIR A HSV
# ============================================================

hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

H, S, V = cv2.split(hsv)


# ============================================================
# 3. OBTENER LIMITES AUTOMÁTICOS DE BRILLO
# ============================================================

# Percentiles calculados sobre toda la imagen.
#
# En lugar de utilizar valores fijos como:
# V > 100
#
# calculamos los límites dependiendo de cada imagen.

limite_oscuro = np.percentile(V, 15)
limite_brillante = np.percentile(V, 95)

print()
print("Límite oscuro:", round(limite_oscuro, 2))
print("Límite brillante:", round(limite_brillante, 2))


# ============================================================
# 4. CREAR MÁSCARA DE CANDIDATOS
# ============================================================

# Conservamos solamente los píxeles que:
#
# - no sean demasiado oscuros
# - no sean excesivamente brillantes
#
# Esto evita inicialmente sombras muy fuertes y reflejos.

mascara = (
    (V >= limite_oscuro) &
    (V <= limite_brillante)
).astype(np.uint8) * 255


# ============================================================
# 5. LIMPIAR LA MÁSCARA
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (7, 7)
)

# Elimina pequeños grupos aislados
mascara = cv2.morphologyEx(
    mascara,
    cv2.MORPH_OPEN,
    kernel
)

# Une pequeñas discontinuidades
mascara = cv2.morphologyEx(
    mascara,
    cv2.MORPH_CLOSE,
    kernel
)


# ============================================================
# 6. BUSCAR REGIONES CONECTADAS
# ============================================================

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mascara,
    connectivity=8
)


# ============================================================
# 7. ANALIZAR LAS REGIONES
# ============================================================

candidatas = []

area_total = alto * ancho

for i in range(1, num_labels):

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    porcentaje = area / area_total

    # Ignorar regiones demasiado pequeñas
    if porcentaje < area_minima:
        continue

    candidatas.append({
        "label": i,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "porcentaje": porcentaje
    })


print()
print("Regiones candidatas:", len(candidatas))


# ============================================================
# 8. SELECCIONAR LA REGIÓN PRINCIPAL
# ============================================================

if len(candidatas) == 0:

    print("No se encontró una región candidata.")

    region_seleccionada = None

else:

    # Por ahora seleccionamos la región candidata
    # de mayor área.
    #
    # IMPORTANTE:
    # Esta es una primera versión.
    # Más adelante podremos agregar otros criterios.

    region_seleccionada = max(
        candidatas,
        key=lambda r: r["area"]
    )

    print()
    print("REGIÓN SELECCIONADA")

    print(
        "x =", region_seleccionada["x"],
        "y =", region_seleccionada["y"]
    )

    print(
        "ancho =", region_seleccionada["w"],
        "alto =", region_seleccionada["h"]
    )

    print(
        "Área:",
        region_seleccionada["area"],
        "píxeles"
    )

    print(
        "Porcentaje:",
        round(region_seleccionada["porcentaje"] * 100, 2),
        "%"
    )


# ============================================================
# 9. CREAR IMAGEN DE LA REGIÓN SELECCIONADA
# ============================================================

resultado = imagen.copy()

if region_seleccionada is not None:

    label = region_seleccionada["label"]

    # Crear máscara solamente de la región seleccionada
    mascara_final = np.zeros_like(mascara)

    mascara_final[labels == label] = 255

    # Crear una imagen de color para visualizar la región
    overlay = resultado.copy()

    overlay[mascara_final > 0] = (0, 255, 0)

    # Mezclar imagen original con región seleccionada
    resultado = cv2.addWeighted(
        resultado,
        0.65,
        overlay,
        0.35,
        0
    )

else:

    mascara_final = np.zeros_like(mascara)


# ============================================================
# 10. MOSTRAR RESULTADOS
# ============================================================

cv2.imshow(
    "1 - Imagen original",
    imagen
)

cv2.imshow(
    "2 - Candidatos",
    mascara
)

cv2.imshow(
    "3 - Region seleccionada",
    resultado
)

cv2.waitKey(0)
cv2.destroyAllWindows()