from pathlib import Path
import cv2
import numpy as np

# ============================================================
# RUTA DEL DATASET
# ============================================================

carpeta_watur = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine" / "0"

ruta_imagen = carpeta_watur / "0.jpg"

# ============================================================
# CARGAR IMAGEN
# ============================================================

imagen = cv2.imread(str(ruta_imagen))

if imagen is None:
    print("ERROR: No se pudo cargar la imagen.")
    exit()

alto, ancho, canales = imagen.shape

print("========================================")
print("ANÁLISIS ESPACIAL DE WATUR")
print("========================================")

print(f"Imagen: {ruta_imagen.name}")
print(f"Ancho: {ancho}")
print(f"Alto: {alto}")

# ============================================================
# DIVIDIR LA IMAGEN EN 4 ZONAS
# ============================================================

mitad_x = ancho // 2
mitad_y = alto // 2

zonas = {
    "Superior izquierda": imagen[0:mitad_y, 0:mitad_x],
    "Superior derecha": imagen[0:mitad_y, mitad_x:ancho],
    "Inferior izquierda": imagen[mitad_y:alto, 0:mitad_x],
    "Inferior derecha": imagen[mitad_y:alto, mitad_x:ancho]
}

print("\n========================================")
print("ESTADÍSTICAS POR ZONA")
print("========================================")

for nombre, zona in zonas.items():

    b, g, r = cv2.split(zona)

    print(f"\n{nombre}")

    print(f"B → media: {b.mean():.2f} | std: {b.std():.2f}")
    print(f"G → media: {g.mean():.2f} | std: {g.std():.2f}")
    print(f"R → media: {r.mean():.2f} | std: {r.std():.2f}")

# ============================================================
# CREAR UNA COPIA PARA DIBUJAR LA CUADRÍCULA
# ============================================================

visualizacion = imagen.copy()

# Línea vertical central
cv2.line(
    visualizacion,
    (mitad_x, 0),
    (mitad_x, alto),
    (0, 255, 0),
    1
)

# Línea horizontal central
cv2.line(
    visualizacion,
    (0, mitad_y),
    (ancho, mitad_y),
    (0, 255, 0),
    1
)

# ============================================================
# MOSTRAR
# ============================================================

cv2.imshow("WaTur - Analisis por zonas", visualizacion)

print("\nPresiona una tecla sobre la imagen para cerrar.")

cv2.waitKey(0)
cv2.destroyAllWindows()