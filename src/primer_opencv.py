"""
####
Proyecto Turbidez - Análisis y procesamiento de imágenes con OpenCV
####
import cv2

imagen = cv2.imread("../imagenes/agua.jpg") # Este paso me dio error
imagen = cv2.imread("D:/Doc/Python/Proyecto_Turbidez/imagenes/agua.jpg")

cv2.imshow("Mi primera imagen con OpenCV", imagen)

cv2.waitKey(0)
cv2.destroyAllWindows()

from pathlib import Path
import cv2
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Obtener la carpeta donde está este programa
carpeta_proyecto = Path(__file__).resolve().parent.parent

# Construir la ruta de la imagen
ruta_imagen = carpeta_proyecto / "imagenes" / "agua.jpg"

print("Ruta de la imagen:")
print(ruta_imagen)

# Leer la imagen
imagen = cv2.imread(str(ruta_imagen))

# Comprobar si la imagen fue cargada correctamente
if imagen is None:
    print("ERROR: No se pudo cargar la imagen.")
    print("Comprueba que el archivo exista y que la ruta sea correcta.")
    exit()


# ==========================================
# SELECCIÓN DE LA REGIÓN DE INTERÉS (ROI)
# ==========================================

print("\n===== SELECCIÓN DEL ROI =====")
print("Selecciona con el mouse la zona que corresponde al agua.")
print("Cuando termines, presiona ENTER.")

# Seleccionar ROI manualmente
x, y, ancho_roi, alto_roi = cv2.selectROI(
    "Selecciona la zona del agua",
    imagen,
    showCrosshair=True,
    fromCenter=False
)

# Cerrar la ventana de selección
cv2.destroyAllWindows()

# Recortar la región seleccionada
roi = imagen[
    y:y + alto_roi,
    x:x + ancho_roi
]

print("\n===== INFORMACIÓN DEL ROI =====")
print(f"Posición X: {x}")
print(f"Posición Y: {y}")
print(f"Ancho del ROI: {ancho_roi} píxeles")
print(f"Alto del ROI: {alto_roi} píxeles")

# Comprobar que el ROI tiene contenido
if roi.size == 0:
    print("ERROR: El ROI está vacío.")
    exit()

# Mostrar el ROI
cv2.imshow("Region de interes - Agua", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ==========================================
# ANÁLISIS DE LOS CANALES DEL ROI
# ==========================================

# Separar los canales BGR del ROI
roi_b, roi_g, roi_r = cv2.split(roi)

print("\n===== INFORMACIÓN DE LOS CANALES DEL ROI =====")

print(
    f"Canal Azul (B): "
    f"mínimo={roi_b.min()}, "
    f"máximo={roi_b.max()}, "
    f"promedio={roi_b.mean():.2f}"
)

print(
    f"Canal Verde (G): "
    f"mínimo={roi_g.min()}, "
    f"máximo={roi_g.max()}, "
    f"promedio={roi_g.mean():.2f}"
)

print(
    f"Canal Rojo (R): "
    f"mínimo={roi_r.min()}, "
    f"máximo={roi_r.max()}, "
    f"promedio={roi_r.mean():.2f}"
)


# ==========================================
# ESTADÍSTICAS DETALLADAS DEL ROI
# ==========================================



print("\n===== ESTADÍSTICAS DETALLADAS DEL ROI =====")

canales = {
    "Azul (B)": roi_b,
    "Verde (G)": roi_g,
    "Rojo (R)": roi_r
}

for nombre, canal in canales.items():

    media = np.mean(canal)
    desviacion = np.std(canal)
    minimo = np.min(canal)
    maximo = np.max(canal)
    mediana = np.median(canal)

    print(f"\n{nombre}")
    print(f"  Media:               {media:.2f}")
    print(f"  Desviación estándar: {desviacion:.2f}")
    print(f"  Mínimo:              {minimo}")
    print(f"  Máximo:              {maximo}")
    print(f"  Mediana:             {mediana:.2f}")

# ==========================================
# HISTOGRAMAS DE LOS CANALES DEL ROI
# ==========================================


# Calcular histogramas
hist_roi_b = cv2.calcHist([roi_b], [0], None, [256], [0, 256])
hist_roi_g = cv2.calcHist([roi_g], [0], None, [256], [0, 256])
hist_roi_r = cv2.calcHist([roi_r], [0], None, [256], [0, 256])

# Crear gráfico
plt.figure(figsize=(10, 6))

# Dibujar los histogramas
plt.plot(hist_roi_b, label="Azul (B)")
plt.plot(hist_roi_g, label="Verde (G)")
plt.plot(hist_roi_r, label="Rojo (R)")

# Configuración
plt.title("Histograma de los canales BGR - ROI")
plt.xlabel("Intensidad del píxel")
plt.ylabel("Cantidad de píxeles")
plt.xlim([0, 256])
plt.legend()
plt.grid()

# Mostrar
plt.show()

# ==========================================
# CONVERSIÓN DEL ROI DE BGR A HSV
# ==========================================

# Convertir el ROI de BGR a HSV
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Separar los tres canales HSV
canal_h, canal_s, canal_v = cv2.split(roi_hsv)

print("\n===== ESTADÍSTICAS DE LOS CANALES HSV =====")

print(
    f"Canal H (Matiz): "
    f"mínimo={canal_h.min()}, "
    f"máximo={canal_h.max()}, "
    f"promedio={canal_h.mean():.2f}"
)

print(
    f"Canal S (Saturación): "
    f"mínimo={canal_s.min()}, "
    f"máximo={canal_s.max()}, "
    f"promedio={canal_s.mean():.2f}"
)

print(
    f"Canal V (Valor): "
    f"mínimo={canal_v.min()}, "
    f"máximo={canal_v.max()}, "
    f"promedio={canal_v.mean():.2f}"
)

# ==========================================
# VISUALIZAR LOS CANALES HSV
# ==========================================

cv2.imshow("Canal H - Matiz", canal_h)
cv2.imshow("Canal S - Saturacion", canal_s)
cv2.imshow("Canal V - Valor", canal_v)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ==========================================
# HISTOGRAMAS DE LOS CANALES HSV - ROI
# ==========================================

# Calcular histogramas
hist_h = cv2.calcHist([canal_h], [0], None, [180], [0, 180])
hist_s = cv2.calcHist([canal_s], [0], None, [256], [0, 256])
hist_v = cv2.calcHist([canal_v], [0], None, [256], [0, 256])

# Crear gráfico
plt.figure(figsize=(10, 6))

# Dibujar histogramas
plt.plot(hist_h, label="Matiz (H)")
plt.plot(hist_s, label="Saturación (S)")
plt.plot(hist_v, label="Valor (V)")

# Configuración
plt.title("Histograma de los canales HSV - ROI")
plt.xlabel("Valor del canal")
plt.ylabel("Cantidad de píxeles")
plt.xlim([0, 256])
plt.legend()
plt.grid()

# Mostrar
plt.show()

# ==========================================
# EXTRACCIÓN DE CARACTERÍSTICAS DEL ROI
# ==========================================

caracteristicas = {
    "B_mean": np.mean(roi_b),
    "G_mean": np.mean(roi_g),
    "R_mean": np.mean(roi_r),

    "B_std": np.std(roi_b),
    "G_std": np.std(roi_g),
    "R_std": np.std(roi_r),

    "H_mean": np.mean(canal_h),
    "S_mean": np.mean(canal_s),
    "V_mean": np.mean(canal_v),

    "H_std": np.std(canal_h),
    "S_std": np.std(canal_s),
    "V_std": np.std(canal_v)
}

print("\n===== VECTOR DE CARACTERÍSTICAS =====")

for nombre, valor in caracteristicas.items():
    print(f"{nombre}: {valor:.2f}")

    # ==========================================
# CREAR TABLA DE CARACTERÍSTICAS
# ==========================================

tabla = pd.DataFrame([caracteristicas])

print("\n===== TABLA DE CARACTERÍSTICAS =====")
print(tabla)

# ==========================================
# GUARDAR CARACTERÍSTICAS EN CSV
# ==========================================

ruta_csv = carpeta_proyecto / "caracteristicas.csv"

tabla.to_csv(
    ruta_csv,
    index=False
)

print("\n===== DATASET GUARDADO =====")
print(f"Archivo: {ruta_csv}")

"""

# Obtener información de la imagen
alto, ancho, canales = imagen.shape

print("===== INFORMACIÓN DE LA IMAGEN =====")
print(f"Ancho: {ancho} píxeles")
print(f"Alto: {alto} píxeles")
print(f"Canales: {canales}")
print(f"Tipo de datos: {imagen.dtype}")

# Mostrar imagen
cv2.imshow("Mi primera imagen con OpenCV", imagen)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Obteneniendo un píxel
pixel = imagen[100, 200]

print("\n===== INFORMACIÓN DEL PÍXEL =====")
print(f"Valor del píxel: {pixel}")
print(f"Tipo del píxel: {type(pixel)}")
print(f"Tipo de los valores: {pixel.dtype}")

# Separar canales BGR
azul = imagen[:, :, 0]
verde = imagen[:, :, 1]
rojo = imagen[:, :, 2]

# Mostrar canales
cv2.imshow("Imagen original", imagen)
cv2.imshow("Canal Azul", azul)
cv2.imshow("Canal Verde", verde)
cv2.imshow("Canal Rojo", rojo)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ==========================================
# SEPARAR LOS CANALES DE COLOR
# ==========================================

# Separar la imagen en sus tres canales
canal_b, canal_g, canal_r = cv2.split(imagen)

print("\n===== INFORMACIÓN DE LOS CANALES =====")

print(f"Canal Azul (B): mínimo={canal_b.min()}, máximo={canal_b.max()}, promedio={canal_b.mean():.2f}")
print(f"Canal Verde (G): mínimo={canal_g.min()}, máximo={canal_g.max()}, promedio={canal_g.mean():.2f}")
print(f"Canal Rojo (R): mínimo={canal_r.min()}, máximo={canal_r.max()}, promedio={canal_r.mean():.2f}")

# ==========================================
# MOSTRAR LOS TRES CANALES
# ==========================================

cv2.imshow("Canal Azul - B", canal_b)
cv2.imshow("Canal Verde - G", canal_g)
cv2.imshow("Canal Rojo - R", canal_r)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ==========================================
# HISTOGRAMAS DE LOS CANALES DE COLOR
# ==========================================

import matplotlib.pyplot as plt

# Calcular histogramas
hist_b = cv2.calcHist([canal_b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([canal_g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([canal_r], [0], None, [256], [0, 256])

# Crear figura
plt.figure(figsize=(10, 6))

# Dibujar histogramas
plt.plot(hist_b, label="Azul (B)")
plt.plot(hist_g, label="Verde (G)")
plt.plot(hist_r, label="Rojo (R)")

# Configurar gráfico
plt.title("Histograma de los canales de color")
plt.xlabel("Intensidad del píxel")
plt.ylabel("Cantidad de píxeles")
plt.xlim([0, 256])
plt.legend()
plt.grid()

# Mostrar gráfico
plt.show()

cv2.selectROI()
"""