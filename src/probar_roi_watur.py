from pathlib import Path
import cv2
import numpy as np

# ==========================================
# RUTA DE LA IMAGEN
# ==========================================

carpeta_watur = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine" / "0"
ruta_imagen = carpeta_watur / "0.jpg"

imagen = cv2.imread(str(ruta_imagen))

if imagen is None:
    print("ERROR: No se pudo cargar la imagen.")
    exit()

# ==========================================
# DEFINIR ROI PROVISIONAL
# ==========================================

# Coordenadas: x_inicio, y_inicio, x_final, y_final
x1, y1 = 15, 75
x2, y2 = 229, 145

roi = imagen[y1:y2, x1:x2]

# ==========================================
# CALCULAR NITIDEZ
# ==========================================

gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# El Laplaciano resalta cambios bruscos de intensidad
laplaciano = cv2.Laplacian(gris, cv2.CV_64F)

# Varianza del Laplaciano
nitidez = laplaciano.var()

print("========================================")
print("PRUEBA DE ROI Y NITIDEZ")
print("========================================")
print(f"ROI: x={x1}:{x2}, y={y1}:{y2}")
print(f"Ancho del ROI: {roi.shape[1]} píxeles")
print(f"Alto del ROI: {roi.shape[0]} píxeles")
print(f"Varianza del Laplaciano: {nitidez:.4f}")

# ==========================================
# MOSTRAR ROI EN LA IMAGEN ORIGINAL
# ==========================================

visualizacion = imagen.copy()

cv2.rectangle(
    visualizacion,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

cv2.imshow("Imagen con ROI", visualizacion)
cv2.imshow("ROI seleccionado", roi)
cv2.imshow("ROI en escala de grises", gris)

print("\nPresiona una tecla para cerrar las ventanas.")
cv2.waitKey(0)
cv2.destroyAllWindows()