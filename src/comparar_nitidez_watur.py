from pathlib import Path
import cv2

# Ruta de la carpeta Formazine
carpeta_formazine = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine"

# Niveles de turbidez disponibles
niveles = ["0", "0.5", "1", "2.5", "4", "5", "7.5", "10", "40"]

# ROI central provisional
x1, y1 = 30, 75
x2, y2 = 214, 145

print("========================================")
print("COMPARACIÓN DE NITIDEZ - WATUR")
print("========================================")
print(f"ROI: x={x1}:{x2}, y={y1}:{y2}\n")

for nivel in niveles:

    carpeta = carpeta_formazine / nivel

    # Buscar imágenes JPG en la carpeta
    imagenes = sorted(carpeta.glob("*.jpg"))

    if not imagenes:
        print(f"Nivel {nivel} NTU: no se encontraron imágenes")
        continue

    ruta = imagenes[0]
    imagen = cv2.imread(str(ruta))

    if imagen is None:
        print(f"Nivel {nivel} NTU: error al cargar {ruta.name}")
        continue

    alto, ancho = imagen.shape[:2]

    # Comprobar que el ROI cabe en la imagen
    if x2 > ancho or y2 > alto:
        print(f"Nivel {nivel} NTU: imagen demasiado pequeña")
        continue

    roi = imagen[y1:y2, x1:x2]

    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplaciano = cv2.Laplacian(gris, cv2.CV_64F)
    nitidez = laplaciano.var()

    print(
        f"Nivel: {nivel:>4} NTU | "
        f"Imagen: {ruta.name:<15} | "
        f"Nitidez: {nitidez:.4f}"
    )