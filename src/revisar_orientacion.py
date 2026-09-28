from pathlib import Path
import cv2
import numpy as np

# ruta de la ubicación de la carpeta WaTur
carpeta = Path(r"C:\Users\Bryan\Desktop\Analisis de Imagenes\WaTur-Water-Turbidity-Dataset-main\Formazine\0")

archivos = sorted(
    carpeta.glob("*.jpg"),
    key=lambda p: int(p.stem)
)[:6]

# Región de interés
x, y, ancho, alto = 20, 10, 204, 40

vistas = []

for archivo in archivos:
    imagen = cv2.imread(str(archivo))

    if imagen is None:
        continue

    numero = int(archivo.stem)

    # Normalizar orientación
    if numero % 2 == 1:
        imagen = cv2.flip(imagen, 0)

    # Dibujar la ROI sobre una copia
    vista = imagen.copy()
    cv2.rectangle(
        vista,
        (x, y),
        (x + ancho, y + alto),
        (0, 0, 255),
        2
    )

    # Nombre de archivo
    cv2.putText(
        vista, archivo.name, (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1
    )

    vistas.append(vista)

# Mostrar las seis en una cuadrícula
if vistas:
    vacia = np.zeros_like(vistas[0])
    while len(vistas) < 6:
        vistas.append(vacia.copy())

    fila1 = np.hstack(vistas[:3])
    fila2 = np.hstack(vistas[3:6])
    mosaico = np.vstack((fila1, fila2))

    cv2.imshow("Comprobacion de ROI", mosaico)
    cv2.waitKey(0)
    cv2.destroyAllWindows()