from pathlib import Path
import cv2
import numpy as np

# ruta de la ubicación de la carpeta WaTur
carpeta_formazine = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine"



# Imágenes para inspeccionar manualmente
pruebas = [
    ("0.5", "314.jpg"),
    ("0.5", "316.jpg"),
    ("0.5", "303.jpg"),
    ("0.5", "305.jpg"),
    ("40", "4.jpg"),
    ("40", "5.jpg"),
    ("1", "22.jpg"),
    ("7.5", "163.jpg")
]

# ROI actual
x, y, ancho, alto = 20, 10, 204, 65

vistas = []

for nivel, nombre in pruebas:
    ruta = carpeta_formazine / nivel / nombre
    imagen = cv2.imread(str(ruta))

    if imagen is None:
        print("No se pudo abrir:", ruta)
        continue

    numero = int(Path(nombre).stem)

    if numero % 2 == 1:
        imagen = cv2.flip(imagen, 0)

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    perfil = gris[:, x:x + ancho].mean(axis=1)

    diferencias = perfil[21:180] - perfil[20:179]
    fila_borde = int(diferencias.argmin() + 21)

    vista = imagen.copy()

    # ROI en rojo
    cv2.rectangle(
        vista, (x, y), (x + ancho, y + alto),
        (0, 0, 255), 2
    )

    # Fila estimada en amarillo
    cv2.line(
        vista, (0, fila_borde), (243, fila_borde),
        (0, 255, 255), 2
    )

    cv2.putText(
        vista, f"{nivel} NTU - {nombre}",
        (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1
    )

    vistas.append(vista)

if vistas:
    ancho_img = 244
    alto_img = 244

    while len(vistas) % 2 != 0:
        vistas.append(np.zeros_like(vistas[0]))

    filas = []
    for i in range(0, len(vistas), 2):
        filas.append(np.hstack(vistas[i:i+2]))

    mosaico = np.vstack(filas)

    cv2.imshow("Revision de detecciones", mosaico)
    cv2.waitKey(0)
    cv2.destroyAllWindows()