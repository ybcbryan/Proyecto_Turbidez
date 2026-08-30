"""
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