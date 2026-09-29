from pathlib import Path
import cv2
import numpy as np

base = Path(__file__).resolve().parents[1] / "WaTur-Water-Turbidity-Dataset-main" / "Formazine"

niveles = ["0", "0.5", "5", "10", "40"]

# Margen para no incluir la frontera
margen = 8

vistas = []

for nivel in niveles:
    carpeta = base / nivel

    archivos = sorted(
        carpeta.glob("*.jpg"),
        key=lambda p: int(p.stem)
    )

    if not archivos:
        print(f"No hay imágenes en {nivel} NTU")
        continue

    # Para 40 NTU usamos la imagen 4, que ya identificaste
    if nivel == "40":
        ruta = carpeta / "4.jpg"
        if not ruta.exists():
            ruta = archivos[0]
    else:
        ruta = archivos[0]

    imagen = cv2.imread(str(ruta))

    if imagen is None:
        print("No se pudo abrir:", ruta)
        continue

    numero = int(ruta.stem)

    # Corregir orientación
    if numero % 2 == 1:
        imagen = cv2.flip(imagen, 0)

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Perfil de intensidad promedio por fila
    perfil = gris[:, 20:224].mean(axis=1).astype(np.float32)

    # Suavizar el perfil para reducir cambios pequeños
    perfil_suave = cv2.GaussianBlur(
        perfil.reshape(-1, 1),
        (1,  nine := 9),
        0
    ).flatten()

    # Buscar el descenso más pronunciado en una zona posible
    inicio, fin = 25, 180
    cambios = np.diff(perfil_suave[inicio:fin])
    fila_borde = int(np.argmin(cambios) + inicio)

    # Dejar un margen por encima de la frontera
    limite = max(0, fila_borde - margen)

    # Máscara: blanco = conservar; negro = excluir
    mascara = np.zeros(gris.shape, dtype=np.uint8)
    mascara[:limite, :] = 255

    # Superposición verde de la zona seleccionada
    superpuesta = imagen.copy()
    verde = np.zeros_like(imagen)
    verde[:, :] = (0, 255, 0)

    seleccion = mascara == 255
    superpuesta[seleccion] = cv2.addWeighted(
        imagen[seleccion], 0.55,
        verde[seleccion], 0.45,
        0
    )

    # Dibujar la frontera estimada en azul
    cv2.line(
        superpuesta,
        (0, fila_borde),
        (243, fila_borde),
        (255, 0, 0),
        2
    )

    # Etiquetas
    cv2.putText(
        imagen, f"{nivel} NTU - Original",
        (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1
    )
    cv2.putText(
        mascara, f"Limite: {limite}",
        (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, 255, 1
    )
    cv2.putText(
        superpuesta, f"Frontera: {fila_borde}",
        (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1
    )

    # Poner las tres vistas en una fila
    fila = np.hstack([
        imagen,
        cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR),
        superpuesta
    ])

    vistas.append(fila)

    print(
        f"{nivel} NTU | archivo: {ruta.name} | "
        f"frontera estimada: {fila_borde} | "
        f"límite de máscara: {limite}"
    )

if vistas:
    mosaico = np.vstack(vistas)
    cv2.imshow("Prueba de segmentacion", mosaico)
    cv2.waitKey(0)
    cv2.destroyAllWindows()