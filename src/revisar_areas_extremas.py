
import pandas as pd
from pathlib import Path

archivo = Path(
    r"C:\Proyecto\src\resultados_caracteristicas"
    r"\caracteristicas_dataset.csv"
)

df = pd.read_csv(archivo, sep=";")

# Ordenar las imágenes por área de la máscara.
df_ordenado = df.sort_values("area_mascara_pct")

# Seleccionar las 20 áreas más pequeñas y las 20 más grandes.
extremas = pd.concat([
    df_ordenado.head(20),
    df_ordenado.tail(20)
])

# Conservar solo las columnas necesarias.
columnas = [
    "nivel_ntu",
    "archivo",
    "ruta",
    "tipo_region",
    "area_mascara_pct"
]

extremas = extremas[columnas]

salida = Path(
    r"C:\Proyecto\src\resultados_caracteristicas"
    r"\areas_extremas.csv"
)

extremas.to_csv(salida, sep=";", index=False)

print("Revisión de áreas terminada.")
print("Imágenes seleccionadas:", len(extremas))
print("Archivo guardado en:", salida)
print()
print(extremas.to_string(index=False))
