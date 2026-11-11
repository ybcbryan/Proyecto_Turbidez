
from pathlib import Path
import pandas as pd
import numpy as np

BASE_PROYECTO = Path(__file__).resolve().parents[1]

CSV_ENTRADA = (
    BASE_PROYECTO
    / "src"
    / "resultados_caracteristicas"
    / "caracteristicas_dataset.csv"
)

if not CSV_ENTRADA.exists():
    print("ERROR: no se encontró el archivo:")
    print(CSV_ENTRADA)
    raise SystemExit(1)

df = pd.read_csv(CSV_ENTRADA, sep=";")

print("\n========== AUDITORÍA DEL DATASET ==========")

print("\n1. DIMENSIONES")
print("Filas:", len(df))
print("Columnas:", len(df.columns))

print("\n2. IMÁGENES POR NIVEL DE TURBIDEZ")
print(df["nivel_ntu"].value_counts().sort_index().to_string())

print("\n3. VALORES VACÍOS")
vacios = df.isna().sum()
vacios = vacios[vacios > 0]

if vacios.empty:
    print("No se encontraron valores vacíos.")
else:
    print(vacios.to_string())

print("\n4. IMÁGENES DUPLICADAS")
duplicados = df.duplicated(subset=["nivel_ntu", "archivo"]).sum()
print("Registros duplicados por nivel y nombre:", duplicados)

print("\n5. NIVELES ENCONTRADOS")
print(sorted(df["nivel_ntu"].dropna().unique()))

print("\n6. ESTADÍSTICAS DE CARACTERÍSTICAS NUMÉRICAS")

columnas_caracteristicas = [
    columna for columna in df.columns
    if columna.endswith("_media")
    or columna.endswith("_std")
    or columna == "area_mascara_pct"
]

numericas = df[columnas_caracteristicas].apply(
    pd.to_numeric, errors="coerce"
)

print(numericas.describe().round(3).to_string())

print("\n7. VALORES NO FINITOS")
no_finitos = (~np.isfinite(numericas)).sum()
no_finitos = no_finitos[no_finitos > 0]

if no_finitos.empty:
    print("No se encontraron valores infinitos.")
else:
    print(no_finitos.to_string())

print("\n8. ÁREA DE LAS MÁSCARAS")
if "area_mascara_pct" in df.columns:
    print("Mínima:", df["area_mascara_pct"].min())
    print("Máxima:", df["area_mascara_pct"].max())
    print("Media:", round(df["area_mascara_pct"].mean(), 3))

print("\n========== FIN DE LA AUDITORÍA ==========")
