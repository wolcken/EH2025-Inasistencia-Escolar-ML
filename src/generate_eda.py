
from pathlib import Path

import duckdb
from data_profiling import ProfileReport

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dataset_escolar.parquet"
OUTPUT = ROOT / "outputs" / "eda"

OUTPUT.mkdir(parents=True, exist_ok=True)

variables = [
    "inasistencia_escolar",
    "s01a_03",  # Edad
    "s01a_02",  # Sexo
    "depto",    # Departamento
    "area",     # Área
    "yhogpc",   # Ingreso per cápita
    "p0",       # Pobreza por ingresos
    "pext0",    # Pobreza extrema
    "tipohogar",
    "factor",
]

print("Cargando dataset analítico...")

con = duckdb.connect()

# Seleccionar variables disponibles en el Parquet
columnas = [
    fila[0]
    for fila in con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [str(DATA)]
    ).fetchall()
]

seleccionadas = [
    variable
    for variable in variables
    if variable in columnas
]

faltantes = sorted(set(variables) - set(seleccionadas))

if faltantes:
    print("Variables no encontradas:", faltantes)

if "inasistencia_escolar" not in seleccionadas:
    raise ValueError("No se encontró la variable objetivo.")

df = con.execute(
    "SELECT * FROM read_parquet(?)",
    [str(DATA)]
).df()

df = df[seleccionadas].copy()

# Categorías: los códigos numéricos no son medidas continuas.
categoricas = [
    "inasistencia_escolar",
    "s01a_02",
    "depto",
    "area",
    "p0",
    "pext0",
    "tipohogar",
]

for columna in categoricas:
    if columna in df.columns:
        df[columna] = (
            df[columna]
            .astype("Int64")
            .astype("string")
        )

print(f"Registros: {len(df)}")
print(f"Variables seleccionadas: {len(df.columns)}")
print("Generando reporte EDA...")

profile = ProfileReport(
    df,
    title="Análisis Exploratorio - EH2025",
    explorative=True,
    minimal=True,
    progress_bar=True
)

destino = OUTPUT / "reporte_eda_eh2025.html"

profile.to_file(str(destino))

print("\nReporte generado correctamente:")
print(destino)

con.close()
