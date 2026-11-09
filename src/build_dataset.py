
from pathlib import Path

import duckdb
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)


def convert_sav(filename, output, prefix=None):
    print(f"Leyendo {filename}...")

    df, _ = pyreadstat.read_sav(
        str(RAW / filename),
        apply_value_formats=False
    )

    if prefix:
        df = df.rename(
            columns={
                col: f"{prefix}{col}"
                for col in df.columns
                if col != "folio"
            }
        )

    df.to_parquet(
        PROCESSED / output,
        index=False,
        engine="pyarrow"
    )

    print(f"Convertido: {output} ({len(df)} registros)")


convert_sav(
    "EH2025_Persona.sav",
    "personas.parquet"
)

convert_sav(
    "EH2025_Vivienda_1.sav",
    "viviendas.parquet",
    prefix="viv_"
)

con = duckdb.connect()

personas_path = (
    PROCESSED / "personas.parquet"
).as_posix()

viviendas_path = (
    PROCESSED / "viviendas.parquet"
).as_posix()

salida = (
    PROCESSED / "dataset_escolar.parquet"
).as_posix()

# Registrar las fuentes Parquet en DuckDB
con.execute(f"""
    CREATE OR REPLACE VIEW personas AS
    SELECT * FROM read_parquet('{personas_path}')
""")

con.execute(f"""
    CREATE OR REPLACE VIEW viviendas AS
    SELECT * FROM read_parquet('{viviendas_path}')
""")

# Validar integridad antes del JOIN
duplicados = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT folio, nro
        FROM personas
        GROUP BY folio, nro
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

duplicados_vivienda = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT folio
        FROM viviendas
        GROUP BY folio
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

sin_vivienda = con.execute("""
    SELECT COUNT(*)
    FROM personas p
    LEFT JOIN viviendas v
        ON p.folio = v.folio
    WHERE v.folio IS NULL
""").fetchone()[0]

if duplicados or duplicados_vivienda or sin_vivienda:
    raise ValueError(
        "Fallo de integridad: revisar claves de unión."
    )

# Construir base de población escolar
# 0 = asiste; 1 = no matriculado o no asiste.
query = """
    SELECT
        p.*,
        v.* EXCLUDE (folio),
        CASE
            WHEN p.cmasi = 2 THEN 0
            WHEN p.cmasi IN (1, 3) THEN 1
            ELSE NULL
        END AS inasistencia_escolar
    FROM personas p
    INNER JOIN viviendas v
        ON p.folio = v.folio
    WHERE p.s01a_03 BETWEEN 6 AND 17
      AND p.cmasi IN (1, 2, 3)
"""

con.execute(
    f"COPY ({query}) TO '{salida}' (FORMAT PARQUET)"
)

# Comprobar resultado
resumen = con.execute("""
    SELECT
        COUNT(*) AS total,
        SUM(inasistencia_escolar) AS casos_inasistencia,
        COUNT(DISTINCT folio) AS hogares
    FROM read_parquet(?)
""", [salida]).fetchone()

print("\n=== DATASET ANALITICO ===")
print("Total estudiantes:", resumen[0])
print("Casos de inasistencia:", resumen[1])
print("Hogares representados:", resumen[2])

if resumen[0] != 9201 or resumen[1] != 208:
    raise ValueError(
        "Las cifras no coinciden con la validación previa."
    )

print("\nDataset generado correctamente:")
print(salida)

con.close()
