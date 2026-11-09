
from pathlib import Path
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "metadata"

persona, _ = pyreadstat.read_sav(
    str(RAW / "EH2025_Persona.sav"),
    usecols=[
        "folio", "nro", "s01a_03", "cmasi",
        "s03a_04", "s03b_10",
        "depto", "area", "factor", "p0"
    ]
)

vivienda, _ = pyreadstat.read_sav(
    str(RAW / "EH2025_Vivienda_1.sav"),
    usecols=["folio"]
)

print("=== VALIDACION DE CLAVES ===")
print("Duplicados persona:",
      persona.duplicated(["folio", "nro"]).sum())
print("Duplicados vivienda:",
      vivienda.duplicated(["folio"]).sum())
print("Personas sin vivienda:",
      (~persona["folio"].isin(vivienda["folio"])).sum())

# Seleccionar población escolar
edad = pd.to_numeric(
    persona["s01a_03"], errors="coerce"
)
escolares = persona.loc[
    edad.between(6, 17)
].copy()

print("\n=== POBLACION DE 6 A 17 ANOS ===")
print("Total:", len(escolares))
print("\nCondicion educativa:")
print(escolares["cmasi"].value_counts(dropna=False))

print("\nCruce matriculacion/asistencia:")
print(pd.crosstab(
    escolares["s03a_04"],
    escolares["cmasi"],
    dropna=False
))

# Frecuencias por departamento y condicion
tabla = pd.crosstab(
    escolares["depto"],
    escolares["cmasi"],
    dropna=False
)

tabla.to_csv(
    OUT / "poblacion_escolar_departamento.csv",
    encoding="utf-8-sig"
)

print("\nDistribucion departamental:")
print(tabla)
print("\nValidacion finalizada.")
