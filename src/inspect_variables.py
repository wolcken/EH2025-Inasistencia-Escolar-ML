
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "metadata"
OUT.mkdir(parents=True, exist_ok=True)

PERSONA = RAW / "EH2025_Persona.sav"
VIVIENDA = RAW / "EH2025_Vivienda_1.sav"

variables = [
    "folio", "nro", "depto", "area",
    "s01a_02", "s01a_03",
    "s03a_04", "s03a_05",
    "s03b_10", "s03b_11", "cmasi",
    "factor", "upm", "estrato",
    "yhogpc", "p0", "pext0",
]

personas, meta = pyreadstat.read_sav(
    str(PERSONA),
    usecols=variables,
    apply_value_formats=False,
)

viviendas, _ = pyreadstat.read_sav(
    str(VIVIENDA),
    usecols=["folio"],
)

# 1. Verificar identificadores
print("\n=== INTEGRIDAD DE IDENTIFICADORES ===")

print("Personas:", len(personas))
print("Viviendas:", len(viviendas))

print(
    "Duplicados folio+nro:",
    personas.duplicated(["folio", "nro"]).sum()
)

print(
    "Duplicados folio en vivienda:",
    viviendas.duplicated(["folio"]).sum()
)

sin_vivienda = ~personas["folio"].isin(
    viviendas["folio"]
)

print(
    "Personas sin vivienda correspondiente:",
    sin_vivienda.sum()
)

# 2. Revisar categorías y sus frecuencias
objetivos = [
    "cmasi", "s03a_04", "s03b_10",
    "s03a_05", "s03b_11",
    "depto", "area", "p0", "pext0"
]

resultados = []

for variable in objetivos:
    print(f"\n=== {variable} ===")

    etiquetas = meta.variable_value_labels.get(
        variable, {}
    )

    conteos = personas[variable].value_counts(
        dropna=False
    )

    for codigo, cantidad in conteos.items():
        etiqueta = etiquetas.get(codigo, "")

        print(codigo, etiqueta, cantidad)

        resultados.append({
            "variable": variable,
            "codigo": codigo,
            "etiqueta": etiqueta,
            "frecuencia": cantidad,
        })

# 3. Inspeccionar población de 6 a 17 años
edad = pd.to_numeric(
    personas["s01a_03"],
    errors="coerce"
)

poblacion = personas[edad.between(6, 17)]

print("\n=== POBLACIÓN DE 6 A 17 AÑOS ===")
print("Registros:", len(poblacion))

for variable in ["cmasi", "s03a_04", "s03b_10"]:
    print(f"\n{variable}")
    print(
        poblacion[variable].value_counts(
            dropna=False
        )
    )

# 4. Exportar frecuencias agregadas
pd.DataFrame(resultados).to_csv(
    OUT / "categorias_variables.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nInspección completada.")
print("Archivo: data/metadata/categorias_variables.csv")
