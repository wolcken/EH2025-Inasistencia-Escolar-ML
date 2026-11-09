
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
METADATA_DIR = ROOT / "data" / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)

summary = []
variables = []

files = sorted(RAW_DIR.glob("*.sav"))

if not files:
    raise FileNotFoundError(
        f"No se encontraron archivos .sav en {RAW_DIR}"
    )

for file in files:
    print(f"\nAnalizando: {file.name}")

    # Leer únicamente metadatos, sin cargar los registros.
    _, meta = pyreadstat.read_sav(
        str(file),
        metadataonly=True
    )

    summary.append({
        "archivo": file.name,
        "registros": meta.number_rows,
        "variables": len(meta.column_names),
        "tamano_mb": round(
            file.stat().st_size / (1024 ** 2), 2
        ),
    })

    for name, label in zip(
        meta.column_names,
        meta.column_labels
    ):
        variables.append({
            "archivo": file.name,
            "variable": name,
            "descripcion": label,
            "tipo": meta.readstat_variable_types.get(name),
        })

    print(
        f"Registros: {meta.number_rows} | "
        f"Variables: {len(meta.column_names)}"
    )

pd.DataFrame(summary).to_csv(
    METADATA_DIR / "resumen_archivos.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(variables).to_csv(
    METADATA_DIR / "diccionario_variables.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAuditoría completada correctamente.")
print(f"Archivos generados en: {METADATA_DIR}")
