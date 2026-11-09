
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT / "data" / "processed"
    / "dataset_escolar.parquet"
)

METADATA = ROOT / "data" / "metadata"

TARGET = "inasistencia_escolar"
GROUP = "folio"

NUMERIC_FEATURES = [
    "s01a_03",
    "ingreso_log",
]

CATEGORICAL_FEATURES = [
    "s01a_02",
    "depto",
    "area",
    "p0",
    "pext0",
    "tipohogar",
]

RAW_FEATURES = [
    "s01a_03",
    "s01a_02",
    "depto",
    "area",
    "yhogpc",
    "p0",
    "pext0",
    "tipohogar",
]


def load_features():
    """Carga y prepara variables sin entrenar modelos."""

    con = duckdb.connect()

    try:
        df = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(DATA)]
        ).df()
    finally:
        con.close()

    required = RAW_FEATURES + [TARGET, GROUP]

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"Faltan variables: {missing}"
        )

    if df[required].isna().any(axis=None):
        print(
            "Aviso: existen valores faltantes; "
            "se imputarán dentro del pipeline de ML."
        )

    # Comprobar variable objetivo
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            "La variable objetivo debe ser binaria (0/1)."
        )

    if df[GROUP].isna().any():
        raise ValueError(
            "Existen hogares sin identificador."
        )

    # Evitar duplicaciones
    if df.duplicated(["folio", "nro"]).any():
        raise ValueError(
            "Se encontraron personas duplicadas."
        )

    # Preparar características
    X = df[RAW_FEATURES].copy()

    # Tratamiento del ingreso
    ingreso = pd.to_numeric(
        X["yhogpc"], errors="coerce"
    )

    # Los ingresos negativos se marcan como faltantes
    ingreso = ingreso.where(ingreso >= 0)

    X["ingreso_log"] = np.log1p(ingreso)

    X = X.drop(columns=["yhogpc"])

    # Edad
    X["s01a_03"] = pd.to_numeric(
        X["s01a_03"], errors="coerce"
    )

    # Variables categóricas:
    # conservar códigos como categorías
    for col in CATEGORICAL_FEATURES:
        X[col] = (
            pd.to_numeric(X[col], errors="coerce")
            .astype("Int64")
            .astype("string")
            .astype(object)
        )

        X[col] = X[col].where(
            pd.notna(X[col]), np.nan
        )

    y = df[TARGET].astype(int)

    groups = df[GROUP].astype(str)

    return X, y, groups


if __name__ == "__main__":

    X, y, groups = load_features()

    print("\n=== DATASET PARA MACHINE LEARNING ===")

    print("Observaciones:", len(X))
    print("Variables predictoras:", X.shape[1])
    print("Hogares:", groups.nunique())

    print("\nDistribución de clases:")
    print(y.value_counts().sort_index())

    print("\nPorcentaje de inasistencia:")
    print(f"{y.mean() * 100:.2f}%")

    print("\nVariables:")
    print(X.dtypes)

    print("\nValores faltantes:")
    print(X.isna().sum())

    METADATA.mkdir(parents=True, exist_ok=True)

    resumen = {
        "observaciones": int(len(X)),
        "variables": list(X.columns),
        "hogares": int(groups.nunique()),
        "clase_negativa": int((y == 0).sum()),
        "clase_positiva": int((y == 1).sum()),
        "porcentaje_positivo": round(
            float(y.mean() * 100), 4
        ),
        "valores_faltantes": {
            k: int(v)
            for k, v in X.isna().sum().items()
        },
    }

    destino = METADATA / "resumen_features.json"

    destino.write_text(
        json.dumps(
            resumen,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nPreparación completada.")
    print("Resumen:", destino)
