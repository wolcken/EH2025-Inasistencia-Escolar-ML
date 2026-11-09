
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dataset_escolar.parquet"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

df = con.execute(
    """
    SELECT
        depto,
        area,
        s01a_03 AS edad,
        inasistencia_escolar AS objetivo,
        factor
    FROM read_parquet(?)
    """,
    [str(DATA)]
).df()

con.close()

# Validar pesos y variable objetivo
df = df.dropna(
    subset=["depto", "area", "edad", "objetivo", "factor"]
).copy()

df = df[df["factor"] > 0].copy()

df["depto"] = df["depto"].astype(int)
df["area"] = df["area"].astype(int)
df["edad"] = df["edad"].astype(int)

DEPARTAMENTOS = {
    1: "Chuquisaca",
    2: "La Paz",
    3: "Cochabamba",
    4: "Oruro",
    5: "Potosí",
    6: "Tarija",
    7: "Santa Cruz",
    8: "Beni",
    9: "Pando",
}

AREAS = {
    1: "Urbana",
    2: "Rural",
}

df["departamento"] = df["depto"].map(DEPARTAMENTOS)
df["tipo_area"] = df["area"].map(AREAS)

# Población ponderada con condición de inasistencia
df["casos_ponderados"] = df["factor"] * df["objetivo"]


def tasa_ponderada(datos, grupo):
    resumen = (
        datos.groupby(grupo, as_index=False)
        .agg(
            muestra=("objetivo", "size"),
            casos=("objetivo", "sum"),
            poblacion_ponderada=("factor", "sum"),
            casos_ponderados=("casos_ponderados", "sum"),
        )
    )

    resumen["tasa"] = (
        resumen["casos_ponderados"]
        / resumen["poblacion_ponderada"]
        * 100
    )

    return resumen


# 1. Comparación departamental
por_depto = tasa_ponderada(df, "departamento")
por_depto = por_depto.sort_values("tasa")

fig1 = px.bar(
    por_depto,
    x="tasa",
    y="departamento",
    orientation="h",
    hover_data=["muestra", "casos"],
    title="Inasistencia escolar por departamento",
    labels={
        "tasa": "Porcentaje ponderado (%)",
        "departamento": "Departamento",
    },
)

fig1.update_layout(template="plotly_white")
fig1.write_html(
    OUT / "01_inasistencia_departamento.html",
    include_plotlyjs=True,
)

# 2. Comparación urbana y rural
por_area = tasa_ponderada(df, "tipo_area")

fig2 = px.bar(
    por_area,
    x="tipo_area",
    y="tasa",
    hover_data=["muestra", "casos"],
    title="Inasistencia escolar según área",
    labels={
        "tipo_area": "Área",
        "tasa": "Porcentaje ponderado (%)",
    },
)

fig2.update_layout(template="plotly_white")
fig2.write_html(
    OUT / "02_inasistencia_area.html",
    include_plotlyjs=True,
)

# 3. Distribución según edad
por_edad = tasa_ponderada(df, "edad")

fig3 = px.line(
    por_edad,
    x="edad",
    y="tasa",
    markers=True,
    hover_data=["muestra", "casos"],
    title="Inasistencia escolar según edad",
    labels={
        "edad": "Edad (años)",
        "tasa": "Porcentaje ponderado (%)",
    },
)

fig3.update_layout(template="plotly_white")
fig3.write_html(
    OUT / "03_inasistencia_edad.html",
    include_plotlyjs=True,
)

# Guardar tablas agregadas, sin registros individuales
por_depto.to_csv(
    OUT / "resumen_departamentos.csv",
    index=False,
    encoding="utf-8-sig",
)

por_area.to_csv(
    OUT / "resumen_areas.csv",
    index=False,
    encoding="utf-8-sig",
)

por_edad.to_csv(
    OUT / "resumen_edades.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n=== VISUALIZACIONES GENERADAS ===")
print("01_inasistencia_departamento.html")
print("02_inasistencia_area.html")
print("03_inasistencia_edad.html")

print("\n=== RESUMEN DEPARTAMENTAL ===")
print(
    por_depto[
        ["departamento", "muestra", "casos", "tasa"]
    ].to_string(index=False)
)

print("\nVisualizaciones generadas correctamente.")
