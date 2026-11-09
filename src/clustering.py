
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/dataset_escolar.parquet"
OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
df = con.execute(
    """
    SELECT
        s01a_03 AS edad,
        area,
        yh ogpc,
        inasistencia_escolar,
        factor
    FROM read_parquet(?)
    """.replace("yh ogpc", "yhogpc"),
    [str(DATA)]
).df()
con.close()

# Preparar variables para el clustering
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
df["area"] = pd.to_numeric(df["area"], errors="coerce")
df["ingreso"] = pd.to_numeric(
    df["yhogpc"], errors="coerce"
)

# El ingreso es asimétrico: transformación logarítmica.
df["ingreso_log"] = np.log1p(
    df["ingreso"].clip(lower=0)
)

# Variables categóricas: representación binaria
df["rural"] = df["area"].map({1: 0, 2: 1})

variables = ["edad", "ingreso_log", "rural"]

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(df[variables])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Comparar varios números de clusters
resultados = []

for k in range(2, 7):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled, labels,
        sample_size=min(3000, len(X_scaled)),
        random_state=42
    )

    resultados.append((k, score))

mejor_k = max(resultados, key=lambda x: x[1])[0]

print("Evaluación silhouette:")
for k, score in resultados:
    print(f"K={k}: {score:.4f}")

print(f"\nNúmero seleccionado de clusters: {mejor_k}")

# Entrenar modelo final
modelo = KMeans(
    n_clusters=mejor_k,
    random_state=42,
    n_init=20
)

df["cluster"] = modelo.fit_predict(X_scaled)

# Resumir perfiles utilizando los pesos de encuesta
df["casos_ponderados"] = (
    df["inasistencia_escolar"] * df["factor"]
)

resumen = df.groupby("cluster").agg(
    muestra=("cluster", "size"),
    edad_promedio=("edad", "mean"),
    ingreso_mediano=("ingreso", "median"),
    porcentaje_rural=("rural", "mean"),
    casos=("inasistencia_escolar", "sum"),
    poblacion_ponderada=("factor", "sum"),
    casos_ponderados=("casos_ponderados", "sum")
).reset_index()

resumen["porcentaje_rural"] *= 100
resumen["tasa_inasistencia"] = (
    100 * resumen["casos_ponderados"]
    / resumen["poblacion_ponderada"]
)

resumen.to_csv(
    OUT / "resumen_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

# Visualización interactiva de perfiles
fig = px.scatter(
    resumen,
    x="edad_promedio",
    y="ingreso_mediano",
    size="muestra",
    color="tasa_inasistencia",
    text="cluster",
    hover_data=[
        "porcentaje_rural",
        "casos",
        "tasa_inasistencia"
    ],
    title="Perfiles socioeconómicos mediante K-Means",
    labels={
        "edad_promedio": "Edad promedio (años)",
        "ingreso_mediano": "Ingreso per cápita mediano",
        "tasa_inasistencia": "Inasistencia ponderada (%)",
        "cluster": "Cluster"
    },
    color_continuous_scale="Viridis"
)

fig.update_traces(textposition="top center")
fig.update_layout(template="plotly_white")

fig.write_html(
    OUT / "04_clustering_perfiles.html",
    include_plotlyjs=True
)

print("\n=== PERFILES IDENTIFICADOS ===")
print(resumen.to_string(index=False))
print("\nVisualización generada correctamente.")
