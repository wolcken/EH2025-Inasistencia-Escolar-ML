
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score

from features import load_features

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs" / "models"
FIGURES = ROOT / "outputs" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

LABELS = {
    "s01a_03": "Edad",
    "s01a_02": "Sexo",
    "depto": "Departamento",
    "area": "Área urbana/rural",
    "p0": "Pobreza",
    "pext0": "Pobreza extrema",
    "tipohogar": "Tipo de hogar",
    "ingreso_log": "Ingreso per cápita (log)",
}

# Recuperar exactamente la partición utilizada
# durante el entrenamiento.
X, y, groups = load_features()

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    cv.split(X, y, groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

modelo = joblib.load(
    MODELS / "mejor_modelo.joblib"
)

print("=== EXPLAINABILITY ===")
print("Observaciones de prueba:", len(X_test))
print("Casos positivos:", int(y_test.sum()))

# Desempeño original
prob = modelo.predict_proba(X_test)[:, 1]

ap_original = average_precision_score(
    y_test, prob
)

print(f"Average Precision original: {ap_original:.4f}")

# Importancia mediante permutaciones
resultado = permutation_importance(
    modelo,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=1,
)

importancia = pd.DataFrame({
    "variable": X_test.columns,
    "importancia_media": resultado.importances_mean,
    "desviacion": resultado.importances_std,
})

importancia["descripcion"] = (
    importancia["variable"].map(LABELS)
)

importancia = importancia.sort_values(
    "importancia_media",
    ascending=False,
)

# Exportar resultados agregados
destino_csv = (
    MODELS / "importancia_variables.csv"
)

importancia.to_csv(
    destino_csv,
    index=False,
    encoding="utf-8-sig",
)

# Visualización interactiva
grafico = importancia.sort_values(
    "importancia_media"
)

fig = px.bar(
    grafico,
    x="importancia_media",
    y="descripcion",
    orientation="h",
    error_x="desviacion",
    title="Importancia de variables — Random Forest",
    labels={
        "importancia_media": "Disminución de Average Precision",
        "descripcion": "Variable",
    },
    hover_data=["variable"],
)

fig.add_vline(
    x=0,
    line_dash="dash",
    line_color="gray",
)

fig.update_layout(
    template="plotly_white",
    height=550,
)

destino_html = (
    FIGURES / "09_importancia_variables.html"
)

fig.write_html(
    destino_html,
    include_plotlyjs=True,
)

print("\n=== IMPORTANCIA DE VARIABLES ===")
print(importancia.to_string(index=False))

print("\nArchivos generados:")
print(destino_csv)
print(destino_html)
