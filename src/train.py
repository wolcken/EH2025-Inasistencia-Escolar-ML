
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import (
    load_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS = OUT / "models"
FIGURES = OUT / "figures"

MODELS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
THRESHOLD = 0.50


def build_pipeline(model):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def model_factory(name):
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=6,
            min_child_samples=30,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbosity=-1,
        )

    raise ValueError(f"Modelo desconocido: {name}")


def metrics(y_true, prob):
    pred = (prob >= THRESHOLD).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(y_true, prob)
        ),
        "average_precision": float(
            average_precision_score(y_true, prob)
        ),
        "f1": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "precision": float(
            precision_score(
                y_true, pred, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, pred, zero_division=0
            )
        ),
    }


def validate(model_name, X, y, groups):
    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups),
        start=1,
    ):
        assert set(groups.iloc[train_idx]).isdisjoint(
            set(groups.iloc[val_idx])
        )

        y_train_fold = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if y_train_fold.nunique() != 2 or y_val.nunique() != 2:
            raise ValueError(
                f"Fold {fold}: falta una clase."
            )

        pipeline = build_pipeline(
            model_factory(model_name)
        )

        pipeline.fit(
            X.iloc[train_idx],
            y_train_fold,
        )

        prob = pipeline.predict_proba(
            X.iloc[val_idx]
        )[:, 1]

        result = metrics(y_val, prob)
        result.update({
            "modelo": model_name,
            "fold": fold,
            "n_validacion": len(val_idx),
            "positivos_validacion": int(y_val.sum()),
        })
        rows.append(result)

        print(
            f"  Fold {fold}: "
            f"AP={result['average_precision']:.4f}, "
            f"AUC={result['roc_auc']:.4f}, "
            f"F1={result['f1']:.4f}"
        )

    return rows


def save_figures(y_test, predictions):
    # Curvas ROC
    roc_fig = go.Figure()

    for name, prob in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)

        roc_fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{name} (AUC={auc:.3f})",
        ))

    roc_fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Clasificador aleatorio",
        line=dict(dash="dash", color="gray"),
    ))

    roc_fig.update_layout(
        title="Comparación ROC — EH2025",
        xaxis_title="Tasa de falsos positivos",
        yaxis_title="Sensibilidad",
        template="plotly_white",
    )

    roc_fig.write_html(
        FIGURES / "06_comparacion_roc.html",
        include_plotlyjs=True,
    )

    # Precision-Recall
    pr_fig = go.Figure()

    for name, prob in predictions.items():
        precision, recall, _ = precision_recall_curve(
            y_test, prob
        )
        ap = average_precision_score(y_test, prob)

        pr_fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"{name} (AP={ap:.3f})",
        ))

    prevalence = float(y_test.mean())

    pr_fig.add_hline(
        y=prevalence,
        line_dash="dash",
        annotation_text="Prevalencia de prueba",
    )

    pr_fig.update_layout(
        title="Precision-Recall — EH2025",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
    )

    pr_fig.write_html(
        FIGURES / "07_precision_recall.html",
        include_plotlyjs=True,
    )

    # Matrices de confusión
    for name, prob in predictions.items():
        pred = (prob >= THRESHOLD).astype(int)

        cm = confusion_matrix(
            y_test, pred, labels=[0, 1]
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Asiste", "Inasistencia"],
            y=["Asiste", "Inasistencia"],
            labels={
                "x": "Predicción",
                "y": "Valor real",
                "color": "Casos",
            },
            title=f"Matriz de confusión — {name}",
            color_continuous_scale="Blues",
        )

        fig.write_html(
            FIGURES / (
                "08_matriz_"
                + name.lower().replace(" ", "_")
                + ".html"
            ),
            include_plotlyjs=True,
        )


def main():
    X, y, groups = load_features()

    print("\n=== DISTRIBUCIÓN ORIGINAL ===")
    print(y.value_counts().sort_index())

    # Partición final independiente.
    outer_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Primera partición prefijada:
    # no elegimos la que obtiene mejores resultados.
    train_idx, test_idx = next(
        outer_cv.split(X, y, groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    assert set(groups_train).isdisjoint(
        set(groups_test)
    )

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "La partición no contiene ambas clases."
        )

    print("\n=== PARTICIÓN POR HOGARES ===")
    print(
        f"Entrenamiento: {len(X_train)} registros, "
        f"{int(y_train.sum())} positivos"
    )
    print(
        f"Prueba: {len(X_test)} registros, "
        f"{int(y_test.sum())} positivos"
    )

    names = ["Random Forest", "LightGBM"]
    cv_rows = []

    for name in names:
        print(f"\n=== VALIDACIÓN: {name} ===")
        cv_rows.extend(
            validate(
                name,
                X_train,
                y_train,
                groups_train,
            )
        )

    cv_df = pd.DataFrame(cv_rows)

    cv_df.to_csv(
        MODELS / "validacion_cruzada.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cv_summary = (
        cv_df.groupby("modelo", as_index=False)
        .agg(
            ap_media=("average_precision", "mean"),
            ap_desviacion=("average_precision", "std"),
            auc_media=("roc_auc", "mean"),
            f1_medio=("f1", "mean"),
            recall_medio=("recall", "mean"),
        )
        .sort_values("ap_media", ascending=False)
    )

    cv_summary.to_csv(
        MODELS / "resumen_validacion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_name = cv_summary.iloc[0]["modelo"]

    print("\n=== COMPARACIÓN VALIDACIÓN ===")
    print(cv_summary.to_string(index=False))
    print("\nSeleccionado por AP:", best_name)

    # Entrenar ambos modelos en toda la partición
    # de entrenamiento, sin tocar datos de prueba.
    predictions = {}
    test_rows = []

    for name in names:
        pipeline = build_pipeline(model_factory(name))
        pipeline.fit(X_train, y_train)

        prob = pipeline.predict_proba(X_test)[:, 1]
        predictions[name] = prob

        result = metrics(y_test, prob)
        result["modelo"] = name
        result["n_test"] = len(y_test)
        result["positivos_test"] = int(y_test.sum())

        test_rows.append(result)

        # Guardar solo el modelo seleccionado
        if name == best_name:
            joblib.dump(
                pipeline,
                MODELS / "mejor_modelo.joblib",
            )

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(
        MODELS / "metricas_prueba.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_figures(y_test, predictions)

    metadata = {
        "modelo_seleccionado": best_name,
        "criterio_seleccion": "average_precision_cv",
        "umbral": THRESHOLD,
        "random_state": RANDOM_STATE,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_positive": int(y_train.sum()),
        "test_positive": int(y_test.sum()),
        "train_households": int(groups_train.nunique()),
        "test_households": int(groups_test.nunique()),
        "test_prevalence": float(y_test.mean()),
        "features": list(X.columns),
    }

    (MODELS / "experimento.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== MÉTRICAS DE PRUEBA ===")
    print(test_df.to_string(index=False))

    print("\n=== EXPERIMENTO FINALIZADO ===")
    print("Modelo seleccionado:", best_name)
    print("Resultados guardados en:", MODELS)
    print("Gráficos guardados en:", FIGURES)


if __name__ == "__main__":
    main()
