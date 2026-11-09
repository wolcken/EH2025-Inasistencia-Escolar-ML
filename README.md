# Inasistencia escolar en Bolivia — EH2025

**Análisis exploratorio, visualización geoespacial, segmentación y aprendizaje automático**

**Autor:** Alfredo Ramos  
**Institución:** Universidad Pública de El Alto (UPEA)  
**Programa:** Maestría en Inteligencia Artificial, Machine Learning y Data Science  
**Módulo:** Minería de Datos y Visualización en la Toma de Decisiones  
**Gestión:** 2026

## Descripción

Este proyecto analiza la **inasistencia escolar de personas de 6 a 17 años en Bolivia** con microdatos de la **Encuesta de Hogares 2025 (EH2025)** del Instituto Nacional de Estadística (INE). Integra auditoría de fuentes, preparación con DuckDB, análisis exploratorio (EDA), visualizaciones interactivas en Plotly, cartografía departamental, clustering K-Means, clasificación mediante Random Forest y LightGBM, e interpretabilidad mediante *Permutation Importance*.

**Pregunta de investigación:** ¿Qué características demográficas, territoriales y socioeconómicas se asocian con la inasistencia escolar de personas de 6 a 17 años en Bolivia y qué desempeño alcanzan los modelos predictivos?

> **Precaución conceptual:** la variable utilizada reúne categorías de no matriculación y no asistencia. Al tratarse de una encuesta transversal, no equivale a una medición longitudinal de abandono o deserción escolar.

## Resultados principales

| Indicador | Resultado |
|---|---:|
| Observaciones de 6 a 17 años | 9.201 |
| Hogares distintos | 5.437 |
| Casos positivos de inasistencia | 208 |
| Proporción muestral **no ponderada** | 2,26 % |
| Tasa nacional **ponderada** | 2,82 % |
| Tasa urbana ponderada | 1,49 % |
| Tasa rural ponderada | 5,61 % |
| Clusters K-Means seleccionados | 2 (Silhouette: 0,4528) |
| Modelo seleccionado por validación cruzada | Random Forest |
| Random Forest: AUC-ROC en prueba | 0,7815 |
| Random Forest: Average Precision en prueba | 0,0976 |
| Random Forest: Precision / Recall en prueba | 0,1027 / 0,4524 |

Las tasas descriptivas usan el **factor de expansión** de la encuesta; las métricas de los clasificadores corresponden a una evaluación muestral, sin ponderación del diseño de encuesta.

### Diferencias territoriales y por edad

- Inasistencia estimada en área rural: **5,61 %**, frente a **1,49 %** en el área urbana.
- Tasas puntuales departamentales más altas: **Pando (5,67 %)**, **Santa Cruz (4,17 %)** y **Chuquisaca (3,90 %)**.
- A los **17 años**, la tasa puntual ponderada estimada alcanza **9,41 %**.

Son comparaciones **descriptivas**, sin intervalos de confianza ajustados por estrato y conglomerado; no deben interpretarse como diferencias estadísticamente confirmadas.

## Fuente de datos

**INE Bolivia — Encuesta de Hogares 2025:** <https://anda.ine.gob.bo/index.php/catalog/256>

El archivo original obtenido del catálogo contiene siete módulos SPSS (`.sav`):

- `EH2025_Persona.sav`
- `EH2025_Vivienda_1.sav`
- `EH2025_Equipamiento.sav`
- `EH2025_Seguridad_Alimentaria.sav`
- `EH2025_Gastos_Alimentarios.sav`
- `EH2025_Gastos_No_Alimentarios.sav`
- `EH2025_Discriminacion.sav`

El conjunto analítico principal integra los módulos de **personas** y **viviendas** mediante `folio`; la clave individual de personas es `folio` + `nro`.

**Los microdatos originales y las bases individuales derivadas no se incluyen en este repositorio.** Deben obtenerse directamente del INE y utilizarse de conformidad con sus condiciones de acceso.

## Metodología

### Preparación y definición del indicador

1. Auditoría de los módulos SPSS mediante `pyreadstat` y extracción de metadatos.
2. Conversión a Parquet y consultas SQL con DuckDB.
3. Integración de personas y viviendas, verificación de claves y filtrado de edades **6–17**.
4. Construcción de la variable binaria `inasistencia_escolar`:

| `cmasi` | Categoría | Objetivo |
|---:|---|---:|
| 1 | No matriculado | 1 |
| 2 | Asiste | 0 |
| 3 | No asiste | 1 |

La tasa descriptiva ponderada se calcula como:

\[
\widehat{p}(\%) = 100\,\frac{\sum_i w_i y_i}{\sum_i w_i}
\]

con `w_i = factor` y `y_i = inasistencia_escolar`.

### Exploración y visualización

Se elaboraron un reporte EDA automatizado, tasas por departamento, área y edad, un mapa coroplético de Bolivia y gráficos de resultados predictivos. La geometría departamental del mapa procede del **servicio cartográfico público de YPFB** y los indicadores educativos se calcularon a partir de la EH2025.

### Segmentación K-Means

Se aplicó K-Means con edad, ingreso per cápita transformado y condición urbana/rural. La comparación de `K=2..6` mediante Silhouette seleccionó **dos clusters**:

| Indicador | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Observaciones | 6.877 | 2.324 |
| Edad promedio | 11,69 | 11,45 |
| Ingreso per cápita mediano | 1.247,26 | 679,17 |
| Proporción rural | 0,00 % | 99,74 % |
| Tasa ponderada de inasistencia | 1,49 % | 5,60 % |

La separación está **dominada por el área urbana/rural**: los clusters no representan tipologías sociales descubiertas independientemente del territorio.

### Modelos supervisados

Se compararon **Random Forest** y **LightGBM** con ocho predictores: edad, sexo, departamento, área, ingreso per cápita transformado, pobreza, pobreza extrema y tipo de hogar.

Se excluyeron de las características los identificadores y las variables educativas que revelan el objetivo. Las transformaciones y la imputación se ajustaron **dentro de pipelines**. La división por hogares evitó que miembros de un mismo hogar aparecieran en entrenamiento y prueba.

- Entrenamiento: **7.363 registros**, 166 positivos.
- Prueba independiente: **1.838 registros**, 42 positivos.
- Selección del algoritmo: **Average Precision media de cuatro folds** de `StratifiedGroupKFold`, únicamente sobre entrenamiento.

**Validación cruzada (promedios):**

| Métrica | Random Forest | LightGBM |
|---|---:|---:|
| AUC-ROC | **0,7955** | 0,7612 |
| Average Precision | **0,1465** | 0,1228 |
| F1 | **0,1658** | 0,1373 |
| Recall | 0,4345 | **0,4829** |

**Prueba independiente:**

| Métrica | Random Forest | LightGBM |
|---|---:|---:|
| AUC-ROC | **0,7815** | 0,7431 |
| Average Precision | **0,0976** | 0,0972 |
| Precision | **0,1027** | 0,0649 |
| Recall | **0,4524** | 0,4048 |
| F1 | **0,1674** | 0,1118 |

Random Forest fue seleccionado por su mayor Average Precision media durante la validación. Aunque supera la prevalencia positiva del conjunto de prueba (aprox. **2,29 %**) como referencia de Average Precision, presenta **baja precisión en las predicciones positivas** y no es adecuado para decisiones individuales automatizadas.

### Interpretabilidad

Se aplicó *Permutation Importance* usando Average Precision como función de evaluación, sin reentrenar los modelos.

| Variable | Importancia media |
|---|---:|
| Edad | +0,0581 |
| Área urbana/rural | +0,0318 |
| Departamento | +0,0315 |
| Tipo de hogar | +0,0286 |
| Sexo | +0,0064 |
| Pobreza | −0,0018 |
| Pobreza extrema | −0,0063 |
| Ingreso per cápita (log) | −0,0202 |

La importancia expresa contribución **predictiva**, no causalidad. Las importancias negativas no significan que una característica sea protectora.

## Estructura del repositorio

```text
.
├── data/
│   ├── raw/                 # Microdatos originales — excluidos de Git
│   ├── processed/           # Datos individuales procesados — excluidos de Git
│   └── metadata/            # Diccionarios y metadatos revisados
├── src/                     # Scripts de auditoría, EDA, gráficos, ML y XAI
├── notebooks/
│   ├── 01_exploracion_eda.ipynb
│   ├── 02_visualizacion_clustering.ipynb
│   └── 03_modelos_xai.ipynb
├── outputs/
│   ├── eda/                 # Reporte exploratorio
│   ├── figures/             # HTML Plotly y tablas agregadas
│   └── models/              # Métricas y metadatos; sin modelos serializados
├── docs/
│   ├── reporte_final.qmd
│   ├── reporte_final.html
│   └── styles.css
├── .gitignore
├── requirements.txt
└── README.md
```

Los archivos que se distribuyan públicamente deben revisarse para confirmar que no incorporan **registros individuales, identificadores o salidas de notebooks sensibles**.

## Instalación y reproducción

### Requisitos

- Python **3.14.3**, versión utilizada en este experimento.
- Quarto **1.11.9**, para generar el informe HTML.
- Git; VS Code con extensiones Python/Jupyter (opcional).
- Dependencias indicadas en `requirements.txt` (verificar sus versiones fijadas antes de replicar resultados).

### Crear entorno en Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Obtener los microdatos

Descargar los archivos SPSS desde el [catálogo oficial del INE](https://anda.ine.gob.bo/index.php/catalog/256) y colocarlos en `data/raw/`. El repositorio público no contiene estos archivos.

### Ejecutar el flujo

Desde la raíz del proyecto, en este orden:

```powershell
# Auditoría y preparación
python src/audit_data.py
python src/inspect_variables.py
python src/validate_school_population.py
python src/build_dataset.py
python src/generate_eda.py

# Visualizaciones y clustering
python src/visualize.py
python src/clustering.py
python src/map_bolivia.py

# Modelos e interpretabilidad
python src/features.py
python src/train.py
python src/explain.py

# Informe Quarto
quarto render docs/reporte_final.qmd --to html
```

**Notebooks:** abrir `notebooks/` en VS Code, seleccionar `.venv` como kernel y ejecutar los archivos numerados en orden. Reutilizan los resultados generados por los scripts y no vuelven a entrenar los clasificadores por defecto.

**Informe interactivo:** `docs/reporte_final.html`. Sus gráficos se cargan mediante rutas relativas a `outputs/figures/`, por lo que para visualizarlo es necesario conservar la estructura de carpetas. La exportación a PDF requeriría sustituir los iframes por imágenes estáticas y comprobar la paginación.

## Recomendaciones de políticas públicas

1. **Acompañamiento en la adolescencia:** evaluar estrategias de seguimiento de estudiantes de 14 a 17 años durante las transiciones educativas.
2. **Diagnóstico de barreras rurales:** investigar transporte, distancias y disponibilidad de servicios antes de diseñar intervenciones territoriales.
3. **Apoyo familiar contextualizado:** estudiar necesidades de acompañamiento de distintos hogares sin atribuir causalidad ni estigmatizar configuraciones familiares.

Estas recomendaciones son hipótesis para evaluación e intervención piloto; **no** constituyen evidencia de efectividad de políticas específicas.

## Limitaciones y uso responsable

- La encuesta es **transversal** y no acredita abandono escolar definitivo.
- La muestra analítica cuenta con **208 casos positivos**, de los cuales solo **42** están en la partición de prueba.
- No se calcularon intervalos de confianza ajustados al diseño complejo de la encuesta (`upm`, `estrato` y factor de expansión).
- Las tasas departamentales pueden presentar alta incertidumbre por la escasez de casos positivos.
- Los modelos no fueron entrenados ni evaluados usando el diseño muestral de la encuesta.
- La validación por hogares reduce fugas entre individuos del mismo hogar, pero no demuestra generalización temporal o territorial.
- Las asociaciones y las importancias predictivas **no establecen causalidad**.
- El clasificador no debe utilizarse para etiquetar estudiantes o automatizar decisiones de acceso a servicios.

## Referencias

- **Instituto Nacional de Estadística (INE), Bolivia.** *Encuesta de Hogares 2025*. <https://anda.ine.gob.bo/index.php/catalog/256>.
- **YPFB.** Servicio cartográfico público de límites administrativos departamentales de Bolivia (geometrías utilizadas para el mapa).
- Herramientas de análisis: Python, Pyreadstat, DuckDB, Pandas, Plotly, Scikit-learn, LightGBM y Quarto.

---

**Alfredo Ramos · Universidad Pública de El Alto (UPEA) · 2026**
