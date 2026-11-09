# Inasistencia escolar en Bolivia — EH2025

**Análisis exploratorio, visualización geoespacial, segmentación y aprendizaje automático**

![Python](https://img.shields.io/badge/Python-3.14.3-3776AB?logo=python&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-SQL-FFF000)
![Plotly](https://img.shields.io/badge/Plotly-Visualizaci%C3%B3n%20interactiva-3F4F75?logo=plotly&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Clasificaci%C3%B3n-2E8B57)
![Quarto](https://img.shields.io/badge/Quarto-HTML5-39729E)
![Estado](https://img.shields.io/badge/Estado-An%C3%A1lisis%20completado-success)

**Autor:** Alfredo Ramos  
**Institución:** Universidad Pública de El Alto (UPEA)  
**Programa:** Maestría en Inteligencia Artificial, Machine Learning y Data Science  
**Módulo:** Minería de Datos y Visualización en la Toma de Decisiones  
**Gestión:** 2026

> **Entrega académica:** el repositorio público contiene el informe final, las evidencias exploratorias, las visualizaciones, los notebooks y el código. **No es necesario ejecutar Python para consultar los resultados ya exportados.** Los archivos HTML interactivos pueden descargarse y abrirse en el navegador; la vista previa de GitHub no ejecuta necesariamente su JavaScript.

## Acceso directo a los entregables

| Entregable | Archivo o carpeta | Finalidad |
|---|---|---|
| **Informe académico final (HTML5)** | [docs/reporte_final.html](docs/reporte_final.html) | Storytelling, metodología, resultados, interpretación y recomendaciones |
| **Reporte exploratorio automatizado (HTML)** | [outputs/eda/reporte_eda_eh2025.html](outputs/eda/reporte_eda_eh2025.html) | Evidencia de exploración y calidad de datos; publicar solo tras revisión de privacidad |
| **Visualizaciones interactivas** | [outputs/figures/](outputs/figures/) | Gráficos por departamento, área y edad, mapa, clustering, ROC/PR y XAI |
| **Notebooks Jupyter** | [notebooks/](notebooks/) | Exploración, visualización y evaluación de modelos |
| **Código modular** | [src/](src/) | Procesamiento, análisis y entrenamiento reproducibles |
| **Resultados agregados y métricas** | [outputs/](outputs/) | Tablas y métricas exportadas |

**Para leer el informe con todos sus gráficos:** descarga el repositorio completo como ZIP (GitHub → **Code → Download ZIP**), descomprímelo sin alterar las carpetas y abre `docs/reporte_final.html` en un navegador. Las visualizaciones se enlazan mediante rutas relativas a `outputs/figures/`; por ello, abrir el HTML aislado puede dejar los gráficos sin cargar.

## Descripción y pregunta de investigación

Este proyecto analiza la **inasistencia escolar de personas de 6 a 17 años en Bolivia** con microdatos de la **Encuesta de Hogares 2025 (EH2025)** del Instituto Nacional de Estadística (INE). Integra auditoría de fuentes, preparación con DuckDB, análisis exploratorio (EDA), visualizaciones interactivas Plotly, cartografía departamental, clustering K-Means, clasificación con Random Forest y LightGBM e interpretabilidad mediante *Permutation Importance*.

**Pregunta de investigación:** ¿Qué características demográficas, territoriales y socioeconómicas se asocian con la inasistencia escolar de personas de 6 a 17 años en Bolivia y qué desempeño alcanzan los modelos predictivos?

> **Precaución conceptual:** el indicador reúne categorías de no matriculación y no asistencia. Al tratarse de una encuesta transversal, **no equivale a una medición longitudinal de abandono escolar definitivo**.

## Resultados principales

| Indicador | Resultado |
|---|---:|
| Observaciones de 6 a 17 años | 9.201 |
| Hogares distintos | 5.437 |
| Casos positivos de inasistencia | 208 |
| Proporción muestral **no ponderada** | 2,26 % |
| Tasa nacional **ponderada** estimada | 2,82 % |
| Tasa urbana ponderada | 1,49 % |
| Tasa rural ponderada | 5,61 % |
| Clusters K-Means seleccionados | 2 (Silhouette: 0,4528) |
| Modelo seleccionado por validación cruzada | Random Forest |
| Random Forest: AUC-ROC en prueba | 0,7815 |
| Random Forest: Average Precision en prueba | 0,0976 |
| Random Forest: Precision / Recall en prueba | 0,1027 / 0,4524 |

Las tasas descriptivas usan el **factor de expansión** de la encuesta; las métricas de los clasificadores corresponden a una evaluación muestral **sin ponderación por el diseño de encuesta**.

### Diferencias territoriales y por edad

- Inasistencia rural ponderada estimada: **5,61 %**, frente a **1,49 %** en el área urbana.
- Tasas puntuales departamentales más altas: **Pando (5,67 %)**, **Santa Cruz (4,17 %)** y **Chuquisaca (3,90 %)**.
- A los **17 años**, la estimación puntual de inasistencia alcanza **9,41 %**.

Son comparaciones **descriptivas**, sin intervalos de confianza ajustados por estratos y conglomerados; no deben interpretarse como diferencias estadísticamente confirmadas.

## Fuente de datos y citación obligatoria

**Fuente oficial:** [INE Bolivia — Encuesta de Hogares 2025](https://anda.ine.gob.bo/index.php/catalog/256).

> **Citación exigida por la actividad:** «Fuente: Instituto Nacional de Estadística, Encuesta de Hogares 2025 - http://anda.ine.gob.bo/index.php/catalog/256. Fecha de acceso: **[COMPLETAR CON LA FECHA REAL DE DESCARGA, DD/MM/AAAA]**».

**Antes de entregar en Moodle, sustituir el marcador por la fecha real de descarga** de los microdatos. Utilizar la misma cita en el informe Quarto (`docs/reporte_final.qmd`) y volver a renderizar su versión HTML para que ambos documentos coincidan.

El conjunto original comprende siete módulos SPSS (`.sav`):

- `EH2025_Persona.sav`
- `EH2025_Vivienda_1.sav`
- `EH2025_Equipamiento.sav`
- `EH2025_Seguridad_Alimentaria.sav`
- `EH2025_Gastos_Alimentarios.sav`
- `EH2025_Gastos_No_Alimentarios.sav`
- `EH2025_Discriminacion.sav`

El conjunto analítico principal integra los módulos de **personas** y **viviendas** mediante `folio`; la clave individual es `folio` + `nro`.

**Los microdatos originales y las bases individuales derivadas no se distribuyen en este repositorio.** Deben obtenerse directamente del INE, respetando las condiciones de acceso y uso. Los CSV publicados contienen resultados agregados, métricas o metadatos revisables; no sustituyen el dataset individual necesario para reproducir el entrenamiento completo.

## Metodología

### Preparación y definición del indicador

1. Auditoría de módulos SPSS mediante `pyreadstat` y extracción de metadatos.
2. Conversión a Parquet y consultas SQL con DuckDB.
3. Integración de personas y viviendas, verificación de claves y filtrado de edades **6–17 años**.
4. Construcción de la variable binaria `inasistencia_escolar` a partir de `cmasi`:

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

Se elaboraron un reporte automatizado de EDA, tasas por departamento, área y edad, un mapa coroplético de Bolivia y gráficos de evaluación predictiva. La geometría departamental del mapa procede de un **servicio cartográfico público de YPFB**; los indicadores educativos se calcularon con la EH2025. Se debe respetar la atribución y condiciones de reutilización de la fuente cartográfica.

### Segmentación K-Means

Se utilizó K-Means con edad, ingreso per cápita transformado y condición urbana/rural. La comparación de `K=2..6` mediante Silhouette seleccionó **dos clusters**:

| Indicador | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Observaciones | 6.877 | 2.324 |
| Edad promedio | 11,69 | 11,45 |
| Ingreso per cápita mediano | 1.247,26 | 679,17 |
| Proporción rural | 0,00 % | 99,74 % |
| Tasa ponderada de inasistencia | 1,49 % | 5,60 % |

La separación está **dominada por el área urbana/rural**: los clusters no constituyen tipologías sociales independientes del territorio.

### Clasificación supervisada

Se compararon **Random Forest** y **LightGBM** con ocho predictores: edad, sexo, departamento, área, ingreso per cápita transformado, pobreza, pobreza extrema y tipo de hogar.

Se excluyeron identificadores y variables educativas que revelan el objetivo. Las transformaciones e imputaciones se ajustaron **dentro de pipelines**. La separación por hogares reduce la fuga de información entre individuos pertenecientes a un mismo hogar.

- **Entrenamiento:** 7.363 registros, 166 positivos.
- **Prueba independiente:** 1.838 registros, 42 positivos.
- **Selección del algoritmo:** Average Precision media de cuatro folds de `StratifiedGroupKFold`, únicamente sobre entrenamiento.

**Validación cruzada (promedios)**

| Métrica | Random Forest | LightGBM |
|---|---:|---:|
| AUC-ROC | **0,7955** | 0,7612 |
| Average Precision | **0,1465** | 0,1228 |
| F1 | **0,1658** | 0,1373 |
| Recall | 0,4345 | **0,4829** |

**Prueba independiente**

| Métrica | Random Forest | LightGBM |
|---|---:|---:|
| AUC-ROC | **0,7815** | 0,7431 |
| Average Precision | **0,0976** | 0,0972 |
| Precision | **0,1027** | 0,0649 |
| Recall | **0,4524** | 0,4048 |
| F1 | **0,1674** | 0,1118 |

Random Forest fue seleccionado por la mayor Average Precision media en validación. Aunque su Average Precision en prueba supera la prevalencia positiva (aproximadamente **2,29 %**) como referencia, presenta **baja precisión entre las predicciones positivas** y no es adecuado para decisiones automatizadas sobre personas.

### Interpretabilidad

Se aplicó *Permutation Importance* con Average Precision como función de evaluación, sin reentrenar los modelos.

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

La importancia expresa contribución **predictiva**, no causalidad. Las importancias negativas no significan efectos protectores.

## Estructura del repositorio

```text
Proyecto_EH2025_ML_Caso1/
├── data/
│   ├── raw/                   # Fuentes SPSS locales; excluidas de Git
│   ├── processed/             # Datos individuales procesados; excluidos de Git
│   └── metadata/              # Diccionarios y metadatos revisados
├── notebooks/
│   ├── 01_exploracion_eda.ipynb
│   ├── 02_visualizacion_clustering.ipynb
│   └── 03_modelos_xai.ipynb
├── src/                       # Scripts modulares de auditoría, EDA, ML y XAI
├── outputs/
│   ├── eda/                   # Reporte exploratorio HTML
│   ├── figures/               # Gráficos HTML y tablas agregadas
│   └── models/                # Métricas; sin modelos serializados
├── docs/
│   ├── reporte_final.qmd
│   ├── reporte_final.html
│   ├── reporte_final_files/   # Recursos HTML de Quarto
│   └── styles.css
├── .gitignore
├── requirements.txt
└── README.md
```

**Nota sobre Git:** las carpetas `data/raw/` y `data/processed/` pueden no verse en GitHub porque están excluidas por `.gitignore`. Si se desea que aparezcan, se pueden incluir archivos `.gitkeep` vacíos sin subir los microdatos. La actividad permite fuentes crudas **o scripts de conexión**; los scripts de procesamiento y la referencia oficial documentan el procedimiento de obtención y uso, sin redistribuir los registros.

## Instalación y reproducción

### Requisitos

- **Python 3.14.3**, versión empleada durante el desarrollo.
- **Quarto 1.11.9**, para generar el informe HTML5.
- Git; VS Code con extensiones de Python/Jupyter (opcional).
- Dependencias Python de `requirements.txt`.

> **Reproducibilidad de dependencias:** el `requirements.txt` inicial enumera paquetes sin fijar versiones. Para cumplir el requisito de *dependencias exactas*, reemplazar esas entradas por las versiones realmente instaladas y comprobadas (`nombre==versión`), antes de la entrega. No se incluyen Dask ni YData-Profiling como dependencias obligatorias si no se utilizaron en el flujo real; el perfilador instalado corresponde al paquete `data-profiling`.

### Crear entorno en Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Obtener los microdatos

Descargar los archivos SPSS desde el [catálogo oficial del INE](https://anda.ine.gob.bo/index.php/catalog/256) y colocarlos en `data/raw/`. Crear la carpeta si no existe. El repositorio público no contiene estos archivos.

### Ejecutar el flujo

Desde la raíz del proyecto:

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

**Notebooks:** abrir `notebooks/` en VS Code, seleccionar el kernel `.venv` y ejecutar los tres archivos numerados en orden. Reutilizan resultados previamente generados y, por defecto, no vuelven a entrenar los clasificadores.

**Informe interactivo:** `docs/reporte_final.html`. Sus gráficos se cargan mediante rutas relativas a `outputs/figures/`, por lo que deben conservarse las carpetas al descargar el repositorio. El HTML y los gráficos están preparados para consulta local sin ejecutar los scripts.

## Recomendaciones de políticas públicas

1. **Acompañamiento en la adolescencia:** evaluar estrategias de seguimiento de estudiantes de 14 a 17 años durante las transiciones educativas.
2. **Diagnóstico de barreras rurales:** investigar transporte, distancias y disponibilidad de servicios antes de diseñar intervenciones territoriales.
3. **Apoyo familiar contextualizado:** estudiar necesidades de distintos tipos de hogar sin atribuir causalidad ni estigmatizar configuraciones familiares.

Estas recomendaciones son hipótesis para evaluación e intervención piloto; **no** constituyen evidencia de efectividad de intervenciones específicas.

## Limitaciones y consideraciones éticas

- La EH2025 es **transversal** y no acredita abandono escolar definitivo.
- La muestra tiene **208 casos positivos**, de los cuales **42** están en la partición de prueba.
- No se calcularon intervalos de confianza ajustados al diseño complejo (`upm`, `estrato`, factor de expansión).
- Las estimaciones departamentales pueden presentar alta incertidumbre por la escasez de casos positivos.
- Los modelos no incorporaron el diseño muestral en entrenamiento y evaluación.
- La validación por hogares reduce fugas, pero no acredita generalización temporal o territorial.
- Las asociaciones e importancias predictivas **no establecen causalidad**.
- No se debe intentar reidentificar personas ni automatizar decisiones individuales con este clasificador.
- Uso **estrictamente académico**. Antes de publicar el reporte EDA y los archivos de metadatos, verificar que no incluyan filas individuales, identificadores ni fragmentos de información sensible.
- Se respetan las condiciones de acceso de los datos del INE y la atribución de fuentes cartográficas.

## Referencias

- **Instituto Nacional de Estadística (INE), Bolivia.** *Encuesta de Hogares 2025*. https://anda.ine.gob.bo/index.php/catalog/256. **Fecha de acceso:** 09/11/2026.
- **YPFB.** Servicio cartográfico público de límites administrativos departamentales de Bolivia (geometrías utilizadas para el mapa; revisar condiciones de reutilización).
- Herramientas: Python, Pyreadstat, DuckDB, Pandas, Plotly, Scikit-learn, LightGBM y Quarto.

---

**Alfredo Ramos · Universidad Pública de El Alto (UPEA) · 2026**
