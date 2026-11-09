
from pathlib import Path
import json
import unicodedata
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import plotly.express as px


# ============================================================
# 1. CONFIGURACIÓN DEL PROYECTO
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "outputs" / "figures"
METADATA = ROOT / "data" / "metadata"

GEO = METADATA / "bolivia_departamentos.geojson"

RESUMEN = OUT / "resumen_departamentos.csv"

DESTINO = OUT / "05_mapa_inasistencia_bolivia.html"

OUT.mkdir(parents=True, exist_ok=True)
METADATA.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FUENTE CARTOGRÁFICA
# ============================================================

# Servicio geográfico de YPFB.
# Capa administrativa: departamentos de Bolivia.

BASE_URL = (
    "https://portalgis.ypfb.gob.bo/arcgisserver/"
    "rest/services/publico/padministrativo/MapServer/1/query"
)

PARAMS = {
    "where": "1=1",
    "outFields": "departamento",
    "returnGeometry": "true",
    "outSR": "4326",
    "f": "geojson",
}


# Departamentos esperados después de normalizar nombres.

DEPARTAMENTOS = {
    "CHUQUISACA",
    "LA PAZ",
    "COCHABAMBA",
    "ORURO",
    "POTOSI",
    "TARIJA",
    "SANTA CRUZ",
    "BENI",
    "PANDO",
}


# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def normalizar(texto):
    """
    Normaliza nombres geográficos:
    - Elimina acentos.
    - Elimina espacios adicionales.
    - Convierte a mayúsculas.
    """

    if texto is None:
        return ""

    texto = unicodedata.normalize(
        "NFKD",
        str(texto),
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    return " ".join(texto.strip().upper().split())


def validar_geojson(geojson):
    """
    Comprueba que el GeoJSON contenga exactamente
    los nueve departamentos de Bolivia.
    """

    if geojson.get("type") != "FeatureCollection":
        raise ValueError(
            "El archivo no es un GeoJSON FeatureCollection."
        )

    features = geojson.get("features", [])

    nombres = []

    for feature in features:

        nombre = normalizar(
            feature.get("properties", {}).get(
                "departamento", ""
            )
        )

        if not feature.get("geometry"):
            raise ValueError(
                f"Geometría ausente para {nombre}"
            )

        nombres.append(nombre)

    if (
        len(features) != 9
        or set(nombres) != DEPARTAMENTOS
        or len(set(nombres)) != 9
    ):
        raise ValueError(
            "El GeoJSON no contiene exactamente "
            "los nueve departamentos válidos."
        )

    return True


# ============================================================
# 4. DESCARGA Y FILTRADO DE GEOMETRÍAS
# ============================================================

def descargar_geojson():
    """
    Descarga la capa geográfica y excluye registros
    correspondientes a lagos, salares y otros
    cuerpos de agua.
    """

    url = BASE_URL + "?" + urlencode(PARAMS)

    print("Descargando geometrías de YPFB...")

    with urlopen(url, timeout=45) as response:
        contenido = json.load(response)

    if contenido.get("type") != "FeatureCollection":
        raise ValueError(
            "El servidor no devolvió un GeoJSON válido. "
            f"Respuesta: {contenido}"
        )

    originales = contenido.get("features", [])

    print("Geometrías originales:", len(originales))

    filtradas = []

    for feature in originales:

        nombre = normalizar(
            feature.get("properties", {}).get(
                "departamento", ""
            )
        )

        # Excluir lagos, salares y registros adicionales.
        if nombre not in DEPARTAMENTOS:
            continue

        if not feature.get("geometry"):
            raise ValueError(
                f"El departamento {nombre} no tiene geometría."
            )

        feature["id"] = nombre
        filtradas.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": filtradas,
    }

    validar_geojson(geojson)

    GEO.write_text(
        json.dumps(
            geojson,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Departamentos válidos:", len(filtradas))
    print("GeoJSON guardado:", GEO)

    return geojson


def cargar_geojson():
    """
    Utiliza el GeoJSON almacenado localmente.
    Si está incompleto, vuelve a descargarlo.
    """

    if GEO.exists():

        print("Cargando GeoJSON local...")

        try:
            geojson = json.loads(
                GEO.read_text(encoding="utf-8")
            )

            validar_geojson(geojson)

            print("GeoJSON local validado.")

        except (ValueError, KeyError, TypeError) as error:

            print(
                "GeoJSON local inválido:",
                error,
            )

            print("Regenerando archivo...")

            geojson = descargar_geojson()

    else:
        geojson = descargar_geojson()

    # Garantizar IDs normalizados.
    for feature in geojson["features"]:
        feature["id"] = normalizar(
            feature["properties"]["departamento"]
        )

    return geojson


# ============================================================
# 5. CARGAR INDICADORES DE LA ENCUESTA EH2025
# ============================================================

def cargar_indicadores():

    if not RESUMEN.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {RESUMEN}"
        )

    df = pd.read_csv(RESUMEN)

    columnas = {
        "departamento",
        "tasa",
        "muestra",
        "casos",
    }

    faltantes = columnas - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Columnas faltantes: {faltantes}"
        )

    # Normalizar identificadores geográficos.
    df["codigo_geo"] = df["departamento"].map(
        normalizar
    )

    # Garantizar tipos numéricos.
    for columna in ["tasa", "muestra", "casos"]:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

    if df[["tasa", "muestra", "casos"]].isna().any().any():
        raise ValueError(
            "Existen indicadores numéricos faltantes."
        )

    if (
        (df["tasa"] < 0).any()
        or (df["tasa"] > 100).any()
        or (df["muestra"] <= 0).any()
        or (df["casos"] < 0).any()
        or (df["casos"] > df["muestra"]).any()
    ):
        raise ValueError(
            "Existen indicadores fuera de rango."
        )

    if (
        len(df) != 9
        or set(df["codigo_geo"]) != DEPARTAMENTOS
        or df["codigo_geo"].duplicated().any()
    ):
        raise ValueError(
            "El resumen no contiene exactamente "
            "los nueve departamentos."
        )

    # Columnas formateadas exclusivamente para el tooltip.
    df["muestra_formato"] = df["muestra"].map(
        lambda valor: f"{int(valor):,}".replace(",", ".")
    )

    df["casos_formato"] = df["casos"].map(
        lambda valor: f"{int(valor):,}".replace(",", ".")
    )

    print("\n=== INDICADORES CARGADOS ===")

    print(
        df[
            ["departamento", "tasa", "muestra", "casos"]
        ].sort_values(
            "tasa",
            ascending=False,
        ).to_string(index=False)
    )

    return df


# ============================================================
# 6. GENERAR MAPA INTERACTIVO
# ============================================================

def generar_mapa():

    geojson = cargar_geojson()

    df = cargar_indicadores()

    ids_geo = {
        feature["id"]
        for feature in geojson["features"]
    }

    ids_datos = set(df["codigo_geo"])

    if ids_geo != ids_datos:
        raise ValueError(
            "Los departamentos no coinciden. "
            f"Sin geometría: {ids_datos - ids_geo}. "
            f"Sin datos: {ids_geo - ids_datos}"
        )

    print("\nGenerando mapa interactivo...")

    # --------------------------------------------------------
    # Mapa coroplético
    # --------------------------------------------------------

    fig = px.choropleth_map(
        df,
        geojson=geojson,
        locations="codigo_geo",
        featureidkey="id",

        # Indicador ponderado de inasistencia escolar.
        color="tasa",

        # Variable principal en el tooltip.
        hover_name="departamento",

        # Datos adicionales en orden controlado.
        custom_data=[
            "departamento",
            "muestra_formato",
            "casos_formato",
        ],

        # Escala de color:
        # amarillo = tasa menor
        # rojo = tasa mayor
        color_continuous_scale="YlOrRd",

        # Encuadre geográfico centrado en Bolivia.
        center={
            "lat": -16.7,
            "lon": -64.7,
        },

        zoom=4.6,

        map_style="carto-positron",

        opacity=0.85,

        title=(
            "Bolivia: inasistencia escolar "
            "estimada por departamento"
        ),

        labels={
            "tasa": "Inasistencia (%)",
        },
    )

    # --------------------------------------------------------
    # Tooltip personalizado
    # --------------------------------------------------------

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b>"
            "<br>"
            "Tasa estimada: %{z:.2f}%"
            "<br>"
            "Personas encuestadas: %{customdata[1]}"
            "<br>"
            "Casos de inasistencia: %{customdata[2]}"
            "<extra></extra>"
        ),

        marker_line_color="#505867",
        marker_line_width=0.8,
    )

    # --------------------------------------------------------
    # Configuración visual general
    # --------------------------------------------------------

    fig.update_layout(

        template="plotly_white",

        height=720,

        margin=dict(
            l=10,
            r=10,
            t=75,
            b=10,
        ),

        title=dict(
            x=0.02,
            xanchor="left",
            font=dict(
                size=21,
                color="#243654",
            ),
        ),

        # Leyenda de valores: obligatoriamente visible.
        coloraxis_showscale=True,

        coloraxis_colorbar=dict(

            title=dict(
                text="Inasistencia (%)",
                side="top",
            ),

            thickness=18,

            len=0.60,

            x=0.98,
            xanchor="right",

            y=0.50,
            yanchor="middle",

            ticksuffix="%",

            tickfont=dict(
                size=12,
            ),

            outlinewidth=0,
        ),

        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#334155",
        ),
    )

    # --------------------------------------------------------
    # Exportación HTML interactiva
    # --------------------------------------------------------

    fig.write_html(
        DESTINO,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "mapa_inasistencia_bolivia_2025",
                "scale": 2,
            },
        },
    )

    print("\n=== MAPA GENERADO ===")

    print("Departamentos:", len(ids_geo))
    print("Archivo:", DESTINO)
    print("Proceso completado correctamente.")


# ============================================================
# 7. EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    generar_mapa()
