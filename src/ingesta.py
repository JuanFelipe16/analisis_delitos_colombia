"""Descubrimiento, lectura y consolidación de los archivos crudos de delitos.

Cualquier .xlsx que se agregue a data/raw/delitos/ (mismo formato de columnas:
DEPARTAMENTO/MUNICIPIO/FECHA HECHO/.../CANTIDAD) se incorpora automáticamente
la próxima vez que se reconstruya el consolidado, sin tocar este código.

data/raw/ conserva cada Excel completo tal como llega (todas las categorías
de delito). data/processed/delitos_consolidado.csv es la versión curada que
usan los notebooks: solo las filas cuyo TIPO_DELITO está en
`config.DELITOS_FILTRO` (editable ahí mismo, sin tocar este archivo).
"""

import re

import pandas as pd

from src.config import (
    ALIAS_COLUMNAS,
    CODIGO_DANE_BOGOTA,
    CONSOLIDADO_PATH,
    DELITOS_FILTRO,
    KEYWORD_ARCHIVO,
    KEYWORD_HOJA,
    NORMALIZACION_DEPARTAMENTO,
    PROCESSED_DIR,
    RAW_DELITOS_DIR,
)

PATRON_DELITO = re.compile(r"^\((?P<codigo>\d+)\)\s*(?P<descripcion>.*)$")


# Solo tildes de vocales: la Ñ es una letra propia del español (no una vocal
# acentuada) y los nombres de lugares del DANE la usan de forma consistente,
# así que no se debe convertir a "N".
_TABLA_TILDES = str.maketrans("áéíóúü", "aeiouu")


def _normalizar_texto(texto: str) -> str:
    """Minúsculas y sin tildes (conserva la Ñ), para comparar texto sin depender de la redacción exacta."""
    return str(texto).strip().lower().translate(_TABLA_TILDES)


def descubrir_archivos_delitos():
    """Devuelve los .xlsx en RAW_DELITOS_DIR cuyo nombre contiene la palabra clave."""
    return sorted(
        p
        for p in RAW_DELITOS_DIR.glob("*.xlsx")
        if KEYWORD_ARCHIVO.upper() in p.stem.upper()
    )

def _encontrar_fila_encabezado(tabla_cruda: pd.DataFrame) -> int:
    for fila, valores in tabla_cruda.iterrows():
        textos = {str(v).strip().upper() for v in valores}
        if "DEPARTAMENTO" in textos and "MUNICIPIO" in textos:
            return fila
    raise ValueError("No se encontró la fila de encabezado (DEPARTAMENTO/MUNICIPIO).")

def _encontrar_hoja(archivo_excel: pd.ExcelFile, ruta) -> str:
    # Formato con una sola hoja de datos (el nombre varía cada año: "Hoja1",
    # "2026", "DELITOS A NIVEL DE REGISTRO", ...): se usa directamente.
    if len(archivo_excel.sheet_names) == 1:
        return archivo_excel.sheet_names[0]

    # Formato antiguo con varias hojas: se prefiere la que menciona "filtrados".
    for nombre in archivo_excel.sheet_names:
        if KEYWORD_HOJA in nombre.strip().lower():
            return nombre

    # Último recurso: probar cada hoja y quedarse con la primera que
    # realmente tenga la fila de encabezado esperada.
    for nombre in archivo_excel.sheet_names:
        try:
            _encontrar_fila_encabezado(archivo_excel.parse(nombre, header=None))
            return nombre
        except ValueError:
            continue

    raise ValueError(
        f"No se encontró una hoja con datos de delitos en {ruta}. "
        f"Hojas disponibles: {archivo_excel.sheet_names}"
    )


def leer_archivo_delitos(ruta) -> pd.DataFrame:
    """Lee un Excel crudo de delitos y devuelve un DataFrame con columnas normalizadas."""
    archivo_excel = pd.ExcelFile(ruta)
    hoja = _encontrar_hoja(archivo_excel, ruta)

    tabla_cruda = archivo_excel.parse(hoja, header=None)
    fila_encabezado = _encontrar_fila_encabezado(tabla_cruda)

    df = archivo_excel.parse(hoja, header=fila_encabezado)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=ALIAS_COLUMNAS)
    df = df.loc[:, [c for c in df.columns if c in set(ALIAS_COLUMNAS.values())]]

    # Al final de cada hoja el archivo trae notas de pie ("Fuente: ...",
    # "Fecha de entrega: ...") que quedan como filas sin DEPARTAMENTO/MUNICIPIO.
    df = df.dropna(subset=["DEPARTAMENTO", "MUNICIPIO"]).reset_index(drop=True)

    extraido = df["DELITO_RAW"].astype(str).str.extract(PATRON_DELITO)
    df["CODIGO_DELITO"] = extraido["codigo"]
    df["TIPO_DELITO"] = extraido["descripcion"].fillna(df["DELITO_RAW"]).str.strip()

    df["ARCHIVO_ORIGEN"] = ruta.name
    return df


def construir_consolidado() -> pd.DataFrame:
    """Reconstruye el dataset consolidado a partir de todos los archivos crudos."""
    archivos = descubrir_archivos_delitos()
    if not archivos:
        raise FileNotFoundError(f"No hay archivos '*{KEYWORD_ARCHIVO}*.xlsx' en {RAW_DELITOS_DIR}")

    df = pd.concat((leer_archivo_delitos(p) for p in archivos), ignore_index=True)

    df["FECHA_HECHO"] = pd.to_datetime(df["FECHA_HECHO"], errors="coerce")
    df["AÑO"] = df["FECHA_HECHO"].dt.year
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce")
    df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"], errors="coerce").fillna(0)

    columnas_texto = ["DEPARTAMENTO", "MUNICIPIO", "GENERO", "GRUPO_EDAD", "TIPO_DELITO", "ARMAS_MEDIOS"]
    for columna in columnas_texto:
        df[columna] = df[columna].astype(str).str.strip()
    df["DEPARTAMENTO"] = df["DEPARTAMENTO"].replace(NORMALIZACION_DEPARTAMENTO)

    df["CODIGO_DANE"] = pd.to_numeric(df["CODIGO_DANE"], errors="coerce")
    df.loc[df["CODIGO_DANE"] == CODIGO_DANE_BOGOTA, "DEPARTAMENTO"] = "BOGOTA"

    # No se deduplica por contenido: cada fila es un caso individual y, al no
    # traer un identificador único, es normal que varios casos distintos
    # coincidan en todas las columnas categóricas (mismo departamento, fecha,
    # tipo de delito, género y grupo de edad).
    df = df.reset_index(drop=True)

    # El CSV procesado solo conserva los delitos de interés social definidos
    # en config.DELITOS_FILTRO (los Excel crudos traen muchas más categorías).
    filtro_normalizado = {_normalizar_texto(d) for d in DELITOS_FILTRO}
    df = df[df["TIPO_DELITO"].map(_normalizar_texto).isin(filtro_normalizado)].reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONSOLIDADO_PATH, index=False)
    return df


def cargar_consolidado(forzar_reconstruccion: bool = False) -> pd.DataFrame:
    """Carga el consolidado desde caché si está actualizado; si no, lo reconstruye."""
    if not forzar_reconstruccion and CONSOLIDADO_PATH.exists():
        archivos = descubrir_archivos_delitos()
        mtime_procesado = CONSOLIDADO_PATH.stat().st_mtime
        mtime_crudos = max((p.stat().st_mtime for p in archivos), default=0)
        if mtime_crudos <= mtime_procesado:
            return pd.read_csv(CONSOLIDADO_PATH, parse_dates=["FECHA_HECHO"])

    return construir_consolidado()
