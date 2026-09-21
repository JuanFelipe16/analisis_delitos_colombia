"""Rutas y constantes compartidas del proyecto."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DELITOS_DIR = BASE_DIR / "data" / "raw" / "delitos"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CONSOLIDADO_PATH = PROCESSED_DIR / "delitos_consolidado.csv"

# Cualquier .xlsx en RAW_DELITOS_DIR cuyo nombre contenga esta palabra
# (sin distinguir mayúsculas) se toma como fuente de delitos.
KEYWORD_ARCHIVO = "DELITOS"

# Fragmento (normalizado a minúsculas, sin tildes) que debe contener el
# nombre de la hoja con la tabla real dentro de cada Excel.
KEYWORD_HOJA = "delitos filtr"

# Alias de columnas observados entre distintas exportaciones (varían
# ligeramente de un año a otro aunque representen lo mismo).
ALIAS_COLUMNAS = {
    "ARMAS MEDIOS": "ARMAS_MEDIOS",
    "ARMA MEDIO": "ARMAS_MEDIOS",
    "DEPARTAMENTO": "DEPARTAMENTO",
    "MUNICIPIO": "MUNICIPIO",
    "FECHA HECHO": "FECHA_HECHO",
    "FECHA ISSO": "FECHA_HECHO",
    "GENERO": "GENERO",
    "GÉNERO": "GENERO",
    "AGRUPA EDAD PERSONA": "GRUPO_EDAD",
    "*AGRUPA EDAD PERSONA*": "GRUPO_EDAD",
    "EDAD ETARIA": "GRUPO_EDAD",
    "CODIGO DANE": "CODIGO_DANE",
    "CÓDIGO DIVIPOLA": "CODIGO_DANE",
    "DELITOS": "DELITO_RAW",
    "MES": "MES",
    "CANTIDAD": "CANTIDAD",
    "ICCS": "ICCS",
}

# El nombre de algunos departamentos varía entre exportaciones (mismo
# departamento, distinta redacción). Se normalizan al nombre más común.
NORMALIZACION_DEPARTAMENTO = {
    "VALLE": "VALLE DEL CAUCA",
    "VALLE DEL CAUCA DEL CAUCA": "VALLE DEL CAUCA",
}

# Bogotá D.C. (código DANE 11001) aparece como DEPARTAMENTO "CUNDINAMARCA" en
# los archivos 2020-2024 y como "BOGOTA" en 2025-2026, aunque es el mismo
# municipio (Cundinamarca es un departamento distinto). Se fuerza por código
# DANE, no por texto, para no tocar los municipios reales de Cundinamarca.
CODIGO_DANE_BOGOTA = 11001

# Delitos que se conservan en el CSV procesado (data/processed/). Los Excel
# crudos traen muchas más categorías; el proyecto se enfoca en estas por su
# relevancia social. Para agregar una categoría nueva, basta con sumarla aquí
# con el mismo texto que aparece en la columna TIPO_DELITO (sin distinguir
# mayúsculas/acentos, ver ingesta._normalizar_texto).
DELITOS_FILTRO = [
    "Otros actos dirigidos a inducir miedo o angustia emocional (Violencia intrafamiliar)",
    "Amenaza",
    "Violación (Delitos sexuales)",
    "Homicidio intencional",
    "Extorsión",
    "Retención ilegal (Secuestro)",
]

PALETA_TENDENCIA = {
    "AUMENTO": "#d62728",
    "DISMINUCIÓN": "#2ca02c",
    "SIN CAMBIO": "#7f7f7f",
}

