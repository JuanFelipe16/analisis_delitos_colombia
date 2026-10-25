"""Agrega el consolidado de casos individuales a conteos mensuales.

Pasa de 1,6 millones de filas (un caso por fila) a una tabla de
departamento x delito x mes x género x grupo de edad, que es la que usan el
modelo y el dashboard. Ejecutar desde la raíz con `python -m src.agregar`.
"""

import pandas as pd

from src.config import CORTE_ANIO, CORTE_MES, DICCIONARIO_PATH, MENSUAL_PATH, NOMBRE_CORTO_DELITO
from src.ingesta import cargar_consolidado

SIN_DATO = "NO REPORTA"


def agregar_mensual(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["anio"] = x["FECHA_HECHO"].dt.year.astype(int)
    x["mes"] = x["FECHA_HECHO"].dt.month.astype(int)
    x["delito"] = x["TIPO_DELITO"].map(NOMBRE_CORTO_DELITO)
    # En el consolidado los vacíos quedaron como texto "nan".
    for col in ["GENERO", "GRUPO_EDAD"]:
        x[col] = x[col].fillna(SIN_DATO).replace({"nan": SIN_DATO})

    # Se descarta lo posterior al corte (agosto 2026 está incompleto).
    x = x[(x["anio"] < CORTE_ANIO) | (x["mes"] <= CORTE_MES)]

    mensual = (
        x.groupby(["DEPARTAMENTO", "delito", "anio", "mes", "GENERO", "GRUPO_EDAD"], as_index=False)["CANTIDAD"]
        .sum()
        .rename(columns={"DEPARTAMENTO": "departamento", "GENERO": "genero", "GRUPO_EDAD": "grupo_edad", "CANTIDAD": "casos"})
    )
    mensual["casos"] = mensual["casos"].astype(int)
    return mensual.sort_values(["departamento", "delito", "anio", "mes"]).reset_index(drop=True)


def diccionario() -> pd.DataFrame:
    return pd.DataFrame([
        ("departamento", "Departamento donde ocurrió el hecho (Bogotá separado de Cundinamarca)", "Categórica", "33 departamentos"),
        ("delito", "Tipo de delito de alto impacto social", "Categórica", ", ".join(NOMBRE_CORTO_DELITO.values())),
        ("anio", "Año del hecho", "Entera", f"2020 a {CORTE_ANIO}"),
        ("mes", "Mes del hecho", "Entera", f"1 a 12 ({CORTE_ANIO}: 1 a {CORTE_MES})"),
        ("genero", "Género de la persona asociada al caso", "Categórica", f"FEMENINO, MASCULINO, {SIN_DATO}"),
        ("grupo_edad", "Grupo de edad de la persona", "Categórica", f"DE 0 A 14 AÑOS, DE 14 A 17 AÑOS, DE 18 AÑOS Y MÁS, {SIN_DATO}"),
        ("casos", "Número de casos registrados por la Policía Nacional", "Entera", ">= 1"),
    ], columns=["variable", "descripcion", "tipo", "rango_o_categorias"])


def main():
    mensual = agregar_mensual(cargar_consolidado())
    mensual.to_csv(MENSUAL_PATH, index=False)
    diccionario().to_csv(DICCIONARIO_PATH, index=False)
    print(f"{MENSUAL_PATH.name}: {len(mensual):,} filas, {mensual.casos.sum():,} casos")


if __name__ == "__main__":
    main()
