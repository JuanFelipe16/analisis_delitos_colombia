# Pronóstico de cierre 2026 de delitos de alto impacto social en Colombia

Proyecto final de Ciencia de Datos II. A partir de los registros de la Policía Nacional (2020 a julio de 2026) se pronostica cuántos casos tendrá cada departamento y tipo de delito al cierre de 2026, se comparan tres iteraciones del modelo y los resultados se comunican en un dashboard interactivo en Dash.

Delitos analizados: violencia intrafamiliar, amenazas, delitos sexuales, homicidio intencional, extorsión y secuestro.

## Estructura

```text
src/config.py                  Rutas, constantes y corte de los datos
src/ingesta.py                 Lectura y consolidación de los Excel crudos
src/agregar.py                 Agregación mensual (departamento x delito x mes)
entrenar_modelo.py             Backtest, comparación de modelos y pronóstico final
app.py                         Dashboard Dash
assets/style.css               Estilos del dashboard
data/delitos_mensual.csv       Datos agregados que usan el modelo y el dashboard
data/diccionario_datos.csv     Diccionario de datos
data/geo/                      Geojson de departamentos
outputs/                       Métricas, backtest, pronósticos y coeficientes
models/                        Modelo seleccionado
tests/test_pipeline.py         Pruebas de integridad
notebooks/                     Portafolio de visualizaciones y prototipo inicial
CASO_ESTUDIO.md                Formulación completa del caso
GUIA_ENTREGA.md                Correspondencia con los entregables
ENLACES.txt                    Enlaces de GitHub y Binder
```

## Ejecución rápida

Requiere Python 3.11 o posterior. Los datos agregados y los resultados ya están en el repositorio, así que para ver el dashboard basta con:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Abra `http://127.0.0.1:8050`.

## Reproducir todo desde los datos crudos

Los Excel originales (unos 270 MB) y el consolidado (458 MB) no se suben a GitHub por su tamaño. Para regenerar todo, copie los archivos `INFORMACIÓN_DE_DELITOS_A_NIVEL_DE_REGISTRO_AÑO_*.xlsx` en `data/raw/delitos/` y ejecute:

```bash
python -m src.ingesta        # consolida los Excel -> data/processed/delitos_consolidado.csv
python -m src.agregar        # agrega por mes -> data/delitos_mensual.csv
python entrenar_modelo.py    # backtest y pronóstico -> outputs/ y models/ (unos 3 minutos)
python app.py
```

## Pruebas

```bash
pytest -q
```

## Binder

El enlace de `ENLACES.txt` abre el dashboard en Binder. `binder/start` lanza `app.py` en segundo plano y Jupyter lo expone en `/proxy/8050/` mediante `jupyter-server-proxy`. La primera vez que se construye la imagen puede tardar varios minutos.

## Reproducibilidad

Los modelos usan semilla fija (42). El backtest y el pronóstico dependen del corte definido en `src/config.py` (julio de 2026). Los resultados pueden variar levemente entre versiones de las dependencias.

## Nota

Los datos son casos registrados por la Policía Nacional, no la criminalidad real: dependen de la denuncia y del rezago de registro. El pronóstico describe una tendencia esperada y no debe usarse como única base para decisiones sobre territorios.
