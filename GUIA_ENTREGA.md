# Guía de entrega y correspondencia con la actividad

## Entregable 1. Informe del proceso

- `INFORME_PROYECTO.pdf`: problema, datos, procesamiento, modelos, regularización, hiperparámetros, métricas, conclusiones, limitaciones y pruebas.
- `CASO_ESTUDIO.md`: formulación completa del caso.
- `notebooks/02_analisis_modelo.ipynb`: recorrido reproducible del análisis para explicarlo en clase.
- `notebooks/01_portafolio_visualizaciones.ipynb`: análisis exploratorio previo (portafolio de visualizaciones).

## Entregable 2. Carpeta del dashboard

- `app.py`: tablero Dash con gráficos, mapa, tablas, filtros y callbacks.
- `assets/style.css`: apariencia y diseño adaptable.
- `data/`: datos agregados, diccionario y geojson.
- `outputs/`: comparación de modelos, backtest, pronósticos, coeficientes, importancia y resumen.
- `models/`: modelo seleccionado.
- `src/` y `entrenar_modelo.py`: reproducción de los datos y del análisis.
- `requirements.txt`, `environment.yml`, `Procfile` y `binder/start`: configuración.
- `tests/`: pruebas automatizadas.

## Entregable 3. Archivo con enlaces

- `ENLACES.txt`: repositorio de GitHub y dashboard en Binder.

## Inicio rápido

1. Cree y active un entorno virtual.
2. Ejecute `pip install -r requirements.txt`.
3. Ejecute `python app.py`.
4. Abra `http://127.0.0.1:8050`.

Los datos y resultados ya están incluidos. Solo ejecute `src/ingesta.py`, `src/agregar.py` y `entrenar_modelo.py` si desea regenerarlos (ver README).
