"""Construye el notebook del análisis del modelo sin depender de nbformat."""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().splitlines()]}


cells = [
    md("""# Pronóstico de cierre 2026 de delitos de alto impacto social

Este notebook documenta el análisis que alimenta el dashboard (`app.py`). Los datos son registros oficiales de la Policía Nacional (2020 a julio de 2026) de seis delitos: violencia intrafamiliar, amenazas, delitos sexuales, homicidio, extorsión y secuestro.

**Pregunta:** con los casos observados hasta julio, ¿cuántos casos tendrá cada departamento y tipo de delito al cierre de 2026?

$$\\text{cierre} = \\text{casos observados ene-jul} + \\text{pronóstico ago-dic}$$"""),
    md("""## 1. Preparación

Los resultados ya están en `outputs/`. Para reproducirlos desde cero: `python -m src.ingesta`, `python -m src.agregar` y `python entrenar_modelo.py` (ver README)."""),
    code("""import sys
from pathlib import Path
import json

RAIZ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from entrenar_modelo import construir_panel
from src.config import CORTE_ANIO, CORTE_MES

data = pd.read_csv(RAIZ / 'data' / 'delitos_mensual.csv')
panel = construir_panel(data)
resultados = pd.read_csv(RAIZ / 'outputs' / 'comparacion_modelos.csv')
backtest = pd.read_csv(RAIZ / 'outputs' / 'backtest_cierres.csv')
cierre = pd.read_csv(RAIZ / 'outputs' / 'pronostico_cierre.csv')
pronostico = pd.read_csv(RAIZ / 'outputs' / 'pronostico_mensual.csv')
coef = pd.read_csv(RAIZ / 'outputs' / 'coeficientes_mlg.csv')
importancia = pd.read_csv(RAIZ / 'outputs' / 'importancia_variables.csv')
resumen = json.loads((RAIZ / 'outputs' / 'resumen_modelo.json').read_text(encoding='utf-8'))
data.head()"""),
    md("## 2. Calidad y descripción de los datos"),
    code("""print(f'Filas agregadas: {len(data):,}')
print(f'Casos: {data.casos.sum():,}')
print(f'Series departamento x delito: {panel.serie.nunique()}')
print(f'Meses: {panel.fecha.nunique()} (ene 2020 a {CORTE_MES:02d}/{CORTE_ANIO})')
print(f'Valores faltantes: {data.isna().sum().sum()}')
print(f'Meses-serie sin casos (ceros completados): {(panel.casos == 0).mean():.1%}')"""),
    code("""tabla = data.pivot_table(index='mes', columns='anio', values='casos', aggfunc='sum')
tabla"""),
    md("""**Lectura:** 2026 va por encima de los años anteriores en casi todos los meses. Diciembre es bajo en todos los archivos anuales (en 2025 apenas 12.313 casos), lo que indica rezago de registro: parte de los hechos de fin de año se cargan después de la fecha de corte del archivo. Por la misma razón, agosto de 2026 (19.672 casos en el archivo original) se excluye y el corte se fija en julio."""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for anio, g in data.groupby('anio'):
    s = g.groupby('mes').casos.sum()
    axes[0].plot(s.index, s.values, marker='o', label=anio, lw=2.5 if anio == CORTE_ANIO else 1.2)
axes[0].set_title('Estacionalidad: casos por mes en cada año'); axes[0].legend(ncol=2, fontsize=8); axes[0].set_xticks(range(1, 13))
data.pivot_table(index='anio', columns='delito', values='casos', aggfunc='sum').plot.bar(stacked=True, ax=axes[1], colormap='tab20c')
axes[1].set_title('Casos por año y delito (2026 parcial)'); axes[1].legend(fontsize=7)
plt.tight_layout();"""),
    md("""## 3. Sobredispersión

Un modelo de Poisson supone que la varianza es igual a la media. Si dentro de cada serie la varianza es mucho mayor, hay sobredispersión y la binomial negativa es más adecuada (Var = μ + αμ²)."""),
    code("""disp = panel.groupby('serie').casos.agg(['mean', 'var'])
disp = disp[disp['mean'] > 0]
print(f"Series con varianza > media: {(disp['var'] > disp['mean']).mean():.0%}")
print(f"Alpha estimado (Cameron-Trivedi): {resumen['alpha_binomial_negativa']:.4f}")
print(f"AIC Poisson: {resumen['aic_poisson']:,.0f} | AIC binomial negativa: {resumen['aic_binomial_negativa']:,.0f}")
ax = disp.plot.scatter(x='mean', y='var', logx=True, logy=True, alpha=.6, figsize=(6, 4.5), title='Media vs varianza mensual por serie')
lim = [disp['mean'].min(), disp['var'].max()]; ax.plot(lim, lim, '--', color='gray', label='Var = media (Poisson)'); ax.legend();"""),
    md("""## 4. Estrategia de evaluación: backtest

El cierre de 2026 aún no se conoce, así que no puede usarse para medir el error. Se simula el mismo problema en años ya cerrados: con datos hasta julio de 2023 (2024, 2025) se pronostica agosto-diciembre y se compara el cierre pronosticado con el real. La métrica principal es el **WAPE** del cierre:

$$\\text{WAPE} = \\frac{\\sum |\\text{cierre pronosticado} - \\text{cierre real}|}{\\sum \\text{cierre real}}$$

## 5. Iteraciones del modelo

1. **Línea base:** casos enero-julio escalados por el peso estacional histórico (últimos 3 años) de cada delito.
2. **MLG binomial negativa:** `casos ~ serie + mes + tendencia por delito + covid`, enlace log. Es la familia de conteos de los modelos lineales generalizados; los coeficientes se interpretan como razones de tasas (IRR = e^β).
3. **Gradient boosting con pérdida Poisson:** rezago de 12 meses, medias móviles de 3 y 12 meses, crecimiento anual, mes objetivo y horizonte. Hiperparámetros ajustados con validación cruzada temporal (`TimeSeriesSplit`) dentro del entrenamiento de cada backtest; la regularización se controla con profundidad, hoja mínima y penalización L2."""),
    code("""cols = ['version', 'regularizacion', 'wape', 'mae_cierre', 'rmse_cierre', 'error_nacional_pct', 'wape_2023', 'wape_2024', 'wape_2025']
resultados[cols].round(3)"""),
    code("""resultados.set_index('version')[['wape_2023', 'wape_2024', 'wape_2025']].plot.bar(figsize=(10, 4), color=['#1976D2', '#26A6A1', '#F4B942'])
plt.ylabel('WAPE del cierre'); plt.title('Error por iteración y año de backtest (menor es mejor)'); plt.xticks(rotation=0); plt.tight_layout();"""),
    code("""bt = backtest[backtest.version == resumen['modelo_seleccionado']]
fig, ax = plt.subplots(figsize=(6, 5))
for anio, g in bt.groupby('anio'):
    ax.scatter(g.cierre_real, g.cierre_pred, alpha=.6, label=anio)
lim = [1, bt[['cierre_real', 'cierre_pred']].max().max()]
ax.plot(lim, lim, '--', color='gray'); ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Cierre real'); ax.set_ylabel('Cierre pronosticado'); ax.set_title(f"Backtest - {resumen['modelo_seleccionado']}"); ax.legend();"""),
    md("""**Lectura:** el boosting y la línea base quedan muy cerca; el MLG es el menos preciso porque impone una tendencia lineal (en escala log) para todo el periodo 2020-2026, y varias series cambiaron de dirección. 2023 es el año más difícil para todos: el segundo semestre de ese año creció más de lo que anticipaba la historia."""),
    md("## 6. Interpretación del MLG (razones de tasas)"),
    code("""tabla_irr = coef[['termino', 'coeficiente', 'valor_p', 'irr', 'irr_inf', 'irr_sup']].copy()
tabla_irr['cambio_pct'] = (tabla_irr.irr - 1) * 100
tabla_irr.round(3)"""),
    md("""- **Tendencia anual:** la extorsión crece en promedio cerca de 11 % por año y el secuestro 26 %, manteniendo constantes la serie y el mes; las amenazas caen 2,4 % y los delitos sexuales 4,3 % por año. La violencia intrafamiliar no muestra tendencia significativa (p ≈ 0,09).
- **Estacionalidad:** frente a enero, agosto a octubre tienen alrededor de 10 % más casos; diciembre tiene 7 % menos (en parte por rezago de registro).
- **Covid:** en los meses de confinamiento estricto de 2020 los casos registrados fueron 23 % menores.

Al igual que el odds ratio en la regresión logística, la IRR describe un cambio multiplicativo, aquí sobre el número esperado de casos."""),
    md("## 7. Importancia de variables del boosting"),
    code("""importancia.sort_values('importancia').plot.barh(x='variable', y='importancia', legend=False, figsize=(7, 4), color='#26A6A1')
plt.title('Importancia por permutación (backtest 2025)'); plt.xlabel('Aumento de la devianza Poisson'); plt.tight_layout();"""),
    md("## 8. Pronóstico de cierre 2026"),
    code("""nac = panel.groupby('fecha').casos.sum()
fut = pronostico.groupby('mes').pred.sum()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nac.index.to_timestamp(), nac.values, color='#1976D2', label='Observado')
fechas_fut = pd.to_datetime([f'{CORTE_ANIO}-{m:02d}-01' for m in fut.index])
ax.plot([nac.index[-1].to_timestamp(), *fechas_fut], [nac.values[-1], *fut.values], '--', color='#D95D5D', lw=2.5, label='Pronóstico')
ax.set_title('Casos mensuales nacionales y pronóstico ago-dic 2026'); ax.legend();
print(f"Cierre {CORTE_ANIO - 1}: {resumen['cierre_nacional_anterior']:,}")
print(f"Cierre {CORTE_ANIO} pronosticado: {resumen['cierre_nacional_pronosticado']:,.0f} "
      f"(rango {resumen['cierre_nacional_inf']:,.0f} - {resumen['cierre_nacional_sup']:,.0f})")"""),
    code("""por_delito = cierre.groupby('delito')[['cierre_anterior', 'observado_ytd', 'cierre_pred']].sum()
por_delito['variacion_pct'] = (por_delito.cierre_pred / por_delito.cierre_anterior - 1) * 100
por_delito.round(1).sort_values('cierre_pred', ascending=False)"""),
    code("""print(cierre.tendencia.value_counts())
cierre.sort_values('variacion_pct', ascending=False)[['departamento', 'delito', 'cierre_anterior', 'cierre_pred', 'variacion_pct']].query('cierre_anterior >= 500').head(10).round(2)"""),
    md("""## 9. Conclusiones

- Con el corte de julio es posible pronosticar el cierre anual por departamento y delito con un error cercano al 8 % (WAPE), y alrededor de 5 % en el total nacional.
- El boosting con pérdida Poisson fue el más preciso, pero la línea base estacional queda muy cerca: la señal principal está en el nivel reciente de cada serie.
- El MLG binomial negativa es menos preciso, pero es el que explica el comportamiento (tendencias por delito, estacionalidad, efecto covid) y confirma la sobredispersión de los conteos.
- Se espera que 2026 cierre con unos 272.700 casos, alrededor de 11 % más que 2025, con aumentos en violencia intrafamiliar, amenazas y extorsión.
- Los datos son casos registrados: el rezago de registro (especialmente en diciembre) y la denuncia afectan tanto el pronóstico como los cierres usados para evaluarlo.

El dashboard `app.py` permite filtrar por departamento y delito, revisar el pronóstico mensual, el mapa de variaciones y el desempeño de cada iteración."""),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
destino = ROOT / "notebooks" / "02_analisis_modelo.ipynb"
destino.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Creado {destino}")
