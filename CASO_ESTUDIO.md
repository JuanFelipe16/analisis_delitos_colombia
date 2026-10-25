# Caso de estudio: pronóstico de cierre anual de delitos de alto impacto social

## 1. Situación problemática

Las cifras de delitos en Colombia se publican por año, pero las decisiones (asignación de pie de fuerza, programas de prevención, presupuesto de comisarías de familia) se toman antes de que el año termine. A mitad de año solo se conoce una parte de los casos y no es evidente si un departamento cerrará por encima o por debajo del año anterior.

## 2. Pregunta analítica

Con los casos registrados hasta julio, ¿cuántos casos tendrá cada departamento y tipo de delito al cierre de 2026, y en cuáles se espera un aumento frente a 2025?

## 3. Objetivo

Construir y evaluar un modelo que pronostique el cierre anual de casos por departamento y tipo de delito, y comunicar sus resultados mediante un dashboard interactivo.

## 4. Datos, unidad de análisis y variable objetivo

- Fuente: Policía Nacional de Colombia, delitos a nivel de registro, archivos anuales 2020-2026 (1,6 millones de casos de los seis delitos de interés).
- Unidad: serie mensual de casos de un departamento y un tipo de delito (33 departamentos x 6 delitos = 193 series con casos; 79 meses de enero 2020 a julio 2026).
- Objetivo: cierre anual = casos observados de enero a julio + casos pronosticados de agosto a diciembre.
- Corte: julio de 2026. Agosto de 2026 viene en el archivo pero con sub-registro (19.672 casos frente a 22.041 en julio, cuando históricamente agosto supera a julio), por lo que se excluye.

## 5. Hipótesis de trabajo

El nivel reciente de cada serie, su estacionalidad mensual y la tendencia de cada tipo de delito permiten anticipar el resto del año. Se espera además que los conteos presenten sobredispersión (varianza mayor que la media), lo que favorece una distribución binomial negativa sobre Poisson.

## 6. Estrategia de solución

1. Consolidar los archivos anuales y normalizar departamentos (Bogotá separado de Cundinamarca por código DANE).
2. Agregar a conteos mensuales y completar con ceros los meses sin casos.
3. Evaluar con backtest: simular el corte de julio en 2023, 2024 y 2025, pronosticar agosto-diciembre con lo disponible hasta ese momento y comparar con el cierre real.
4. Comparar tres iteraciones:
   - **Línea base:** casos enero-julio escalados por el peso estacional histórico de cada delito.
   - **MLG binomial negativa:** `casos ~ serie + mes + tendencia por delito + covid`, enlace log. La sobredispersión se estima con la prueba de Cameron y Trivedi. Es la extensión a conteos de los modelos lineales generalizados vistos en Análisis Cuantitativo (logístico binario y multinomial): mismo marco, distinta familia y enlace; los coeficientes se interpretan como razones de tasas (IRR = e^β) en lugar de odds ratios.
   - **Gradient boosting con pérdida Poisson:** variables de rezago (mismo mes del año anterior), medias móviles de 3 y 12 meses, crecimiento anual, mes objetivo y horizonte. Hiperparámetros (profundidad, hoja mínima, regularización L2, tasa de aprendizaje) ajustados con validación cruzada temporal.
5. Seleccionar el modelo con menor WAPE del cierre (suma de errores absolutos / suma de cierres reales) y reentrenarlo con todo hasta julio de 2026.

## 7. Resultados principales

| Iteración | WAPE cierre | WAPE 2023 | WAPE 2024 | WAPE 2025 |
|---|---|---|---|---|
| 3. Boosting Poisson (seleccionado) | 7,6 % | 10,8 % | 6,8 % | 5,5 % |
| 1. Línea base | 7,8 % | 11,7 % | 6,3 % | 5,7 % |
| 2. MLG binomial negativa | 10,1 % | 11,5 % | 10,2 % | 8,6 % |

- Cierre nacional pronosticado 2026: unos 272.700 casos frente a 244.872 en 2025 (+11 %), con un rango de 268.000 a 301.000 basado en el error nacional del backtest.
- La binomial negativa reduce el AIC de 192.503 (Poisson) a 109.784, lo que confirma la sobredispersión.
- Según el MLG, la extorsión crece en promedio 11 % al año y el secuestro 26 %, mientras las amenazas caen 2,4 %; el confinamiento de 2020 redujo las denuncias un 23 %.
- La ventaja del boosting sobre la línea base es pequeña: la mayor parte de la señal está en el nivel reciente de cada serie (las medias móviles son las variables más importantes).

## 8. Uso del tablero

Los filtros de departamento y delito actualizan los indicadores, el pronóstico mensual, el mapa de variación y la tabla de series. Los filtros de género, edad y años solo afectan la pestaña Panorama, porque el modelo pronostica totales. La pestaña de rendimiento permite revisar el backtest de cada iteración y los coeficientes del MLG.

## 9. Limitaciones y uso responsable

- Los datos son casos registrados, no la criminalidad real: dependen de la denuncia, que varía por territorio y tipo de delito.
- Diciembre llega sub-registrado en los archivos anuales (por ejemplo 12.313 casos en diciembre de 2025), lo que afecta también los cierres reales usados en el backtest.
- El backtest tiene solo tres años; el error de 2023 fue mayor porque el segundo semestre de ese año creció más de lo esperado.
- Una ventana de entrenamiento de 36 meses mejora el MLG (WAPE 8,8 %), pero no se adoptó porque se identificó mirando los mismos años de prueba.
- Las series pequeñas (secuestro, departamentos de baja población) tienen errores relativos altos.
- El modelo no incorpora población, contexto de orden público ni cambios en la forma de registrar. El pronóstico apoya la planeación, pero no explica causas ni reemplaza el análisis territorial.
