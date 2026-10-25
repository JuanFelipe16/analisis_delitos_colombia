"""Crea el informe PDF del proyecto y sus gráficos de apoyo."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import CORTE_ANIO, CORTE_MES


# Editar con los nombres de todos los integrantes del grupo.
AUTORES = "Juan Felipe Moreno"
ASIGNATURA = "Ciencia de Datos II"

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "tmp" / "report"
TMP.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(ROOT / "data" / "delitos_mensual.csv")
models = pd.read_csv(ROOT / "outputs" / "comparacion_modelos.csv")
cierre = pd.read_csv(ROOT / "outputs" / "pronostico_cierre.csv")
forecast = pd.read_csv(ROOT / "outputs" / "pronostico_mensual.csv")
coef = pd.read_csv(ROOT / "outputs" / "coeficientes_mlg.csv")
importance = pd.read_csv(ROOT / "outputs" / "importancia_variables.csv")
summary = json.loads((ROOT / "outputs" / "resumen_modelo.json").read_text(encoding="utf-8"))

NAVY, BLUE, CYAN, GOLD, RED, LIGHT = "#12304A", "#1976D2", "#26A6A1", "#F4B942", "#D95D5D", "#F4F7FA"
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MES_CORTE = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"][CORTE_MES - 1]
ANIO_ANT = CORTE_ANIO - 1
best = models.iloc[0]
irr = coef.set_index("termino")["irr"]
var_nac = summary["cierre_nacional_pronosticado"] / summary["cierre_nacional_anterior"] - 1


def num(x, dec=0):
    """Formato colombiano: punto de miles y coma decimal."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, dec=1):
    return num(x * 100, dec) + " %"


def chart_series(path):
    nac = data.groupby(["anio", "mes"]).casos.sum()
    fechas = pd.to_datetime([f"{a}-{m:02d}-01" for a, m in nac.index])
    fut = forecast.groupby("mes").pred.sum()
    fechas_fut = pd.to_datetime([f"{CORTE_ANIO}-{m:02d}-01" for m in fut.index])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(fechas, nac.values, color=BLUE, lw=1.8, label="Observado")
    ax.plot([fechas[-1], *fechas_fut], [nac.values[-1], *fut.values], "--", color=RED, lw=2.4, label="Pronóstico ago-dic")
    ax.set_title("Casos mensuales nacionales y pronóstico del resto de 2026", loc="left", weight="bold", color=NAVY)
    ax.set_ylabel("Casos"); ax.legend(frameon=False); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def chart_models(path):
    m = models.set_index("version")[[f"wape_{a}" for a in summary["anios_backtest"]]]
    ax = m.plot.bar(figsize=(9, 4), color=[BLUE, CYAN, GOLD])
    ax.set_ylabel("WAPE del cierre"); ax.set_xlabel(""); ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.set_title("Error del cierre por iteración y año de backtest (menor es mejor)", loc="left", weight="bold", color=NAVY)
    ax.legend([f"Backtest {a}" for a in summary["anios_backtest"]], frameon=False, fontsize=8)
    plt.xticks(rotation=0); fig = ax.get_figure(); fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def chart_irr(path):
    c = coef.copy()
    c["etiqueta"] = (c.termino.str.replace(r"C\(mes\)\[T\.(\d+)\]", lambda m: f"Mes {MESES[int(m.group(1)) - 1]} vs ene", regex=True)
                     .str.replace(r"C\(delito\)\[(.+)\]:t", r"Tendencia anual: \1", regex=True)
                     .str.replace("covid", "Confinamiento covid"))
    c = c.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.errorbar(c.irr, c.etiqueta, xerr=[c.irr - c.irr_inf, c.irr_sup - c.irr], fmt="o", color=NAVY, ecolor=NAVY, capsize=3)
    ax.axvline(1, ls="--", color=GOLD)
    ax.set_title("MLG binomial negativa: razones de tasas (IRR) e IC 95 %", loc="left", weight="bold", color=NAVY)
    ax.set_xlabel("IRR = exp(coeficiente); 1 = sin efecto"); ax.tick_params(axis="y", labelsize=8); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def chart_importance(path):
    s = importance.sort_values("importancia")
    fig, ax = plt.subplots(figsize=(8.4, 3.6)); ax.barh(s.variable, s.importancia, color=CYAN)
    ax.set_title("Boosting: importancia por permutación (backtest 2025)", loc="left", weight="bold", color=NAVY)
    ax.set_xlabel("Aumento de la devianza Poisson al permutar la variable"); fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def chart_crimes(path):
    s = cierre.groupby("delito")[["cierre_anterior", "cierre_pred"]].sum().sort_values("cierre_pred")
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    y = range(len(s))
    ax.barh([i + .2 for i in y], s.cierre_anterior, height=.4, color="#B8C7D6", label=f"Cierre {ANIO_ANT}")
    ax.barh([i - .2 for i in y], s.cierre_pred, height=.4, color=CYAN, label=f"Pronóstico {CORTE_ANIO}")
    ax.set_yticks(list(y)); ax.set_yticklabels(s.index)
    for i, (a, p) in enumerate(zip(s.cierre_anterior, s.cierre_pred)):
        ax.text(p, i - .2, f"  {(p / a - 1):+.0%}", va="center", fontsize=8, color=NAVY)
    ax.set_title(f"Cierre por tipo de delito: {ANIO_ANT} vs pronóstico {CORTE_ANIO}", loc="left", weight="bold", color=NAVY)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


chart_series(TMP / "serie.png"); chart_models(TMP / "modelos.png"); chart_irr(TMP / "irr.png")
chart_importance(TMP / "importancia.png"); chart_crimes(TMP / "delitos.png")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleWhite", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=26, leading=31, textColor=colors.white, alignment=TA_LEFT, spaceAfter=14))
styles.add(ParagraphStyle(name="SubWhite", parent=styles["BodyText"], fontSize=12, leading=18, textColor=colors.HexColor("#DDEBF4")))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor(NAVY), spaceBefore=8, spaceAfter=10))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor(BLUE), spaceBefore=9, spaceAfter=6))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.5, leading=14, textColor=colors.HexColor("#283B4A"), spaceAfter=7))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#5F7180")))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=11, leading=16, textColor=colors.HexColor(NAVY), alignment=TA_CENTER))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def tabla(rows, widths):
    t = Table([[P(str(x), "Smallx") for x in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#C7D7E2")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    # La cabecera debe ir en blanco aunque el estilo Smallx sea gris.
    for i in range(len(rows[0])):
        t._cellvalues[0][i] = P(f"<font color='white'><b>{rows[0][i]}</b></font>", "Smallx")
    return t


def header_footer(canvas, doc):
    canvas.saveState(); w, h = A4
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#DDE6EE")); canvas.line(1.6*cm, h-1.25*cm, w-1.6*cm, h-1.25*cm)
        canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#667986"))
        canvas.drawString(1.6*cm, h-1.05*cm, f"Pronóstico de cierre {CORTE_ANIO} de delitos de alto impacto social - {ASIGNATURA}")
        canvas.drawRightString(w-1.6*cm, .85*cm, f"Página {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(ROOT / "INFORME_PROYECTO.pdf"), pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm, topMargin=1.55*cm, bottomMargin=1.35*cm,
                        title=f"Informe - Pronóstico de cierre {CORTE_ANIO} de delitos", author=AUTORES)
story = []

# Portada
cover = Table([[P("PROYECTO FINAL", "SubWhite")], [P(f"Pronóstico de cierre {CORTE_ANIO} de delitos de alto impacto social en Colombia", "TitleWhite")],
               [P("Modelado, evaluación por backtest y comunicación interactiva de resultados", "SubWhite")],
               [Spacer(1, 4.0*cm)], [P(f"{AUTORES}<br/>{ASIGNATURA}", "SubWhite")],
               [P("Datos Policía Nacional 2020-2026 | Dash + Plotly + scikit-learn + statsmodels", "SubWhite")]],
              colWidths=[17.8*cm], rowHeights=[1.4*cm, 3.4*cm, 1.5*cm, 4.0*cm, 1.6*cm, 1.2*cm])
cover.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(NAVY)), ("LEFTPADDING", (0, 0), (-1, -1), 28), ("RIGHTPADDING", (0, 0), (-1, -1), 28),
                           ("TOPPADDING", (0, 0), (-1, -1), 14), ("BOTTOMPADDING", (0, 0), (-1, -1), 14)]))
story += [Spacer(1, 1.5*cm), cover, Spacer(1, .7*cm),
          P("Actividad final: presentación de los resultados obtenidos durante el proceso de modelado mediante un dashboard interactivo.", "Smallx"), PageBreak()]

# Resumen ejecutivo
kpis = Table([[P(f"Cierre {ANIO_ANT} registrado", "Smallx"), P(f"Cierre {CORTE_ANIO} pronosticado", "Smallx"), P("Error del modelo (WAPE)", "Smallx")],
              [P(num(summary["cierre_nacional_anterior"]), "Callout"), P(f"{num(summary['cierre_nacional_pronosticado'])} (+{pct(var_nac)})", "Callout"), P(pct(best.wape), "Callout")]],
             colWidths=[5.9*cm]*3,
             style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5EEF5")), ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#C7D7E2")),
                               ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#C7D7E2")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
story += [P("Resumen ejecutivo", "H1x"),
          P(f"A mitad de año solo se conoce una parte de los delitos registrados y no es evidente si un departamento cerrará por encima o por debajo del año anterior. "
            f"Con <b>{num(data.casos.sum())} casos</b> de seis delitos de alto impacto social registrados por la Policía Nacional entre enero de 2020 y {MES_CORTE} de {CORTE_ANIO}, "
            f"se pronosticó el cierre de {CORTE_ANIO} para <b>{summary['series']} series</b> (departamento por tipo de delito) y se compararon tres iteraciones del modelo."),
          kpis,
          P("Hallazgo principal", "H2x"),
          P(f"El modelo seleccionado, <b>{best.version[3:]}</b>, pronostica el cierre por serie con un error de {pct(best.wape)} en el backtest de {', '.join(map(str, summary['anios_backtest']))}. "
            f"Se espera que {CORTE_ANIO} cierre con unos {num(summary['cierre_nacional_pronosticado'])} casos (rango {num(summary['cierre_nacional_inf'])} a {num(summary['cierre_nacional_sup'])}), "
            f"{pct(var_nac)} más que {ANIO_ANT}, impulsado por violencia intrafamiliar, amenazas y extorsión."),
          P("Interpretación", "H2x"),
          P(f"El MLG binomial negativa, aunque menos preciso, explica el comportamiento: la extorsión crece {pct(irr['C(delito)[Extorsión]:t'] - 1, 0)} al año y el secuestro "
            f"{pct(irr['C(delito)[Secuestro]:t'] - 1, 0)}, las amenazas caen {pct(1 - irr['C(delito)[Amenaza]:t'])} y el confinamiento de 2020 redujo las denuncias un {pct(1 - irr['covid'], 0)}."),
          P("Entregables incluidos", "H2x"),
          P("Caso de estudio, datos agregados y diccionario, notebook de análisis, scripts reproducibles (ingesta, agregación, entrenamiento), modelo serializado, resultados, dashboard, pruebas, archivos para GitHub y Binder y este informe."),
          PageBreak()]

# 1. Caso
story += [P("1. Definición del caso", "H1x"), P("Pregunta", "H2x"),
          P(f"Con los casos registrados hasta {MES_CORTE}, ¿cuántos casos tendrá cada departamento y tipo de delito al cierre de {CORTE_ANIO}, y en cuáles se espera un aumento frente a {ANIO_ANT}?"),
          P("Objetivo analítico", "H2x"),
          P("Construir y evaluar un modelo que pronostique el cierre anual de casos por departamento y delito, y comunicar los resultados en un dashboard que permita identificar dónde se esperan aumentos."),
          P("Unidad de análisis y variable objetivo", "H2x"),
          P(f"La unidad es la serie mensual de casos de un departamento y un delito. El objetivo es el cierre anual: <b>casos observados de enero a {MES_CORTE} + casos pronosticados del resto del año</b>. "
            "Se pronostica el resto del año mes a mes y se suma a lo observado."),
          P("Hipótesis", "H2x"),
          P("El nivel reciente de cada serie, su estacionalidad y la tendencia de cada tipo de delito permiten anticipar el resto del año. Los conteos presentarían sobredispersión, lo que favorece la binomial negativa sobre Poisson."),
          P("Criterio de éxito", "H2x"),
          P("La métrica principal es el WAPE del cierre (suma de errores absolutos sobre la suma de cierres reales), que pondera cada serie por su tamaño. Se complementa con MAE y RMSE por serie, error del total nacional y error por año de backtest."),
          Image(str(TMP / "serie.png"), width=17*cm, height=7.2*cm),
          P("Figura 1. Serie nacional. Se observa estacionalidad (picos en marzo y agosto-octubre), la caída por el confinamiento de 2020 y el nivel más alto de 2026.", "Smallx"),
          PageBreak()]

# 2. Datos
story += [P("2. Datos y procesamiento", "H1x"),
          P("Los datos provienen de los archivos anuales de delitos a nivel de registro de la Policía Nacional (2020-2026). Cada fila del archivo original es un caso individual."),
          tabla([["Etapa", "Script", "Resultado"],
                 ["Ingesta", "src/ingesta.py", "Detecta la hoja y el encabezado de cada Excel, unifica nombres de columnas entre años y filtra los seis delitos (1,6 millones de casos)"],
                 ["Normalización", "src/config.py", "Unifica nombres de departamentos y separa Bogotá de Cundinamarca por código DANE (en 2020-2024 venía como Cundinamarca)"],
                 ["Agregación", "src/agregar.py", "Conteos por departamento, delito, mes, género y edad (47.884 filas, 3 MB, versionado en GitHub)"],
                 ["Panel", "entrenar_modelo.py", "193 series x 79 meses, completando con cero los meses sin casos (12,9 % de los meses-serie)"]],
                [2.7*cm, 3.4*cm, 11*cm]),
          P("Decisión de corte", "H2x"),
          P(f"Agosto de {CORTE_ANIO} aparece en el archivo con 19.672 casos frente a 22.041 en julio, cuando históricamente agosto supera a julio. Se interpretó como sub-registro (denuncias aún no cargadas) y el corte se fijó en {MES_CORTE} de {CORTE_ANIO}."),
          P("Rezago de registro en diciembre", "H2x"),
          P("Diciembre es bajo en todos los archivos anuales (12.313 casos en 2025 y 15.608 en 2024, frente a más de 20.000 en meses normales). Parte de los hechos de fin de año se registran después de la fecha de corte del archivo. "
            "Los modelos aprenden este patrón, por lo que el pronóstico corresponde a casos <i>registrados</i> con el mismo criterio de los años anteriores."),
          P("Sobredispersión", "H2x"),
          P(f"El 98 % de las series tiene varianza mensual mayor que su media. La prueba de Cameron y Trivedi estima alpha = {summary['alpha_binomial_negativa']:.3f} "
            f"(Var = mu + alpha mu<super>2</super>) y la binomial negativa reduce el AIC de {num(summary['aic_poisson'])} (Poisson) a {num(summary['aic_binomial_negativa'])}."),
          P("Calidad", "H2x"),
          P("El conjunto agregado no tiene faltantes; los casos sin género o edad se agrupan como NO REPORTA. Las pruebas automáticas verifican el esquema, que no haya meses posteriores al corte y que el panel conserve el total de casos."),
          PageBreak()]

# 3. Iteraciones
rows = [["Iteración", "Variables", "Regularización / ajuste", "WAPE", "MAE serie", "Error nacional"]]
for r in models.itertuples():
    rows.append([r.version, r.variables, r.regularizacion, pct(r.wape), num(r.mae_cierre), pct(r.error_nacional_pct)])
story += [P("3. Mejora iterativa del modelo", "H1x"),
          P("Evaluación por backtest", "H2x"),
          P(f"El cierre de {CORTE_ANIO} aún no se conoce, así que no sirve para medir el error. Se simuló el mismo problema en años ya cerrados: con los datos hasta {MES_CORTE} de 2023, 2024 y 2025 "
            "se entrenó cada modelo, se pronosticó el resto del año y se comparó con el cierre real. Ningún modelo ve información posterior al corte simulado."),
          Image(str(TMP / "modelos.png"), width=12.6*cm, height=5.6*cm),
          P("Figura 2. Error por iteración y año. 2023 es el año más difícil para todos: su segundo semestre creció más de lo que anticipaba la historia.", "Smallx"),
          tabla(rows, [3.4*cm, 4.6*cm, 3.9*cm, 1.6*cm, 1.8*cm, 1.8*cm]),
          P("Iteraciones", "H2x"),
          P("<b>1. Línea base:</b> casos de enero a julio escalados por el peso estacional promedio de cada delito en los tres años previos. Es simple y difícil de superar porque ya incorpora el nivel del año en curso. "
            "<b>2. MLG binomial negativa:</b> log(casos esperados) = efecto de la serie + mes + tendencia lineal por delito + indicador de confinamiento, con alpha estimado por Cameron y Trivedi. "
            "<b>3. Gradient boosting con pérdida Poisson:</b> para cada mes de origen y horizonte de 1 a 5 meses usa el rezago de 12 meses, las medias móviles de 3 y 12 meses, el crecimiento anual, el mes objetivo, el departamento y el delito."),
          P("Ajuste de hiperparámetros y regularización", "H2x"),
          P(f"En el boosting se buscaron profundidad (3, 6), tasa de aprendizaje (0,05; 0,1), número de árboles (200, 400), tamaño mínimo de hoja (20, 60) y penalización L2 (0, 1) con validación cruzada temporal "
            f"(TimeSeriesSplit de 4 pliegues) dentro del entrenamiento de cada backtest. La combinación elegida en el modelo final ({', '.join(f'{k}={v}' for k, v in summary['parametros_modelo_final'].items())}) "
            "es la más regularizada de la rejilla: árboles poco profundos, hojas grandes y L2 activa, lo que indica que más complejidad producía sobreajuste. "
            "En el MLG la regularización proviene del parámetro de dispersión, que evita intervalos y errores estándar artificialmente estrechos."),
          P("Selección", "H2x"),
          P(f"Se seleccionó <b>{best.version}</b> por el menor WAPE ({pct(best.wape)}). La ventaja sobre la línea base es pequeña ({num((models.iloc[1].wape - best.wape) * 100, 1)} puntos porcentuales), así que la conclusión honesta es que ambos son competitivos; "
            "el boosting gana en 2023 y 2025 y la línea base en 2024. Una ventana de entrenamiento de 36 meses mejora el MLG a 8,8 %, pero no se adoptó porque se identificó mirando los mismos años de prueba."),
          PageBreak()]

# 4. Interpretación
story += [P("4. Interpretación", "H1x"),
          Image(str(TMP / "irr.png"), width=15.5*cm, height=10.3*cm),
          P("Figura 3. Coeficientes del MLG como razones de tasas. Un valor de 1,10 significa 10 % más casos esperados manteniendo constante lo demás.", "Smallx"),
          P("Lectura de las razones de tasas", "H2x"),
          P("Igual que el odds ratio en la regresión logística (vista en Análisis Cuantitativo), la razón de tasas (IRR = exp(coeficiente)) es un cambio multiplicativo, aquí sobre el número esperado de casos. "
            f"Frente a enero, agosto a octubre tienen cerca de 10 % más casos y diciembre {pct(1 - irr['C(mes)[T.12]'], 0)} menos. La extorsión ({pct(irr['C(delito)[Extorsión]:t'] - 1, 0)} anual) y el secuestro "
            f"({pct(irr['C(delito)[Secuestro]:t'] - 1, 0)} anual) son los delitos con mayor crecimiento; los delitos sexuales ({pct(1 - irr['C(delito)[Delitos sexuales]:t'])}) y las amenazas ({pct(1 - irr['C(delito)[Amenaza]:t'])}) disminuyen."),
          Image(str(TMP / "importancia.png"), width=15.5*cm, height=6.6*cm),
          P("Figura 4. Importancia por permutación medida fuera de muestra. Las medias móviles dominan: el nivel reciente de la serie es la información más útil, lo que explica por qué la línea base compite tan bien.", "Smallx"),
          PageBreak()]

# 5. Pronóstico y dashboard
top = cierre[cierre.cierre_anterior >= 1000].sort_values("variacion_pct", ascending=False).head(5)
story += [P(f"5. Pronóstico de cierre {CORTE_ANIO} y dashboard", "H1x"),
          Image(str(TMP / "delitos.png"), width=16.4*cm, height=7.4*cm),
          P("Figura 5. Cierre por delito. La barra turquesa es el pronóstico (observado enero-julio más pronóstico agosto-diciembre).", "Smallx"),
          P("Resultados", "H2x"),
          P(f"De las {summary['series']} series, {(cierre.tendencia == 'AUMENTO').sum()} se clasifican en AUMENTO (más de 5 % sobre {ANIO_ANT}), {(cierre.tendencia == 'DISMINUCIÓN').sum()} en DISMINUCIÓN y "
            f"{(cierre.tendencia == 'SIN CAMBIO').sum()} SIN CAMBIO. Entre las series con más de 1.000 casos, los mayores aumentos esperados son: "
            + "; ".join(f"{r.departamento.title()} - {r.delito.lower()} ({r.variacion_pct:+.0%})" for r in top.itertuples()) + "."),
          P("Rango del pronóstico", "H2x"),
          P("Para cada serie, el rango usa los cuantiles 10 % y 90 % del error relativo del resto del año observado en el backtest, por delito. Para el total nacional no se suman esos rangos "
            "(supondría que todas las series fallan al extremo a la vez); se usa el error nacional del backtest ("
            + ", ".join(f"{a}: {v:+.0%}" for a, v in summary["error_nacional_resto_backtest"].items()) + " sobre lo pronosticado de agosto a diciembre)."),
          P("Narrativa del tablero", "H2x"),
          P("<b>Panorama:</b> estacionalidad por año, casos por delito, mapa y departamentos con más casos. <b>Pronóstico:</b> serie mensual observada con el pronóstico, mapa de variación frente al año anterior, "
            "cierre por delito con su rango y tabla filtrable de las 193 series. <b>Rendimiento del modelo:</b> métricas, error por iteración, dispersión real contra pronosticado del backtest de cada iteración, razones de tasas del MLG e importancia de variables."),
          P("Interactividad y diseño", "H2x"),
          P("Los filtros de departamento y delito actualizan indicadores, gráficos y tabla. Género, edad y años filtran solo Panorama porque el modelo pronostica totales, y el tablero lo indica. "
            "Rojo representa aumento y verde disminución; el dashboard carga resultados precalculados y no reentrena en cada interacción. Funciona en local y en Binder (detecta el prefijo del proxy)."),
          PageBreak()]

# 6. Conclusiones
checks = [["Componente", "Verificación"],
          ["Datos", "33 departamentos, 6 delitos, sin faltantes ni meses posteriores al corte; el panel conserva el total de casos"],
          ["Modelo", "Tres iteraciones con backtest 2023-2025; el ganador coincide con el menor WAPE"],
          ["Pronóstico", "Cierre = observado + pronóstico en cada serie; rango mínimo <= máximo; tendencias válidas"],
          ["Dashboard", "Carga, callbacks de las tres pestañas, filtros, mapa y tabla; prueba visual de cada pestaña"],
          ["Código", "pytest (4 pruebas) y ejecución completa del notebook sin errores"]]
story += [P("6. Conclusiones, recomendaciones y pruebas", "H1x"),
          P("Conclusiones", "H2x"),
          P(f"1. Con el corte de {MES_CORTE} es posible pronosticar el cierre anual por departamento y delito con un error cercano al {num(best.wape * 100, 0)} % y de {pct(best.error_nacional_pct, 0)} en el total nacional. "
            "2. El boosting fue el más preciso, pero la línea base queda muy cerca: la señal principal es el nivel reciente de cada serie. "
            "3. El MLG binomial negativa es el que explica el comportamiento y confirma la sobredispersión de los conteos. "
            f"4. Se espera un cierre {CORTE_ANIO} alrededor de {pct(var_nac, 0)} superior a {ANIO_ANT}. 5. Los resultados describen casos registrados y tendencias, no causas."),
          P("Recomendaciones para una siguiente iteración", "H2x"),
          P("Normalizar por población (tasas por 100.000 habitantes con proyecciones del DANE); modelar el rezago de registro para corregir los últimos meses; incorporar variables de contexto (orden público, festividades); "
            "ampliar el backtest con cortes en distintos meses; y evaluar la multinomial AUMENTO / SIN CAMBIO / DISMINUCIÓN como modelo directo de la tendencia."),
          P("Pruebas realizadas", "H2x"),
          tabla(checks, [3.4*cm, 13.7*cm]),
          P("Limitaciones", "H2x"),
          P("Los datos son casos registrados y dependen de la denuncia, que varía por territorio y delito. Diciembre llega sub-registrado, lo que afecta también los cierres reales del backtest. "
            "El backtest tiene solo tres años y las series pequeñas (secuestro, departamentos de baja población) tienen errores relativos altos. El modelo no incorpora población ni contexto."),
          P("Uso responsable", "H2x"),
          P("El pronóstico apoya la planeación (por ejemplo, anticipar demanda en comisarías de familia o priorizar prevención de extorsión), pero no debe usarse para estigmatizar territorios ni como única base de decisiones. "
            "Debe acompañarse de análisis de contexto y revisión de expertos."),
          Spacer(1, .4*cm), P("Fin del informe", "Callout")]

doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"Creado {ROOT / 'INFORME_PROYECTO.pdf'}")
