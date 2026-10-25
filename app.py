"""Dashboard interactivo del pronóstico de cierre 2026 de delitos de alto impacto social."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

from src.config import CORTE_ANIO, CORTE_MES, PALETA_TENDENCIA


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = pd.read_csv(ROOT / "data" / "delitos_mensual.csv")
FORECAST = pd.read_csv(OUT / "pronostico_mensual.csv")
CIERRE = pd.read_csv(OUT / "pronostico_cierre.csv")
MODELS = pd.read_csv(OUT / "comparacion_modelos.csv")
BACKTEST = pd.read_csv(OUT / "backtest_cierres.csv")
COEF = pd.read_csv(OUT / "coeficientes_mlg.csv")
IMPORTANCE = pd.read_csv(OUT / "importancia_variables.csv")
SUMMARY = json.loads((OUT / "resumen_modelo.json").read_text(encoding="utf-8"))
GEO = json.loads((ROOT / "data" / "geo" / "colombia_departamentos.geojson").read_text(encoding="utf-8"))
COLORS = {"navy": "#12304A", "blue": "#1976D2", "cyan": "#26A6A1", "gold": "#F4B942", "red": "#D95D5D", "paper": "#F4F7FA"}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
GANADOR = SUMMARY["modelo_seleccionado"]
ANIO_ANT = CORTE_ANIO - 1

# El geojson usa nombres oficiales con tildes distintas y 3 alias (igual que en el portafolio).
_TILDES = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
_ALIAS_GEO = {"BOGOTA": "SANTAFE DE BOGOTA D.C", "GUAJIRA": "LA GUAJIRA",
              "SAN ANDRES": "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA"}
_GEO_POR_NOMBRE = {f["properties"]["NOMBRE_DPT"].translate(_TILDES).upper(): f["properties"]["NOMBRE_DPT"] for f in GEO["features"]}


# Vista fija del territorio continental más San Andrés (con fitbounds el archipiélago
# aleja tanto el zoom que el mapa queda diminuto).
GEO_VISTA = dict(visible=False, projection_type="mercator", lonaxis_range=[-82, -66.5], lataxis_range=[-4.3, 13.6])


def nombre_geo(depto):
    clave = depto.translate(_TILDES).upper()
    return _GEO_POR_NOMBRE.get(_ALIAS_GEO.get(clave, clave))


app = Dash(__name__, title="Pronostico de cierre 2026", suppress_callback_exceptions=True)
server = app.server


def card(title, value_id, note_id):
    return html.Div([html.P(title, className="kpi-title"), html.H2(id=value_id), html.P(id=note_id, className="kpi-note")], className="kpi-card")


app.layout = html.Div([
    html.Header([
        html.Div([html.P("DATOS ABIERTOS · POLICIA NACIONAL DE COLOMBIA", className="eyebrow"),
                  html.H1(f"Pronóstico de cierre {CORTE_ANIO} de delitos de alto impacto social"),
                  html.P(f"Casos observados de enero a {MESES[CORTE_MES - 1].lower()} y pronóstico del resto del año por departamento y tipo de delito. "
                         "Violencia intrafamiliar, amenazas, delitos sexuales, homicidio, extorsión y secuestro.", className="subtitle")]),
        html.Div("DASH + PLOTLY", className="badge"),
    ], className="hero"),
    html.Div([
        html.Div([
            html.H3("Filtros interactivos"),
            html.Label("Departamento"),
            dcc.Dropdown(sorted(DATA.departamento.unique()), multi=True, id="departamento", placeholder="Todos los departamentos"),
            html.Label("Tipo de delito"),
            dcc.Dropdown(sorted(DATA.delito.unique()), multi=True, id="delito", placeholder="Todos los delitos"),
            html.Hr(),
            html.Label("Género (solo Panorama)"),
            dcc.Dropdown(sorted(DATA.genero.unique()), multi=True, id="genero", placeholder="Todos"),
            html.Label("Grupo de edad (solo Panorama)"),
            dcc.Dropdown(sorted(DATA.grupo_edad.unique()), multi=True, id="edad", placeholder="Todos"),
            html.Label("Años (solo Panorama)"),
            dcc.RangeSlider(2020, CORTE_ANIO, 1, value=[2020, CORTE_ANIO], marks={a: str(a)[2:] for a in range(2020, CORTE_ANIO + 1)}, id="anios"),
            html.P("El modelo pronostica el total de cada departamento y delito, por eso género, edad y años solo filtran la pestaña Panorama.", className="help"),
            html.Button("Restablecer filtros", id="reset", n_clicks=0),
        ], className="sidebar"),
        html.Main([
            html.Div([card(f"Casos ene-{MESES[CORTE_MES - 1].lower()} {CORTE_ANIO}", "kpi-ytd", "note-ytd"),
                      card(f"Cierre {ANIO_ANT}", "kpi-prev", "note-prev"),
                      card(f"Cierre {CORTE_ANIO} pronosticado", "kpi-close", "note-close"),
                      card(f"Variación vs {ANIO_ANT}", "kpi-var", "note-var")], className="kpi-grid"),
            dcc.Tabs(id="tabs", value="panorama", children=[
                dcc.Tab(label="Panorama", value="panorama", children=[
                    html.Div([dcc.Graph(id="season-chart"), dcc.Graph(id="crime-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="map-cases"), dcc.Graph(id="top-chart")], className="grid-2"),
                ]),
                dcc.Tab(label=f"Pronóstico {CORTE_ANIO}", value="pronostico", children=[
                    html.Div([dcc.Graph(id="forecast-chart")], className="wide"),
                    html.Div([dcc.Graph(id="map-var"), dcc.Graph(id="close-chart")], className="grid-2"),
                    html.H3("Cierre pronosticado por departamento y delito", className="section-title"),
                    dash_table.DataTable(id="close-table", page_size=12, sort_action="native", filter_action="native",
                        style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "8px"},
                        style_header={"backgroundColor": COLORS["navy"], "color": "white", "fontWeight": "bold"},
                        style_data_conditional=[
                            {"if": {"filter_query": "{tendencia} = AUMENTO"}, "backgroundColor": "#FCE8E8"},
                            {"if": {"filter_query": "{tendencia} = DISMINUCIÓN"}, "backgroundColor": "#E6F4EA"},
                        ]),
                ]),
                dcc.Tab(label="Rendimiento del modelo", value="modelo", children=[
                    html.Div(id="metrics-strip", className="metric-strip"),
                    html.Div([dcc.Graph(id="model-chart"), html.Div([
                        html.Label("Iteración a revisar en el backtest", className="inline-label"),
                        dcc.Dropdown(MODELS.version.tolist(), GANADOR, id="version", clearable=False),
                        dcc.Graph(id="backtest-chart"),
                    ])], className="grid-2"),
                    html.Div([dcc.Graph(id="irr-chart"), dcc.Graph(id="importance-chart")], className="grid-2"),
                    html.Div([html.Div(id="narrative", className="narrative")], className="wide"),
                ]),
            ]),
            html.Footer(f"Fuente: Policía Nacional de Colombia, delitos a nivel de registro 2020-{CORTE_ANIO}. "
                        f"Corte observado: {MESES[CORTE_MES - 1].lower()} {CORTE_ANIO}. Evaluación con backtest de cierre en {', '.join(map(str, SUMMARY['anios_backtest']))}."),
        ], className="content")
    ], className="shell")
], className="app")


def filtrar(df, deptos, delitos):
    d = df
    if deptos: d = d[d.departamento.isin(deptos)]
    if delitos: d = d[d.delito.isin(delitos)]
    return d


@app.callback(
    Output("departamento", "value"), Output("delito", "value"), Output("genero", "value"), Output("edad", "value"), Output("anios", "value"),
    Input("reset", "n_clicks"), prevent_initial_call=True)
def reset_filters(_):
    return [], [], [], [], [2020, CORTE_ANIO]


@app.callback(
    Output("kpi-ytd", "children"), Output("note-ytd", "children"), Output("kpi-prev", "children"), Output("note-prev", "children"),
    Output("kpi-close", "children"), Output("note-close", "children"), Output("kpi-var", "children"), Output("note-var", "children"),
    Output("forecast-chart", "figure"), Output("map-var", "figure"), Output("close-chart", "figure"),
    Output("close-table", "data"), Output("close-table", "columns"),
    Input("departamento", "value"), Input("delito", "value"))
def update_forecast(deptos, delitos):
    template = "plotly_white"
    c = filtrar(CIERRE, deptos, delitos)
    ytd, prev, close = c.observado_ytd.sum(), c.cierre_anterior.sum(), c.cierre_pred.sum()
    if not deptos and not delitos:
        inf, sup = SUMMARY["cierre_nacional_inf"], SUMMARY["cierre_nacional_sup"]
    else:
        inf, sup = c.cierre_pred_inf.sum(), c.cierre_pred_sup.sum()
    ytd_prev = filtrar(DATA, deptos, delitos)
    ytd_prev = ytd_prev[(ytd_prev.anio == ANIO_ANT) & (ytd_prev.mes <= CORTE_MES)].casos.sum()
    var = (close - prev) / prev if prev else np.nan

    hist = filtrar(DATA, deptos, delitos).groupby(["anio", "mes"], as_index=False).casos.sum()
    hist["fecha"] = pd.to_datetime(dict(year=hist.anio, month=hist.mes, day=1))
    fut = filtrar(FORECAST, deptos, delitos).groupby(["anio", "mes"], as_index=False).pred.sum()
    fut["fecha"] = pd.to_datetime(dict(year=fut.anio, month=fut.mes, day=1))
    puente = hist.tail(1).rename(columns={"casos": "pred"})
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=hist.fecha, y=hist.casos, name="Observado", line=dict(color=COLORS["blue"], width=2.5)))
    fig_fc.add_trace(go.Scatter(x=pd.concat([puente.fecha, fut.fecha]), y=pd.concat([puente.pred, fut.pred]), name=f"Pronóstico ({GANADOR[3:]})",
                                line=dict(color=COLORS["red"], width=3, dash="dash")))
    fig_fc.update_layout(template=template, title="Casos mensuales observados y pronóstico del resto del año", yaxis_title="Casos", hovermode="x unified",
                         legend=dict(orientation="h", y=-0.15))

    by_depto = filtrar(CIERRE, None, delitos).groupby("departamento", as_index=False)[["cierre_anterior", "cierre_pred"]].sum()
    by_depto["variacion"] = (by_depto.cierre_pred - by_depto.cierre_anterior) / by_depto.cierre_anterior
    by_depto["geo"] = by_depto.departamento.map(nombre_geo)
    lim = max(0.05, float(by_depto.variacion.abs().quantile(0.9)))
    fig_map = px.choropleth(by_depto, geojson=GEO, locations="geo", featureidkey="properties.NOMBRE_DPT", color="variacion",
                            color_continuous_scale="RdYlGn_r", range_color=[-lim, lim], hover_name="departamento",
                            hover_data={"geo": False, "cierre_anterior": ":,.0f", "cierre_pred": ":,.0f", "variacion": ":.1%"},
                            title=f"Variación pronosticada {CORTE_ANIO} vs {ANIO_ANT} por departamento")
    fig_map.update_geos(**GEO_VISTA)
    fig_map.update_layout(template=template, coloraxis_colorbar=dict(title="", tickformat=".0%"), margin=dict(l=0, r=0, t=50, b=0))

    by_crime = c.groupby("delito", as_index=False)[["cierre_anterior", "cierre_pred", "cierre_pred_inf", "cierre_pred_sup"]].sum().sort_values("cierre_pred")
    fig_close = go.Figure()
    fig_close.add_trace(go.Bar(y=by_crime.delito, x=by_crime.cierre_anterior, name=f"Cierre {ANIO_ANT}", orientation="h", marker_color="#B8C7D6"))
    fig_close.add_trace(go.Bar(y=by_crime.delito, x=by_crime.cierre_pred, name=f"Pronóstico {CORTE_ANIO}", orientation="h", marker_color=COLORS["cyan"],
                               error_x=dict(type="data", symmetric=False, array=by_crime.cierre_pred_sup - by_crime.cierre_pred,
                                            arrayminus=by_crime.cierre_pred - by_crime.cierre_pred_inf, color=COLORS["navy"])))
    fig_close.update_layout(template=template, barmode="group", title="Cierre por tipo de delito", xaxis_title="Casos", legend=dict(orientation="h", y=-0.15))

    cols = ["departamento", "delito", "cierre_anterior", "observado_ytd", "cierre_pred", "cierre_pred_inf", "cierre_pred_sup", "variacion_pct", "tendencia"]
    table = c.sort_values("cierre_pred", ascending=False)[cols].copy()
    for col in ["cierre_pred", "cierre_pred_inf", "cierre_pred_sup"]:
        table[col] = table[col].round(0)
    table["variacion_pct"] = (table["variacion_pct"] * 100).round(1)
    nombres = {"departamento": "Departamento", "delito": "Delito", "cierre_anterior": f"Cierre {ANIO_ANT}", "observado_ytd": f"Ene-{MESES[CORTE_MES - 1]} {CORTE_ANIO}",
               "cierre_pred": f"Cierre {CORTE_ANIO}", "cierre_pred_inf": "Mínimo", "cierre_pred_sup": "Máximo", "variacion_pct": "Variación %", "tendencia": "Tendencia"}
    columns = [{"name": nombres[c_], "id": c_} for c_ in cols]

    return (f"{ytd:,.0f}", f"{(ytd - ytd_prev) / ytd_prev:+.1%} frente a ene-{MESES[CORTE_MES - 1].lower()} {ANIO_ANT}" if ytd_prev else "",
            f"{prev:,.0f}", "casos registrados en el año",
            f"{close:,.0f}", f"rango {inf:,.0f} - {sup:,.0f}",
            f"{var:+.1%}" if not np.isnan(var) else "--", "AUMENTO" if var > 0.05 else "DISMINUCIÓN" if var < -0.05 else "SIN CAMBIO",
            fig_fc, fig_map, fig_close, table.to_dict("records"), columns)


@app.callback(
    Output("season-chart", "figure"), Output("crime-chart", "figure"), Output("map-cases", "figure"), Output("top-chart", "figure"),
    Input("departamento", "value"), Input("delito", "value"), Input("genero", "value"), Input("edad", "value"), Input("anios", "value"))
def update_panorama(deptos, delitos, generos, edades, anios):
    template = "plotly_white"
    d = filtrar(DATA, deptos, delitos)
    if generos: d = d[d.genero.isin(generos)]
    if edades: d = d[d.grupo_edad.isin(edades)]
    d = d[d.anio.between(anios[0], anios[1])]

    season = d.groupby(["anio", "mes"], as_index=False).casos.sum()
    season["anio"] = season.anio.astype(str)
    fig_season = px.line(season, x="mes", y="casos", color="anio", markers=True, title="Casos por mes en cada año",
                         color_discrete_sequence=px.colors.sequential.Blues[2:] + [COLORS["red"]])
    fig_season.update_layout(template=template, xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=MESES), xaxis_title="",
                             yaxis_title="Casos", legend_title="Año")

    crime = d.groupby(["anio", "delito"], as_index=False).casos.sum()
    fig_crime = px.bar(crime, x="anio", y="casos", color="delito", title="Casos por año y tipo de delito",
                       color_discrete_sequence=[COLORS["navy"], COLORS["blue"], COLORS["cyan"], COLORS["gold"], COLORS["red"], "#8E7CC3"])
    fig_crime.update_layout(template=template, xaxis_title="", yaxis_title="Casos", legend_title="")
    if CORTE_ANIO in crime.anio.values:
        fig_crime.add_annotation(x=CORTE_ANIO, y=crime[crime.anio == CORTE_ANIO].casos.sum(), text=f"ene-{MESES[CORTE_MES - 1].lower()}", showarrow=False, yshift=10)

    by_depto = d.groupby("departamento", as_index=False).casos.sum()
    by_depto["geo"] = by_depto.departamento.map(nombre_geo)
    fig_map = px.choropleth(by_depto, geojson=GEO, locations="geo", featureidkey="properties.NOMBRE_DPT", color="casos",
                            color_continuous_scale="Reds", hover_name="departamento", hover_data={"geo": False, "casos": ":,.0f"},
                            title="Distribución geográfica de casos")
    fig_map.update_geos(**GEO_VISTA)
    fig_map.update_layout(template=template, coloraxis_colorbar=dict(title=""), margin=dict(l=0, r=0, t=50, b=0))

    top = by_depto.sort_values("casos").tail(12)
    fig_top = px.bar(top, x="casos", y="departamento", orientation="h", title="Departamentos con más casos", color_discrete_sequence=[COLORS["cyan"]],
                     text=top.casos.map(lambda x: f"{x:,.0f}"))
    fig_top.update_layout(template=template, xaxis_title="Casos", yaxis_title="")
    return fig_season, fig_crime, fig_map, fig_top


@app.callback(
    Output("metrics-strip", "children"), Output("model-chart", "figure"), Output("backtest-chart", "figure"),
    Output("irr-chart", "figure"), Output("importance-chart", "figure"), Output("narrative", "children"),
    Input("version", "value"))
def update_model(version):
    template = "plotly_white"
    best = MODELS.iloc[0]
    cells = [("Ganador", best.version[3:]), ("WAPE cierre", f"{best.wape:.1%}"), ("MAE por serie", f"{best.mae_cierre:,.0f}"),
             ("Error nacional", f"{best.error_nacional_pct:.1%}"), ("Series", f"{SUMMARY['series']}")]
    strip = [html.Div([html.Span(k), html.Strong(v)]) for k, v in cells]

    wape_cols = [f"wape_{a}" for a in SUMMARY["anios_backtest"]]
    mm = MODELS.melt(id_vars="version", value_vars=wape_cols, var_name="anio", value_name="error")
    mm["anio"] = mm.anio.str.replace("wape_", "Backtest ")
    fig_models = px.bar(mm, x="version", y="error", color="anio", barmode="group", title="Error del cierre (WAPE) por iteración y año",
                        color_discrete_sequence=[COLORS["blue"], COLORS["cyan"], COLORS["gold"]])
    fig_models.update_layout(template=template, yaxis_tickformat=".0%", xaxis_title="", yaxis_title="WAPE (menor es mejor)", legend_title="")

    bt = BACKTEST[BACKTEST.version == version].copy()
    bt["anio"] = bt.anio.astype(str)
    fig_bt = px.scatter(bt, x="cierre_real", y="cierre_pred", color="anio", log_x=True, log_y=True, opacity=0.7,
                        hover_data=["departamento", "delito"], title="Cierre real vs pronosticado por serie (backtest)",
                        color_discrete_sequence=[COLORS["blue"], COLORS["cyan"], COLORS["gold"]])
    lo, hi = max(1, bt[["cierre_real", "cierre_pred"]].min().min()), bt[["cierre_real", "cierre_pred"]].max().max()
    fig_bt.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="Igualdad"))
    fig_bt.update_layout(template=template, xaxis_title="Cierre real", yaxis_title="Cierre pronosticado", legend_title="")

    irr = COEF.copy()
    irr["etiqueta"] = (irr.termino.str.replace(r"C\(mes\)\[T\.(\d+)\]", lambda m: f"Mes: {MESES[int(m.group(1)) - 1]} vs Ene", regex=True)
                       .str.replace(r"C\(delito\)\[(.+)\]:t", r"Tendencia anual: \1", regex=True)
                       .str.replace("covid", "Confinamiento covid"))
    irr = irr.iloc[::-1]
    fig_irr = go.Figure(go.Scatter(x=irr.irr, y=irr.etiqueta, mode="markers", marker=dict(color=COLORS["navy"], size=8),
                                   error_x=dict(type="data", symmetric=False, array=irr.irr_sup - irr.irr, arrayminus=irr.irr - irr.irr_inf)))
    fig_irr.add_vline(x=1, line_dash="dash", line_color=COLORS["gold"])
    fig_irr.update_layout(template=template, title="MLG binomial negativa: razones de tasas (IRR) e IC 95%", xaxis_title="IRR = e^β (1 = sin efecto)",
                          height=520, margin=dict(l=10))

    imp = IMPORTANCE.sort_values("importancia")
    fig_imp = px.bar(imp, x="importancia", y="variable", orientation="h", title="Boosting: importancia por permutación (backtest 2025)",
                     color_discrete_sequence=[COLORS["cyan"]])
    fig_imp.update_layout(template=template, xaxis_title="Aumento de la devianza Poisson", yaxis_title="", height=520)

    tend = COEF.set_index("termino")
    extorsion, secuestro, amenaza = (tend.loc[f"C(delito)[{d}]:t", "irr"] for d in ["Extorsión", "Secuestro", "Amenaza"])
    covid = tend.loc["covid", "irr"]
    err = SUMMARY["error_nacional_resto_backtest"]
    narrative = [html.H3("Lectura ejecutiva"),
        html.P(f"Se compararon tres iteraciones simulando el corte de julio en {', '.join(map(str, SUMMARY['anios_backtest']))}. "
               f"El modelo {GANADOR[3:]} obtuvo el menor error del cierre por serie (WAPE {best.wape:.1%}), seguido de cerca por la línea base estacional; "
               "la diferencia entre ambos es pequeña, lo que indica que buena parte de la señal está en el nivel reciente de cada serie."),
        html.P(f"El cierre nacional pronosticado para {CORTE_ANIO} es de {SUMMARY['cierre_nacional_pronosticado']:,.0f} casos frente a "
               f"{SUMMARY['cierre_nacional_anterior']:,} en {ANIO_ANT}. El rango se construye con el error nacional observado en el backtest "
               f"({', '.join(f'{a}: {v:+.0%}' for a, v in err.items())} sobre lo pronosticado de agosto a diciembre)."),
        html.P(f"El MLG binomial negativa no fue el más preciso, pero explica el comportamiento: la extorsión crece en promedio {extorsion - 1:.0%} al año y el secuestro "
               f"{secuestro - 1:.0%}, mientras las amenazas caen {1 - amenaza:.1%}. El confinamiento de 2020 redujo las denuncias un {1 - covid:.0%}. "
               f"La sobredispersión (alpha = {SUMMARY['alpha_binomial_negativa']:.3f}) justifica la binomial negativa sobre Poisson "
               f"(AIC {SUMMARY['aic_binomial_negativa']:,.0f} frente a {SUMMARY['aic_poisson']:,.0f})."),
        html.P("Los datos son casos registrados, no la criminalidad real: dependen de la denuncia y del rezago de registro (diciembre suele cargarse incompleto). "
               "El pronóstico describe una tendencia esperada y no explica causas; no reemplaza el análisis de contexto de cada territorio."),
    ]
    return strip, fig_models, fig_bt, fig_irr, fig_imp, narrative


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
