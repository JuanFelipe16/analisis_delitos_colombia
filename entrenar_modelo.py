"""Entrena, compara y documenta tres iteraciones del pronóstico de cierre anual.

Pregunta: con los casos observados hasta julio, ¿cuántos casos tendrá cada
departamento y tipo de delito al cierre del año?

    cierre = casos observados ene-jul + pronóstico ago-dic

Como el cierre de 2026 aún no se conoce, cada modelo se evalúa con un
backtest: se simula el mismo corte (julio) en 2023, 2024 y 2025, se
pronostica ago-dic y se compara con el cierre real. El modelo ganador se
reentrena con todo hasta julio de 2026 para producir el pronóstico final.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.config import CORTE_ANIO, CORTE_MES, MENSUAL_PATH


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
ANIOS_BACKTEST = [2023, 2024, 2025]
MESES_RESTO = list(range(CORTE_MES + 1, 13))
HORIZONTE = len(MESES_RESTO)
SERIE = ["departamento", "delito"]
# Meses de confinamiento estricto por covid (caída atípica de denuncias).
COVID = pd.period_range("2020-03", "2020-08", freq="M")
UMBRAL_TENDENCIA = 0.05


def construir_panel(mensual):
    """Una fila por serie (departamento x delito) y mes, con ceros donde no hubo casos."""
    total = mensual.groupby(SERIE + ["anio", "mes"], as_index=False)["casos"].sum()
    total["fecha"] = pd.PeriodIndex.from_fields(year=total["anio"], month=total["mes"], freq="M")
    fechas = pd.period_range("2020-01", f"{CORTE_ANIO}-{CORTE_MES:02d}", freq="M")
    series = total[SERIE].drop_duplicates()
    idx = pd.MultiIndex.from_tuples(
        [(d, t, f) for d, t in series.itertuples(index=False) for f in fechas], names=SERIE + ["fecha"]
    )
    panel = total.set_index(SERIE + ["fecha"])["casos"].reindex(idx, fill_value=0).reset_index()
    panel["anio"] = panel["fecha"].dt.year
    panel["mes"] = panel["fecha"].dt.month
    panel["idx"] = (panel["fecha"] - fechas[0]).apply(lambda x: x.n)
    panel["t"] = panel["idx"] / 12
    panel["covid"] = panel["fecha"].isin(COVID).astype(int)
    panel["serie"] = panel["departamento"] + " | " + panel["delito"]
    return panel


def filas_resto(panel, anio):
    """Filas ago-dic del año a pronosticar (con los mismos campos que el panel)."""
    series = panel[SERIE + ["serie"]].drop_duplicates()
    fechas = pd.period_range(f"{anio}-{CORTE_MES + 1:02d}", f"{anio}-12", freq="M")
    futuro = series.merge(pd.DataFrame({"fecha": fechas}), how="cross")
    futuro["anio"] = anio
    futuro["mes"] = futuro["fecha"].dt.month
    futuro["idx"] = (futuro["fecha"] - pd.Period("2020-01", freq="M")).apply(lambda x: x.n)
    futuro["t"] = futuro["idx"] / 12
    futuro["covid"] = 0
    return futuro


# ---------------------------------------------------------------------------
# Iteración 1. Línea base: proporción estacional histórica por delito
# ---------------------------------------------------------------------------
def modelo_base(train, anio, anios_hist=3):
    hist = train[train["anio"].between(anio - anios_hist, anio - 1)]
    por_mes = hist.groupby(["delito", "anio", "mes"])["casos"].sum()
    peso_mes = (por_mes / por_mes.groupby(["delito", "anio"]).transform("sum")).groupby(["delito", "mes"]).mean()
    peso_ytd = peso_mes[peso_mes.index.get_level_values("mes") <= CORTE_MES].groupby("delito").sum()

    ytd = train[train["anio"] == anio].groupby(SERIE)["casos"].sum().rename("ytd").reset_index()
    futuro = filas_resto(train, anio).merge(ytd, on=SERIE, how="left")
    futuro["pred"] = [
        y * peso_mes[(d, m)] / peso_ytd[d] for y, d, m in zip(futuro["ytd"].fillna(0), futuro["delito"], futuro["mes"])
    ]
    params = {"anios_historicos": anios_hist}
    return futuro, params


# ---------------------------------------------------------------------------
# Iteración 2. MLG binomial negativa (familia de conteos, enlace log)
# ---------------------------------------------------------------------------
FORMULA_MLG = "casos ~ C(serie) + C(mes) + C(delito):t + covid"


def alpha_cameron_trivedi(y, mu):
    """Prueba de sobredispersión: Var(y) = mu + alpha * mu^2 (regresión auxiliar sin intercepto)."""
    aux = ((y - mu) ** 2 - y) / mu
    return float(np.asarray(sm.OLS(aux, mu).fit().params)[0])


def ajustar_mlg(train):
    poisson = smf.glm(FORMULA_MLG, data=train, family=sm.families.Poisson()).fit()
    alpha = max(alpha_cameron_trivedi(train["casos"].to_numpy(), poisson.mu), 1e-4)
    nb = smf.glm(FORMULA_MLG, data=train, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    return nb, poisson, alpha


def modelo_mlg(train, anio):
    nb, poisson, alpha = ajustar_mlg(train)
    futuro = filas_resto(train, anio)
    futuro["pred"] = nb.predict(futuro).to_numpy()
    params = {"familia": "Binomial negativa", "enlace": "log", "alpha": round(alpha, 4),
              "aic_poisson": round(poisson.aic, 1), "aic_binomial_negativa": round(nb.aic, 1)}
    return futuro, params, nb


# ---------------------------------------------------------------------------
# Iteración 3. Gradient boosting con pérdida Poisson y rezagos
# ---------------------------------------------------------------------------
VARIABLES_GB = ["departamento", "delito", "mes_objetivo", "horizonte", "rezago_12", "media_3", "media_12", "crecimiento_anual"]


def matriz_series(panel):
    wide = panel.pivot_table(index=SERIE, columns="idx", values="casos", aggfunc="sum").fillna(0)
    codigos = {c: {v: i for i, v in enumerate(sorted(panel[c].unique()))} for c in SERIE}
    return wide, codigos


def features_origen(wide, codigos, origen, horizontes):
    """Variables calculadas solo con la historia disponible hasta el mes de origen."""
    y = wide.to_numpy()
    filas = []
    for h in horizontes:
        objetivo = origen + h
        mes_obj = objetivo % 12 + 1
        f = pd.DataFrame({
            "departamento": [codigos["departamento"][d] for d, _ in wide.index],
            "delito": [codigos["delito"][t] for _, t in wide.index],
            "mes_objetivo": mes_obj,
            "horizonte": h,
            "rezago_12": y[:, objetivo - 12],
            "media_3": y[:, origen - 2:origen + 1].mean(axis=1),
            "media_12": y[:, origen - 11:origen + 1].mean(axis=1),
            "crecimiento_anual": y[:, origen - 11:origen + 1].sum(axis=1) / (y[:, origen - 23:origen - 11].sum(axis=1) + 1),
            "origen": origen,
            "objetivo": objetivo,
        })
        f["departamento_nombre"] = [d for d, _ in wide.index]
        f["delito_nombre"] = [t for _, t in wide.index]
        if objetivo < y.shape[1]:
            f["casos"] = y[:, objetivo]
        filas.append(f)
    return pd.concat(filas, ignore_index=True)


def modelo_gb(train, anio):
    wide, codigos = matriz_series(train)
    fin = wide.columns.max()
    # Orígenes desde que hay 24 meses de historia; objetivos dentro del entrenamiento.
    data = pd.concat(
        [features_origen(wide, codigos, o, [h for h in range(1, HORIZONTE + 1) if o + h <= fin]) for o in range(23, fin)],
        ignore_index=True,
    ).sort_values("objetivo", kind="stable")
    search = GridSearchCV(
        HistGradientBoostingRegressor(loss="poisson", categorical_features=[0, 1], random_state=42),
        {"max_depth": [3, 6], "learning_rate": [0.05, 0.1], "max_iter": [200, 400],
         "min_samples_leaf": [20, 60], "l2_regularization": [0.0, 1.0]},
        scoring="neg_mean_poisson_deviance", cv=TimeSeriesSplit(n_splits=4), n_jobs=-1, refit=True,
    )
    search.fit(data[VARIABLES_GB], data["casos"])
    fut = features_origen(wide, codigos, fin, range(1, HORIZONTE + 1))
    fut["pred"] = search.best_estimator_.predict(fut[VARIABLES_GB])
    futuro = pd.DataFrame({
        "departamento": fut["departamento_nombre"], "delito": fut["delito_nombre"],
        "anio": anio, "mes": fut["mes_objetivo"], "pred": fut["pred"],
    })
    return futuro, search.best_params_, search.best_estimator_


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def cierres(panel, futuro, anio):
    ytd = panel[(panel["anio"] == anio) & (panel["mes"] <= CORTE_MES)].groupby(SERIE)["casos"].sum().rename("observado_ytd")
    resto = futuro.groupby(SERIE)["pred"].sum().rename("pronostico_resto")
    c = pd.concat([ytd, resto], axis=1).fillna(0)
    c["cierre_pred"] = c["observado_ytd"] + c["pronostico_resto"]
    return c


def metricas(nombre, variables, regularizacion, params, tabla):
    """tabla: una fila por serie y año de backtest, con cierre_real y cierre_pred."""
    err = tabla["cierre_pred"] - tabla["cierre_real"]
    por_anio = tabla.assign(abs_err=err.abs()).groupby("anio").apply(
        lambda g: g["abs_err"].sum() / g["cierre_real"].sum(), include_groups=False
    )
    nacional = tabla.groupby("anio")[["cierre_pred", "cierre_real"]].sum()
    row = {
        "version": nombre,
        "variables": variables,
        "regularizacion": regularizacion,
        "hiperparametros": json.dumps(params, ensure_ascii=False, default=str),
        "mae_cierre": err.abs().mean(),
        "rmse_cierre": np.sqrt((err ** 2).mean()),
        "wape": err.abs().sum() / tabla["cierre_real"].sum(),
        "error_nacional_pct": ((nacional["cierre_pred"] - nacional["cierre_real"]) / nacional["cierre_real"]).abs().mean(),
    }
    row.update({f"wape_{a}": v for a, v in por_anio.items()})
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    panel = construir_panel(pd.read_csv(MENSUAL_PATH))
    real = panel.groupby(SERIE + ["anio"])["casos"].sum().rename("cierre_real")

    versiones = {"1. Linea base": [], "2. MLG binomial negativa": [], "3. Boosting Poisson": []}
    params_ver, mensual_bt = {}, []
    for anio in ANIOS_BACKTEST:
        train = panel[panel["fecha"] <= pd.Period(f"{anio}-{CORTE_MES:02d}", freq="M")]
        salidas = {
            "1. Linea base": modelo_base(train, anio),
            "2. MLG binomial negativa": modelo_mlg(train, anio)[:2],
            "3. Boosting Poisson": modelo_gb(train, anio)[:2],
        }
        for nombre, (futuro, params) in salidas.items():
            c = cierres(panel, futuro, anio).join(real.xs(anio, level="anio"))
            versiones[nombre].append(c.assign(anio=anio, version=nombre).reset_index())
            params_ver.setdefault(nombre, {})[anio] = params
            mensual_bt.append(futuro[SERIE + ["anio", "mes", "pred"]].assign(version=nombre))
        print(f"Backtest {anio} listo")

    backtest = pd.concat([pd.concat(v) for v in versiones.values()], ignore_index=True)
    descripcion = {
        "1. Linea base": ("Casos ene-jul + peso estacional por delito", "No aplica"),
        "2. MLG binomial negativa": ("Serie + mes + tendencia por delito + covid", "Dispersion alpha (Cameron-Trivedi)"),
        "3. Boosting Poisson": ("Rezagos, medias moviles, crecimiento, mes, horizonte", "Profundidad, hoja minima y L2"),
    }
    resultados = pd.DataFrame([
        metricas(n, *descripcion[n], params_ver[n], backtest[backtest["version"] == n]) for n in versiones
    ]).sort_values("wape").reset_index(drop=True)
    ganador = resultados.iloc[0]["version"]

    # Reentrenamiento con todo hasta el corte de 2026 para el pronóstico final.
    finales = {
        "1. Linea base": lambda: (*modelo_base(panel, CORTE_ANIO), None),
        "2. MLG binomial negativa": lambda: modelo_mlg(panel, CORTE_ANIO),
        "3. Boosting Poisson": lambda: modelo_gb(panel, CORTE_ANIO),
    }
    fut_final, params_final, modelo_final = finales[ganador]()
    nb_final, poisson_final, alpha_final = ajustar_mlg(panel)

    # Intervalo del cierre: cuantiles 10%-90% del error relativo del resto (ago-dic)
    # observado en el backtest del modelo ganador, por delito.
    bt_g = backtest[backtest["version"] == ganador].copy()
    resto_real = bt_g["cierre_real"] - bt_g["observado_ytd"]
    bt_g["error_rel_resto"] = (resto_real - bt_g["pronostico_resto"]) / bt_g["pronostico_resto"].clip(lower=1)
    q = bt_g.groupby("delito")["error_rel_resto"].quantile([0.10, 0.90]).unstack()
    # Para el total nacional no se suman los intervalos de las series (supondría
    # que todas fallan al extremo a la vez): se usa el rango del error nacional.
    nac = bt_g.groupby("anio")[["observado_ytd", "pronostico_resto", "cierre_real"]].sum()
    err_nac = (nac["cierre_real"] - nac["observado_ytd"] - nac["pronostico_resto"]) / nac["pronostico_resto"]

    cierre = cierres(panel, fut_final, CORTE_ANIO).reset_index()
    cierre = cierre.merge(real.xs(CORTE_ANIO - 1, level="anio").rename("cierre_anterior").reset_index(), on=SERIE, how="left")
    cierre["cierre_pred_inf"] = cierre["observado_ytd"] + cierre["pronostico_resto"] * (1 + cierre["delito"].map(q[0.10]))
    cierre["cierre_pred_sup"] = cierre["observado_ytd"] + cierre["pronostico_resto"] * (1 + cierre["delito"].map(q[0.90]))
    cierre["variacion_pct"] = (cierre["cierre_pred"] - cierre["cierre_anterior"]) / cierre["cierre_anterior"].replace(0, np.nan)
    cierre["tendencia"] = np.select(
        [cierre["variacion_pct"] > UMBRAL_TENDENCIA, cierre["variacion_pct"] < -UMBRAL_TENDENCIA],
        ["AUMENTO", "DISMINUCIÓN"], default="SIN CAMBIO",
    )

    # Coeficientes del MLG como razones de tasas (IRR = e^beta), sin los efectos fijos de serie.
    ci = nb_final.conf_int()
    coef = pd.DataFrame({
        "termino": nb_final.params.index, "coeficiente": nb_final.params.values,
        "error_estandar": nb_final.bse.values, "valor_p": nb_final.pvalues.values,
        "irr": np.exp(nb_final.params.values), "irr_inf": np.exp(ci[0].values), "irr_sup": np.exp(ci[1].values),
    })
    coef = coef[~coef["termino"].str.startswith(("C(serie)", "Intercept"))]

    # Importancia por permutación del boosting, medida fuera de muestra (backtest 2025).
    train_25 = panel[panel["fecha"] <= pd.Period(f"2025-{CORTE_MES:02d}", freq="M")]
    _, _, gb_25 = modelo_gb(train_25, 2025)
    wide, codigos = matriz_series(panel)
    origen_25 = int(train_25["idx"].max())
    eval_25 = features_origen(wide, codigos, origen_25, range(1, HORIZONTE + 1))
    perm = permutation_importance(gb_25, eval_25[VARIABLES_GB], eval_25["casos"], scoring="neg_mean_poisson_deviance",
                                  n_repeats=8, random_state=42, n_jobs=-1)
    importance = pd.DataFrame({"variable": VARIABLES_GB, "importancia": perm.importances_mean}).sort_values("importancia", ascending=False)

    resultados.to_csv(OUT / "comparacion_modelos.csv", index=False)
    backtest.to_csv(OUT / "backtest_cierres.csv", index=False)
    pd.concat(mensual_bt).to_csv(OUT / "backtest_mensual.csv", index=False)
    fut_final[SERIE + ["anio", "mes", "pred"]].to_csv(OUT / "pronostico_mensual.csv", index=False)
    cierre.to_csv(OUT / "pronostico_cierre.csv", index=False)
    coef.to_csv(OUT / "coeficientes_mlg.csv", index=False)
    importance.to_csv(OUT / "importancia_variables.csv", index=False)
    # El MLG no se serializa (guarda toda la matriz de diseño, >100 MB); sus
    # coeficientes quedan en coeficientes_mlg.csv.
    joblib.dump({"modelo": modelo_final, "ganador": ganador, "variables": VARIABLES_GB, "corte": f"{CORTE_ANIO}-{CORTE_MES:02d}"},
                MODEL_DIR / "modelo_cierre.joblib")
    resumen = {
        "modelo_seleccionado": ganador,
        "corte": f"{CORTE_ANIO}-{CORTE_MES:02d}",
        "series": int(panel["serie"].nunique()),
        "meses": int(panel["fecha"].nunique()),
        "anios_backtest": ANIOS_BACKTEST,
        "mejor_wape": float(resultados.iloc[0]["wape"]),
        "alpha_binomial_negativa": alpha_final,
        "aic_poisson": float(poisson_final.aic),
        "aic_binomial_negativa": float(nb_final.aic),
        "cierre_nacional_anterior": int(cierre["cierre_anterior"].sum()),
        "cierre_nacional_pronosticado": float(cierre["cierre_pred"].sum()),
        "cierre_nacional_inf": float(cierre["observado_ytd"].sum() + cierre["pronostico_resto"].sum() * (1 + min(err_nac.min(), 0))),
        "cierre_nacional_sup": float(cierre["observado_ytd"].sum() + cierre["pronostico_resto"].sum() * (1 + max(err_nac.max(), 0))),
        "error_nacional_resto_backtest": {int(a): round(float(e), 4) for a, e in err_nac.items()},
        "parametros_modelo_final": params_final,
        "semilla": 42,
        "criterio_seleccion": "Menor WAPE del cierre anual en backtest con corte en julio (2023-2025)",
    }
    (OUT / "resumen_modelo.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(resultados[["version", "wape", "mae_cierre", "error_nacional_pct"] + [f"wape_{a}" for a in ANIOS_BACKTEST]].to_string())
    print(json.dumps(resumen, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
