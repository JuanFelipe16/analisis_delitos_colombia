from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CORTE_ANIO, CORTE_MES  # noqa: E402


def test_datos_mensuales():
    data = pd.read_csv(ROOT / "data" / "delitos_mensual.csv")
    assert data.isna().sum().sum() == 0
    assert (data.casos >= 1).all()
    assert data.departamento.nunique() == 33
    assert data.delito.nunique() == 6
    assert data.anio.between(2020, CORTE_ANIO).all()
    # Nada posterior al corte (agosto 2026 viene incompleto y se descarta).
    assert not ((data.anio == CORTE_ANIO) & (data.mes > CORTE_MES)).any()


def test_panel_completo():
    from entrenar_modelo import construir_panel
    data = pd.read_csv(ROOT / "data" / "delitos_mensual.csv")
    panel = construir_panel(data)
    series = data[["departamento", "delito"]].drop_duplicates().shape[0]
    meses = (CORTE_ANIO - 2020) * 12 + CORTE_MES
    assert len(panel) == series * meses
    assert panel.casos.sum() == data.casos.sum()


def test_resultados_modelo():
    results = pd.read_csv(ROOT / "outputs" / "comparacion_modelos.csv")
    summary = json.loads((ROOT / "outputs" / "resumen_modelo.json").read_text(encoding="utf-8"))
    assert len(results) == 3
    assert results.wape.between(0, 1).all()
    assert summary["modelo_seleccionado"] == results.sort_values("wape").iloc[0]["version"]


def test_pronostico_cierre():
    cierre = pd.read_csv(ROOT / "outputs" / "pronostico_cierre.csv")
    mensual = pd.read_csv(ROOT / "outputs" / "pronostico_mensual.csv")
    assert cierre[["departamento", "delito"]].duplicated().sum() == 0
    assert (cierre.pronostico_resto >= 0).all()
    assert ((cierre.observado_ytd + cierre.pronostico_resto - cierre.cierre_pred).abs() < 1e-6).all()
    assert (cierre.cierre_pred_inf <= cierre.cierre_pred_sup).all()
    assert set(cierre.tendencia) <= {"AUMENTO", "DISMINUCIÓN", "SIN CAMBIO"}
    assert sorted(mensual.mes.unique()) == list(range(CORTE_MES + 1, 13))
    assert abs(mensual.pred.sum() - cierre.pronostico_resto.sum()) < 1e-3
