from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from pipeline.limpieza import ejecutar_pipeline
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from pipeline.limpieza import ejecutar_pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PROCESADOS = BASE_DIR / "datos" / "procesados"


def cargar_indicadores() -> pd.DataFrame:
    ruta = DATOS_PROCESADOS / "indicadores_limpios.csv"
    if not ruta.exists():
        ejecutar_pipeline()
    return pd.read_csv(ruta)


def entrenar_modelo_indicador(df: pd.DataFrame) -> dict:
    df = df.sort_values("anio").reset_index(drop=True)
    x = df[["anio"]].values
    y = df["valor"].values

    modelo = LinearRegression()
    modelo.fit(x, y)
    predicciones = modelo.predict(x)

    if len(df) >= 6:
        corte = max(len(df) - 2, 3)
        modelo_validacion = LinearRegression()
        modelo_validacion.fit(x[:corte], y[:corte])
        pred_holdout = modelo_validacion.predict(x[corte:])
        mae = mean_absolute_error(y[corte:], pred_holdout)
        r2_validacion = r2_score(y[corte:], pred_holdout)
    else:
        mae = float("nan")
        r2_validacion = float("nan")

    return {
        "modelo": modelo,
        "r2_entrenamiento": r2_score(y, predicciones),
        "mae_holdout": mae,
        "r2_holdout": r2_validacion,
        "serie_historica": df,
    }


def pronosticar_indicador(indicador: str, horizonte: int = 3) -> dict:
    indicadores = cargar_indicadores()
    df = indicadores[indicadores["indicador"] == indicador].copy()
    entrenamiento = entrenar_modelo_indicador(df)
    modelo = entrenamiento["modelo"]

    ultimo_anio = int(df["anio"].max())
    anios_futuros = np.arange(ultimo_anio + 1, ultimo_anio + horizonte + 1)
    forecast = modelo.predict(anios_futuros.reshape(-1, 1))

    proyecciones = pd.DataFrame(
        {
            "anio": anios_futuros.astype(int),
            "valor_predicho": np.round(forecast, 2),
            "indicador": indicador,
        }
    )

    return {
        **entrenamiento,
        "proyecciones": proyecciones,
    }


def generar_resumen_predicciones(horizonte: int = 3) -> pd.DataFrame:
    indicadores = cargar_indicadores()
    resultados = []

    for indicador in indicadores["indicador"].unique():
        pred = pronosticar_indicador(indicador, horizonte=horizonte)
        ultima_fila = pred["serie_historica"].sort_values("anio").iloc[-1]
        proxima = pred["proyecciones"].iloc[0]
        delta = round(proxima["valor_predicho"] - ultima_fila["valor"], 2)

        resultados.append(
            {
                "indicador": indicador,
                "anio_base": int(ultima_fila["anio"]),
                "valor_actual": round(float(ultima_fila["valor"]), 2),
                "anio_prediccion": int(proxima["anio"]),
                "valor_predicho": round(float(proxima["valor_predicho"]), 2),
                "cambio_absoluto": delta,
                "r2_entrenamiento": round(float(pred["r2_entrenamiento"]), 3),
                "mae_holdout": round(float(pred["mae_holdout"]), 3)
                if not np.isnan(pred["mae_holdout"])
                else np.nan,
            }
        )

    return pd.DataFrame(resultados)


if __name__ == "__main__":
    ejecutar_pipeline()
    print(generar_resumen_predicciones())
