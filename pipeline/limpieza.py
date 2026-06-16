from pathlib import Path

import pandas as pd

try:
    from .ingesta import cargar_desempleo, cargar_ipc_2025, cargar_ipc_historico
except ImportError:
    from ingesta import cargar_desempleo, cargar_ipc_2025, cargar_ipc_historico


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PROCESADOS = BASE_DIR / "datos" / "procesados"
DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)

MAPA_MESES = {
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
}


def limpiar_ipc_historico() -> pd.DataFrame:
    df = cargar_ipc_historico().copy()
    df.columns = ["anio", "ipc", "poder_adquisitivo"]
    df = df[["anio", "ipc"]].dropna()
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df["ipc"] = pd.to_numeric(df["ipc"], errors="coerce")
    df = df.dropna().astype({"anio": int})
    df["inflacion_pct"] = df["ipc"].pct_change().mul(100).round(2)
    df = df.dropna(subset=["inflacion_pct"]).reset_index(drop=True)
    df.to_csv(DATOS_PROCESADOS / "ipc_limpio.csv", index=False)
    return df


def limpiar_ipc_2025() -> pd.DataFrame:
    df = cargar_ipc_2025().copy()
    df.columns = [col.strip() for col in df.columns]
    df = df[["Mes", "Total"]].dropna()

    def parsear_mes(valor: str) -> pd.Timestamp:
        abreviatura, anio = valor.split("-")
        anio = int(f"20{anio}")
        return pd.Timestamp(year=anio, month=MAPA_MESES[abreviatura.lower()], day=1)

    df["fecha"] = df["Mes"].map(parsear_mes)
    df["ipc_total"] = pd.to_numeric(df["Total"], errors="coerce")
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df = df[["fecha", "anio", "mes", "ipc_total"]].dropna().reset_index(drop=True)
    df.to_csv(DATOS_PROCESADOS / "ipc_2025_mensual.csv", index=False)
    return df


def limpiar_desempleo() -> pd.DataFrame:
    df = cargar_desempleo().copy()
    df = df[df["Country Name"] == "Panama"]
    anios = [str(anio) for anio in range(1991, 2026)]
    disponibles = [col for col in anios if col in df.columns]
    df = df[disponibles].T.reset_index()
    df.columns = ["anio", "desempleo_pct"]
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df["desempleo_pct"] = pd.to_numeric(df["desempleo_pct"], errors="coerce")
    df = df.dropna().astype({"anio": int})
    df["desempleo_pct"] = df["desempleo_pct"].round(2)
    df.to_csv(DATOS_PROCESADOS / "desempleo_limpio.csv", index=False)
    return df


def consolidar_indicadores() -> pd.DataFrame:
    ipc = limpiar_ipc_historico()[["anio", "inflacion_pct"]].rename(
        columns={"inflacion_pct": "valor"}
    )
    ipc["indicador"] = "Inflacion (IPC)"
    ipc["fuente"] = "INEC"
    ipc["unidad"] = "%"

    desempleo = limpiar_desempleo()[["anio", "desempleo_pct"]].rename(
        columns={"desempleo_pct": "valor"}
    )
    desempleo["indicador"] = "Desempleo"
    desempleo["fuente"] = "Banco Mundial"
    desempleo["unidad"] = "%"

    consolidado = pd.concat([ipc, desempleo], ignore_index=True)
    consolidado = consolidado[["anio", "indicador", "valor", "unidad", "fuente"]]
    consolidado = consolidado.sort_values(["indicador", "anio"]).reset_index(drop=True)
    consolidado.to_csv(DATOS_PROCESADOS / "indicadores_limpios.csv", index=False)
    return consolidado


def ejecutar_pipeline() -> dict[str, pd.DataFrame]:
    return {
        "ipc": limpiar_ipc_historico(),
        "ipc_2025": limpiar_ipc_2025(),
        "desempleo": limpiar_desempleo(),
        "indicadores": consolidar_indicadores(),
    }


if __name__ == "__main__":
    resultados = ejecutar_pipeline()
    for nombre, df in resultados.items():
        print(f"\n--- {nombre} ---")
        print(df.tail())
