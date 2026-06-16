from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_BRUTOS = BASE_DIR / "datos" / "brutos"

RUTA_IPC_HISTORICO = DATOS_BRUTOS / "ipc_historico_2013_2024.csv"
RUTA_IPC_2025 = DATOS_BRUTOS / "ipc_2025.csv"
RUTA_DESEMPLEO = DATOS_BRUTOS / "desempleo_panama_historico.csv"


def cargar_ipc_historico() -> pd.DataFrame:
    """Carga el CSV historico del IPC publicado por INEC."""
    return pd.read_csv(RUTA_IPC_HISTORICO, encoding="latin-1", sep=";")


def cargar_ipc_2025() -> pd.DataFrame:
    """Carga el CSV mensual de IPC 2025."""
    return pd.read_csv(RUTA_IPC_2025, encoding="latin-1", sep=";")


def cargar_desempleo() -> pd.DataFrame:
    """Carga el CSV historico de desempleo para Panama."""
    return pd.read_csv(RUTA_DESEMPLEO, skiprows=4, encoding="utf-8")


if __name__ == "__main__":
    for nombre, funcion in [
        ("IPC historico", cargar_ipc_historico),
        ("IPC 2025", cargar_ipc_2025),
        ("Desempleo", cargar_desempleo),
    ]:
        df = funcion()
        print(f"\n--- {nombre} ---")
        print(df.head())
