from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error, request

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from modelos.prediccion import generar_resumen_predicciones
    from pipeline.limpieza import ejecutar_pipeline
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from modelos.prediccion import generar_resumen_predicciones
    from pipeline.limpieza import ejecutar_pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PROCESADOS = BASE_DIR / "datos" / "procesados"
INDICADORES = ("Inflacion (IPC)", "Desempleo")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MESES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def _asegurar_datos() -> None:
    rutas = [
        DATOS_PROCESADOS / "indicadores_limpios.csv",
        DATOS_PROCESADOS / "ipc_2025_mensual.csv",
    ]
    if not all(ruta.exists() for ruta in rutas):
        ejecutar_pipeline()


def cargar_dataset() -> pd.DataFrame:
    _asegurar_datos()
    return pd.read_csv(DATOS_PROCESADOS / "indicadores_limpios.csv")


def cargar_ipc_mensual() -> pd.DataFrame:
    _asegurar_datos()
    df = pd.read_csv(DATOS_PROCESADOS / "ipc_2025_mensual.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha").reset_index(drop=True)


def _variacion_texto(actual: float, anterior: float) -> str:
    delta = round(float(actual) - float(anterior), 2)
    if delta > 0:
        return f"subio {abs(delta):.2f} pp"
    if delta < 0:
        return f"bajo {abs(delta):.2f} pp"
    return "se mantuvo sin cambio"


def construir_documentos() -> pd.DataFrame:
    df = cargar_dataset()
    ipc_mensual = cargar_ipc_mensual()
    resumen_pred = generar_resumen_predicciones()
    documentos = []

    for indicador in INDICADORES:
        serie = df[df["indicador"] == indicador].sort_values("anio")
        actual = serie.iloc[-1]
        anterior = serie.iloc[-2] if len(serie) > 1 else actual
        minimo = serie.loc[serie["valor"].idxmin()]
        maximo = serie.loc[serie["valor"].idxmax()]
        pred = resumen_pred[resumen_pred["indicador"] == indicador].iloc[0]

        documentos.append(
            {
                "tipo": "resumen",
                "indicador": indicador,
                "periodo": f"{int(serie['anio'].min())}-{int(serie['anio'].max())}",
                "texto": (
                    f"{indicador} Panama resumen. Fuente {actual['fuente']}. "
                    f"Serie anual disponible desde {int(serie['anio'].min())} hasta {int(serie['anio'].max())}. "
                    f"Ultimo valor {actual['valor']:.2f}{actual['unidad']} en {int(actual['anio'])}. "
                    f"Respecto a {int(anterior['anio'])} {_variacion_texto(actual['valor'], anterior['valor'])}. "
                    f"Minimo {minimo['valor']:.2f}{minimo['unidad']} en {int(minimo['anio'])}. "
                    f"Maximo {maximo['valor']:.2f}{maximo['unidad']} en {int(maximo['anio'])}. "
                    f"Prediccion {int(pred['anio_prediccion'])}: {pred['valor_predicho']:.2f}{actual['unidad']}."
                ),
                "fuente": actual["fuente"],
            }
        )

        for _, fila in serie.iterrows():
            documentos.append(
                {
                    "tipo": "dato_anual",
                    "indicador": indicador,
                    "periodo": str(int(fila["anio"])),
                    "texto": (
                        f"{indicador} Panama dato anual {int(fila['anio'])}: "
                        f"{fila['valor']:.2f}{fila['unidad']}. Fuente {fila['fuente']}."
                    ),
                    "fuente": fila["fuente"],
                }
            )

    for _, fila in ipc_mensual.iterrows():
        documentos.append(
            {
                "tipo": "dato_mensual",
                "indicador": "IPC mensual 2025",
                "periodo": fila["fecha"].strftime("%Y-%m"),
                "texto": (
                    f"IPC mensual 2025 Panama {MESES[int(fila['mes'])]} {int(fila['anio'])}: "
                    f"indice total {fila['ipc_total']:.1f}. Fuente INEC."
                ),
                "fuente": "INEC",
            }
        )

    documentos.append(
        {
            "tipo": "metodologia",
            "indicador": "General",
            "periodo": "proyecto",
            "texto": (
                "El dashboard integra INEC para inflacion basada en IPC, INEC para IPC mensual 2025 "
                "y Banco Mundial para desempleo total de Panama. El modelo predictivo usa regresion lineal "
                "sobre series anuales. El asistente usa recuperacion TF-IDF sobre documentos generados desde "
                "los CSV procesados y sintetiza la respuesta con un modelo local de Ollama."
            ),
            "fuente": "INEC, Banco Mundial",
        }
    )

    return pd.DataFrame(documentos)


def recuperar_contexto(pregunta: str, top_k: int = 5) -> pd.DataFrame:
    documentos = construir_documentos()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    matriz = vectorizer.fit_transform(documentos["texto"])
    consulta = vectorizer.transform([pregunta])
    scores = cosine_similarity(consulta, matriz).flatten()
    documentos = documentos.copy()
    documentos["score"] = scores
    return documentos.sort_values(["score", "tipo"], ascending=[False, True]).head(top_k)


def _ollama_request(path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(f"{OLLAMA_URL}{path}", data=data, headers=headers, method="POST" if data else "GET")
    try:
        with request.urlopen(req, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(
            "No pude conectarme a Ollama. Asegurate de tener la app corriendo en "
            f"{OLLAMA_URL}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama devolvio una respuesta invalida.") from exc


def _modelo_disponible() -> bool:
    try:
        respuesta = _ollama_request("/api/tags")
    except RuntimeError:
        return False

    modelos = [modelo.get("name", "") for modelo in respuesta.get("models", [])]
    return OLLAMA_MODEL in modelos


def _construir_prompt(pregunta: str, contexto: pd.DataFrame) -> str:
    bloques = []
    for _, fila in contexto.iterrows():
        bloques.append(
            "\n".join(
                [
                    f"Indicador: {fila['indicador']}",
                    f"Periodo: {fila['periodo']}",
                    f"Fuente: {fila['fuente']}",
                    f"Score: {float(fila['score']):.3f}",
                    f"Texto: {fila['texto']}",
                ]
            )
        )

    contexto_texto = "\n\n".join(bloques)
    return (
        "Responde en espanol usando solo el contexto proporcionado.\n"
        "Si la informacion no alcanza, dilo con claridad y no inventes.\n"
        "Prioriza datos, periodos, comparaciones y fuentes.\n"
        "No menciones que usaste TF-IDF ni detalles internos del sistema salvo que el usuario pregunte por metodologia.\n"
        "Mantente concreto y profesional.\n\n"
        f"Pregunta del usuario:\n{pregunta}\n\n"
        f"Contexto recuperado:\n{contexto_texto}\n"
    )


def _generar_respuesta_llm(pregunta: str, contexto: pd.DataFrame) -> str:
    if not _modelo_disponible():
        raise RuntimeError(
            f"El modelo '{OLLAMA_MODEL}' no esta disponible en Ollama. Ejecuta: ollama pull {OLLAMA_MODEL}"
        )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente economico de Panama. "
                    "Respondes solo con base en la evidencia entregada y nunca inventas cifras."
                ),
            },
            {"role": "user", "content": _construir_prompt(pregunta, contexto)},
        ],
        "options": {"temperature": 0.2},
    }
    respuesta = _ollama_request("/api/chat", payload)
    contenido = respuesta.get("message", {}).get("content", "").strip()
    if not contenido:
        raise RuntimeError("Ollama no devolvio contenido para esta consulta.")
    return contenido


def responder_pregunta(pregunta: str) -> dict:
    pregunta = pregunta.strip()
    contexto = recuperar_contexto(pregunta)

    if contexto.empty:
        raise RuntimeError("No encontre contexto para responder con los datos disponibles.")

    respuesta = _generar_respuesta_llm(pregunta, contexto)

    fuentes = []
    for fuente in contexto["fuente"].drop_duplicates().tolist():
        for item in [parte.strip() for parte in str(fuente).split(",")]:
            if item and item not in fuentes:
                fuentes.append(item)

    evidencia = [
        {
            "indicador": fila["indicador"],
            "periodo": fila["periodo"],
            "texto": fila["texto"],
            "score": round(float(fila["score"]), 3),
        }
        for _, fila in contexto.head(3).iterrows()
    ]

    return {
        "respuesta": respuesta,
        "contexto": contexto[["tipo", "indicador", "periodo", "texto", "fuente", "score"]].to_dict("records"),
        "evidencia": evidencia,
        "fuentes": fuentes,
        "intencion": "rag_ollama",
        "indicadores": sorted(contexto["indicador"].dropna().unique().tolist()),
        "modelo": OLLAMA_MODEL,
    }


if __name__ == "__main__":
    print(responder_pregunta("Como ha evolucionado la inflacion en Panama en los ultimos 5 anos?")["respuesta"])
