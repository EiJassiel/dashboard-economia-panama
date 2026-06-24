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
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
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
                "los CSV procesados y sintetiza la respuesta con Groq usando solo la evidencia recuperada."
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


def _obtener_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        import streamlit as st

        secret_key = st.secrets.get("GROQ_API_KEY", "")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    raise RuntimeError(
        "Falta configurar GROQ_API_KEY. Agrega la clave en tu entorno local o en Streamlit Secrets."
    )


def _groq_request(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{GROQ_API_URL}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_obtener_groq_api_key()}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detalle = exc.read().decode("utf-8", errors="ignore")
        if exc.code in (401, 403):
            raise RuntimeError("La clave de Groq no es valida o no tiene acceso al modelo configurado.") from exc
        raise RuntimeError(f"Groq devolvio un error HTTP {exc.code}. Detalle: {detalle}") from exc
    except error.URLError as exc:
        raise RuntimeError("No pude conectarme a Groq. Revisa tu conexion a internet.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Groq devolvio una respuesta invalida.") from exc


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
        "Si el usuario saluda o hace una pregunta muy general, responde breve y orienta la consulta hacia los indicadores disponibles.\n"
        "No menciones detalles internos del sistema salvo que el usuario pregunte por metodologia.\n"
        "Cierra con una mini linea de fuentes si aplica.\n\n"
        f"Pregunta del usuario:\n{pregunta}\n\n"
        f"Contexto recuperado:\n{contexto_texto}\n"
    )


def _generar_respuesta_llm(pregunta: str, contexto: pd.DataFrame) -> str:
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente economico de Panama. "
                    "Respondes solo con base en la evidencia entregada y nunca inventas cifras. "
                    "Menciona claramente si la pregunta excede el contexto recuperado."
                ),
            },
            {"role": "user", "content": _construir_prompt(pregunta, contexto)},
        ],
    }
    respuesta = _groq_request("/chat/completions", payload)
    contenido = respuesta.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not contenido:
        raise RuntimeError("Groq no devolvio contenido para esta consulta.")
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
        "intencion": "rag_groq",
        "indicadores": sorted(contexto["indicador"].dropna().unique().tolist()),
        "modelo": GROQ_MODEL,
    }


if __name__ == "__main__":
    print(responder_pregunta("Como ha evolucionado la inflacion en Panama en los ultimos 5 anos?")["respuesta"])
