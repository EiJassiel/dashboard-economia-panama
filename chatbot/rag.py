from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from modelos.prediccion import generar_resumen_predicciones, pronosticar_indicador
    from pipeline.limpieza import ejecutar_pipeline
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from modelos.prediccion import generar_resumen_predicciones, pronosticar_indicador
    from pipeline.limpieza import ejecutar_pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PROCESADOS = BASE_DIR / "datos" / "procesados"
INDICADORES = ("Inflacion (IPC)", "Desempleo")
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


def _fmt(valor: float, unidad: str = "%") -> str:
    return f"{float(valor):.2f}{unidad}"


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
                "y Banco Mundial para desempleo total de Panama. El modelo usa regresion lineal "
                "sobre series anuales y el RAG recupera evidencia local con TF-IDF."
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


def _normalizar(texto: str) -> str:
    reemplazos = str.maketrans("áéíóúüñ", "aeiouun")
    return texto.lower().translate(reemplazos)


def _detectar_indicadores(pregunta: str) -> list[str]:
    texto = _normalizar(pregunta)
    indicadores = []
    if "inflacion" in texto or "ipc" in texto or "precio" in texto:
        indicadores.append("Inflacion (IPC)")
    if "desempleo" in texto or "paro" in texto or "laboral" in texto:
        indicadores.append("Desempleo")
    if any(palabra in texto for palabra in ["compara", "comparacion", "ambos", "indicadores"]):
        indicadores = list(INDICADORES)
    return indicadores


def _detectar_intencion(pregunta: str) -> str:
    texto = _normalizar(pregunta)
    if any(palabra in texto for palabra in ["predice", "prediccion", "pronostico", "proyeccion", "futuro"]):
        return "prediccion"
    if any(palabra in texto for palabra in ["evolucion", "tendencia", "cambio", "ultimos", "historica"]):
        return "tendencia"
    if any(palabra in texto for palabra in ["maximo", "mayor", "pico", "mas alto"]):
        return "maximo"
    if any(palabra in texto for palabra in ["minimo", "menor", "mas bajo"]):
        return "minimo"
    if any(palabra in texto for palabra in ["ultimo", "actual", "reciente", "disponible"]):
        return "ultimo"
    if any(palabra in texto for palabra in ["fuente", "metodologia", "metodo", "modelo", "pipeline"]):
        return "metodologia"
    return "general"


def _detectar_anios(pregunta: str) -> list[int]:
    return [int(anio) for anio in re.findall(r"\b(19\d{2}|20\d{2})\b", pregunta)]


def _serie(indicador: str) -> pd.DataFrame:
    return cargar_dataset().query("indicador == @indicador").sort_values("anio").reset_index(drop=True)


def _respuesta_mensual(pregunta: str) -> str | None:
    texto = _normalizar(pregunta)
    if "2025" not in texto and "mensual" not in texto:
        return None
    if "ipc" not in texto and "inflacion" not in texto:
        return None

    mensual = cargar_ipc_mensual()
    mensual_2025 = mensual[mensual["anio"] == 2025].copy()
    ultimo = mensual.iloc[-1]
    primero_2025 = mensual_2025.iloc[0]
    maximo = mensual_2025.loc[mensual_2025["ipc_total"].idxmax()]
    minimo = mensual_2025.loc[mensual_2025["ipc_total"].idxmin()]
    delta = round(float(ultimo["ipc_total"]) - float(primero_2025["ipc_total"]), 2)
    direccion = "subio" if delta > 0 else "bajo" if delta < 0 else "se mantuvo"

    return (
        f"Para el IPC mensual 2025, el ultimo corte disponible es "
        f"{MESES[int(ultimo['mes'])]} de {int(ultimo['anio'])} con indice {ultimo['ipc_total']:.1f}. "
        f"Frente a enero de 2025 ({primero_2025['ipc_total']:.1f}), {direccion} {abs(delta):.1f} puntos de indice. "
        f"El mayor valor mensual del archivo es {maximo['ipc_total']:.1f} en {MESES[int(maximo['mes'])]} "
        f"y el menor es {minimo['ipc_total']:.1f} en {MESES[int(minimo['mes'])]}. Fuente: INEC."
    )


def _responder_indicador(indicador: str, intencion: str, anios: list[int]) -> str:
    serie = _serie(indicador)
    unidad = serie.iloc[-1]["unidad"]
    fuente = serie.iloc[-1]["fuente"]

    if anios:
        filas = serie[serie["anio"].isin(anios)]
        if not filas.empty:
            datos = ", ".join(
                f"{int(fila['anio'])}: {_fmt(fila['valor'], fila['unidad'])}" for _, fila in filas.iterrows()
            )
            return f"{indicador}: {datos}. Fuente: {fuente}."

    primero = serie.iloc[0]
    ultimo = serie.iloc[-1]
    anterior = serie.iloc[-2] if len(serie) > 1 else ultimo
    maximo = serie.loc[serie["valor"].idxmax()]
    minimo = serie.loc[serie["valor"].idxmin()]

    if intencion == "maximo":
        return f"{indicador}: el maximo de la serie es {_fmt(maximo['valor'], unidad)} en {int(maximo['anio'])}. Fuente: {fuente}."
    if intencion == "minimo":
        return f"{indicador}: el minimo de la serie es {_fmt(minimo['valor'], unidad)} en {int(minimo['anio'])}. Fuente: {fuente}."
    if intencion == "ultimo":
        return (
            f"{indicador}: el ultimo dato anual disponible es {_fmt(ultimo['valor'], unidad)} en {int(ultimo['anio'])}; "
            f"frente a {int(anterior['anio'])} {_variacion_texto(ultimo['valor'], anterior['valor'])}. Fuente: {fuente}."
        )
    if intencion == "prediccion":
        pred = pronosticar_indicador(indicador, horizonte=3)
        proxima = pred["proyecciones"].iloc[0]
        cambio = _variacion_texto(proxima["valor_predicho"], ultimo["valor"])
        return (
            f"{indicador}: la regresion lineal proyecta {_fmt(proxima['valor_predicho'], unidad)} para "
            f"{int(proxima['anio'])}; frente al ultimo dato observado ({_fmt(ultimo['valor'], unidad)} en "
            f"{int(ultimo['anio'])}) {cambio}. R2 entrenamiento: {pred['r2_entrenamiento']:.3f}."
        )

    ventana = serie.tail(5)
    cambio_total = round(float(ventana.iloc[-1]["valor"]) - float(ventana.iloc[0]["valor"]), 2)
    direccion = "al alza" if cambio_total > 0 else "a la baja" if cambio_total < 0 else "estable"
    return (
        f"{indicador}: entre {int(ventana.iloc[0]['anio'])} y {int(ventana.iloc[-1]['anio'])} la tendencia fue "
        f"{direccion}, pasando de {_fmt(ventana.iloc[0]['valor'], unidad)} a {_fmt(ventana.iloc[-1]['valor'], unidad)} "
        f"({cambio_total:+.2f} pp). En toda la serie, minimo {_fmt(minimo['valor'], unidad)} en {int(minimo['anio'])} "
        f"y maximo {_fmt(maximo['valor'], unidad)} en {int(maximo['anio'])}. Fuente: {fuente}."
    )


def responder_pregunta(pregunta: str) -> dict:
    pregunta = pregunta.strip()
    intencion = _detectar_intencion(pregunta)
    indicadores = _detectar_indicadores(pregunta)
    anios = _detectar_anios(pregunta)
    respuesta_mensual = _respuesta_mensual(pregunta)

    if not indicadores and intencion == "prediccion":
        indicadores = list(INDICADORES)

    contexto = recuperar_contexto(" ".join([pregunta, *indicadores]) if indicadores else pregunta)
    if respuesta_mensual:
        contexto_mensual = recuperar_contexto(f"{pregunta} ultimo agosto 2025 IPC mensual indice")
        contexto_filtrado = contexto_mensual[contexto_mensual["indicador"].eq("IPC mensual 2025")]
        if not contexto_filtrado.empty:
            contexto = contexto_filtrado.head(3)
    elif indicadores:
        bloques = []
        for indicador in indicadores:
            contexto_indicador = recuperar_contexto(f"{pregunta} {indicador}")
            contexto_filtrado = contexto_indicador[contexto_indicador["indicador"].eq(indicador)].head(2)
            if not contexto_filtrado.empty:
                bloques.append(contexto_filtrado)
        if bloques:
            contexto = pd.concat(bloques, ignore_index=True)

    if respuesta_mensual:
        respuesta = respuesta_mensual
    elif intencion == "metodologia":
        respuesta = (
            "El proyecto usa un pipeline local: carga datos brutos, limpia IPC historico, IPC mensual 2025 "
            "y desempleo, consolida indicadores anuales, entrena regresiones lineales por indicador y usa "
            "TF-IDF para recuperar evidencia textual desde los CSV procesados. Fuentes: INEC y Banco Mundial."
        )
    elif indicadores:
        partes = [_responder_indicador(indicador, intencion, anios) for indicador in indicadores]
        respuesta = " ".join(partes)
    else:
        respuesta = (
            "Puedo responder con datos locales sobre Inflacion (IPC), IPC mensual 2025 y Desempleo de Panama. "
            "Pregunta por ultimo dato, maximo, minimo, tendencia, comparacion, ano especifico, prediccion o metodologia."
        )

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
        "intencion": intencion,
        "indicadores": indicadores,
    }


if __name__ == "__main__":
    print(responder_pregunta("Como ha evolucionado la inflacion en Panama en los ultimos 5 anos?")["respuesta"])
