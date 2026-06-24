from pathlib import Path
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chatbot.rag import responder_pregunta
from modelos.prediccion import generar_resumen_predicciones
from pipeline.limpieza import ejecutar_pipeline


BASE_DIR = Path(__file__).resolve().parent
DATOS_PROCESADOS = BASE_DIR / "datos" / "procesados"

st.set_page_config(
    page_title="Dashboard Economico de Panama",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cargar_datos() -> dict[str, pd.DataFrame]:
    ejecutar_pipeline()
    return {
        "ipc": pd.read_csv(DATOS_PROCESADOS / "ipc_limpio.csv"),
        "ipc_2025": pd.read_csv(DATOS_PROCESADOS / "ipc_2025_mensual.csv"),
        "desempleo": pd.read_csv(DATOS_PROCESADOS / "desempleo_limpio.csv"),
        "indicadores": pd.read_csv(DATOS_PROCESADOS / "indicadores_limpios.csv"),
        "predicciones": generar_resumen_predicciones(),
    }


def inyectar_estilos() -> None:
    st.markdown(
        """
        <style>
            :root {
                color-scheme: light;
                --bg: #F6F3EC;
                --surface: #FFFFFF;
                --surface-2: #FAF8F1;
                --line: #DFE5DC;
                --text: #17211B;
                --muted: #66736A;
                --green: #18382F;
                --green-soft: #E7EEE9;
                --gold: #B58B3B;
                --red: #8F1515;
                --radius: 8px;
                --shadow: 0 8px 24px rgba(23,33,27,0.06);
            }

            html,
            body,
            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"] {
                color-scheme: light;
            }

            .stApp {
                background: var(--bg);
                color: var(--text);
                font-family: Arial, sans-serif;
            }

            .block-container {
                max-width: 1360px;
                padding: 24px 32px 48px;
            }

            h1, h2, h3, h4,
            .hero h1,
            .info-value {
                color: var(--text);
                font-family: Georgia, "Times New Roman", serif;
                letter-spacing: 0;
            }

            section[data-testid="stSidebar"] {
                background: var(--green);
                border-right: 4px solid var(--gold);
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                color: #FFFFFF;
            }

            section[data-testid="stSidebar"] hr {
                border-color: rgba(255,255,255,0.18);
            }

            section[data-testid="stSidebar"] button {
                background: rgba(255,255,255,0.08);
                border: 1px solid var(--gold);
                border-radius: var(--radius);
                color: #FFFFFF;
                font-weight: 700;
            }

            section[data-testid="stSidebar"] button p,
            section[data-testid="stSidebar"] button span {
                color: #FFFFFF !important;
            }

            section[data-testid="stSidebar"] button:not(:has(p)) {
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
            }

            section[data-testid="stSidebar"] div[data-baseweb="select"] span,
            section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
                color: var(--text) !important;
                fill: var(--text) !important;
            }

            section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
                color: rgba(255,255,255,0.72) !important;
            }

            .hero {
                position: relative;
                background:
                    linear-gradient(135deg, #18382F 0%, #244A3F 100%);
                border-top: 6px solid var(--red);
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: var(--radius);
                color: #FFFFFF;
                margin-bottom: 24px;
                padding: 32px;
                box-shadow: var(--shadow);
            }

            .hero::after {
                content: "";
                position: absolute;
                right: 32px;
                top: 32px;
                width: 80px;
                height: 3px;
                background: var(--gold);
            }

            .hero-kicker {
                color: var(--gold);
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                margin-bottom: 8px;
                text-transform: uppercase;
            }

            .hero h1 {
                color: #FFFFFF;
                font-size: 2.4rem;
                line-height: 1.05;
                margin: 0 0 16px;
            }

            .hero p {
                color: rgba(255,255,255,0.82);
                font-size: 1rem;
                line-height: 1.6;
                margin: 0;
                max-width: 64rem;
            }

            .hero-subtitle {
                color: #FFFFFF;
                font-weight: 700;
                margin-bottom: 8px;
            }

            .info-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-top: 4px solid var(--gold);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                height: 100%;
                padding: 16px;
            }

            .info-label {
                color: var(--muted);
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                margin-bottom: 8px;
                text-transform: uppercase;
            }

            .info-value {
                color: var(--text);
                font-size: 2rem;
                line-height: 1;
                margin-bottom: 8px;
            }

            .info-sub {
                color: var(--muted);
                font-size: 0.92rem;
            }

            .section-note {
                color: var(--muted);
                font-size: 0.95rem;
                line-height: 1.6;
                margin-bottom: 16px;
            }

            .chat-panel {
                background: rgba(255,255,255,0.48);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                gap: 14px;
                margin: 16px 0 0;
                max-height: 480px;
                max-width: 100%;
                overflow-y: auto;
                overflow-x: hidden;
                padding: 14px;
                width: 100%;
            }

            .chat-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                margin: 0 0 6px;
                text-transform: uppercase;
            }

            .chat-user {
                background: var(--green-soft);
                border: 1px solid var(--line);
                border-left: 4px solid var(--green);
                border-radius: 14px 14px 4px 14px;
                color: var(--green);
                margin: 0 0 0 auto;
                box-sizing: border-box;
                max-width: min(680px, 72%);
                overflow-wrap: anywhere;
                padding: 12px 14px;
            }

            .chat-assistant {
                background: var(--surface);
                border: 1px solid var(--line);
                border-left: 4px solid var(--gold);
                border-radius: 14px 14px 14px 4px;
                color: var(--text);
                margin: 0 auto 0 0;
                box-sizing: border-box;
                max-width: min(760px, 78%);
                overflow-wrap: anywhere;
                padding: 12px 14px;
            }

            .source-chip {
                background: var(--surface-2);
                border: 1px solid var(--line);
                border-radius: 999px;
                color: var(--muted);
                display: inline-block;
                font-size: 0.8rem;
                margin-right: 8px;
                margin-top: 8px;
                padding: 4px 8px;
            }

            .evidence-box {
                background: transparent;
                border: 0;
                border-top: 1px solid var(--line);
                border-radius: var(--radius);
                box-sizing: border-box;
                color: var(--muted);
                font-size: 0.78rem;
                line-height: 1.45;
                margin-top: 10px;
                padding: 8px 0 0;
            }

            .evidence-meta {
                color: var(--red);
                font-size: 0.78rem;
                letter-spacing: 0.06em;
                margin-bottom: 8px;
                text-transform: uppercase;
            }

            .chat-input-shell {
                background: linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(236,232,223,0.96) 100%);
                border: 1px solid #D8D1C5;
                border-radius: 18px;
                box-sizing: border-box;
                box-shadow: 0 12px 28px rgba(23,33,27,0.06);
                margin: 14px 0 0;
                max-width: 100%;
                padding: 12px;
                width: 100%;
            }

            .chat-input-shell [data-testid="stForm"] {
                border: 0;
                padding: 0;
                background: transparent !important;
            }

            .chat-input-shell [data-testid="stForm"] > div:first-child {
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
                padding: 0 !important;
            }

            .chat-input-shell [data-testid="stForm"] > div:last-child {
                background: transparent !important;
                border: 0 !important;
                box-shadow: none !important;
                padding: 0 !important;
            }

            .chat-input-shell div[data-baseweb="input"] {
                background: transparent !important;
            }

            .chat-input-shell .stTextInput input {
                background: #FFFFFF !important;
                border: 1px solid #D8D1C5 !important;
                border-radius: 14px !important;
                box-shadow: none !important;
                color: var(--text) !important;
                min-height: 46px;
                padding-right: 14px !important;
                transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
            }

            .chat-input-shell .stTextInput input:focus,
            .chat-input-shell .stTextInput input:focus-visible,
            .chat-input-shell div[data-baseweb="input"]:focus-within {
                border-color: var(--gold) !important;
                box-shadow: 0 0 0 3px rgba(181,139,59,0.16) !important;
                outline: none !important;
            }

            .chat-input-shell .stTextInput input::selection {
                background: rgba(181,139,59,0.18);
                color: var(--text);
            }

            .chat-input-shell .stFormSubmitButton button {
                background: #17211B;
                border-color: #17211B;
                border-radius: 12px;
                box-shadow: none !important;
                color: #FFFFFF;
                min-height: 46px;
                max-width: 150px !important;
                min-width: 150px !important;
                transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
                width: 150px !important;
            }

            .chat-input-shell .stFormSubmitButton button:hover {
                background: #244A3F;
                border-color: #244A3F;
                box-shadow: 0 10px 22px rgba(24,56,47,0.16) !important;
                transform: translateY(-1px);
            }

            .chat-input-shell .stFormSubmitButton button:focus,
            .chat-input-shell .stFormSubmitButton button:focus-visible {
                box-shadow: 0 0 0 3px rgba(181,139,59,0.18) !important;
                outline: none !important;
            }

            div[data-testid="stFormSubmitButton"] button {
                max-width: 150px !important;
                min-width: 150px !important;
                width: 150px !important;
            }

            .chat-input-shell .stFormSubmitButton {
                display: flex;
                justify-content: flex-end;
            }

            .suggestion-label {
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 700;
                margin: 18px 0 8px;
                max-width: 100%;
            }

            .rag-loader {
                align-items: center;
                background: rgba(255,255,255,0.82);
                border: 1px solid rgba(181,139,59,0.28);
                border-radius: 14px;
                box-shadow: 0 10px 24px rgba(23,33,27,0.06);
                display: inline-flex;
                gap: 12px;
                margin: 14px 0 6px;
                padding: 12px 16px;
            }

            .rag-loader-ring {
                width: 18px;
                height: 18px;
                border-radius: 50%;
                border: 2px solid rgba(24,56,47,0.14);
                border-top-color: var(--gold);
                animation: rag-spin 0.85s linear infinite;
                flex-shrink: 0;
            }

            .rag-loader-copy {
                color: var(--text);
                font-size: 0.9rem;
                line-height: 1.3;
            }

            .rag-loader-copy span {
                color: var(--muted);
                display: block;
                font-size: 0.78rem;
                margin-top: 2px;
            }

            @keyframes rag-spin {
                from {
                    transform: rotate(0deg);
                }
                to {
                    transform: rotate(360deg);
                }
            }

            @media (max-width: 760px) {
                .chat-user,
                .chat-assistant {
                    max-width: 100%;
                }

                .chat-input-shell .stFormSubmitButton button {
                    max-width: 100% !important;
                    min-width: 100% !important;
                    width: 100%;
                }
            }

            div[data-testid="stTabs"] button {
                border-bottom: 3px solid transparent;
                border-radius: 0;
                color: var(--muted);
                font-weight: 700;
            }

            div[data-testid="stTabs"] button[aria-selected="true"] {
                border-bottom-color: var(--red);
                color: var(--red);
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stPlotlyChart"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                padding: 8px;
            }

            .stButton button,
            .stFormSubmitButton button {
                background: var(--surface-2);
                border: 1px solid var(--line);
                border-radius: var(--radius);
                color: var(--text);
                font-weight: 700;
            }

            .stButton button p,
            .stFormSubmitButton button p {
                color: inherit !important;
            }

            .stTextInput input,
            div[data-baseweb="select"] > div {
                background: #FFFFFF !important;
                border-color: var(--line) !important;
                color: var(--text) !important;
            }

            div[data-testid="InputInstructions"] {
                color: transparent;
                font-size: 0;
            }

            textarea,
            input,
            select {
                color-scheme: light;
            }

            div[data-testid="stExpander"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: var(--radius);
            }

            @media (max-width: 900px) {
                .hero h1 {
                    font-size: 1.65rem;
                }

                .block-container {
                    padding: 16px 16px 32px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inicializar_estado() -> None:
    if "chat_history" not in st.session_state:
        pregunta = "Como ha evolucionado la inflacion en Panama en los ultimos 5 anos?"
        try:
            respuesta = responder_pregunta(pregunta)
            contenido = respuesta["respuesta"]
            fuentes = respuesta["fuentes"]
            evidencia = respuesta["evidencia"]
            intencion = respuesta["intencion"]
        except Exception as exc:
            contenido = (
                "No pude iniciar el asistente RAG porque falta configurar Groq o hubo un problema con la API. "
                f"Detalle: {exc}"
            )
            fuentes = []
            evidencia = []
            intencion = "error"
        st.session_state.chat_history = [
            {"role": "user", "content": pregunta, "sources": []},
            {
                "role": "assistant",
                "content": contenido,
                "sources": fuentes,
                "evidence": evidencia,
                "intent": intencion,
            },
        ]
    if "rag_cargando" not in st.session_state:
        st.session_state.rag_cargando = False
    if "rag_pregunta_pendiente" not in st.session_state:
        st.session_state.rag_pregunta_pendiente = ""


def guardar_interaccion(pregunta: str) -> None:
    st.session_state.rag_cargando = True
    try:
        respuesta = responder_pregunta(pregunta)
        contenido = respuesta["respuesta"]
        fuentes = respuesta["fuentes"]
        evidencia = respuesta["evidencia"]
        intencion = respuesta["intencion"]
    except Exception as exc:
        contenido = f"No pude consultar Groq. Detalle: {exc}"
        fuentes = []
        evidencia = []
        intencion = "error"
    finally:
        st.session_state.rag_cargando = False
    st.session_state.chat_history.append({"role": "user", "content": pregunta, "sources": []})
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": contenido,
            "sources": fuentes,
            "evidence": evidencia,
            "intent": intencion,
        }
    )


def construir_grafico_comparativo(df: pd.DataFrame) -> go.Figure:
    figura = go.Figure()
    colores = {"Inflacion (IPC)": "#B58B3B", "Desempleo": "#18382F"}

    for indicador in df["indicador"].unique():
        serie = df[df["indicador"] == indicador].sort_values("anio")
        figura.add_trace(
            go.Scatter(
                x=serie["anio"],
                y=serie["valor"],
                mode="lines+markers",
                name=indicador,
                line={"width": 3, "color": colores.get(indicador, "#66736A")},
                marker={"size": 7, "line": {"width": 1, "color": "#FFFFFF"}},
            )
        )

    figura.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=420,
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"color": "#17211B"}},
        font={"color": "#17211B", "family": "Arial, sans-serif"},
        xaxis={"title": "", "showgrid": False, "color": "#66736A"},
        yaxis={"title": "", "gridcolor": "#E7EEE9", "zeroline": False, "color": "#66736A"},
    )
    return figura


def construir_grafico_ipc_2025(df: pd.DataFrame) -> go.Figure:
    figura = go.Figure(
        data=[
            go.Scatter(
                x=df["fecha"],
                y=df["ipc_total"],
                mode="lines+markers",
                line={"width": 3, "color": "#18382F"},
                marker={"size": 7, "color": "#B58B3B", "line": {"width": 1, "color": "#FFFFFF"}},
                fill="tozeroy",
                fillcolor="rgba(24,56,47,0.10)",
                name="IPC mensual 2025",
            )
        ]
    )
    figura.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=320,
        showlegend=False,
        font={"color": "#17211B", "family": "Arial, sans-serif"},
        xaxis={"title": "", "showgrid": False, "color": "#66736A"},
        yaxis={"title": "", "gridcolor": "#E7EEE9", "zeroline": False, "color": "#66736A"},
    )
    return figura


def mostrar_fuentes() -> None:
    st.markdown("### Fuentes del proyecto")
    st.markdown(
        """
        - `INEC`: IPC historico 2013-2024 e IPC mensual 2025.
        - `Banco Mundial`: desempleo historico de Panama.
        """
    )


def main() -> None:
    inyectar_estilos()
    inicializar_estado()
    datos = cargar_datos()

    indicadores = datos["indicadores"].copy()
    ipc = datos["ipc"].copy().sort_values("anio")
    desempleo = datos["desempleo"].copy().sort_values("anio")
    ipc_2025 = datos["ipc_2025"].copy().sort_values("fecha")
    predicciones = datos["predicciones"].copy()

    min_anio = int(indicadores["anio"].min())
    max_anio = int(indicadores["anio"].max())
    opciones = sorted(indicadores["indicador"].unique().tolist())

    with st.sidebar:
        st.title("Economia Panama")
        st.caption("Dashboard del proyecto integrador")

        if st.button("Actualizar pipeline", width="stretch"):
            st.cache_data.clear()
            ejecutar_pipeline()
            st.rerun()

        indicador_sel = st.selectbox("Indicador", ["Todos"] + opciones, index=0)
        rango = st.slider("Rango de anos", min_value=min_anio, max_value=max_anio, value=(2014, max_anio))

        st.markdown("---")
        st.markdown("**Cobertura**")
        st.write(f"{min_anio} - {max_anio}")
        st.markdown("**Tecnicas usadas**")
        st.write("- Pipeline de datos")
        st.write("- Regresion lineal")
        st.write("- RAG con evidencia + Groq")
        st.markdown("---")
        mostrar_fuentes()

    seleccion = opciones if indicador_sel == "Todos" else [indicador_sel]
    filtrado = indicadores[
        (indicadores["indicador"].isin(seleccion))
        & (indicadores["anio"].between(rango[0], rango[1]))
    ].copy()

    ipc_filtrado = ipc[ipc["anio"].between(*rango)]
    desempleo_filtrado = desempleo[desempleo["anio"].between(*rango)]
    inflacion_actual = ipc_filtrado.iloc[-1]
    inflacion_previa = ipc_filtrado.iloc[-2]
    desempleo_actual = desempleo_filtrado.iloc[-1]
    desempleo_previo = desempleo_filtrado.iloc[-2]
    ultimo_ipc_2025 = ipc_2025.iloc[-1]

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Informe economico interactivo</div>
            <h1>Dashboard Economico de Panama</h1>
            <p class="hero-subtitle">Plataforma institucional para analisis economico y consulta asistida con evidencia.</p>
            <p>Visualiza inflacion, desempleo, predicciones y trazabilidad de datos oficiales mediante un dashboard ejecutivo construido en Streamlit.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    cards = [
        (
            "Inflacion (IPC)",
            f"{inflacion_actual['inflacion_pct']:.2f}%",
            f"Cambio vs {int(inflacion_previa['anio'])}: {inflacion_actual['inflacion_pct'] - inflacion_previa['inflacion_pct']:+.2f} pp",
        ),
        (
            "Desempleo",
            f"{desempleo_actual['desempleo_pct']:.2f}%",
            f"Cambio vs {int(desempleo_previo['anio'])}: {desempleo_actual['desempleo_pct'] - desempleo_previo['desempleo_pct']:+.2f} pp",
        ),
        (
            "IPC mensual 2025",
            f"{ultimo_ipc_2025['ipc_total']:.1f}",
            f"Ultimo corte: {pd.to_datetime(ultimo_ipc_2025['fecha']).strftime('%b %Y')}",
        ),
        (
            "Indicadores modelados",
            str(len(predicciones)),
            "Inflacion y desempleo",
        ),
    ]
    for col, (label, value, sub) in zip(metric_cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">{label}</div>
                    <div class="info-value">{value}</div>
                    <div class="info-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    tab_rag, tab_resumen, tab_hist, tab_pred, tab_pipeline = st.tabs(
        ["Asistente RAG", "Resumen", "Comparacion historica", "Predicciones", "Datos y pipeline"]
    )

    with tab_resumen:
        st.markdown("### Resumen ejecutivo")
        st.markdown(
            '<div class="section-note">Este bloque resume el estado actual de los indicadores y ofrece una lectura rapida del corte mensual 2025 para IPC.</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([0.63, 0.37], gap="large")
        with left:
            st.plotly_chart(
                construir_grafico_comparativo(filtrado),
                width="stretch",
                config={"displayModeBar": False},
                key="grafico_resumen_comparativo",
            )
        with right:
            st.markdown("#### Lectura mensual IPC 2025")
            st.plotly_chart(
                construir_grafico_ipc_2025(ipc_2025),
                width="stretch",
                config={"displayModeBar": False},
                key="grafico_resumen_ipc_2025",
            )
            st.markdown(
                f"""
                - Ultimo valor mensual disponible: `{ultimo_ipc_2025['ipc_total']:.1f}`
                - Fecha del corte: `{pd.to_datetime(ultimo_ipc_2025['fecha']).strftime('%Y-%m')}`
                - Fuente: `INEC`
                """
            )

    with tab_hist:
        st.markdown("### Comparacion historica")
        st.markdown(
            '<div class="section-note">Compara la trayectoria anual de inflacion y desempleo dentro del rango seleccionado.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            construir_grafico_comparativo(filtrado),
            width="stretch",
            config={"displayModeBar": False},
            key="grafico_historico_comparativo",
        )
        st.dataframe(
            filtrado.sort_values(["anio", "indicador"]).reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )

    with tab_pred:
        st.markdown("### Predicciones")
        st.markdown(
            '<div class="section-note">El modelo usa regresion lineal sobre cada serie historica para estimar el siguiente corte disponible.</div>',
            unsafe_allow_html=True,
        )
        pred_cols = st.columns(len(predicciones))
        for col, (_, fila) in zip(pred_cols, predicciones.iterrows()):
            delta = fila["cambio_absoluto"]
            signo = "sube" if delta >= 0 else "baja"
            with col:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-label">{fila['indicador']}</div>
                        <div class="info-value">{fila['valor_predicho']:.2f}%</div>
                        <div class="info-sub">Prediccion {int(fila['anio_prediccion'])}</div>
                        <div class="info-sub">El modelo {signo} {abs(delta):.2f} pp frente al valor base.</div>
                        <div class="info-sub">R² entrenamiento: {fila['r2_entrenamiento']:.3f}</div>
                        <div class="info-sub">MAE holdout: {fila['mae_holdout']:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_rag:
        st.markdown("### Asistente RAG")
        st.markdown(
            '<div class="section-note">Haz preguntas sobre tendencia, maximos, minimos, ultimo valor, ano especifico, comparacion, IPC mensual 2025 o prediccion. Cada respuesta muestra fuentes y evidencia recuperada.</div>',
            unsafe_allow_html=True,
        )

        chat_html = ['<div class="chat-panel">']
        for mensaje in st.session_state.chat_history[-6:]:
            if mensaje["role"] == "user":
                chat_html.append(
                    '<div class="chat-user">'
                    '<div class="chat-label">Tu</div>'
                    f'{escape(mensaje["content"])}'
                    "</div>"
                )
            else:
                fuentes_html = ""
                if mensaje.get("sources"):
                    fuentes_html = "".join(
                        [f'<span class="source-chip">{escape(fuente)}</span>' for fuente in mensaje["sources"]]
                    )
                evidencia_html = ""
                if mensaje.get("evidence"):
                    items = []
                    for evidencia in mensaje["evidence"][:1]:
                        items.append(
                            f"{escape(evidencia['indicador'])} | {escape(str(evidencia['periodo']))} | score {evidencia['score']:.3f}"
                        )
                    evidencia_html = (
                        '<div class="evidence-box">'
                        '<div class="evidence-meta">Evidencia consultada</div>'
                        + "<br>".join(items)
                        + "</div>"
                    )
                chat_html.append(
                    '<div class="chat-assistant">'
                    '<div class="chat-label">Asistente economico</div>'
                    f'{escape(mensaje["content"])}'
                    f'<div>{fuentes_html}</div>'
                    f"{evidencia_html}"
                    "</div>"
                )
        chat_html.append("</div>")
        st.markdown("".join(chat_html), unsafe_allow_html=True)

        if st.session_state.get("rag_cargando"):
            st.markdown(
                """
                <div class="rag-loader">
                    <div class="rag-loader-ring"></div>
                    <div class="rag-loader-copy">
                        Consultando Groq
                        <span>Recuperando evidencia y generando respuesta con la API...</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="chat-input-shell">', unsafe_allow_html=True)
        with st.form("rag_form", clear_on_submit=True):
            pregunta = st.text_input(
                "Pregunta",
                placeholder="Escribe tu pregunta economica sobre Panama...",
                label_visibility="collapsed",
            )
            enviar = st.form_submit_button("Enviar")
        st.markdown("</div>", unsafe_allow_html=True)

        if enviar and pregunta.strip():
            st.session_state.rag_pregunta_pendiente = pregunta.strip()
            st.session_state.rag_cargando = True
            st.rerun()
        if st.session_state.get("rag_cargando") and st.session_state.get("rag_pregunta_pendiente"):
            guardar_interaccion(st.session_state.rag_pregunta_pendiente)
            st.session_state.rag_pregunta_pendiente = ""
            st.rerun()

        st.markdown('<div class="suggestion-label">Preguntas rapidas</div>', unsafe_allow_html=True)
        quick_cols = st.columns(5)
        preguntas_demo = [
            "Evolucion de la inflacion",
            "Maximo historico del desempleo",
            "Compara indicadores actuales",
            "Ultimo IPC mensual 2025",
            "Prediccion del modelo",
        ]
        preguntas_reales = [
            "Como ha evolucionado la inflacion en Panama en los ultimos 5 anos?",
            "Cual fue el maximo historico del desempleo?",
            "Compara inflacion y desempleo en el ultimo dato disponible",
            "Cual es el ultimo IPC mensual 2025?",
            "Que predice el modelo para el siguiente corte?",
        ]
        for col, label_demo, pregunta_demo in zip(quick_cols, preguntas_demo, preguntas_reales):
            with col:
                if st.button(label_demo, key=f"demo_{pregunta_demo}", width="stretch"):
                    guardar_interaccion(pregunta_demo)
                    st.rerun()

    with tab_pipeline:
        st.markdown("### Datos y pipeline")
        st.markdown(
            '<div class="section-note">Esta seccion documenta el flujo del proyecto: ingesta, limpieza, consolidacion y salida procesada para dashboard, modelo y RAG.</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(
                """
                **Etapas del pipeline**

                1. Carga de IPC historico desde INEC.
                2. Carga de IPC mensual 2025 desde INEC.
                3. Carga de desempleo historico desde Banco Mundial.
                4. Limpieza, conversion de tipos y normalizacion.
                5. Consolidacion en `indicadores_limpios.csv`.
                6. Consumo por dashboard, modelo predictivo y RAG.
                """
            )
        with c2:
            st.markdown(
                """
                **Archivos generados**

                - `datos/procesados/ipc_limpio.csv`
                - `datos/procesados/ipc_2025_mensual.csv`
                - `datos/procesados/desempleo_limpio.csv`
                - `datos/procesados/indicadores_limpios.csv`
                """
            )

        st.markdown("#### Vista del dataset consolidado")
        st.dataframe(indicadores, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
