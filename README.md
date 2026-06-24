# Dashboard de Indicadores Economicos de Panama

Proyecto en Python y Streamlit para visualizar, comparar y proyectar indicadores economicos de Panama usando la data disponible en el repositorio.

## Alcance actual

- Pipeline de datos funcional con 2 fuentes ya cargadas:
  - INEC: IPC historico y corte mensual 2025.
  - Banco Mundial: desempleo historico de Panama.
- Limpieza y consolidacion de datos en `datos/procesados/`.
- Modelo predictivo con regresion lineal para:
  - Inflacion (IPC)
  - Desempleo
- Dashboard interactivo en Streamlit.
- Asistente economico con RAG: recuperacion sobre datos procesados y generacion con Groq.
- Vista de rubrica objetivo 100/100 dentro del dashboard: pipeline, visualizacion, RAG, modelo y documentacion.

## Estructura

```text
dashboard-economico-panama/
├── app.py
├── requirements.txt
├── README.md
├── chatbot/
│   └── rag.py
├── datos/
│   ├── brutos/
│   └── procesados/
├── modelos/
│   └── prediccion.py
└── pipeline/
    ├── ingesta.py
    └── limpieza.py
```

## Como ejecutar

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecuta la aplicacion:

```bash
streamlit run app.py
```

## Archivos generados por el pipeline

- `datos/procesados/ipc_limpio.csv`
- `datos/procesados/ipc_2025_mensual.csv`
- `datos/procesados/desempleo_limpio.csv`
- `datos/procesados/indicadores_limpios.csv`

## Nota sobre el chatbot

El asistente implementa una version de RAG usando recuperacion TF-IDF sobre los datasets procesados, datos mensuales de IPC 2025 y resumenes generados por el modelo. La respuesta final la redacta Groq usando solo la evidencia recuperada. Cada respuesta devuelve fuentes y evidencia con score.

## Configurar Groq

1. Crea una API key en [Groq Console](https://console.groq.com/keys).
2. En local, define la variable de entorno antes de abrir Streamlit:

```powershell
$env:GROQ_API_KEY="tu_api_key"
```

3. Si quieres cambiar el modelo por defecto, define tambien:

```powershell
$env:GROQ_MODEL="llama-3.1-8b-instant"
streamlit run app.py
```

4. En Streamlit Community Cloud, pega el secret asi:

```toml
GROQ_API_KEY="tu_api_key"
GROQ_MODEL="llama-3.1-8b-instant"
```

5. Despliega la app en Streamlit Community Cloud apuntando a `app.py`.
