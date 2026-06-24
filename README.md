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
- Asistente economico con RAG local: recuperacion sobre datos procesados y generacion con un modelo real via Ollama.
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

El asistente implementa una version local de RAG usando recuperacion TF-IDF sobre los datasets procesados, datos mensuales de IPC 2025 y resumenes generados por el modelo. La respuesta final la redacta un modelo real servido localmente por Ollama, usando solo la evidencia recuperada. Cada respuesta devuelve fuentes y evidencia recuperada con score. No depende de APIs pagas.

## Configurar Ollama

1. Instala Ollama desde [ollama.com](https://ollama.com/).
2. Descarga al menos un modelo local:

```bash
ollama pull llama3.1:8b
```

3. Si quieres usar otro modelo, define la variable de entorno antes de abrir Streamlit:

```bash
$env:OLLAMA_MODEL="qwen2.5:7b"
streamlit run app.py
```

4. Ejecuta la app con Ollama corriendo localmente en `http://127.0.0.1:11434`.
