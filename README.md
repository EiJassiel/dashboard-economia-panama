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
- Asistente economico con RAG local, deteccion de intencion, respuesta calculada desde CSV y evidencia recuperada.
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

El asistente implementa una version local de RAG usando recuperacion TF-IDF sobre los datasets procesados, datos mensuales de IPC 2025 y resumenes generados por el modelo. Detecta preguntas sobre ultimo dato, tendencia, maximos, minimos, anos especificos, comparacion, prediccion y metodologia. Cada respuesta devuelve fuentes y evidencia recuperada con score. No depende de APIs externas.
