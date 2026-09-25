# 📊 YouTube Analytics Agent · Análisis Predictivo e Inteligencia Conversacional

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io/)
[![BigQuery ML](https://img.shields.io/badge/BigQuery%20ML-XGBoost-orange)](https://cloud.google.com/bigquery-ml)
[![Gemini API](https://img.shields.io/badge/Gemini%20API-RAG-blueviolet)](https://ai.google.dev/)
[![Looker Studio](https://img.shields.io/badge/Looker%20Studio-Dashboard-blue)](https://datastudio.google.com/reporting/1bcb2a87-8ba6-43f7-8405-52fc05328a60)

## 📌 Descripción del proyecto

**YouTube Analytics Agent** es un agente conversacional impulsado por IA que permite a los creadores de contenido analizar el rendimiento de su canal de YouTube de forma intuitiva. El sistema combina **análisis de datos**, **procesamiento de lenguaje natural** y **machine learning** para ofrecer recomendaciones estratégicas personalizadas.

El agente interpreta preguntas en lenguaje natural, consulta métricas almacenadas en BigQuery y responde con insights accionables para mejorar el alcance y el engagement del canal.

## 🎯 Funcionalidades clave

- **Análisis de rendimiento:** Consulta métricas clave (engagement, vistas, likes, comentarios) organizadas por tema, día de la semana y evolución temporal.
- **Búsqueda semántica en transcripciones:** Encuentra el momento exacto en que se menciona un tema específico dentro de un video, utilizando embeddings y técnicas de **Retrieval-Augmented Generation (RAG)**.
- **Modelo predictivo de rendimiento:** Implementación de un modelo **XGBoost** dentro de **BigQuery ML** para predecir las vistas de un video en función de variables como duración, tema y engagement previo.
- **Panel de control interactivo:** Visualizaciones dinámicas con gráficas de barras y líneas que permiten filtrar y explorar el rendimiento del canal desde diferentes perspectivas.
- **Interfaz conversacional:** Chat lateral con **Gemini** para responder preguntas en lenguaje natural y proporcionar insights accionables.
- **Detección de patrones:** Identifica qué temas, formatos y días de publicación generan mayor engagement, y recomienda estrategias basadas en datos históricos.

## 🛠️ Tecnologías utilizadas

| Categoría | Herramientas |
|-----------|--------------|
| **Lenguajes** | Python |
| **Interfaz web** | Streamlit |
| **Visualización** | Plotly |
| **Machine Learning** | XGBoost (BigQuery ML) |
| **NLP y RAG** | Gemini API, Sentence-Transformers, Embeddings |
| **Infraestructura de datos** | Google BigQuery, YouTube Data API v3 |
| **Despliegue** | Streamlit Cloud, GitHub |

## 📁 Estructura del repositorio
youtube-analytics-agent/
├── app.py # Interfaz principal de Streamlit
├── agent.py # Lógica del agente RAG y modelos ML
├── secrets.toml # Configuración de credenciales (no versionado)
├── .gitignore # Archivos excluidos del repositorio
├── requirements.txt # Dependencias del proyecto
└── README.md

## 🚀 Cómo ejecutar localmente

1. Clona el repositorio:
   ```bash
   git clone https://github.com/koseguera-r/youtube-analytics-agent.git
   cd youtube-analytics-agent
2. Crea un entorno virtual e instala dependencias:
    python -m venv venv
    source venv/bin/activate        # Linux/Mac
    pip install -r requirements.txt
3. Configura tus credenciales en secrets.toml:
    PROJECT_ID = "tu-proyecto-gcp"
    DATASET_ID = "youtube"
    TABLE_NAME = "fact_final"
    SEGMENTS_TABLE_NAME = "transcript_segments_transformers"
    CHANNEL_ID = "tu-channel-id"
    GOOGLE_API_KEY = "tu-api-key"
4. Ejecuta la aplicación:
    streamlit run app.py
🌐 Aplicación en vivo

https://static.streamlit.io/badges/streamlit_badge_black_white.svg

Pruébala aquí: YouTube Analytics Agent
📊 Dashboard interactivo en Looker Studio

Explora las métricas del canal de YouTube de forma visual e interactiva:

https://img.shields.io/badge/Looker%20Studio-Dashboard-blue

🔗 Ver Dashboard en Looker Studio

💡 Ejemplos de preguntas que puedes hacerle al agente

    "¿Qué temas tienen mejor engagement en mi canal?"

    "¿En qué minuto hablé de relaciones tóxicas en el video de la semana pasada?"

    "¿Qué día me recomiendas subir un video?"

    "Dame el top 5 de videos con más views por minuto"

    "¿Qué videos superaron la predicción del modelo?"

    "¿Cómo puedo mejorar el alcance de mi canal?"

🧠 ¿Cómo funciona el agente?

    Interpretación de la pregunta: El agente clasifica la intención del usuario (ranking, búsqueda de momentos, análisis de temas, etc.) usando Gemini.

    Recuperación de datos: Consulta métricas en BigQuery y, si es necesario, realiza búsqueda semántica en transcripciones con embeddings.

    Re-ranking contextual: Filtra y reordena los resultados considerando jerga mexicana, relevancia semántica y métricas de rendimiento.
📈 Resultados destacados

    Procesamiento de +250 videos y +300 registros de métricas del canal.

    Modelo XGBoost con 90% de certeza en la predicción de vistas.

    Búsqueda semántica con 98% de precisión al localizar el minuto exacto de un tema en transcripciones.

    Detección de patrones: los Shorts sobre dinero generan 44% más engagement que los podcasts de finanzas.

📫 Autor

Kevin Yael Oseguera Reyes
kevinose1666@gmail.com
LinkedIn | GitHub
⭐ Si este proyecto te ha sido útil, no olvides darle una estrella en GitHub.
