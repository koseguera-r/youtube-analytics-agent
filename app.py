# -*- coding: utf-8 -*-

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery


# =========================
# 1. CONFIGURACION DE PAGINA
# =========================

st.set_page_config(
    page_title="Las Damitas Histeria | Agente YouTube",
    page_icon="play",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 2. CREDENCIALES
# =========================

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("Error critico: no se encontro GOOGLE_API_KEY en Secrets o en el archivo .env.")
    st.stop()

try:
    has_gcp_secret = "gcp_service_account" in st.secrets
except Exception:
    has_gcp_secret = False

if not has_gcp_secret and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    st.warning(
        "No se encontro gcp_service_account en Secrets ni GOOGLE_APPLICATION_CREDENTIALS. "
        "En local se intentara usar credenciales ADC de Google."
    )




# =========================
# 3. IMPORTACION DEL AGENTE
# =========================

try:
    from agent import (
        CHANNEL_ID,
        DATASET_ID,
        PROJECT_ID,
        SEGMENTS_TABLE_ID,
        TABLE_NAME,
        get_agent,
        get_retriever,
    )
except Exception as exc:
    st.error("Error al importar el agente desde agent.py.")
    st.exception(exc)
    st.stop()

# =========================
# 4. RECURSOS
# =========================

try:
    retriever = get_retriever()
    agent = get_agent()
except Exception as exc:
    st.error(
        "No se pudo inicializar BigQuery. Revisa gcp_service_account en "
        "Streamlit Secrets o configura credenciales ADC."
    )
    st.exception(exc)
    st.stop()


def format_compact_number(value):
    try:
        value = float(value or 0)
    except Exception:
        return "0"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{int(value):,}"


# =========================
# FUNCIONES DE GRÁFICAS
# =========================


# Grafica de evolución de suscriptores, con validaciones para evitar errores por datos faltantes o nulos.
@st.cache_data(ttl=600)
def plot_subscriber_growth():
    """Gráfica de líneas: evolución de suscriptores"""
    try:
        # 1. Verificar la existencia de la columna 'suscriptores_canal'
        check_column_query = f"""
        SELECT column_name
        FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'fact_metricas_variables'
          AND column_name = 'suscriptores_canal'
        """
        col_result = retriever.client.query(check_column_query).result()
        if not any(col_result):
            st.warning("La columna 'suscriptores_canal' no existe en la tabla.")
            return None

        # 2. Consulta los datos de suscriptores
        query = f"""
        SELECT fecha_publicacion, suscriptores_canal
        FROM `{PROJECT_ID}.{DATASET_ID}.fact_metricas_variables`
        WHERE suscriptores_canal IS NOT NULL
          AND fecha_publicacion IS NOT NULL
        ORDER BY fecha_publicacion ASC
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]

        if not rows:
            st.info("No hay datos históricos de suscriptores para mostrar evolución.")
            return None

        # 3. Crear DataFrame y validar
        df = pd.DataFrame(rows)
        df["fecha_publicacion"] = pd.to_datetime(df["fecha_publicacion"])
        df = df.sort_values("fecha_publicacion")

        if df.empty:
            st.info("No hay datos suficientes para generar la gráfica de suscriptores.")
            return None

        # 4. Crear la gráfica de líneas con Plotly
        import plotly.express as px

        fig = px.line(
            df,
            x="fecha_publicacion",
            y="suscriptores_canal",
            title="Evolución de suscriptores",
            labels={"fecha_publicacion": "Fecha", "suscriptores_canal": "Suscriptores"},
        )
        fig.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font_color="#ffffff",
            title_font_color="#ffffff",
        )
        return fig

    except Exception as e:
        st.error(f"Error al generar la gráfica de suscriptores: {e}")
        return None


@st.cache_data(ttl=600)
def plot_views_by_topic():
    """Gráfica de barras horizontales: vistas totales por tema (Plotly)"""
    try:
        topics = retriever.topic_performance(limit=8, order_by="views")
        if not topics:
            return None
        df = pd.DataFrame(topics)
        import plotly.express as px

        fig = px.bar(
            df,
            x="views_totales",
            y="tema_legible",
            orientation="h",
            title="📊 Vistas totales por tema",
            labels={"views_totales": "Vistas", "tema_legible": "Tema"},
            color="views_totales",
            color_continuous_scale="Reds",
        )
        fig.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font_color="white",
            title_font_color="white",
        )
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de vistas por tema: {e}")
        return None


@st.cache_data(ttl=600)
def plot_engagement_by_topic():
    """Gráfica de barras horizontales: engagement promedio por tema (Plotly)"""
    try:
        topics = retriever.topic_performance(limit=8, order_by="engagement")
        if not topics:
            return None
        df = pd.DataFrame(topics)
        import plotly.express as px

        fig = px.bar(
            df,
            x="engagement_promedio",
            y="tema_legible",
            orientation="h",
            title="🔥 Engagement por tema",
            labels={"engagement_promedio": "Engagement (%)", "tema_legible": "Tema"},
            color="engagement_promedio",
            color_continuous_scale="Reds",
        )
        fig.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font_color="white",
            title_font_color="white",
        )
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de engagement por tema: {e}")
        return None


@st.cache_data(show_spinner=False, ttl=900)
def load_sidebar_stats():
    try:
        metrics = retriever.analytics_summary() or {}
    except Exception:
        metrics = {}

    try:
        segment_stats = retriever.transcript_segments_stats()
    except Exception:
        segment_stats = {
            "existe": False,
            "videos": 0,
            "segmentos": 0,
            "actualizado": None,
            "embedding_model": None,
        }

    return metrics, segment_stats

# ============================================================
# NUEVAS FUNCIONES: LIKES y WATCH TIME (tema, día, evolución)
# ============================================================

@st.cache_data(ttl=600)
def plot_likes_by_topic():
    """Gráfica de barras horizontales: likes totales por tema"""
    try:
        topics = retriever.topic_performance(limit=8, order_by="likes")
        if not topics:
            return None
        df = pd.DataFrame(topics)
        fig = px.bar(
            df, x='likes_totales', y='tema_legible', orientation='h',
            title="👍 Likes totales por tema",
            labels={'likes_totales': 'Likes', 'tema_legible': 'Tema'},
            color='likes_totales', color_continuous_scale='Reds'
        )
        fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                          font_color='white', title_font_color='white')
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de likes por tema: {e}")
        return None

@st.cache_data(ttl=600)
def plot_likes_by_weekday():
    """Gráfica de barras: likes promedio por día de la semana"""
    try:
        query = f"""
        SELECT 
            FORMAT_TIMESTAMP('%A', SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion)) as dia_semana,
            AVG(likes) as avg_likes,
            COUNT(*) as num_videos
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE channel_id = '{CHANNEL_ID}'
          AND fecha_publicacion IS NOT NULL
        GROUP BY dia_semana
        ORDER BY CASE dia_semana
            WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2 WHEN 'Wednesday' THEN 3
            WHEN 'Thursday' THEN 4 WHEN 'Friday' THEN 5 WHEN 'Saturday' THEN 6 WHEN 'Sunday' THEN 7 END
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]
        if not rows:
            st.info("No hay datos suficientes para mostrar likes por día.")
            return None
        df = pd.DataFrame(rows)
        dias_es = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miércoles',
                   'Thursday':'Jueves','Friday':'Viernes','Saturday':'Sábado','Sunday':'Domingo'}
        df['dia_semana'] = df['dia_semana'].map(dias_es)
        fig = px.bar(df, x='dia_semana', y='avg_likes',
                     title="👍 Likes promedio por día de publicación",
                     labels={'dia_semana':'Día','avg_likes':'Likes promedio'},
                     color='avg_likes', color_continuous_scale='Reds', text='num_videos')
        fig.update_traces(textposition='outside')
        fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                          font_color='white', title_font_color='white')
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de likes por día: {e}")
        return None

@st.cache_data(ttl=600)
def plot_likes_over_time():
    """Gráfica de líneas: evolución de likes a lo largo del tiempo"""
    try:
        query = f"""
        SELECT 
            SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion) as fecha_ts,
            SUM(likes) as likes_diarios
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE channel_id = '{CHANNEL_ID}'
          AND fecha_publicacion IS NOT NULL
        GROUP BY fecha_ts
        ORDER BY fecha_ts ASC
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]
        if not rows:
            st.info("No hay datos suficientes para mostrar evolución de likes.")
            return None
        df = pd.DataFrame(rows).dropna(subset=['fecha_ts']).sort_values('fecha_ts')
        fig = px.line(df, x='fecha_ts', y='likes_diarios',
                      title="📈 Evolución de likes a través del tiempo",
                      labels={'fecha_ts':'Fecha','likes_diarios':'Likes diarios'},
                      markers=True)
        fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                          font_color='white', title_font_color='white')
        fig.update_traces(line=dict(color='#e63946', width=2), marker=dict(size=4, color='#ff6b6b'))
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de evolución de likes: {e}")
        return None

@st.cache_data(ttl=600)
def plot_watchtime_by_topic():
    """Gráfica de barras horizontales: watch time total por tema (en horas)"""
    try:
        # Obtenemos los datos desde topic_performance y calculamos watch_time = views * duracion_minutos / 60
        # Pero topic_performance no tiene duración, así que necesitamos una consulta personalizada.
        # Usaremos una consulta directa a BigQuery agregando por tema.
        query = f"""
        SELECT 
            tema_legible,
            SUM(views * duracion_minutos / 60) as watch_time_horas,
            COUNT(DISTINCT video_id) as num_videos
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE channel_id = '{CHANNEL_ID}'
          AND tema_legible IS NOT NULL
        GROUP BY tema_legible
        ORDER BY watch_time_horas DESC
        LIMIT 8
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]
        if not rows:
            st.info("No hay datos suficientes para mostrar watch time por tema.")
            return None
        df = pd.DataFrame(rows)
        fig = px.bar(df, x='watch_time_horas', y='tema_legible', orientation='h',
                     title="⏱️ Watch time total por tema (horas)",
                     labels={'watch_time_horas': 'Horas', 'tema_legible': 'Tema'},
                     color='watch_time_horas', color_continuous_scale='Reds')
        fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                          font_color='white', title_font_color='white')
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de watch time por tema: {e}")
        return None

@st.cache_data(ttl=600)
def plot_watchtime_by_weekday():
    """Gráfica de barras: watch time promedio por día de la semana (horas)"""
    try:
        query = f"""
        SELECT 
            FORMAT_TIMESTAMP('%A', SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion)) as dia_semana,
            AVG(views * duracion_minutos / 60) as avg_watch_time,
            COUNT(*) as num_videos
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE channel_id = '{CHANNEL_ID}'
          AND fecha_publicacion IS NOT NULL
        GROUP BY dia_semana
        ORDER BY CASE dia_semana
            WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2 WHEN 'Wednesday' THEN 3
            WHEN 'Thursday' THEN 4 WHEN 'Friday' THEN 5 WHEN 'Saturday' THEN 6 WHEN 'Sunday' THEN 7 END
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]
        if not rows:
            st.info("No hay datos suficientes para mostrar watch time por día.")
            return None
        df = pd.DataFrame(rows)
        dias_es = {'Monday':'Lunes','Tuesday':'Martes','Wednesday':'Miércoles',
                   'Thursday':'Jueves','Friday':'Viernes','Saturday':'Sábado','Sunday':'Domingo'}
        df['dia_semana'] = df['dia_semana'].map(dias_es)
        fig = px.bar(df, x='dia_semana', y='avg_watch_time',
                     title="⏱️ Watch time promedio por día (horas)",
                     labels={'dia_semana':'Día','avg_watch_time':'Horas promedio'},
                     color='avg_watch_time', color_continuous_scale='Reds', text='num_videos')
        fig.update_traces(textposition='outside')
        fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                          font_color='white', title_font_color='white')
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de watch time por día: {e}")
        return None

@st.cache_data(ttl=600)
def plot_watchtime_over_time():
    """Gráfica de líneas: evolución de watch time diario (horas)"""
    try:
        query = f"""
        SELECT 
            SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion) as fecha_ts,
            SUM(views * duracion_minutos / 60) as watch_time_diario
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE channel_id = '{CHANNEL_ID}'
          AND fecha_publicacion IS NOT NULL
        GROUP BY fecha_ts
        ORDER BY fecha_ts ASC
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]
        if not rows:
            st.info("No hay datos suficientes para mostrar evolución de watch time.")
            return None
        df = pd.DataFrame(rows).dropna(subset=['fecha_ts']).sort_values('fecha_ts')
        fig = px.line(df, x='fecha_ts', y='watch_time_diario',
                      title="📈 Evolución de watch time diario (horas)",
                      labels={'fecha_ts':'Fecha','watch_time_diario':'Horas diarias'},
                      markers=True)
        fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                          font_color='white', title_font_color='white')
        fig.update_traces(line=dict(color='#e63946', width=2), marker=dict(size=4, color='#ff6b6b'))
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de evolución de watch time: {e}")
        return None

@st.cache_data(ttl=600)
def plot_engagement_over_time():
    """Gráfica de líneas: evolución de engagement promedio diario"""
    try:
        query = f"""
        SELECT 
            SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion) as fecha_ts,
            AVG(engagement) as engagement_diario
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE channel_id = '{CHANNEL_ID}'
          AND fecha_publicacion IS NOT NULL
        GROUP BY fecha_ts
        ORDER BY fecha_ts ASC
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]
        if not rows:
            st.info("No hay datos suficientes para mostrar evolución de engagement.")
            return None
        df = pd.DataFrame(rows).dropna(subset=['fecha_ts']).sort_values('fecha_ts')
        fig = px.line(df, x='fecha_ts', y='engagement_diario',
                      title="📈 Evolución del engagement promedio diario",
                      labels={'fecha_ts':'Fecha','engagement_diario':'Engagement (%)'},
                      markers=True)
        fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                          font_color='white', title_font_color='white')
        fig.update_traces(line=dict(color='#e63946', width=2), marker=dict(size=4, color='#ff6b6b'))
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de evolución de engagement: {e}")
        return None

metrics, segment_stats = load_sidebar_stats()


@st.cache_data(ttl=600)
def plot_views_by_weekday():
    """Gráfica de barras: vistas promedio por día de la semana (timestamp string)"""
    try:
        query = f"""
        SELECT
            FORMAT_TIMESTAMP('%A', SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion)) as dia_semana,
            AVG(views) as avg_views,
            COUNT(*) as num_videos
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE channel_id = '{CHANNEL_ID}'
          AND fecha_publicacion IS NOT NULL
          AND SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion) IS NOT NULL
        GROUP BY dia_semana
        ORDER BY
            CASE dia_semana
                WHEN 'Monday' THEN 1
                WHEN 'Tuesday' THEN 2
                WHEN 'Wednesday' THEN 3
                WHEN 'Thursday' THEN 4
                WHEN 'Friday' THEN 5
                WHEN 'Saturday' THEN 6
                WHEN 'Sunday' THEN 7
            END
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]
        if not rows:
            st.info("No hay datos suficientes para mostrar vistas por día.")
            return None

        df = pd.DataFrame(rows)
        # Traducir días al español
        dias_es = {
            "Monday": "Lunes",
            "Tuesday": "Martes",
            "Wednesday": "Miércoles",
            "Thursday": "Jueves",
            "Friday": "Viernes",
            "Saturday": "Sábado",
            "Sunday": "Domingo",
        }
        df["dia_semana"] = df["dia_semana"].map(dias_es).fillna(df["dia_semana"])

        import plotly.express as px

        fig = px.bar(
            df,
            x="dia_semana",
            y="avg_views",
            title="📅 Vistas promedio por día de publicación",
            labels={"dia_semana": "Día", "avg_views": "Vistas promedio"},
            color="avg_views",
            color_continuous_scale="Reds",
            text="num_videos",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#1e1e1e",
            font_color="white",
            title_font_color="white",
            xaxis=dict(tickangle=0),
        )
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de vistas por día: {e}")
        return None

@st.cache_data(ttl=600)
def plot_engagement_by_weekday():
    """Gráfica de barras: engagement promedio por día de la semana"""
    try:
        query = f"""
        SELECT 
            FORMAT_TIMESTAMP('%A', SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion)) as dia_semana,
            AVG(engagement) as avg_engagement,
            COUNT(*) as num_videos
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE channel_id = '{CHANNEL_ID}'
          AND fecha_publicacion IS NOT NULL
          AND SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion) IS NOT NULL
        GROUP BY dia_semana
        ORDER BY 
            CASE dia_semana
                WHEN 'Monday' THEN 1
                WHEN 'Tuesday' THEN 2
                WHEN 'Wednesday' THEN 3
                WHEN 'Thursday' THEN 4
                WHEN 'Friday' THEN 5
                WHEN 'Saturday' THEN 6
                WHEN 'Sunday' THEN 7
            END
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]
        if not rows:
            st.info("No hay datos suficientes para mostrar engagement por día.")
            return None
        
        df = pd.DataFrame(rows)
        # Traducir días al español
        dias_es = {
            'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
            'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        }
        df['dia_semana'] = df['dia_semana'].map(dias_es).fillna(df['dia_semana'])
        
        import plotly.express as px
        fig = px.bar(
            df,
            x='dia_semana',
            y='avg_engagement',
            title="❤️ Engagement promedio por día de publicación",
            labels={'dia_semana': 'Día', 'avg_engagement': 'Engagement promedio (%)'},
            color='avg_engagement',
            color_continuous_scale='Reds',
            text='num_videos'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font_color='white',
            title_font_color='white',
            xaxis=dict(tickangle=0)
        )
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de engagement por día: {e}")
        return None

@st.cache_data(ttl=600)
def plot_views_over_time():
    """Gráfica de líneas: evolución de views a lo largo del tiempo"""
    try:
        query = f"""
        SELECT 
            SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', fecha_publicacion) as fecha_ts,
            SUM(views) as views_diarias
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE channel_id = '{CHANNEL_ID}'
          AND fecha_publicacion IS NOT NULL
        GROUP BY fecha_ts
        ORDER BY fecha_ts ASC
        """
        result = retriever.client.query(query).result()
        rows = [dict(row) for row in result]
        if not rows:
            st.info("No hay datos suficientes para mostrar evolución de vistas.")
            return None
        
        df = pd.DataFrame(rows)
        df = df.dropna(subset=['fecha_ts'])
        df = df.sort_values('fecha_ts')
        
        import plotly.express as px
        fig = px.line(
            df, x='fecha_ts', y='views_diarias',
            title="📈 Evolución de visualizaciones a través del tiempo",
            labels={'fecha_ts': 'Fecha', 'views_diarias': 'Vistas diarias'},
            markers=True
        )
        fig.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font_color='white',
            title_font_color='white',
            xaxis=dict(tickangle=45)
        )
        fig.update_traces(line=dict(color='#e63946', width=2), marker=dict(size=4, color='#ff6b6b'))
        return fig
    except Exception as e:
        st.error(f"Error en gráfica de evolución de vistas: {e}")
        return None


# =========================
# 5. ESTILOS GENERALES DE LA APP
# =========================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body, .stApp {
        background-color: #0f0f0f;  /* Fondo oscuro principal */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #f0f0f0;
    }

    /* Ocultar elementos nativos de Streamlit */
    header, footer, #MainMenu {
        display: none !important;
    }

    .block-container {
        padding: 1rem 1.2rem 5rem 1.2rem;
        max-width: 1680px;
        margin: 0 auto;
    }

    /* Tarjetas oscuras */
    .card, .metric-card, .video-card, .topic-card {
        background-color: #1e1e1e;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        transition: all 0.2s ease;
        border: 1px solid #2c2c2c;
    }
    .card:hover, .video-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.6);
        border-color: #e63946;
    }
    .metric-card {
        padding: 0.85rem 0.8rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #a0a0a0;
    }
    .metric-change {
        color: #10b981;
        font-size: 0.7rem;
        margin-top: 0.3rem;
    }

    /* Header */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.1rem;
        flex-wrap: wrap;
    }
    .logo-area {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .logo-icon {
        background: #e63946;
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.6rem;
        box-shadow: 0 4px 12px rgba(230,57,70,0.4);
    }
    .logo-text h1 {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    .logo-text p {
        font-size: 0.7rem;
        color: #a0a0a0;
    }
    .badge {
        background: #2c2c2c;
        padding: 0.4rem 1rem;
        border-radius: 40px;
        font-size: 0.75rem;
        color: #e63946;
        font-weight: 700;
        border: 1px solid #3a3a3a;
    }

    /* Grid de videos */
    .videos-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1.2rem;
        margin-top: 1rem;
    }
   .video-card {
    background-color: #1e1e1e;
    border-radius: 16px;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid #2c2c2c;
    cursor: pointer;
    }
    .video-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.4);
        border-color: #e63946;
    }
    .video-thumb {
        width: 100%;
        aspect-ratio: 16/9;
        object-fit: cover;
        background: #2a2a2a;
    }
    .video-info {
        padding: 0.8rem;
    }
    .video-title {
        font-weight: 700;
        font-size: 0.85rem;
        color: #f0f0f0;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        line-height: 1.3;
        margin-bottom: 0.3rem;
    }
    .video-stats {
        font-size: 0.7rem;
        color: #a0a0a0;
        display: flex;
        justify-content: space-between;
    }

    /* Gráfico (matplotlib) - se ajustará con estilo propio */
    .chart-container {
        background: #1e1e1e;
        border-radius: 20px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #2c2c2c;
    }

    /* Tendencias */
    .topic-grid {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .topic-card {
        flex: 1;
        padding: 0.8rem;
        text-align: center;
    }

    /* Chat - fondos oscuros */
    .stChatInput input {
        background-color: #2c2c2c !important;
        border: 1px solid #3a3a3a !important;
        color: #ffffff !important;
        border-radius: 40px !important;
        padding: 0.6rem 1rem !important;
    }
    .stChatInput input::placeholder {
        color: #a0a0a0 !important;
    }
    .stChatInput button {
        background-color: #e63946 !important;
        border-radius: 40px !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stChatInput button:hover {
        background-color: #c1121f !important;
    }

    /* Mensajes del chat */
    [data-testid="stChatMessageContent"] {
        background-color: #1e1e1e !important;
        color: #f0f0f0 !important;
        border: 1px solid #2c2c2c !important;
        border-radius: 12px !important;
        padding: 0.75rem 0.85rem !important;
        overflow-wrap: anywhere;
    }
    .topic-card-full {
    background: #1e1e1e;
    border-radius: 16px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    border: 1px solid #2c2c2c;
    word-wrap: break-word;
    white-space: normal;
    }
    .stButton button {
    background-color: #e63946 !important;
    color: white !important;
    border: none !important;
    }
    .stButton button:hover {
        background-color: #c1121f !important;
    }

    .thinking-box {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        background: #1e1e1e;
        border: 1px solid #2c2c2c;
        border-radius: 12px;
        padding: 0.75rem 0.85rem;
        color: #f0f0f0;
        font-size: 0.85rem;
    }
    .thinking-dot {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: #e63946;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.35; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.1); }
        100% { opacity: 0.35; transform: scale(0.9); }
    }

    /* Estilo para el label del selectbox */
    div[data-testid="stSelectbox"] label {
        color: #ffffff !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Estilo para el selectbox en sí (el recuadro) */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #f0f0f0 !important;
        border-color: #e63946 !important;
        border-radius: 8px !important;
    }


    /* Estilo para el ícono de dropdown */
    div[data-testid="stSelectbox"] svg {
        fill: #e63946 !important;
    }


    /* Estilo para el menú desplegable (las opciones) */
    div[data-testid="stSelectbox"] ul {
        background-color: #1e1e1e !important;
        border: 1px solid #e63946 !important;
    }

    /* Estilo para cada opción individual */
    div[data-testid="stSelectbox"] ul li {
        color: #ffffff !important;
        background-color: #1e1e1e !important;
        font-size: 0.8rem !important;
    }

    /* Estilo cuando pasas el mouse sobre una opción */
    div[data-testid="stSelectbox"] ul li:hover {
        background-color: #e63946 !important;
        color: #ffffff !important;
    }

    /* Estilo para el ícono de flecha */
    div[data-testid="stSelectbox"] svg {
    fill: #e63946 !important;
    }

    /* Estilo para el texto dentro del selectbox */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
        color: #000000 !important;        /* Cambiado a negro */
        font-size: 0.85rem !important;
        font-weight: bold !important;     /* Añadido negritas */
    }

    /* Hacer visible el texto del selectbox cuando está cerrado */
    .stSelectbox [data-baseweb="select"] div:first-child {
        color: #000000 !important;        /* Cambiado a negro */
        font-weight: bold !important;     /* Añadido negritas */
    }

    /* Botones del panel interactivo */
    .stButton button {
        background-color: #2c2c2c !important;
        color: #ffffff !important;
        border: 1px solid #e63946 !important;
        border-radius: 40px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton button:hover {
        background-color: #e63946 !important;
        color: #ffffff !important;
        border-color: #ffffff !important;
        transform: translateY(-2px);
    }

    .video-card.viral {
    border: 2px solid #e63946;
    box-shadow: 0 0 8px #e63946;
    }

    /* Color blanco para las etiquetas de number_input y slider */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Crear dos columnas: izquierda (2/3 del ancho) y derecha (1/3)
col_izq, col_der = st.columns([2.35, 1.05], gap="medium")

with col_izq:

    # =========================
    # 6. HEADER
    # =========================


    @st.cache_data(ttl=600)
    def get_dashboard_data():
        try:
            top_videos = retriever.ranked_videos(order_by="views", limit=6)
            channel_profile = retriever.channel_profile() or {}
            analytics = retriever.analytics_summary() or {}
            topics = retriever.topic_performance(limit=5, order_by="engagement")
            subs_df = []
            # ... (código para subs_df) ...
            return top_videos, channel_profile, analytics, topics, subs_df
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return [], {}, {}, [], []

    # Cargar datos
    top_videos, channel_profile, analytics, topics, subs_df = get_dashboard_data()

    st.markdown(
        f"""
    <div class="dashboard-header">
        <div class="logo-area">
            <div class="logo-icon">▶</div>
            <div class="logo-text">
                <h1>INFLUENCER INSIGHTS LAB</h1>
                <p>YouTube Analytics · Gemini AI · BigQuery</p>
            </div>
        </div>
        <div class="badge">Pro Analyst</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- Video destacado y metricas ---
    if top_videos:
        featured = top_videos[0]
        video_url = featured.get("url_video", "")
        video_id = video_url.split("v=")[-1].split("&")[0] if "v=" in video_url else ""

        col1, col2 = st.columns([1.6, 1])
        with col1:
            if video_id:
                st.markdown(
                    f"""
                <div class="card">
                    <iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}"
                    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowfullscreen></iframe>
                    <div style="margin-top: 0.8rem;">
                        <strong>{featured.get('titulo_video', 'Top video')}</strong><br>
                        <span style="font-size:0.8rem; color:#606060;">{featured.get('views', 0):,} vistas · {featured.get('engagement', 0):.1%} engagement</span>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Video destacado no disponible (falta URL)")
        with col2:
            st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
            metrics_data = [
                ("Engagement", f"{analytics.get('engagement_promedio', 0):.1f}%", "+2.1%"),
                ("Watch Time", f"{featured.get('views', 0) * (featured.get('duracion_minutos', 10) / 60):,.0f} hrs", "+14%"),
                ("Views", f"{analytics.get('views', 0):,.0f}", "+8.5%"),
                ("Subscribers Gained", f"+{channel_profile.get('suscriptores_canal', 0)-100000:,}", "+38%"),
            ]
            for label, value, change in metrics_data:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-change">▲ {change}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No hay videos para mostrar. Verifica la conexión con BigQuery.")

    # --- Top videos de crecimiento ---
    if top_videos:
        st.markdown("## 🚀 Videos Más Vistos")
        cols = st.columns(3)
        for idx, video in enumerate(top_videos[:6]):
            with cols[idx % 3]:
                titulo = video.get("titulo_video", "Sin título")
                views = video.get("views", 0)
                likes = video.get("likes", 0)
                engagement = video.get("engagement", 0)
                url_video = video.get("url_video", "")

                # Extraer video_id de la URL de YouTube
                video_id = None
                if url_video:
                    if "v=" in url_video:
                        video_id = url_video.split("v=")[1].split("&")[0]
                    elif "youtu.be/" in url_video:
                        video_id = url_video.split("youtu.be/")[1].split("?")[0]

                # Generar miniatura desde YouTube (calidad media)
                if video_id:
                    thumb_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                else:
                    thumb_url = "https://placehold.co/320x180/1e1e1e/e63946?text=Preview+No+Disponible"

                # Tarjeta cliqueable
                st.markdown(
                    f"""
                <a href="{url_video}" target="_blank" style="text-decoration: none;">
                    <div class="video-card">
                        <img class="video-thumb" src="{thumb_url}" loading="lazy"
                            onerror="this.src='https://placehold.co/320x180/1e1e1e/e63946?text=Error+Cargar'">
                        <div class="video-info">
                            <div class="video-title">{titulo}</div>
                            <div class="video-stats">
                                <span>👁 {views:,}</span>
                                <span>❤️ {likes:,}</span>
                                <span>📈 {engagement:.1%}</span>
                            </div>
                        </div>
                    </div>
                </a>
                """,
                    unsafe_allow_html=True,
                )

    
    # --- Tendencias de contenido ---
    if topics:
        st.markdown("## 📊 Tendencias de Contenido")
        # Usamos 2 columnas en lugar de 4 para que quepa mejor el texto completo
        topic_cols = st.columns(2)
        for i, topic in enumerate(topics[:4]):
            with topic_cols[i % 2]:
                # HTML personalizado sin truncamiento
                st.markdown(
                    f"""
                <div class="topic-card-full">
                    <div style="font-weight: 700; color: #e63946; margin-bottom: 4px;">{topic.get('tema_legible', 'Tema')}</div>
                    <div style="font-size: 1.2rem; font-weight: 800;">{topic.get('engagement_promedio', 0):.1f}% <span style="font-size: 0.8rem;">engagement</span></div>
                    <div style="color: #10b981; font-size: 0.8rem;">▲ {topic.get('videos', 0)} videos</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # ==================== PANEL INTERACTIVO DE GRÁFICAS ====================
    st.markdown("## 🎛️ Panel de análisis interactivo")

    # --- Botones para seleccionar métrica ---
    st.markdown("### 📈 Selecciona la métrica:")
    col_metricas = st.columns(4)
    metricas_opciones = ["Views", "Engagement", "Likes", "Watch Time"]
    metricas_icons = ["👁️", "❤️", "👍", "⏱️"]

    for idx, (metrica, icono) in enumerate(zip(metricas_opciones, metricas_icons)):
        with col_metricas[idx]:
            if st.button(f"{icono} {metrica}", key=f"btn_{metrica}", use_container_width=True):
                st.session_state['metrica_seleccionada'] = metrica.lower()

    # Valor por defecto
    if 'metrica_seleccionada' not in st.session_state:
        st.session_state['metrica_seleccionada'] = 'views'

    # --- Botones para seleccionar tipo de visualización ---
    st.markdown("### 📊 Visualizar por:")
    col_tipos = st.columns(3)
    tipos_opciones = ["Tema", "Día de semana", "Evolución temporal"]
    tipos_icons = ["📂", "📅", "📈"]

    for idx, (tipo, icono) in enumerate(zip(tipos_opciones, tipos_icons)):
        with col_tipos[idx]:
            if st.button(f"{icono} {tipo}", key=f"btn_{tipo}", use_container_width=True):
                st.session_state['tipo_visualizacion'] = tipo.lower()

    if 'tipo_visualizacion' not in st.session_state:
        st.session_state['tipo_visualizacion'] = 'tema'

    # --- Mostrar la gráfica según selección ---
    st.markdown("---")
    st.markdown(f"#### 🔍 Mostrando: **{st.session_state['metrica_seleccionada'].capitalize()}** por **{st.session_state['tipo_visualizacion']}**")

    # Función maestra para generar gráfica
    def mostrar_grafica_dinamica(metrica, tipo):
    # Normalizar tipo (puede venir con espacios o guiones)
        tipo = tipo.lower().replace(' ', '_')
        if tipo == 'tema':
            if metrica == 'views':
                fig = plot_views_by_topic()
            elif metrica == 'engagement':
                fig = plot_engagement_by_topic()
            elif metrica == 'likes':
                fig = plot_likes_by_topic()
            elif metrica == 'watch time':
                fig = plot_watchtime_by_topic()
            else:
                st.info(f"📌 {metrica.capitalize()} por tema no disponible aún.")
                return None
        elif tipo == 'día_de_semana' or tipo == 'día de semana':
            if metrica == 'views':
                fig = plot_views_by_weekday()
            elif metrica == 'engagement':
                fig = plot_engagement_by_weekday()
            elif metrica == 'likes':
                fig = plot_likes_by_weekday()
            elif metrica == 'watch time':
                fig = plot_watchtime_by_weekday()
            else:
                st.info(f"📌 {metrica.capitalize()} por día no disponible aún.")
                return None
        elif tipo == 'evolución_temporal' or tipo == 'evolución temporal':
            if metrica == 'views':
                fig = plot_views_over_time()
            elif metrica == 'engagement':
                fig = plot_engagement_over_time()
            elif metrica == 'likes':
                fig = plot_likes_over_time()
            elif metrica == 'watch time':
                fig = plot_watchtime_over_time()
            else:
                st.info(f"📌 Evolución temporal de {metrica.capitalize()} no disponible aún.")
                return None
        else:
            st.info("Tipo de visualización no reconocido.")
            return None
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ No se pudo generar la gráfica. Puede faltar datos en la tabla.")

    # Ejecutar la visualización
    mostrar_grafica_dinamica(st.session_state['metrica_seleccionada'], st.session_state['tipo_visualizacion'])

    # ============================================
    # SECCIÓN: EXPLORADOR DE VIDEOS (con filtros y miniaturas pequeñas)
    # ============================================
    st.markdown("## 🔥 Explorador de Videos")
    st.markdown("_Filtra y ordena los videos según métricas clave_")

    # --- Controles interactivos (filtros y orden) ---
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        orden_por = st.selectbox(
            "Ordenar por",
            options=["views", "engagement", "likes", "duracion_minutos"],
            format_func=lambda x: {"views": "👁️ Vistas", "engagement": "📈 Engagement", "likes": "❤️ Likes", "duracion_minutos": "⏱️ Duración"}[x],
            key="orden_videos"
        )
    with col2:
        min_views = st.number_input("Vistas mínimas (miles)", min_value=0, value=100, step=50, key="min_views") * 1000
    with col3:
        min_engagement_pct = st.slider("Engagement mínimo (%)", 0.0, 100.0, 0.0, 5.0, key="min_eng")
        min_engagement = min_engagement_pct / 100.0  # convertir a decimal (ej: 10% -> 0.10)
    with col4:
        max_resultados = st.number_input("Mostrar", min_value=5, max_value=50, value=15, step=5, key="max_res")

    # Obtener lista amplia de videos (por ejemplo, 50) ordenada según selección
    videos_crudos = retriever.ranked_videos(order_by=orden_por, limit=50)

    # Aplicar filtros (vistas mínimas y engagement mínimo)
    videos_filtrados = []
    for v in videos_crudos:
        if v.get('views', 0) >= min_views and v.get('engagement', 0) >= min_engagement:
            videos_filtrados.append(v)

    # Limitar número de resultados
    videos_filtrados = videos_filtrados[:max_resultados]

    if not videos_filtrados:
        st.info("No hay videos que cumplan los filtros seleccionados. Ajusta los umbrales.")
    else:
        # Mostrar cada video en una fila con miniatura pequeña
        for video in videos_filtrados:
            titulo = video.get('titulo_video', 'Sin título')
            views = video.get('views', 0)
            likes = video.get('likes', 0)
            engagement = video.get('engagement', 0) * 100  # a porcentaje
            duracion = video.get('duracion_minutos', 0)
            url_video = video.get('url_video', '')

            # Extraer ID de la miniatura
            video_id = None
            if url_video:
                if 'v=' in url_video:
                    video_id = url_video.split('v=')[1].split('&')[0]
                elif 'youtu.be/' in url_video:
                    video_id = url_video.split('youtu.be/')[1].split('?')[0]
            if video_id:
                thumb_url = f"https://img.youtube.com/vi/{video_id}/default.jpg"  # miniatura pequeña (120x90)
            else:
                thumb_url = "https://placehold.co/120x90/1e1e1e/e63946?text=No"

            # Formatear duración
            if duracion >= 60:
                horas = int(duracion // 60)
                minutos = int(duracion % 60)
                duracion_str = f"{horas}h {minutos}m"
            else:
                duracion_str = f"{int(duracion)} min"

            # Crear una fila con columnas: miniatura | info
            c1, c2 = st.columns([1, 5])
            with c1:
                st.markdown(f'<a href="{url_video}" target="_blank"><img src="{thumb_url}" width="80" style="border-radius: 8px;"></a>', unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                **{titulo}**  
                <span style="font-size:0.75rem; color:#a0a0a0;">⏱️ {duracion_str}</span> &nbsp;|&nbsp;
                <span style="font-size:0.75rem; color:#a0a0a0;">👁️ {views:,}</span> &nbsp;|&nbsp;
                <span style="font-size:0.75rem; color:#a0a0a0;">❤️ {likes:,}</span> &nbsp;|&nbsp;
                <span style="font-size:0.75rem; color:#10b981;">📈 {engagement:.1f}%</span>
                """, unsafe_allow_html=True)
            st.divider()

# Actualizacion, movimiento del chat bot a la parte derecha, para que quede mas visible y con mas espacio para las respuestas largas, ademas de que se vea mas como un asistente personal que siempre esta a la mano.
with col_der:
    # =========================
    # MEMORIA Y CHAT
    # =========================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Bienvenida si no hay mensajes (solo se muestra cuando está vacío)
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-logo">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" style="width:34px;height:34px;fill:white;">
                    <path d="M19.59 7a2.5 2.5 0 0 0-1.76-1.76C16.46 5 12 5 12 5s-4.46 0-5.83.24A2.5 2.5 0 0 0 4.41 7 26 26 0 0 0 4.17 12a26 26 0 0 0 .24 5 2.5 2.5 0 0 0 1.76 1.76C7.54 19 12 19 12 19s4.46 0 5.83-.24A2.5 2.5 0 0 0 19.59 17 26 26 0 0 0 19.83 12a26 26 0 0 0-.24-5zM10 15v-6l5 3-5 3z"/>
                </svg>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="empty-title">Hola, soy tu agente de YouTube</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="empty-text">
                Puedo analizar el rendimiento de <b>Las Damitas Histeria</b>, encontrar
                en qué episodio hablaron de un tema y recomendarte decisiones con datos.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Botón para limpiar conversación
    if st.button("🗑️ Limpiar conversación", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Input del chat
    prompt = st.chat_input("Pregunta sobre el canal... ej: ¿Qué temas tuvieron más engagement?")

    # Procesar el prompt
    if prompt:
        history_for_agent = st.session_state.messages[-8:]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(
                """
                <div class="thinking-box">
                    <div class="thinking-dot"></div>
                    Analizando métricas y transcripciones...
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                answer = agent.answer(prompt, history=history_for_agent)
                thinking_placeholder.empty()
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                thinking_placeholder.empty()
                error_message = (
                    "**Ocurrió un error al procesar tu pregunta.**\n\n"
                    f"`{str(exc)}`\n\n"
                    "Revisa Secrets, permisos de BigQuery y la tabla de segmentos."
                )
                st.error(error_message)
                st.exception(exc)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
