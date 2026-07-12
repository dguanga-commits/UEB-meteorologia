# -*- coding: utf-8 -*-
"""
Dashboard Interactivo de Clima y Agroproducción - Provincia de Bolívar, Ecuador
Desarrollado para el Trabajo Final del Curso de Especialización en Forecasting.
Estudiante: Deysi Margoth Guanga Chunata
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Dashboard Agroclimático Bolívar",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. PALETA DE COLORES PROFESIONAL (PEGA ESTO AQUÍ)
# ---------------------------------------------------------
# Colores recomendados
COLORS = {
    'primary': '#00F2FE',      # Cyan brillante
    'secondary': '#4FACFE',    # Azul claro
    'dark': '#0e1117',         # Negro suave
    'light': '#ffffff',        # Blanco
    'sidebar_bg': '#f8f9fa',   # Gris muy claro (para el sidebar)
    'text': '#262730',         # Gris oscuro (para el texto)
    'success': '#10B981',      # Verde
    'warning': '#F59E0B',      # Naranja
    'error': '#EF4444'         # Rojo
}

# ---------------------------------------------------------
# ESTILOS CSS PERSONALIZADOS (Glassmorphism & Aesthetics)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .kpi-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.25);
        transition: transform 0.3s ease, border-color 0.3s ease;
        margin-bottom: 1.5rem;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 254, 0.4);
    }
    .kpi-title {
        color: #F8FAFC;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        color: #94A3B8;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 10px 0 5px 0;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    .kpi-delta-up {
        color: #10B981;
    }
    .kpi-delta-down {
        color: #EF4444;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# INFORMACIÓN GEOGRÁFICA DE LOS CANTONES DE BOLÍVAR
# ---------------------------------------------------------
cantones_info = {
    "Guaranda": {"lat": -1.5905, "lon": -79.0024, "alt": 2668, "tipo": "Sierra (Frío-Templado)", "cultivo_principal": "Maíz, Trigo, Cebada, Papa", "lapse_relative": 0.0, "rain_factor": 1.0},
    "San Miguel": {"lat": -1.7083, "lon": -79.0433, "alt": 2470, "tipo": "Sierra (Templado)", "cultivo_principal": "Maíz suave, Fréjol, Trigo", "lapse_relative": 1.2, "rain_factor": 1.2},
    "Chimbo": {"lat": -1.6833, "lon": -79.0333, "alt": 2450, "tipo": "Sierra (Templado)", "cultivo_principal": "Maíz, Hortalizas, Frutales", "lapse_relative": 1.3, "rain_factor": 1.2},
    "Chillanes": {"lat": -1.9428, "lon": -79.0664, "alt": 2300, "tipo": "Sierra-Subtrópico (Húmedo)", "cultivo_principal": "Maíz duro, Mora, Café", "lapse_relative": 2.2, "rain_factor": 1.5},
    "Caluma": {"lat": -1.6333, "lon": -79.2500, "alt": 350, "tipo": "Subtrópico (Cálido-Húmedo)", "cultivo_principal": "Cacao, Naranja, Banano", "lapse_relative": 13.9, "rain_factor": 2.5},
    "Echeandía": {"lat": -1.4333, "lon": -79.2833, "alt": 300, "tipo": "Subtrópico (Cálido-Húmedo)", "cultivo_principal": "Cacao, Café, Caña de azúcar", "lapse_relative": 14.2, "rain_factor": 2.5},
    "Las Naves": {"lat": -1.3333, "lon": -79.3167, "alt": 200, "tipo": "Costa/Tropical (Cálido)", "cultivo_principal": "Cacao, Arroz, Maíz duro", "lapse_relative": 14.8, "rain_factor": 2.7}
}

# ---------------------------------------------------------
# FUNCIONES DE CARGA DE DATOS CON CACHÉ
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df_diario = pd.read_csv("datos_preprocesados_diarios_py.csv")
        df_test = pd.read_csv("predicciones_test.csv")
        df_anual = pd.read_csv("datos_anuales_var.csv")
        df_irf = pd.read_csv("var_irf_results.csv")
        
        with open("metricas_modelos.json", "r", encoding='utf-8') as f:
            metrics = json.load(f)
            
        with open("bolivar_cantones.geojson", "r", encoding='utf-8') as f:
            geojson = json.load(f)
        
        # Estandarizar columnas de fecha
        def standardize_date_column(df, possible_names):
            for col in possible_names:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    return col
            for col in df.columns:
                if 'fecha' in col.lower() or 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col])
                    return col
            return None
        
        date_col_diario = standardize_date_column(df_diario, ['Fecha_Dia', 'fecha_dia', 'Fecha', 'Date'])
        if date_col_diario and date_col_diario != 'Fecha_Dia':
            df_diario = df_diario.rename(columns={date_col_diario: 'Fecha_Dia'})
        
        date_col_test = standardize_date_column(df_test, ['Fecha_Dia', 'fecha_dia', 'Fecha', 'Date'])
        if date_col_test and date_col_test != 'Fecha_Dia':
            df_test = df_test.rename(columns={date_col_test: 'Fecha_Dia'})
        
        df_test_pred = df_test.copy()
        
        return df_diario, df_test, df_anual, df_irf, metrics, geojson, df_test_pred
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        st.stop()

# Cargar datos
df_diario, df_test, df_anual, df_irf, metrics_summary, bolivar_geojson, df_test_pred = load_data()

# ---------------------------------------------------------
# ENTRENAMIENTO DE MODELOS PARA EL SIMULADOR
# ---------------------------------------------------------
@st.cache_resource
def train_simulator_models():
    try:
        features = ["Temp_Media_Lag1", "Temp_Media_Lag2", "Precip_Acum_Lag1", "Precip_Acum_Lag2", "Hum_Media"]
        
        # Verificar que las columnas existan
        missing_cols = [col for col in features if col not in df_diario.columns]
        if missing_cols:
            st.warning(f"Columnas faltantes en df_diario: {missing_cols}. Usando valores por defecto.")
            # Crear datos dummy para el simulador
            X = pd.DataFrame(np.random.rand(100, 5), columns=features)
            y = np.random.rand(100) * 10 + 10
        else:
            X = df_diario[features].dropna()
            y = df_diario.loc[X.index, 'Temp_Media']
        
        model_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        model_rf.fit(X, y)
        
        model_xgb = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
        model_xgb.fit(X, y)
        
        return model_rf, model_xgb
    except Exception as e:
        st.error(f"Error al entrenar modelos: {e}")
        return None, None

sim_model_rf, sim_model_xgb = train_simulator_models()

# ---------------------------------------------------------
# SIDEBAR / PANEL DE CONTROL
# ---------------------------------------------------------
#st.sidebar.image("https://img.icons8.com/clouds/200/000000/weather.png", width=120)
st.sidebar.markdown("### ⛈️")
st.sidebar.markdown("<h2 style='color: #00F2FE; font-weight: 700; margin-bottom: 0px;'>Pronóstico Bolívar</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #94A3B8; font-size: 0.85rem; margin-top: 0px;'>Programa Experto en Forecasting</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Configuración")
model_selected = st.sidebar.selectbox(
    "Modelo de Predicción Base",
    options=["XGBoost (Recomendado)", "Random Forest", "LSTM", "SARIMA"],
    index=0
)

model_col_map = {
    "XGBoost (Recomendado)": "XGBoost",
    "Random Forest": "Random_Forest",
    "LSTM": "LSTM",
    "SARIMA": "SARIMA"
}
model_col = model_col_map[model_selected]

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Estación Base")
st.sidebar.markdown("""
- **Nombre:** Guaranda UEB
- **Ubicación:** Cantón Guaranda
- **Altitud:** 2,668 m s.n.m.
- **Rango de Datos:** 2016 - 2023
""")

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color: #64748B; font-size: 0.8rem;'>Diseñado por: Deysi Guanga<br>Experto en Forecasting / IA</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# CUERPO PRINCIPAL DEL DASHBOARD
# ---------------------------------------------------------
st.markdown("<h1 class='title-gradient'>Monitoreo y Pronóstico Climático de Bolívar, Ecuador</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Modelado predictivo del clima con Machine Learning y Deep Learning y su impacto en la productividad agropecuaria regional.</p>", unsafe_allow_html=True)

tabs = st.tabs([
    "🗺️ Mapa y Resumen del Clima", 
    "📈 Comparativa de Modelos (Forecasting)", 
    "🌱 Impacto Agroproductivo (VAR)", 
    "🔮 Simulador Predictivo"
])

# ==============================================================================
# PESTAÑA 1: MAPA Y RESUMEN DEL CLIMA
# ==============================================================================
with tabs[0]:
    st.markdown("### 📅 Comportamiento Diario del Clima (Test 2023)")
    st.markdown("Seleccione una fecha dentro del conjunto de prueba (2023) para visualizar la predicción y el comportamiento estimado por cantones.")
    
    # Detectar columna de fecha
    date_column = None
    possible_date_columns = ['Fecha_Dia', 'fecha_dia', 'Fecha', 'fecha', 'Date', 'date']
    
    for col in possible_date_columns:
        if col in df_test_pred.columns:
            date_column = col
            break
    
    if date_column is None:
        for col in df_test_pred.columns:
            if 'fecha' in col.lower() or 'date' in col.lower():
                date_column = col
                break
    
    if date_column is None:
        st.error("❌ No se encontró una columna de fecha en los datos")
        st.stop()
    
    min_date = df_test_pred[date_column].min().to_pydatetime()
    max_date = df_test_pred[date_column].max().to_pydatetime()
    
    selected_date = st.slider(
        "Línea de Tiempo (Fecha)",
        min_value=min_date,
        max_value=max_date,
        value=min_date,
        format="YYYY-MM-DD"
    )
    
    day_row = df_test_pred[df_test_pred[date_column] == selected_date].iloc[0]
    
    kpi_cols = st.columns(4)
    
    temp_real = day_row.get('Temp_Real', 0)
    temp_pred = day_row.get(model_col, 0)
    temp_diff = temp_pred - temp_real
    color_class = "kpi-delta-up" if temp_diff >= 0 else "kpi-delta-down"
    sign = "+" if temp_diff >= 0 else ""
    
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Temp. Media (Guaranda)</div>
            <div class="kpi-value">{temp_real:.1f} °C</div>
            <div class="kpi-delta {color_class}">Pronóstico ({model_selected}): <b>{temp_pred:.1f} °C</b> ({sign}{temp_diff:.2f} °C)</div>
        </div>
        """, unsafe_allow_html=True)
    
    hum_val = day_row.get('Hum_Media', 0)
    hum_mean = df_diario['Hum_Media'].mean() if 'Hum_Media' in df_diario.columns else 80
    hum_diff = hum_val - hum_mean
    color_class = "kpi-delta-up" if hum_diff >= 0 else "kpi-delta-down"
    sign = "+" if hum_diff >= 0 else ""
    
    with kpi_cols[1]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Humedad Relativa</div>
            <div class="kpi-value">{hum_val:.1f} %</div>
            <div class="kpi-delta {color_class}">Desviación vs Promedio: <b>{sign}{hum_diff:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    
    precip_val = day_row.get('Precip_Acum', 0)
    with kpi_cols[2]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Lluvia Diaria (Estación)</div>
            <div class="kpi-value">{precip_val:.2f} mm</div>
            <div class="kpi-delta" style="color: #64748B;">Estado: <b>{"Lluvia" if precip_val > 0.1 else "Seco"}</b></div>
        </div>
        """, unsafe_allow_html=True)
    
    gdd_val = max(temp_real - 10, 0)
    with kpi_cols[3]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">GDD Diario (Cultivos)</div>
            <div class="kpi-value">{gdd_val:.2f} GDD</div>
            <div class="kpi-delta" style="color: #00F2FE;">Base biológica: <b>10.0 °C</b></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    map_cols = st.columns([3, 2])
    
    cantones_list = []
    for c_name, c_data in cantones_info.items():
        c_temp = temp_pred + c_data['lapse_relative']
        c_rain = precip_val * c_data['rain_factor']
        
        cantones_list.append({
            "Cantón": c_name,
            "Temperatura_Est (°C)": round(c_temp, 2),
            "Precipitación_Est (mm)": round(c_rain, 2),
            "Altitud (m)": c_data['alt'],
            "Latitud": c_data['lat'],
            "Longitud": c_data['lon'],
            "Microclima": c_data['tipo'],
            "Cultivo Principal": c_data['cultivo_principal']
        })
    df_cantones = pd.DataFrame(cantones_list)
    
    with map_cols[0]:
        st.markdown("#### 🗺️ Mapa Choropleth de Predicción por Cantón")
        map_var = st.radio("Variable a visualizar en el mapa:", ["Temperatura_Est (°C)", "Precipitación_Est (mm)"], horizontal=True)
        
        color_scale = "RdYlBu_r" if "Temperatura" in map_var else "Blues"
        
        fig_map = px.choropleth_mapbox(
            df_cantones,
            geojson=bolivar_geojson,
            locations="Cantón",
            featureidkey="properties.name",
            color=map_var,
            color_continuous_scale=color_scale,
            mapbox_style="carto-positron",
            zoom=8.8,
            center={"lat": -1.60, "lon": -79.12},
            opacity=0.75,
            hover_name="Cantón",
            hover_data={
                "Altitud (m)": True, 
                "Microclima": True,
                "Cultivo Principal": True,
                "Temperatura_Est (°C)": True,
                "Precipitación_Est (mm)": True,
                "Cantón": False
            },
            height=500
        )
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(
                title=map_var,
                thicknessmode="pixels", thickness=15,
                lenmode="fraction", len=0.8,
                yanchor="top", y=0.9,
                ticks="outside"
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    with map_cols[1]:
        st.markdown("#### 📊 Distribución Térmica por Cantón")
        
        fig_bar = px.bar(
            df_cantones.sort_values("Altitud (m)", ascending=False),
            x="Temperatura_Est (°C)",
            y="Cantón",
            orientation='h',
            color="Temperatura_Est (°C)",
            color_continuous_scale="RdYlBu_r",
            text="Temperatura_Est (°C)",
            hover_data=["Altitud (m)", "Microclima"],
            height=450
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Temperatura Estimada (°C)",
            yaxis_title=None,
            coloraxis_showscale=False
        )
        fig_bar.update_traces(texttemplate='%{text} °C', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

# ==============================================================================
# PESTAÑA 2: COMPARATIVA DE MODELOS
# ==============================================================================
# ==============================================================================
# PESTAÑA 2: COMPARATIVA DE MODELOS
# ==============================================================================
with tabs[1]:
    st.markdown("### 📈 Evaluación Histórica de Modelos Predictivos")
    
    metrics_cols = st.columns([1, 2])
    
    with metrics_cols[0]:
        st.markdown("#### 📋 Tabla General de Métricas")
        df_metrics = pd.DataFrame(metrics_summary).T
        df_metrics.columns = ["RMSE (°C)", "MAE (°C)", "MAPE (%)", "R² (Coef. Det.)"]
        st.dataframe(
            df_metrics.style.format({
                "RMSE (°C)": "{:.4f}",
                "MAE (°C)": "{:.4f}",
                "MAPE (%)": "{:.2f}%",
                "R² (Coef. Det.)": "{:.4f}"
            }).highlight_min(subset=["RMSE (°C)", "MAE (°C)", "MAPE (%)"], color="#1E293B")
              .highlight_max(subset=["R² (Coef. Det.)"], color="#1E293B"),
            use_container_width=True
        )
        
        st.markdown("""
        > **Conclusión:**
        > **XGBoost** presenta el mejor balance con RMSE = 0.75 y R² = 0.67, superando ampliamente a SARIMA (R² negativo).
        """)
    
    with metrics_cols[1]:
        st.markdown("#### 🏆 Comparación Visual de Métricas")
        
        fig_metrics = go.Figure()
        
        fig_metrics.add_trace(go.Bar(
            x=df_metrics.index,
            y=df_metrics["RMSE (°C)"],
            name="RMSE (°C)",
            marker_color="#EF4444",
            yaxis="y1"
        ))
        
        fig_metrics.add_trace(go.Bar(
            x=df_metrics.index,
            y=df_metrics["R² (Coef. Det.)"],
            name="R² Score",
            marker_color="#00F2FE",
            yaxis="y2"
        ))
        
        fig_metrics.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title="Modelo", 
                gridcolor='rgba(255,255,255,0.05)'
            ),
            yaxis=dict(
                title=dict(text="RMSE (Menor es mejor)", font=dict(color="#EF4444")),
                tickfont=dict(color="#EF4444"),
                gridcolor='rgba(255,255,255,0.05)'
            ),
            yaxis2=dict(
                title=dict(text="R² (Cercano a 1 es mejor)", font=dict(color="#00F2FE")),
                tickfont=dict(color="#00F2FE"),
                overlaying="y1",
                side="right",
                range=[-1, 1]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 🔍 Visualizador de Serie de Tiempo (Test 2023)")
    
    models_to_plot = st.multiselect(
        "Modelos a desplegar:",
        options=["SARIMA", "Random Forest", "XGBoost", "LSTM"],
        default=["XGBoost", "LSTM"]
    )
    
    fig_time = go.Figure()
    
    # Curva real
    fig_time.add_trace(go.Scatter(
        x=df_test_pred[date_column],
        y=df_test_pred['Temp_Real'],
        mode='lines',
        name='Temp Real',
        line=dict(color='#F8FAFC', width=2),
        opacity=0.8
    ))
    
    colors_map = {
        "SARIMA": "#64748B",
        "Random Forest": "#10B981",
        "XGBoost": "#00F2FE",
        "LSTM": "#A855F7"
    }
    
    for m in models_to_plot:
        col = m.replace(" ", "_")
        if col in df_test_pred.columns:
            fig_time.add_trace(go.Scatter(
                x=df_test_pred[date_column],
                y=df_test_pred[col],
                mode='lines',
                name=m,
                line=dict(color=colors_map[m], width=1.5)
            ))
    
    fig_time.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Fecha",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(
            title="Temperatura (°C)",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

# ==============================================================================
# PESTAÑA 3: IMPACTO AGROPRODUCTIVO
# ==============================================================================
with tabs[2]:
    st.markdown("### 🌾 Interacción Clima-Cultivos y Modelo VAR")
    
    st.markdown("#### ⚡ Funciones de Impulso-Respuesta")
    
    fig_irf = go.Figure()
    fig_irf.add_trace(go.Scatter(
        x=df_irf['Periodo'],
        y=df_irf['Impulso_Temp'],
        mode='lines+markers',
        name='Respuesta a Temperatura',
        line=dict(color='#EF4444', width=2)
    ))
    fig_irf.add_trace(go.Scatter(
        x=df_irf['Periodo'],
        y=df_irf['Impulso_Lluvia'],
        mode='lines+markers',
        name='Respuesta a Precipitación',
        line=dict(color='#00F2FE', width=2)
    ))
    
    fig_irf.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Periodo"),
        yaxis=dict(title="Variación en Rendimiento (Ton/Ha)")
    )
    st.plotly_chart(fig_irf, use_container_width=True)

# ==============================================================================
# PESTAÑA 4: SIMULADOR PREDICTIVO
# ==============================================================================
with tabs[3]:
    st.markdown("### 🔮 Calculadora Agroclimática en Tiempo Real")
    
    if sim_model_rf is None or sim_model_xgb is None:
        st.error("Los modelos no están disponibles. Verifique los datos de entrada.")
    else:
        sim_cols = st.columns([1, 1])
        
        with sim_cols[0]:
            st.markdown("#### ⚙️ Parámetros de Entrada")
            
            temp_lag1 = st.slider("Temperatura de Ayer (°C)", 10.0, 25.0, 14.5, 0.1)
            temp_lag2 = st.slider("Temperatura hace 2 Días (°C)", 10.0, 25.0, 14.3, 0.1)
            rain_lag1 = st.slider("Precipitación de Ayer (mm)", 0.0, 50.0, 2.5, 0.1)
            rain_lag2 = st.slider("Precipitación hace 2 Días (mm)", 0.0, 50.0, 1.0, 0.1)
            hum_actual = st.slider("Humedad Relativa (%)", 30.0, 100.0, 82.0, 0.5)
            
            sim_model_type = st.radio("Modelo:", ["XGBoost", "Random Forest"], horizontal=True)
        
        with sim_cols[1]:
            st.markdown("#### 📊 Predicciones")
            
            features_input = pd.DataFrame([{
                "Temp_Media_Lag1": temp_lag1,
                "Temp_Media_Lag2": temp_lag2,
                "Precip_Acum_Lag1": rain_lag1,
                "Precip_Acum_Lag2": rain_lag2,
                "Hum_Media": hum_actual
            }])
            
            if sim_model_type == "XGBoost":
                pred_temp_base = sim_model_xgb.predict(features_input)[0]
            else:
                pred_temp_base = sim_model_rf.predict(features_input)[0]
            
            st.metric("Temperatura Predicha", f"{pred_temp_base:.2f} °C")
