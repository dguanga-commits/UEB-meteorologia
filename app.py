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
from datetime import datetime, timedelta

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Dashboard Agroclimático Bolívar 2026",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# ESTILOS CSS PERSONALIZADOS (Con contraste mejorado en Sidebar)
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
    .kpi-delta-up { color: #10B981; }
    .kpi-delta-down { color: #EF4444; }
    
    /* Sidebar con mejor contraste */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] a {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #00F2FE !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .stSelectbox > div,
    [data-testid="stSidebar"] .stTextInput > div {
        background-color: #334155 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] ul li {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] a {
        color: #00F2FE !important;
    }
    
    .prediction-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 20px 0;
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
# FUNCIONES DE CARGA DE DATOS Y MODELOS
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
        
        return df_diario, df_test, df_anual, df_irf, metrics, geojson
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        st.stop()

# Cargar datos
df_diario, df_test, df_anual, df_irf, metrics_summary, bolivar_geojson = load_data()

@st.cache_resource
def train_simulator_models():
    try:
        features = ["Temp_Media_Lag1", "Temp_Media_Lag2", "Precip_Acum_Lag1", "Precip_Acum_Lag2", "Hum_Media"]
        missing_cols = [col for col in features if col not in df_diario.columns]
        if missing_cols:
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
# FUNCIÓN DE PREDICCIÓN PARA 2026 (SIN CACHÉ)
# ---------------------------------------------------------
def generate_2026_forecast(model, days=365):
    if model is None:
        return None
    
    start_date = datetime(2026, 1, 1)
    last_known = df_diario.tail(10).copy()
    
    predictions = []
    current_lags = {
        'Temp_Media_Lag1': last_known['Temp_Media'].iloc[-1],
        'Temp_Media_Lag2': last_known['Temp_Media'].iloc[-2],
        'Precip_Acum_Lag1': last_known['Precip_Acum'].iloc[-1],
        'Precip_Acum_Lag2': last_known['Precip_Acum'].iloc[-2],
        'Hum_Media': last_known['Hum_Media'].iloc[-1]
    }
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        features = pd.DataFrame([{
            'Temp_Media_Lag1': current_lags['Temp_Media_Lag1'],
            'Temp_Media_Lag2': current_lags['Temp_Media_Lag2'],
            'Precip_Acum_Lag1': current_lags['Precip_Acum_Lag1'],
            'Precip_Acum_Lag2': current_lags['Precip_Acum_Lag2'],
            'Hum_Media': current_lags['Hum_Media']
        }])
        
        temp_pred = model.predict(features)[0]
        month = current_date.month
        rain_probs = {1: 0.6, 2: 0.7, 3: 0.8, 4: 0.5, 5: 0.3, 6: 0.2, 7: 0.1, 8: 0.1, 9: 0.2, 10: 0.4, 11: 0.5, 12: 0.6}
        
        if np.random.random() < rain_probs[month]:
            precip_pred = np.random.exponential(5.0)
        else:
            precip_pred = 0.0
            
        hum_pred = 80 - (temp_pred - 13) * 2 + (precip_pred * 0.5)
        hum_pred = np.clip(hum_pred, 30, 100)
        
        predictions.append({
            'Fecha': current_date, 'Temp_Pred': temp_pred,
            'Precip_Pred': precip_pred, 'Hum_Pred': hum_pred,
            'GDD': max(temp_pred - 10, 0)
        })
        
        current_lags['Temp_Media_Lag2'] = current_lags['Temp_Media_Lag1']
        current_lags['Temp_Media_Lag1'] = temp_pred
        current_lags['Precip_Acum_Lag2'] = current_lags['Precip_Acum_Lag1']
        current_lags['Precip_Acum_Lag1'] = precip_pred
        current_lags['Hum_Media'] = hum_pred
        
    return pd.DataFrame(predictions)

# ---------------------------------------------------------
# INICIALIZACIÓN DE PRONÓSTICO 2026 USANDO SESSION_STATE
# ---------------------------------------------------------
# Esto evita el UnhashableParamError y el NameError
if 'forecast_2026' not in st.session_state:
    with st.spinner("🔄 Generando predicciones para el año 2026..."):
        st.session_state['forecast_2026'] = generate_2026_forecast(sim_model_xgb, days=365)

forecast_2026 = st.session_state['forecast_2026']

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown("### ️")
st.sidebar.markdown("<h2 style='color: #00F2FE; font-weight: 700; margin-bottom: 0px;'>Pronóstico Bolívar 2026</h2>", unsafe_allow_html=True)
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
st.sidebar.subheader(" Estación Base")
st.sidebar.markdown("""
- **Nombre:** Guaranda UEB
- **Ubicación:** Cantón Guaranda
- **Altitud:** 2,668 m s.n.m.
- **Rango de Datos:** 2016 - 2023
- **Predicción:** 2026
""")
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color: #64748B; font-size: 0.8rem;'>Diseñado por: Deysi Guanga<br>Experto en Forecasting </p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# CUERPO PRINCIPAL
# ---------------------------------------------------------
st.markdown("<h1 class='title-gradient'>Monitoreo y Pronóstico Climático de Bolívar, Ecuador</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Modelado predictivo del clima con Machine Learning y Deep Learning - Predicciones 2026</p>", unsafe_allow_html=True)

tabs = st.tabs([
    "🔮 Pronóstico 2026",
    "🗺️ Mapa y Resumen del Clima", 
    " Comparativa de Modelos (Forecasting)", 
    "🌱 Impacto Agroproductivo (VAR)", 
    "⚙️ Simulador Predictivo"
])

# ==============================================================================
# PESTAÑA 0: PRONÓSTICO 2026
# ==============================================================================
with tabs[0]:
    st.markdown("### 🔮 Pronóstico Climático para el Año 2026")
    
    if forecast_2026 is None:
        st.error("❌ No se pudo generar el pronóstico para 2026.")
    else:
        st.markdown("#### 📊 Resumen Anual 2026")
        summary_cols = st.columns(4)
        with summary_cols[0]: st.metric("Temperatura Promedio", f"{forecast_2026['Temp_Pred'].mean():.1f} °C")
        with summary_cols[1]: st.metric("Precipitación Total", f"{forecast_2026['Precip_Pred'].sum():.0f} mm")
        with summary_cols[2]: st.metric("Humedad Promedio", f"{forecast_2026['Hum_Pred'].mean():.1f} %")
        with summary_cols[3]: st.metric("GDD Acumulado", f"{forecast_2026['GDD'].sum():.0f}")
        
        st.markdown("---")
        st.markdown("#### 📈 Evolución Mensual de Variables Climáticas")
        
        forecast_2026['Mes'] = forecast_2026['Fecha'].dt.month
        monthly_stats = forecast_2026.groupby('Mes').agg({
            'Temp_Pred': 'mean', 'Precip_Pred': 'sum', 'Hum_Pred': 'mean', 'GDD': 'sum'
        }).reset_index()
        
        month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        monthly_stats['Mes_Nombre'] = [month_names[i-1] for i in monthly_stats['Mes']]
        
        fig_2026 = go.Figure()
        fig_2026.add_trace(go.Bar(x=monthly_stats['Mes_Nombre'], y=monthly_stats['Precip_Pred'], name='Precipitación (mm)', marker_color='#00F2FE', yaxis='y2', opacity=0.6))
        fig_2026.add_trace(go.Scatter(x=monthly_stats['Mes_Nombre'], y=monthly_stats['Temp_Pred'], name='Temperatura (°C)', mode='lines+markers', line=dict(color='#EF4444', width=3), marker=dict(size=10), yaxis='y1'))
        
        fig_2026.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Mes", gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title=dict(text="Temperatura (°C)", font=dict(color="#EF4444")), tickfont=dict(color="#EF4444"), gridcolor='rgba(255,255,255,0.05)'),
            yaxis2=dict(title=dict(text="Precipitación (mm)", font=dict(color="#00F2FE")), tickfont=dict(color="#00F2FE"), overlaying='y1', side='right', gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500
        )
        st.plotly_chart(fig_2026, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 🗺️ Predicciones por Cantón - Año 2026")
        
        canton_forecast = []
        avg_temp_2026 = forecast_2026['Temp_Pred'].mean()
        total_rain_2026 = forecast_2026['Precip_Pred'].sum()
        
        for c_name, c_data in cantones_info.items():
            canton_forecast.append({
                "Cantón": c_name,
                "Temp_Promedio_2026 (°C)": round(avg_temp_2026 + c_data['lapse_relative'], 2),
                "Precip_Total_2026 (mm)": round(total_rain_2026 * c_data['rain_factor'], 0),
                "Altitud (m)": c_data['alt'], "Microclima": c_data['tipo'], "Cultivo Principal": c_data['cultivo_principal']
            })
        df_canton_2026 = pd.DataFrame(canton_forecast)
        
        map_cols = st.columns([2, 1])
        with map_cols[0]:
            map_var = st.radio("Variable a visualizar:", ["Temp_Promedio_2026 (°C)", "Precip_Total_2026 (mm)"], horizontal=True)
            color_scale = "RdYlBu_r" if "Temp" in map_var else "Blues"
            
            fig_map_2026 = px.choropleth_mapbox(
                df_canton_2026, geojson=bolivar_geojson, locations="Cantón", featureidkey="properties.name",
                color=map_var, color_continuous_scale=color_scale, mapbox_style="carto-positron",
                zoom=8.8, center={"lat": -1.60, "lon": -79.12}, opacity=0.75, hover_name="Cantón",
                hover_data={"Altitud (m)": True, "Microclima": True, "Cultivo Principal": True, "Cantón": False}, height=500
            )
            fig_map_2026.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_map_2026, use_container_width=True)
            
        with map_cols[1]:
            st.markdown("##### 📊 Ranking por Temperatura")
            df_sorted = df_canton_2026.sort_values("Temp_Promedio_2026 (°C)", ascending=False)
            fig_bar_2026 = px.bar(df_sorted, x="Temp_Promedio_2026 (°C)", y="Cantón", orientation='h', color="Temp_Promedio_2026 (°C)", color_continuous_scale="RdYlBu_r", text="Temp_Promedio_2026 (°C)", height=400)
            fig_bar_2026.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, yaxis_title=None)
            st.plotly_chart(fig_bar_2026, use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### 🌾 Estimación de Impacto Agrícola 2026")
        
        avg_temp_year = forecast_2026['Temp_Pred'].mean()
        total_rain_year = forecast_2026['Precip_Pred'].sum()
        avg_hum_year = forecast_2026['Hum_Pred'].mean()
        
        rend_maiz_2026 = 3.4 + 0.05 * (avg_temp_year - 14.2) + 0.002 * (total_rain_year - 1000)
        rend_cacao_2026 = 0.46 + 0.005 * (avg_hum_year - 80) + 0.0001 * (total_rain_year - 1000)
        rend_arroz_2026 = 4.2 + 0.06 * (avg_temp_year - 14.2) + 0.003 * (total_rain_year - 1000)
        
        crop_cols = st.columns(3)
        with crop_cols[0]: st.metric("🌽 Maíz (Sierra)", f"{max(rend_maiz_2026, 1.5):.2f} Ton/Ha", delta=f"{rend_maiz_2026 - 3.4:.2f} vs Prom")
        with crop_cols[1]: st.metric("🍫 Cacao (Subtrópico)", f"{max(rend_cacao_2026, 0.15):.2f} Ton/Ha", delta=f"{rend_cacao_2026 - 0.46:.2f} vs Prom")
        with crop_cols[2]: st.metric("🌾 Arroz (Llanura)", f"{max(rend_arroz_2026, 2.0):.2f} Ton/Ha", delta=f"{rend_arroz_2026 - 4.2:.2f} vs Prom")
        
        st.markdown("---")
        st.markdown("####  Descargar Predicciones 2026")
        csv = forecast_2026.to_csv(index=False).encode('utf-8')
        st.download_button(label="📊 Descargar Predicciones Diarias (CSV)", data=csv, file_name="predicciones_climaticas_bolivar_2026.csv", mime="text/csv")

# ==============================================================================
# PESTAÑA 1: MAPA Y RESUMEN DEL CLIMA (HISTÓRICO)
# ==============================================================================
with tabs[1]:
    st.markdown("### 📅 Comportamiento Diario del Clima (Test 2023)")
    
    date_column = None
    for col in ['Fecha_Dia', 'fecha_dia', 'Fecha', 'fecha', 'Date', 'date']:
        if col in df_test.columns:
            date_column = col
            break
            
    if date_column:
        min_date = df_test[date_column].min().to_pydatetime()
        max_date = df_test[date_column].max().to_pydatetime()
        selected_date = st.slider("Línea de Tiempo (Fecha)", min_value=min_date, max_value=max_date, value=min_date, format="YYYY-MM-DD")
        day_row = df_test[df_test[date_column] == selected_date].iloc[0]
        
        kpi_cols = st.columns(4)
        temp_real = day_row.get('Temp_Real', 0)
        temp_pred = day_row.get(model_col, 0)
        
        with kpi_cols[0]:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Temp. Media (Guaranda)</div><div class="kpi-value">{temp_real:.1f} °C</div><div class="kpi-delta">Pronóstico: <b>{temp_pred:.1f} °C</b></div></div>""", unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Humedad Relativa</div><div class="kpi-value">{day_row.get('Hum_Media', 0):.1f} %</div></div>""", unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Lluvia Diaria</div><div class="kpi-value">{day_row.get('Precip_Acum', 0):.2f} mm</div></div>""", unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-title">GDD Diario</div><div class="kpi-value">{max(temp_real - 10, 0):.2f} GDD</div></div>""", unsafe_allow_html=True)
            
        # Mapa histórico
        map_cols = st.columns([3, 2])
        cantones_list = []
        for c_name, c_data in cantones_info.items():
            cantones_list.append({
                "Cantón": c_name,
                "Temperatura_Est (°C)": round(temp_pred + c_data['lapse_relative'], 2),
                "Precipitación_Est (mm)": round(day_row.get('Precip_Acum', 0) * c_data['rain_factor'], 2),
                "Altitud (m)": c_data['alt']
            })
        df_cantones = pd.DataFrame(cantones_list)
        
        with map_cols[0]:
            map_var = st.radio("Variable a visualizar:", ["Temperatura_Est (°C)", "Precipitación_Est (mm)"], horizontal=True)
            color_scale = "RdYlBu_r" if "Temperatura" in map_var else "Blues"
            fig_map = px.choropleth_mapbox(df_cantones, geojson=bolivar_geojson, locations="Cantón", featureidkey="properties.name", color=map_var, color_continuous_scale=color_scale, mapbox_style="carto-positron", zoom=8.8, center={"lat": -1.60, "lon": -79.12}, opacity=0.75, height=500)
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_map, use_container_width=True)
            
        with map_cols[1]:
            fig_bar = px.bar(df_cantones.sort_values("Altitud (m)", ascending=False), x="Temperatura_Est (°C)", y="Cantón", orientation='h', color="Temperatura_Est (°C)", color_continuous_scale="RdYlBu_r", height=450)
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title=None, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

# ==============================================================================
# PESTAÑA 2: COMPARATIVA DE MODELOS
# ==============================================================================
with tabs[2]:
    st.markdown("### 📈 Evaluación Histórica de Modelos Predictivos")
    metrics_cols = st.columns([1, 2])
    
    with metrics_cols[0]:
        st.markdown("#### 📋 Tabla General de Métricas")
        df_metrics = pd.DataFrame(metrics_summary).T
        df_metrics.columns = ["RMSE (°C)", "MAE (°C)", "MAPE (%)", "R² (Coef. Det.)"]
        st.dataframe(df_metrics.style.format({"RMSE (°C)": "{:.4f}", "MAE (°C)": "{:.4f}", "MAPE (%)": "{:.2f}%", "R² (Coef. Det.)": "{:.4f}"}), use_container_width=True)
        
    with metrics_cols[1]:
        st.markdown("#### 🏆 Comparación Visual de Métricas")
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(x=df_metrics.index, y=df_metrics["RMSE (°C)"], name="RMSE (°C)", marker_color="#EF4444", yaxis="y1"))
        fig_metrics.add_trace(go.Bar(x=df_metrics.index, y=df_metrics["R² (Coef. Det.)"], name="R² Score", marker_color="#00F2FE", yaxis="y2"))
        
        fig_metrics.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Modelo", gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title=dict(text="RMSE (Menor es mejor)", font=dict(color="#EF4444")), tickfont=dict(color="#EF4444"), gridcolor='rgba(255,255,255,0.05)'),
            yaxis2=dict(title=dict(text="R² (Cercano a 1 es mejor)", font=dict(color="#00F2FE")), tickfont=dict(color="#00F2FE"), overlaying="y1", side="right", range=[-1, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=400
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

# ==============================================================================
# PESTAÑA 3: IMPACTO AGROPRODUCTIVO
# ==============================================================================
with tabs[3]:
    st.markdown("### 🌾 Interacción Clima-Cultivos y Modelo VAR")
    st.markdown("#### ⚡ Funciones de Impulso-Respuesta")
    
    fig_irf = go.Figure()
    fig_irf.add_trace(go.Scatter(x=df_irf['Periodo'], y=df_irf['Impulso_Temp'], mode='lines+markers', name='Respuesta a Temperatura', line=dict(color='#EF4444', width=2)))
    fig_irf.add_trace(go.Scatter(x=df_irf['Periodo'], y=df_irf['Impulso_Lluvia'], mode='lines+markers', name='Respuesta a Precipitación', line=dict(color='#00F2FE', width=2)))
    
    fig_irf.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Periodo"), yaxis=dict(title="Variación en Rendimiento (Ton/Ha)")
    )
    st.plotly_chart(fig_irf, use_container_width=True)

# ==============================================================================
# PESTAÑA 4: SIMULADOR PREDICTIVO
# ==============================================================================
with tabs[4]:
    st.markdown("### ⚙️ Calculadora Agroclimática en Tiempo Real")
    
    if sim_model_rf is None or sim_model_xgb is None:
        st.error("Los modelos no están disponibles.")
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
            st.markdown("####  Predicciones")
            features_input = pd.DataFrame([{
                "Temp_Media_Lag1": temp_lag1, "Temp_Media_Lag2": temp_lag2,
                "Precip_Acum_Lag1": rain_lag1, "Precip_Acum_Lag2": rain_lag2, "Hum_Media": hum_actual
            }])
            
            if sim_model_type == "XGBoost":
                pred_temp_base = sim_model_xgb.predict(features_input)[0]
            else:
                pred_temp_base = sim_model_rf.predict(features_input)[0]
                
            st.metric("Temperatura Predicha", f"{pred_temp_base:.2f} °C")
