# ⛈️ Dashboard de Pronóstico Climático e Impacto Agroproductivo (Bolívar, Ecuador)

Este repositorio contiene la solución completa para la implementación de un **Dashboard Interactivo y Predictivo** que analiza el comportamiento del clima en la provincia de Bolívar, Ecuador, y cuantifica su impacto sobre los rendimientos de cultivos agrícolas clave (Maíz, Cacao y Arroz). 

Este desarrollo ha sido estructurado y diseñado bajo estándares profesionales y científicos como **trabajo final de la Especialización en Forecasting**.

---

## 📁 Estructura del Proyecto

El proyecto está organizado en los siguientes archivos clave:

1.  **`app.py`**: Código principal del dashboard desarrollado en Streamlit. Contiene la interfaz interactiva, los gráficos dinámicos de Plotly, el mapa de calor de correlaciones y el simulador de escenarios agroclimáticos en tiempo real.
2.  **`get_geojson.py`**: Script de automatización que descarga los límites cantonales de Ecuador, filtra exclusivamente los 7 cantones de la provincia de Bolívar (código político 02 del INEC), limpia sus nombres y los guarda en el archivo local ligero `bolivar_cantones.geojson`.
3.  **`prepare_data.py`**: Script que realiza el preprocesamiento de los datos climáticos diarios, limpia outliers por el método IQR, imputa datos vacíos usando KNN Imputer, entrena los modelos predictivos (SARIMA, Random Forest, XGBoost y LSTM) y exporta los archivos optimizados (menos de 5 MB) para que el servidor gratuito del dashboard responda de forma instantánea.
4.  **`requirements.txt`**: Archivo de configuración con las dependencias y librerías de Python requeridas para ejecutar y desplegar la aplicación en servidores gratuitos.
5.  **`.streamlit/config.toml`**: Archivo que define el estilo visual de la interfaz web, configurando un tema estético oscuro (Dark Mode) con contrastes modernos de azul pizarra y cian.

---

## 🛠️ Paso 1: Instalación y Configuración Local

Siga estos pasos detallados para configurar la aplicación en su máquina:

### 1. Clonar o acceder a la carpeta del proyecto
Abra su terminal (en macOS) o consola de comandos y navegue a la carpeta del proyecto:
```bash
cd "/Users/deysi/Documents/Forecasting"
```

### 2. Crear un entorno virtual de Python (Recomendado)
Para mantener las dependencias aisladas y evitar conflictos:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias necesarias
Instale las librerías requeridas utilizando el archivo `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Paso 2: Preparación del Entorno y Datos

Para generar los datos procesados y el mapa local que alimentan la interfaz, ejecute los siguientes dos comandos:

### 1. Descargar y estructurar el GeoJSON de Bolívar
```bash
python3 get_geojson.py
```
*Este comando generará el archivo `bolivar_cantones.geojson` con los límites de los 7 cantones (Guaranda, Chimbo, San Miguel, Chillanes, Caluma, Echeandía y Las Naves).*

### 2. Ejecutar la limpieza y entrenamiento de modelos
```bash
python3 prepare_data.py
```
*Este comando leerá la base de datos cruda `2016-2023.txt`, aplicará la ingeniería de características (lags, GDD), ajustará los modelos predictivos y exportará los resultados en CSV y JSON. Una vez ejecutado, no requerirá volver a procesar el archivo pesado de 235 MB.*

---

## 🚀 Paso 3: Ejecución Local de la Aplicación

Para abrir la interfaz web interactiva en su navegador local, ejecute:
```bash
streamlit run app.py
```
Streamlit abrirá automáticamente la aplicación en la dirección local: `http://localhost:8501`.

---

## ☁️ Paso 4: Despliegue Público Gratuito

Una de las ventajas del diseño ligero de este dashboard es que puede subirse de forma gratuita a la nube. A continuación se presentan las dos mejores opciones para su presentación final:

### Opción A: Streamlit Community Cloud (Recomendada)
Es la plataforma gratuita oficial de Streamlit para desplegar aplicaciones directamente desde un repositorio de GitHub.

1.  **Subir el código a GitHub:**
    *   Cree un repositorio público en GitHub (por ejemplo, `bolivar-climate-dashboard`).
    *   Suba los siguientes archivos a la raíz del repositorio:
        *   `app.py`
        *   `requirements.txt`
        *   `bolivar_cantones.geojson`
        *   `predicciones_test.csv`
        *   `metricas_modelos.json`
        *   `datos_anuales_var.csv`
        *   `var_irf_results.csv`
        *   `datos_preprocesados_diarios_py.csv`
        *   `.streamlit/config.toml`
    *   *Nota: No suba el archivo crudo de 235 MB `2016-2023.txt` a GitHub.*
2.  **Desplegar en Streamlit Cloud:**
    *   Ingrese a [share.streamlit.io](https://share.streamlit.io/) e inicie sesión con su cuenta de GitHub.
    *   Haga clic en **"New App"**.
    *   Seleccione su repositorio, la rama (`main` o `master`) y escriba `app.py` en el campo "Main file path".
    *   Haga clic en **"Deploy"**.
    *   ¡Listo! En menos de 2 minutos su aplicación estará disponible en una URL pública compartible (ejemplo: `https://bolivar-climate-dashboard.streamlit.app`).

### Opción B: Hugging Face Spaces (Alternativa Estética)
Hugging Face ofrece hosting gratuito para dashboards de Python con excelente rendimiento.

1.  Cree una cuenta en [Hugging Face](https://huggingface.co/).
2.  Haga clic en su perfil en la esquina superior derecha y seleccione **"New Space"**.
3.  Asigne un nombre a su Space, seleccione **Streamlit** como la SDK, y defina la licencia (por ejemplo, MIT). Seleccione el plan gratuito "CPU Basic".
4.  Clone el repositorio Git del Space en su máquina local o suba directamente los archivos a través del navegador.
5.  Hugging Face detectará automáticamente el archivo `app.py` y levantará la aplicación web con una URL pública gratuita.

---

## 🧪 Rigor Científico del Dashboard (Para su Defensa)

Al defender este trabajo final de Forecasting, asegúrese de destacar los siguientes tres pilares metodológicos que dotan de rigor científico a su dashboard:

1.  **Gradiente Altitudinal Relativo:** Dado que no existe una red densa de estaciones para cada cantón y los datos crudos provienen de la estación base en Guaranda, la visualización en el mapa calcula las temperaturas locales utilizando la tasa de variación ambiental húmeda de la atmósfera (\(-6.0^\circ\text{C}\) por cada 1,000 metros de altitud). Esto permite extrapolar con alta fidelidad las condiciones cálidas de Caluma o Las Naves en base a la predicción andina fría de Guaranda.
2.  **Comparación Multimodelo:** La pestaña de forecasting contrasta los modelos tradicionales autorregresivos (SARIMA) con técnicas no lineales de ensamble (Random Forest y XGBoost) y redes neuronales recurrentes para series temporales (LSTM). Esto demuestra un dominio profundo de los diferentes paradigmas predictivos.
3.  **Análisis Multivariante Dinámico (VAR):** La integración de un modelo de Vectores Autorregresivos (VAR(1)) permite modelar la retroalimentación y cuantificar el impacto dinámico a través de las Funciones de Impulso-Respuesta (IRF), mostrando que los choques de precipitación de un año impactan de forma acumulativa el rendimiento agrícola (Maíz) en los años subsiguientes.
