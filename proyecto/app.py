# app.py
import streamlit as st
import pickle
import pandas as pd
from sklearn.datasets import load_iris

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Iris",
    page_icon="🌼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar modelo y datos
@st.cache_resource
def load_model():
    try:
        with open('iris_model.pkl', 'rb') as file:
            model = pickle.load(file)
        iris = load_iris()
        return model, iris
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None

model, iris = load_model()

# Título principal
st.title("🌼 Clasificador de Flores Iris")
st.markdown("""
Esta aplicación predice la especie de una flor Iris basándose en sus medidas.
Utiliza un modelo de Machine Learning (Random Forest) entrenado con el dataset clásico de Iris.
""")

# Sidebar con inputs
st.sidebar.header("📏 Medidas de la Flor")
st.sidebar.markdown("Ajusta las medidas usando los sliders:")

def get_user_input():
    sepal_length = st.sidebar.slider(
        'Longitud del sépalo (cm)',
        min_value=4.0, max_value=8.0, value=5.4, step=0.1
    )
    sepal_width = st.sidebar.slider(
        'Ancho del sépalo (cm)',
        min_value=2.0, max_value=4.5, value=3.4, step=0.1
    )
    petal_length = st.sidebar.slider(
        'Longitud del pétalo (cm)',
        min_value=1.0, max_value=7.0, value=1.3, step=0.1
    )
    petal_width = st.sidebar.slider(
        'Ancho del pétalo (cm)',
        min_value=0.1, max_value=2.5, value=0.2, step=0.1
    )
    
    return {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

user_input = get_user_input()

# Mostrar inputs del usuario
st.subheader("📋 Medidas Ingresadas")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Longitud Sépalo", f"{user_input['sepal_length']} cm")
with col2:
    st.metric("Ancho Sépalo", f"{user_input['sepal_width']} cm")
with col3:
    st.metric("Longitud Pétalo", f"{user_input['petal_length']} cm")
with col4:
    st.metric("Ancho Pétalo", f"{user_input['petal_width']} cm")

# Botón de predicción
if st.sidebar.button("🎯 Predecir Especie", type="primary"):
    if model is None or iris is None:
        st.error("El modelo no está disponible. Por favor, verifica que el archivo del modelo existe.")
    else:
        try:
            # Convertir a DataFrame para la predicción
            input_df = pd.DataFrame([user_input])
            
            # Hacer predicción
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Mostrar resultados
            st.subheader("📊 Resultados de la Predicción")
            
            # Tarjeta con la especie predicha
            species = iris.target_names[prediction][0]
            st.success(f"### Especie Predicha: **{species}**")
            
            # Mostrar probabilidades
            st.subheader("📈 Probabilidades por Especie")
            
            proba_df = pd.DataFrame({
                'Especie': iris.target_names,
                'Probabilidad': prediction_proba[0]
            })
            
            # Gráfico de barras
            st.bar_chart(proba_df.set_index('Especie'))
            
            # Tabla de probabilidades
            st.dataframe(
                proba_df.style.format({'Probabilidad': '{:.2%}'}),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

# Información adicional
with st.expander("ℹ️ Acerca de esta aplicación"):
    st.markdown("""
    **Características técnicas:**
    - **Dataset:** Iris de sklearn (150 muestras)
    - **Modelo:** Random Forest Classifier (100 árboles)
    - **Precisión:** 96.67% en conjunto de prueba
    - **Framework:** Streamlit
    - **Despliegue:** Render.com
    
    **Especies de Iris:**
    - **0: Setosa** - Flores pequeñas, sépalos grandes
    - **1: Versicolor** - Flores medianas
    - **2: Virginica** - Flores grandes, pétalos anchos
    """)

# Footer
st.markdown("---")
st.caption("Desplegado usando Streamlit y Render | Modelo de Machine Learning")