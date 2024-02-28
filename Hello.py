import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# Configurar la página de Streamlit
st.title('Análisis y Predicción de Reservas de Hotel')

# Cargar datos
uploaded_file = st.file_uploader("Elige un archivo CSV para análisis", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Convertir 'booking_status' a numérico: 1 para "Canceled", 0 para "Not_Canceled"
    data['booking_status_numeric'] = data['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)

    # Mostrar información básica del DataFrame
    if st.checkbox('Mostrar descripción de los datos'):
        st.write(data.describe())

    # Análisis Exploratorio de Datos (EDA)
    if st.checkbox('Mostrar análisis exploratorio de datos'):
        st.subheader('Distribución del Estado de Reservas')
        fig, ax = plt.subplots()
        sns.countplot(x='booking_status', data=data, ax=ax)
        st.pyplot(fig)

        st.subheader('Heatmap de Correlación')
        # Necesitas definir 'numeric_features' basado en tu dataset
        numeric_features = st.multiselect('Selecciona características numéricas', options=data.columns, default=data.columns[0])
        correlation_matrix = data[numeric_features + ['booking_status_numeric']].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    # Preparación de Datos y Modelado
    if st.checkbox('Entrenar modelo de predicción'):
        st.subheader('Entrenamiento de Modelo de Bosque Aleatorio')

        # Dividir los datos
        X = data.drop(['booking_status', 'booking_status_numeric'], axis=1)  # Ajusta esta línea según sea necesario
        y = data['booking_status_numeric']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocesamiento
        # Debes definir numeric_features y categorical_features correctamente basado en tu dataset
        numeric_features = st.multiselect('Selecciona características numéricas para el modelo', options=X.columns, default=X.columns[0])
        categorical_features = st.multiselect('Selecciona características categóricas para el modelo', options=X.columns, default=X.columns[1])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

        # Entrenar el modelo
        rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        rand_forest.fit(preprocessor.fit_transform(X_train), y_train)

        # Predicciones y evaluación
        y_pred = rand_forest.predict(preprocessor.transform(X_test))
        accuracy = accuracy_score(y_test, y_pred)

        st.write(f'Precisión del modelo: {accuracy:.4f}')
        st.text(classification_report(y_test, y_pred))

# Instrucciones para ejecutar Streamlit
st.sidebar.subheader("Instrucciones para ejecutar:")
st.sidebar.write("1. Guarda este script como `hotel_booking_analysis.py`.")
st.sidebar.write("2. Abre tu terminal o cmd.")
st.sidebar.write("3. Navega al directorio donde guardaste el script.")
st.sidebar.write("4. Escribe `streamlit run hotel_booking_analysis.py` y presiona enter.")

