import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Configuración de la página
st.set_page_config(page_title="Análisis de Reservas de Hotel", layout="wide")

# Cargar y explorar el dataset
@st.cache(allow_output_mutation=True)
def load_and_explore_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    return df

# Preprocesamiento de datos
def preprocess_data(df):
    df['booking_status_numeric'] = df['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)
    return df.drop(['Booking_ID', 'booking_status'], axis=1), df['booking_status_numeric']

# Visualización de la matriz de correlación solo para variables numéricas
def plot_correlation_matrix(df):
    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Heatmap de Correlación')
    st.pyplot(plt)


# Preparación de los datos para el modelo
def prepare_data_for_model(X, y):
    categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    numeric_features = ['lead_time', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'no_of_special_requests', 'required_car_parking_space', 'repeated_guest']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_preprocessed = preprocessor.fit_transform(X)
    
    return train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Entrenamiento y evaluación de modelos
def train_and_evaluate_models(X_train, X_test, y_train, y_test, exclude_svm=False):
    models = [
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (DecisionTreeClassifier(), "Decision Tree"),
        (RandomForestClassifier(), "Random Forest"),
        (GradientBoostingClassifier(), "Gradient Boosting"),
        (SVC(), "Support Vector Machine")
    ]
    
    if exclude_svm:
        models = models[:-1]  # Excluir SVM
    
    results = []
    for model, name in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results.append((name, accuracy, report))
    
    return results

def interactive_correlation_matrix(df):
    st.subheader("Mapa de Calor de Correlación Interactivo")
    all_columns = df.columns.tolist()
    selected_vars = st.multiselect("Selecciona variables para correlacionar", options=all_columns, default=all_columns[:5])
    
    # Filtrar solo las columnas numéricas de las seleccionadas
    numeric_vars = df[selected_vars]._get_numeric_data().columns.tolist()
    
    if numeric_vars:  # Asegurarse de que hay variables numéricas seleccionadas
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_vars].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Por favor, selecciona al menos una variable numérica.")



def predict_cancellation(model, preprocessor, input_data):
    # Preprocesar los datos de entrada con el preprocesador ya ajustado
    processed_input = preprocessor.transform(input_data)
    # Realizar la predicción
    prediction = model.predict(processed_input)
    return prediction

def prediction_form(model, preprocessor):
    st.subheader("Predicción de Cancelación de Reserva")
    # Ejemplo de campos para el formulario de predicción
    lead_time = st.number_input("Lead Time", value=10)
    adults = st.number_input("Número de Adultos", value=2)
    children = st.number_input("Número de Niños", value=0)
    # Añadir más campos según sea necesario

    if st.button("Predecir Cancelación"):
        input_data = pd.DataFrame([[lead_time, adults, children]], columns=["lead_time", "adults", "children"])
        prediction = predict_cancellation(model, preprocessor, input_data)
        if prediction == 1:
            st.error("La reserva es probable que sea cancelada.")
        else:
            st.success("La reserva tiene alta probabilidad de ser confirmada.")


def display_insights():
    st.subheader("Principales Hallazgos y Recomendaciones")
    st.write("""
    - **Hallazgo 1:** Las reservas realizadas con más de 60 días de antelación tienen una mayor tasa de cancelación.
    - **Recomendación 1:** Implementar políticas de depósito no reembolsable para reservas hechas con mucha antelación.
    
    - **Hallazgo 2:** Los clientes repetidos tienden a cancelar menos.
    - **Recomendación 2:** Crear un programa de fidelización para incentivar a los clientes a repetir estancias.
    """)

def prepare_data_for_model(X, y):
    categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    numeric_features = ['lead_time', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'no_of_special_requests', 'required_car_parking_space', 'repeated_guest']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_preprocessed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    
    # Asegúrate de retornar el preprocessor también aquí
    return X_train, X_test, y_train, y_test, preprocessor

def prediction_form(model, preprocessor):
    st.subheader("Predicción de Cancelación de Reserva")
    # Ejemplo de campos para el formulario de predicción
    lead_time = st.number_input("Lead Time", value=10)
    adults = st.number_input("Número de Adultos", value=2)
    children = st.number_input("Número de Niños", value=0)
    
    if st.button("Predecir Cancelación"):
        input_data = pd.DataFrame([[lead_time, adults, children]], columns=["lead_time", "adults", "children"])
        # Asegúrate de transformar los datos de entrada usando el preprocesador
        processed_input = preprocessor.transform(input_data)
        prediction = model.predict(processed_input)
        
        if prediction == 1:
            st.error("La reserva es probable que sea cancelada.")
        else:
            st.success("La reserva tiene alta probabilidad de ser confirmada.")



# Interfaz de usuario en Streamlit
def main():
    st.title("Análisis y Predicción de Reservas de Hotel")

    uploaded_file = st.file_uploader("Elige un archivo CSV para análisis", type="csv", key="unique_file_uploader_key")

    if uploaded_file is not None:
        df = load_and_explore_data(uploaded_file)
        if st.button("Mostrar datos"):
            st.write(df.head())
            st.write("Resumen estadístico:", df.describe())
            st.write("Valores faltantes:", df.isnull().sum())
        
        X, y = preprocess_data(df)
        
        # Asegurándonos de que prepare_data_for_model retorne también el preprocessor
        X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_model(X, y)
        
        if st.button("Mostrar Correlación"):
            plot_correlation_matrix(X)
        
        # Sección para entrenar y evaluar modelos
        if st.button("Entrenar y Evaluar Modelos"):
            exclude_svm = st.checkbox("Excluir SVM para reducir tiempo de ejecución", value=True)
            results = train_and_evaluate_models(X_train, X_test, y_train, y_test, exclude_svm)
            for result in results:
                st.write(f"Modelo: {result[0]}, Precisión: {result[1]:.4f}")
                st.text(f"Informe de Clasificación:\n{result[2]}")
        
        # Entrenar el modelo RandomForest específicamente para la función de predicción
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)  # Asegurarse de que X_train ya esté preprocesado si es necesario

        # Mostrar insights y correlaciones
        display_insights()
        interactive_correlation_matrix(df)

        # Formulario de predicción
        st.write("Realizar una predicción de cancelación de reserva")
        prediction_form(model, preprocessor)  # Pasamos el preprocessor correctamente

if __name__ == "__main__":
    main()

