import streamlit as st
import pickle
import pandas as pd

# Cargar los modelos entrenados
with open('regresión_logística.pkl', 'rb') as file:
    log_reg = pickle.load(file)

with open('árboles_de_decisión.pkl', 'rb') as file:
    arbol = pickle.load(file)

with open('random_forest.pkl', 'rb') as file:
    rf = pickle.load(file)

with open('xgboost.pkl', 'rb') as file:
    xgb = pickle.load(file)

# %%
def classify(pred):
    return 'Hará un depósito' if pred == 1 else 'No hará un depósito'

# %%
def main():
    st.title('Predicción de Depósito Bancario')
    st.sidebar.header('Parámetros del Usuario')

    def user_input():
        age = st.sidebar.slider('Edad', 18, 95, 30)   ## Listo
        job = st.sidebar.selectbox('Trabajo', ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'student', 'self-employed', 'unknown'])       # Listo
        marital = st.sidebar.selectbox('Estado civil', ['married', 'single', 'divorced'])           # Listo
        education = st.sidebar.selectbox('Educación', ['unknown', 'primary', 'secondary', 'tertiary'])  # Listo
        default = st.sidebar.selectbox('¿Tiene default?', ['yes', 'no'])                            # Listo
        balance = st.sidebar.number_input('Balance', -6847, 82204, 1000)                            # Listo
        housing = st.sidebar.selectbox('¿Tiene préstamo de vivienda?', ['yes', 'no'])               # Listo
        loan = st.sidebar.selectbox('¿Tiene otro préstamo?', ['yes', 'no'])                         # Listo
        contact = st.sidebar.selectbox('Medio de contacto', ['cellular', 'telephone','unknown'])    # Listo
        day = st.sidebar.selectbox('Día del mes', list(range(1, 32)))                               # Listo
        month = st.sidebar.selectbox('Mes de contacto', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])       # Listo
        duration = st.sidebar.slider('Duración de contacto (segundos)', 2, 3881, 200)               # Listo
        campaign = st.sidebar.slider('Número de contactos en esta campaña', 1, 63, 20)               # Listo
        pdays = st.sidebar.slider('Días desde la última campaña (-1 si no se contactó)', -1, 854, 5)       # Listo
        previous = st.sidebar.slider('Número de contactos anteriores', 0, 58, 10)                    # Listo
        poutcome = st.sidebar.selectbox('Resultado campaña anterior', ['unknown', 'failure', 'other', 'success'])       # Listo

        data = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }

        return pd.DataFrame(data, index=[0])

    input_df = user_input()

    st.subheader('Datos ingresados:')
    st.write(input_df)

    modelo_seleccionado = st.sidebar.selectbox('Elige el modelo', ['Regresión Logística', 'Árboles de Decisión', 'Random Forest', 'XGBoost'])

    if st.button('Predecir'):
        if modelo_seleccionado == 'Regresión Logística':
            pred = log_reg.predict(input_df)
        elif modelo_seleccionado == 'Árboles de Decisión':
            pred = arbol.predict(input_df)
        elif modelo_seleccionado == 'Random Forest':
            pred = rf.predict(input_df)
        else:
            pred = xgb.predict(input_df)

        st.success(f'Resultado: {classify(pred[0])}')

if __name__ == '__main__':
    main()
