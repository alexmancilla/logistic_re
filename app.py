import numpy as np
import pandas as pd
import streamlit as st

# Widget de carga de archivos
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    # Leer los datos
    titanic_df = pd.read_csv(archivo)

    # Convertir la variable 'Sex' en variables ficticias (dummy variables)
    dummies_sexo = pd.get_dummies(titanic_df['Sex'], prefix='Sexo')
    titanic_df = pd.concat([titanic_df, dummies_sexo], axis=1)

    # Seleccionar características y eliminar valores faltantes
    caracteristicas_seleccionadas = ['Edad', 'Sexo_female', 'Sexo_male', 'Clase', 'Tarifa']
    titanic_df = titanic_df[caracteristicas_seleccionadas + ['Sobrevivio']].dropna()

    # Definir la función sigmoide
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Definir la función de costo
    def funcion_de_costo(X, y, pesos):
        m = len(y)
        h = sigmoid(X.dot(pesos))
        costo = -(1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
        gradiente = (1/m) * X.T.dot(h-y)
        return costo, gradiente

    # Definir la función de entrenamiento
    def entrenar(X, y, tasa_de_aprendizaje, num_iteraciones):
        m, n = X.shape
        pesos = np.zeros(n)
        for i in range(num_iteraciones):
            costo, gradiente = funcion_de_costo(X, y, pesos)
            pesos = pesos - tasa_de_aprendizaje * gradiente
            if i % 1000 == 0:
                print(f'Costo después de la iteración {i}: {costo}')
        return pesos

    # Preparar los datos
    X = titanic_df[caracteristicas_seleccionadas]
    y = titanic_df['Sobrevivio']
    X = np.hstack((np.ones((len(X), 1)), X))
    pesos = entrenar(X, y, 0.01, 10000)

    # Definir la función de predicción
    def predecir_supervivencia(edad, sexo, clase, tarifa):
        es_hombre = 0
        es_mujer = 0
        if sexo == 'male':
            es_hombre = 1
        elif sexo == 'female':
            es_mujer = 1
        data = np.array([[1, edad, es_mujer, es_hombre, clase, tarifa]])
        prediccion = np.round(sigmoid(data.dot(pesos)))
        if prediccion[0] == 1:
            return "¡Este pasajero habría sobrevivido!"
        else:
            return "Este pasajero no habría sobrevivido."

    # Streamlit app
    st.title("Predicción de supervivencia en el Titanic")
    edad = st.slider("Edad:", min_value=0, max_value=100, value=30, step=1)
    sexo = st.radio("Sexo:", options=['male', 'female'], index=

