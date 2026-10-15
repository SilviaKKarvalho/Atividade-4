import streamlit as st
import tensorflow as tf
import numpy as np



st.title("Previsor de Horas de Estudo para o ENEM 🎯")
st.write("Um modelo simples com TensorFlow para estimar o esforço necessário.")


# 1. Dados e Normalização (Dividimos por 1000 para estabilizar o treinamento)
notas = np.array([500.0, 600.0, 700.0, 800.0, 900.0, 950.0], dtype=float) / 1000.0
horas_estudo = np.array([5.0, 10.0, 18.0, 25.0, 35.0, 42.0], dtype=float)


# 2. Criando e Treinando o Modelo (com cache para não re-treinar a cada clique)
@st.cache_resource
def carregar_modelo():
    modelo = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    modelo.fit(notas, horas_estudo, epochs=1000, verbose=0)
    return modelo


modelo = carregar_modelo()


# 3. Interface Interativa
st.subheader("Simule sua Meta")
nota_desejada = st.slider("Qual nota você quer tirar no ENEM?", 500.0, 1000.0, 850.0, 10.0)


# Normalizando a entrada do usuário da mesma forma que no treino
nota_normalizada = np.array([nota_desejada / 1000.0])
horas_previstas = modelo.predict(nota_normalizada)[0][0]


# Exibindo o resultado corrigido
st.success (horas_previstas)
