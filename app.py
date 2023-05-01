#Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv(r"C:\Users\cauai\Desktop\Projetos\crop_recomendation\Crop_recommendation.csv")

st.header("Teste de Algoritmos para um Sistema de recomendação de cultura")
texto = "O Objetivo desse projeto foi testar 2 algoritmos de classificação,onde baseado nos parâmetros nutrientes,temperatura,umidade, ph e água foi recomendado qual cultura plantar tomando como base a classificação feita pelos algoritmos de Árvore de Decisão e Random Forest."

st.write("<div style='text-align: justify;'>{}</div>".format(texto), unsafe_allow_html=True)
st.write("    ")

st.write('<p style="font-size:20px;">1 - Tabela com os Dados</p>', unsafe_allow_html=True)
st.dataframe(dataset)

data_corr = pd.DataFrame(dataset.corr(numeric_only=True))

st.write('<p style="font-size:20px;">2 - Matriz de Correlação das Variáveis</p>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(dataset.corr(numeric_only=True),annot=True)
st.pyplot(fig)

st.write('<p style="font-size:20px;">3 - Análises Gráficas</p>', unsafe_allow_html=True)
df_mean = pd.pivot_table(dataset,index=['label'], aggfunc='mean').reset_index()

fig_1 = go.Figure()
fig_1.add_trace(go.Bar(x =df_mean.label,y=df_mean['N'],name='Nitrogênio'))
fig_1.add_trace(go.Bar(x = df_mean.label,y=df_mean['P'],name='Fósforo'))
fig_1.add_trace(go.Bar(x=df_mean.label, y=df_mean['K'],name='Potássio'))
fig_1.update_layout(title='Comparação entre os Nutrientes',title_x=0.3)
st.plotly_chart(fig_1)

fig_2 = px.box(dataset,x =dataset.label,y='ph',color='label')
fig_2.update_layout(title='Ph',title_x=0.45)
st.plotly_chart(fig_2)

fig_3 = px.box(dataset,x =dataset.label,y='rainfall',color='label')
fig_3.update_layout(title='Precipitação',title_x=0.45)
st.plotly_chart(fig_3)

#Modelos
alvo = dataset['label']
features = dataset[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
labels = dataset['label']

crop = []
modelo = []

x_treino,x_teste,y_treino,y_teste = train_test_split(features,alvo,test_size=0.30,random_state = 0)

#Implementação do modelo de arvore de decisão
DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=5,max_depth=5)
DecisionTree.fit(x_treino,y_treino)

predicted_values_dt = DecisionTree.predict(x_teste)
x_dt = (accuracy_score(y_teste,predicted_values_dt)*100).round(4)
crop.append(x_dt)
modelo.append("Árvore de Decisão")

#report do modelo de arvore de decisão em um dataframe pandas
texto_dt = classification_report(y_teste,predicted_values_dt,output_dict=True)
classification_report_dt = pd.DataFrame(texto_dt).transpose()

#Implementação do modelo de random forest

random_forest = RandomForestClassifier(n_estimators=10,random_state=0)
random_forest.fit(x_treino,y_treino)
predicte_values_rf = random_forest.predict(x_teste)
x_rf = (accuracy_score(y_teste,predicte_values_rf)*100).round(4)
crop.append(x_rf)
modelo.append("Random Forest")

texto_rf = classification_report(y_teste,predicte_values_rf,output_dict=True)
classification_report_rf = pd.DataFrame(texto_rf).transpose()



st.write('<p style="font-size:20px;">3 - Resultado dos Algoritmos</p>', unsafe_allow_html=True)


col1,col2 = st.columns(2)

with col1:
    st.text("   Árvore de Decisão    ")
    st.dataframe(classification_report_dt.round(2))
with col2:
    st.text("    Random Forest       ")
    st.dataframe(classification_report_rf.round(2))

st.text(f"O Algoritmo de Árvore de Decisão apresenta uma acurácia de: {x_dt}")
st.text(f"O Algoritmo de Random Forest apresenta uma acurácia de: {x_rf}")




















