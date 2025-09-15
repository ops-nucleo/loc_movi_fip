import streamlit as st
import pandas as pd

# Carrega seu DataFrame, ex: do GitHub
df = pd.read_csv("teste.csv")

# Filtro dinâmico para uma coluna categórica
fundos = df["Fundo"].unique().tolist()
fundo_selecionado = st.multiselect("Selecione o Fundo", fundos)

# Filtro para Data
datas = sorted(df["Data"].unique())
data_selecionada = st.selectbox("Selecione a Data", datas)

# Filtro por Id
ids = df["Id"].unique().tolist()
id_selecionado = st.multiselect("Selecione o Id", ids)

# Filtra dinamicamente o dataframe
df_filtrado = df.copy()

if fundo_selecionado:
    df_filtrado = df_filtrado[df_filtrado["Fundo"].isin(fundo_selecionado)]
if data_selecionada:
    df_filtrado = df_filtrado[df_filtrado["Data"] == data_selecionada]
if id_selecionado:
    df_filtrado = df_filtrado[df_filtrado["Id"].isin(id_selecionado)]

st.write(df_filtrado)
