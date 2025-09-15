import time
import os
import sqlite3
from datetime import datetime, date
import pandas as pd
import streamlit as st
from io import BytesIO
from collections import defaultdict
import base64

st.set_page_config(layout="wide")

# ================================
# Autenticação com senha
# ================================
if 'acesso_permitido' not in st.session_state:
    st.session_state['acesso_permitido'] = False

if not st.session_state['acesso_permitido']:
    senha_usuario = st.text_input("Digite a senha para acessar o dashboard:", type="password")
    if senha_usuario:
        if senha_usuario == st.secrets["access_token"]:
            st.session_state['acesso_permitido'] = True
            st.rerun()
        else:
            st.error("Senha incorreta.")
            st.stop()
    else:
        st.stop()

# ================================
# Estilo customizado e fundo com logo
# ================================
def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background(logo_path):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{logo_path}");
            background-size: cover;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background(get_image_as_base64("nucleo.png"))

st.markdown("""
<style>
div[role="radiogroup"] label {
    background-color: rgb(0, 32, 96);
    color: white !important;
    padding: 10px 21px;
    border-radius: 8px;
    font-weight: normal;
    cursor: pointer;
    transition: 0.3s;
    border: 2px solid transparent;
}
div[role="radiogroup"] div {
    color: white;
}
div[role="radiogroup"] label span {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================================
# Classe de backend
# ================================
class dashboard_localiza_movida:
    def __init__(self, caminho_db):
        self.caminho_db = caminho_db
        self.df = None
        self.modelos_escolhidos_localiza = sorted([
            "MOBI LIKE 1.0", "TRACKER PREMIER 1.2", "ONIX PLUS LTZ 1.0",
            "ONIX LT 1.0", "ONIX PLUS LT 1.0", "HB20 COMFORT PLUS FLEX 1.0",
            "ARGO DRIVE 1.0", "T-CROSS 1.0", "COMPASS LONGITUDE 1.3",
            "CRONOS DRIVE 1.0"
        ])
        self.modelos_escolhidos_movida = sorted([
            "Volkswagen Polo","Renault Kwid","Fiat Mobi","Fiat Argo",
            "Chevrolet Onix","Fiat Pulse","Renault Logan",
            "Volkswagen T-Cross","Peugeot 208","Hyundai Hb20"
        ])

    def conectar_df(self):
        # Etapa 1 – conexão
        conn = sqlite3.connect(self.caminho_db)

        # Etapa 2 – leitura
        df_base = pd.read_sql_query("SELECT * FROM cars;", conn)

        # Fecha conexão imediatamente
        conn.close()

        # Etapa 3 – tratamento
        df_base["Data"] = pd.to_datetime(df_base["Data"], errors="coerce")
        df_base = df_base.dropna(subset=["ID"])

        # Etapa final
        self.df = df_base
        return self.df

    def formatar_dataframe_valores(self, df_mes, colunas_valores, tipo_analise, variavel):
        if tipo_analise == 'variacao_media':
            return df_mes[colunas_valores].applymap(
                lambda x: f"{x:,.2f}%"
                        .replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "-"
            )
        elif variavel == "qtd":
            return df_mes[colunas_valores].applymap(
                lambda x: f"{int(x):,}".replace(",", ".") if pd.notnull(x) else "-"
            )
        else:
            return df_mes[colunas_valores].applymap(
                lambda x: f"R$ {x:,.2f}"
                        .replace(",", "X").replace(".", ",").replace("X", ".") if pd.notnull(x) else "-"
            )

    def calcular_variacao_semanal(self, filtro=None, site=None, tipo_analise=None, variavel=None, data_corte=None):
        self.df["Data"] = pd.to_datetime(self.df["Data"]).dt.normalize()

        if site:
            df = self.df[self.df["Site"].str.lower() == site.lower()].copy()
        else:
            df = self.df.copy()

        modelos = self.modelos_escolhidos_localiza if site == "localiza" else self.modelos_escolhidos_movida
        df = df[df["Model"].isin(modelos)]

        if filtro in ['vendidos', 'novos']:
            df = df.dropna(subset=['ID'])

        if data_corte is not None:
            data_corte = pd.to_datetime(data_corte).normalize()
            df = df[df["Data"] >= data_corte]
        with st.expander(f"🧪 DEBUG: DF após filtros - Site: {site} | Filtro: {filtro} | Corte: {data_corte}"):
            st.write(f"Registros após filtros: {len(df)}")
            st.dataframe(df)
        df["Data"] = pd.to_datetime(df["Data"]).dt.normalize()
        df_ids = df.groupby('Data')['ID'].apply(set).to_dict()
        all_dates = df["Data"].sort_values().unique()

        meses = defaultdict(list)
        for dt in all_dates:
            chave_mes = dt.strftime('%Y-%m')
            meses[chave_mes].append(dt)

        meses_ordenados = sorted(meses.keys())
        intervalos = defaultdict(list)

        for i, chave_mes in enumerate(meses_ordenados):
            datas_mes = sorted(meses[chave_mes])
            if len(datas_mes) < 2:
                continue
            start = datas_mes[0] if i == 0 else max(meses[meses_ordenados[i - 1]])
            for end in datas_mes:
                if end <= start or end.weekday() != 0:
                    continue
                if tipo_analise == 'variacao_media':
                    label = f"Variação desde ({start.strftime('%d/%m')} até {end.strftime('%d/%m')})"
                else:
                    label = f"{variavel} de carros {filtro} de ({start.strftime('%d/%m')} até {end.strftime('%d/%m')})"
                intervalos[chave_mes].append((label, start, end))

        def preencher_nulos(row_dict):
            for modelo in modelos:
                row_dict[modelo] = None
            row_dict['avg'] = None

        dfs_por_mes = {}
        for mes, lista_intervalos in intervalos.items():
            results = []
            prev_ids = None
            for (rotulo, dt_ini, dt_fim) in lista_intervalos:
                row_dict = {'Intervalo': rotulo}

                ids_ini = df_ids.get(dt_ini, set())
                ids_fim = df_ids.get(dt_fim, set())

                if prev_ids is None:
                    novos = ids_fim - ids_ini
                    vendidos = ids_ini - ids_fim
                else:
                    novos = ids_fim - prev_ids
                    vendidos = prev_ids - ids_fim

                prev_ids = ids_fim

                if filtro == 'novos':
                    ids_validos = novos
                elif filtro == 'vendidos':
                    ids_validos = vendidos
                else:
                    ids_validos = ids_fim

                df_filtrado = df[df["ID"].isin(ids_validos)]

                if not df_filtrado.empty:
                    try:
                        if tipo_analise == 'variacao_media':
                            df_pivot = df.pivot_table(
                                index=['Data'], columns='Model', values='Price', aggfunc='mean'
                            ).sort_index().reset_index().set_index("Data")

                            if dt_ini in df_pivot.index and dt_fim in df_pivot.index:
                                prices_ini = df_pivot.loc[dt_ini]
                                prices_fim = df_pivot.loc[dt_fim]
                                modelos_validos = df[df["ID"].isin(ids_validos)]["Model"].unique()
                                prices_ini = prices_ini[modelos_validos]
                                prices_fim = prices_fim[modelos_validos]
                                variacoes = (prices_fim / prices_ini - 1) * 100

                                q_fim = df[(df["Data"] == dt_fim) & (df["ID"].isin(ids_validos))].groupby("Model").size()
                                media_pond = (variacoes * q_fim).sum() / q_fim.sum() if q_fim.sum() != 0 else 0

                                for modelo in modelos:
                                    row_dict[modelo] = variacoes.get(modelo, None)
                                row_dict['avg'] = media_pond
                            else:
                                preencher_nulos(row_dict)
                        else:
                            if variavel == "qtd":
                                qtd_modelos = df_filtrado.groupby("Model")["ID"].nunique()
                                total = len(df_filtrado["ID"].unique())
                                for modelo in modelos:
                                    row_dict[modelo] = qtd_modelos.get(modelo, 0)
                            else:
                                media_modelos = df_filtrado.groupby("Model")["Price"].mean()
                                total = df_filtrado["Price"].mean()
                                for modelo in modelos:
                                    row_dict[modelo] = media_modelos.get(modelo, None)
                            row_dict['avg'] = total
                    except Exception:
                        preencher_nulos(row_dict)
                else:
                    preencher_nulos(row_dict)

                results.append(row_dict)

            df_mes = pd.DataFrame(results)
            df_mes = df_mes[['Intervalo'] + modelos + ['avg']]
            colunas_valores = [col for col in df_mes.columns if col != "Intervalo"]
            df_mes[colunas_valores] = self.formatar_dataframe_valores(df_mes, colunas_valores, tipo_analise, variavel)
            df_mes = df_mes.iloc[::-1].reset_index(drop=True)
            dfs_por_mes[mes] = df_mes

        return dfs_por_mes

# ================================
# Classe de Tabelas HTML
# ================================
class TabelaLocalizaMovida:
    def __init__(self, df_localiza, df_movida):
        self.df_localiza = df_localiza
        self.df_movida = df_movida
    
    def gerar_html_tabela(self, df, titulo, cor_rgb):
        html = f"""
        <table style="width: 100%; border-collapse: collapse; font-family: Calibri, sans-serif; font-size: 11px;">
            <thead>
                <!-- TÍTULO MESCLADO -->
                <tr>
                    <th colspan="{df.shape[1]}" style="background-color: {cor_rgb};
                                                        color: white;
                                                        font-weight: bold;
                                                        font-size: 15px;
                                                        text-align: center;
                                                        padding: 8px;">
                        {titulo}
                    </th>
                </tr>
                <!-- CABEÇALHO DAS COLUNAS -->
                <tr style="background-color: rgb(242, 242, 242); color: black;">
        """
        for col in df.columns:
            html += f'<th style="border: 1px solid #000; padding: 6px; text-align: center;">{col}</th>'
        html += '</tr></thead><tbody>'
    
        for _, row in df.iterrows():
            html += f'<tr style="background-color: white;">'
            for col in df.columns:
                valor = row[col]
                cor_valor = (
                    "rgb(0,176,80)" if isinstance(valor, str) and valor.startswith("+")
                    else "rgb(255,0,0)" if isinstance(valor, str) and valor.startswith("-")
                    else "black"
                )
                html += f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center; color:{cor_valor};">{valor}</td>'
            html += '</tr>'
        html += '</tbody></table>'
        return html

    
    def gerar_html_tabelas_lado_a_lado(self, df_esquerda, df_direita, titulo_esquerda, titulo_direita, cor_esquerda, cor_direita):
        tabela_esquerda = self.gerar_html_tabela(df_esquerda, titulo_esquerda, cor_esquerda)
        tabela_direita = self.gerar_html_tabela(df_direita, titulo_direita, cor_direita)
    
        html = f"""
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="width: 50%; vertical-align: top; padding-right: 10px;">
                    {tabela_esquerda}
                </td>
                <td style="width: 50%; vertical-align: top; padding-left: 10px;">
                    {tabela_direita}
                </td>
            </tr>
        </table>
        <br><br>
        """
        return html

    def mostrar_tabelas(self):
        col1, col2 = st.columns(2)
        with col1:
            html_loc = self.gerar_html_tabela(self.df_localiza, "Localiza", "rgb(0, 176, 80)")
            st.markdown(html_loc, unsafe_allow_html=True)
        with col2:
            html_mov = self.gerar_html_tabela(self.df_movida, "Movida", "rgb(237, 125, 49)")
            st.markdown(html_mov, unsafe_allow_html=True)

    def download_excel(self):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            self.df_localiza.to_excel(writer, sheet_name="Localiza", index=False)
            self.df_movida.to_excel(writer, sheet_name="Movida", index=False)
        st.download_button(
            label="Baixar tabelas em Excel",
            data=output.getvalue(),
            file_name="tabelas_localiza_movida.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ================================
# Cache: carregar base SQLite
# ================================
@st.cache_data
def carregar_base():
    caminho = os.path.join("data", "car_prices_database.db")
    dash = dashboard_localiza_movida(caminho)
    return dash

# ================================
# Sidebar (filtros)
# ================================
st.sidebar.header("Opções do Dashboard")
site = st.sidebar.selectbox("Selecione o site:", ["localiza", "movida"])
filtro = st.sidebar.selectbox("Filtro de carros:", ["total", "novos", "vendidos"])
tipo_analise = st.sidebar.selectbox("Tipo de análise:", ["variacao_media", "preco", "qtd"])
variavel = st.sidebar.selectbox("Variável:", ["preco", "qtd"])
data_corte = st.sidebar.date_input("Data de corte:", value=date(2025, 4, 30))
modo_visualizacao = st.sidebar.radio("Modo de visualização:", ["Todos os meses", "Filtrar mês específico"])

# ================================
# Exibir resultados
# ================================
dash = carregar_base()
dash.conectar_df()
dfs_por_mes = dash.calcular_variacao_semanal(filtro=filtro, site=site, tipo_analise=tipo_analise, variavel=variavel, data_corte=data_corte)

if modo_visualizacao == "Todos os meses":
    for mes, df_mes in dfs_por_mes.items():
        st.subheader(f"Resultados - {site.title()} - {mes}")
        if site == "localiza":
            tabelas = TabelaLocalizaMovida(df_mes, pd.DataFrame())
        else:
            tabelas = TabelaLocalizaMovida(pd.DataFrame(), df_mes)
        tabelas.mostrar_tabelas()
else:
    mes_escolhido = st.sidebar.selectbox("Selecione o mês:", list(dfs_por_mes.keys()))
    df_mes = dfs_por_mes[mes_escolhido]
    st.subheader(f"Resultados - {site.title()} - {mes_escolhido}")
    if site == "localiza":
        tabelas = TabelaLocalizaMovida(df_mes, pd.DataFrame())
    else:
        tabelas = TabelaLocalizaMovida(pd.DataFrame(), df_mes)
    tabelas.mostrar_tabelas()










