import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from scipy.stats import percentileofscore

st.set_page_config(layout="wide", page_title="Perfiles Jugadores")

# --- CARGA DATOS ---
@st.cache_data(show_spinner=False)
def cargar_datos(path="estadisticas_segunda_feb_2025.csv"):
    df = pd.read_csv(path)
    # Limpieza básica
    for col in ['Ast/TO', 'Stl/TO']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

    cols_pct = ['FG%', '3P%', 'FT%', 'TS%', 'eFG%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%']
    for col in cols_pct:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

df = cargar_datos()

# --- FILTROS ---
st.sidebar.title("Configuración")

# Inicializar session_state para filtros
if "posiciones" not in st.session_state:
    st.session_state.posiciones = []
if "equipos" not in st.session_state:
    st.session_state.equipos = []
if "minutos" not in st.session_state:
    st.session_state.minutos = (int(df['MIN'].min()), int(df['MIN'].max()))

posiciones = st.sidebar.multiselect(
    "Filtrar por posición",
    sorted(df['Pos'].dropna().unique()),
    default=st.session_state.posiciones,
    key="posiciones"
)

equipos = st.sidebar.multiselect(
    "Filtrar por equipo",
    sorted(df['Team_completo'].dropna().unique()),
    default=st.session_state.equipos,
    key="equipos"
)

min_min = int(df['MIN'].min())
max_min = int(df['MIN'].max())
# Inicializa el valor solo si no está ya en session_state
if "minutos" not in st.session_state:
    st.session_state["minutos"] = (int(df['MIN'].min()), int(df['MIN'].max()))

minutos_seleccionados = st.sidebar.slider(
    "Filtrar por minutos jugados (MIN)",
    int(df['MIN'].min()),
    int(df['MIN'].max()),
    key="minutos"
)


def aplicar_filtros(df, posiciones, equipos, minutos):
    df_filt = df.copy()
    if posiciones:
        df_filt = df_filt[df_filt['Pos'].isin(posiciones)]
    if equipos:
        df_filt = df_filt[df_filt['Team_completo'].isin(equipos)]
    if minutos:
        df_filt = df_filt[(df_filt['MIN'] >= minutos[0]) & (df_filt['MIN'] <= minutos[1])]
    return df_filt

df_filtrado = aplicar_filtros(df, posiciones, equipos, minutos_seleccionados)

# --- VARIABLES Y PARÁMETROS ---
columnas_excluir = ['#_prom', 'Player', 'Team_prom', '#_adv', 'Team_adv', 'Team_completo', 'Pos']
columnas_numericas = df_filtrado.select_dtypes(include='number').columns
variables = [c for c in columnas_numericas if c not in columnas_excluir]

vars_seleccionadas = st.sidebar.multiselect(
    "Variables para clustering",
    variables,
    default=st.session_state.get("vars_seleccionadas", variables[:5]),
    key="vars_seleccionadas"
)

k = st.sidebar.slider(
    "Número de clusters",
    2, 10,
    value=st.session_state.get("k", 3),
    key="k"
)

mostrar_radar = st.sidebar.checkbox("Mostrar Radar Charts", True, key="mostrar_radar")
mostrar_dendros = st.sidebar.checkbox("Mostrar Dendrogramas", True, key="mostrar_dendros")
mostrar_similares = st.sidebar.checkbox("Mostrar Jugadores Similares", True, key="mostrar_similares")
mostrar_corr = st.sidebar.checkbox("Mostrar Correlaciones", True, key="mostrar_corr")

if len(vars_seleccionadas) < 2:
    st.error("Selecciona al menos 2 variables.")
    st.stop()

# --- PROCESAMIENTOS CACHEADOS ---
@st.cache_data(show_spinner=False)
def preprocesar(df_local, variables_local):
    df_local = df_local.dropna(subset=variables_local)
    scaler_local = StandardScaler()
    X_scaled = scaler_local.fit_transform(df_local[variables_local])
    return df_local, X_scaled, scaler_local

df_clustered, X_scaled, scaler = preprocesar(df_filtrado, vars_seleccionadas)

@st.cache_data(show_spinner=False)
def aplicar_kmeans(X_scaled_local, k_local):
    kmeans_local = KMeans(n_clusters=k_local, random_state=42, n_init='auto')
    clusters_local = kmeans_local.fit_predict(X_scaled_local)
    return clusters_local, kmeans_local

clusters, kmeans = aplicar_kmeans(X_scaled, k)

@st.cache_data(show_spinner=False)
def aplicar_pca(X_scaled_local):
    pca_local = PCA(n_components=2)
    X_pca_local = pca_local.fit_transform(X_scaled_local)
    return X_pca_local, pca_local

X_pca, pca = aplicar_pca(X_scaled)

df_clustered = df_clustered.reset_index(drop=True)
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = X_pca[:, 0]
df_clustered['PCA2'] = X_pca[:, 1]

# --- VISUALIZACIONES ---
tabs = st.tabs([
    "📊 Clusters",
    "🌳 Dendrogramas",
    "📈 Radar",
    "📏 Diferentes",
    "🎯 Similares",
    "🔥 Correlaciones",
    "📝 Scouting Report"
])

# TAB 1: Clusters
with tabs[0]:
    st.subheader("Jugadores por Cluster")
    st.dataframe(df_clustered[['Player', 'Team_completo', 'Pos'] + vars_seleccionadas])

    st.subheader("Perfil promedio por Cluster")
    resumen = df_clustered.groupby('Cluster')[vars_seleccionadas].mean().round(2)
    st.dataframe(resumen)

    fig = px.scatter(
        df_clustered,
        x='PCA1', y='PCA2',
        color=df_clustered['Cluster'].astype(str),
        hover_data=['Player', 'Team_completo', 'Pos'],
        title="PCA 2D - Clustering de Jugadores",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text='Cluster')
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Dendrogramas
with tabs[1]:
    if mostrar_dendros:
        clusters_unicos = sorted(df_clustered['Cluster'].unique())
        cluster_sel = st.selectbox(
            "Filtrar dendrograma por cluster",
            options=[-1] + clusters_unicos,
            format_func=lambda x: "Todos" if x == -1 else f"Cluster {x}",
            key="cluster_dendro"
        )

        if cluster_sel == -1:
            df_dendro = df_clustered
        else:
            df_dendro = df_clustered[df_clustered['Cluster'] == cluster_sel]

        if len(df_dendro) > 2:
            linkage_matrix = linkage(df_dendro[vars_seleccionadas], method='ward')
            fig = ff.create_dendrogram(
                df_dendro[vars_seleccionadas],
                labels=df_dendro['Player'].values,
                linkagefun=lambda x: linkage_matrix
            )
            fig.update_layout(width=1000, height=600)
            st.plotly_chart(fig)
        else:
            st.info("Pocos datos para dendrograma en este cluster.")

# TAB 3: Radar
with tabs[2]:
    if mostrar_radar:
        st.subheader("Radar Charts por Cluster")
        colores = plt.cm.viridis(np.linspace(0, 1, k))

        scaler_radar = MinMaxScaler((0, 100))
        for cluster_id in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            if len(cluster_data) < 2:
                continue
            mean_vals = cluster_data[vars_seleccionadas].mean().values.reshape(-1, 1)
            normalized = scaler_radar.fit_transform(mean_vals).flatten()

            labels = vars_seleccionadas
            values = list(normalized) + [normalized[0]]
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
            ax.plot(angles, values, color=to_hex(colores[cluster_id]), linewidth=2)
            ax.fill(angles, values, color=to_hex(colores[cluster_id]), alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_yticklabels([])
            ax.set_title(f"Radar (0-100) - Cluster {cluster_id}")
            st.pyplot(fig)

# TAB 4: Más alejados del centroide
with tabs[3]:
    st.subheader("Jugadores más alejados del centroide")
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        if len(cluster_data) <= 2:
            st.info(f"Pocos jugadores en cluster {cluster_id} para mostrar.")
            continue
        centroid = cluster_data[vars_seleccionadas].mean().values
        distances = np.linalg.norm(cluster_data[vars_seleccionadas] - centroid, axis=1)
        cluster_data = cluster_data.copy()
        cluster_data['DistanciaCentroide'] = distances
        top5 = cluster_data.sort_values(by='DistanciaCentroide', ascending=False).head(5)
        st.write(f"**Cluster {cluster_id}**")
        st.dataframe(top5[['Player', 'Pos', 'Team_completo', 'DistanciaCentroide'] + vars_seleccionadas])

# TAB 5: Jugadores similares
with tabs[4]:
    if mostrar_similares:
        st.subheader("Buscar jugadores similares")
        jugador = st.selectbox("Selecciona un jugador", df_clustered['Player'].sort_values().unique())
        jugador_data = df_clustered[df_clustered['Player'] == jugador][vars_seleccionadas]
        if jugador_data.empty:
            st.warning("Jugador no encontrado.")
        else:
            jugador_vals = jugador_data.values[0]
            df_clustered['DistSim'] = np.linalg.norm(df_clustered[vars_seleccionadas] - jugador_vals, axis=1)
            similares = df_clustered[df_clustered['Player'] != jugador].sort_values('DistSim').head(10)
            st.dataframe(similares[['Player', 'Pos', 'Team_completo', 'Cluster', 'DistSim'] + vars_seleccionadas])

# TAB 6: Correlaciones
with tabs[5]:
    if mostrar_corr:
        st.subheader("Mapa de calor de correlaciones")
        corr = df_clustered[vars_seleccionadas].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# TAB 7: Scouting Report
with tabs[6]:
    def generar_texto_scouting(fortalezas, debilidades, percentiles):
        texto = ""
        if fortalezas:
            texto += "🟢 **Fortalezas:** Destaca en " + ", ".join(
                [f"{v} (percentil {int(percentiles[v])})" for v in fortalezas]) + ".\n\n"
        if debilidades:
            texto += "🔴 **Debilidades:** Puede mejorar en " + ", ".join(
                [f"{v} (percentil {int(percentiles[v])})" for v in debilidades]) + ".\n\n"
        if not fortalezas and not debilidades:
            texto += "Perfil equilibrado, sin variables particularmente altas o bajas.\n\n"
        return texto

    jugadora = st.selectbox("Selecciona una jugadora", df_clustered['Player'].unique(), key="scouting_player")
    fila = df_clustered[df_clustered['Player'] == jugadora].iloc[0]
    percentiles = {var: percentileofscore(df_clustered[var].dropna(), fila[var]) for var in vars_seleccionadas}

    fortalezas = [var for var, pct in percentiles.items() if pct >= 75]
    debilidades = [var for var, pct in percentiles.items() if pct <= 25]

    valores_normalizados = MinMaxScaler((0, 100)).fit_transform(df_clustered[vars_seleccionadas]).T
    valores_dict = dict(zip(df_clustered['Player'], valores_normalizados.T))
    valores_radar = valores_dict[jugadora].tolist()
    valores_radar += valores_radar[:1]

    labels = vars_seleccionadas
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, valores_radar, linewidth=2, label=jugadora)
    ax.fill(angles, valores_radar, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f"Radar de {jugadora}")

    st.pyplot(fig)
    st.markdown("_Valores normalizados (0-100) para comparación entre variables._")

    texto = f"**Informe de {jugadora}**\n\n" + generar_texto_scouting(fortalezas, debilidades, percentiles)
    st.markdown(texto)
