import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px
import seaborn as sns
from scipy.stats import percentileofscore
import io

# --- CONFIGURACION INICIAL ---
st.set_page_config(layout="wide", page_title="Perfiles Jugadores - Segunda FEB")

# --- CARGA DE DATOS ---
df = pd.read_csv("estadisticas_segunda_feb_2025.csv")

# LIMPIEZA DE DATOS
for col in ['Ast/TO', 'Stl/TO']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

# Columnas porcentaje, aseguramos formato float sin dividir
columnas_porcentaje = ['FG%', '3P%', 'FT%', 'TS%', 'eFG%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%']
for col in columnas_porcentaje:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Asegurar columnas numéricas importantes
df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
if 'GP' in df.columns:
    df['GP'] = pd.to_numeric(df['GP'], errors='coerce')
if 'Edad' in df.columns:
    df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')

# --- SIDEBAR: FILTROS DINÁMICOS ---
st.sidebar.title("Configuración de Filtros")

posiciones = st.sidebar.multiselect("Filtrar por posición", sorted(df['Pos'].dropna().unique()))
equipos = st.sidebar.multiselect("Filtrar por equipo", sorted(df['Team_completo'].dropna().unique()))

min_minutos = st.sidebar.slider("Mínimo minutos totales jugados", int(df['MIN'].min() or 0), int(df['MIN'].max() or 1000), 0)
min_partidos = None
if 'GP' in df.columns:
    min_partidos = st.sidebar.slider("Mínimo partidos jugados (GP)", int(df['GP'].min() or 0), int(df['GP'].max() or 1000), 0)
edad_rango = None
if 'Edad' in df.columns:
    edad_rango = st.sidebar.slider("Rango de edad", int(df['Edad'].min() or 15), int(df['Edad'].max() or 50), (int(df['Edad'].min() or 15), int(df['Edad'].max() or 50)))

# Aplicar filtros
df_filtrado = df.copy()
if posiciones:
    df_filtrado = df_filtrado[df_filtrado['Pos'].isin(posiciones)]
if equipos:
    df_filtrado = df_filtrado[df_filtrado['Team_completo'].isin(equipos)]
if min_minutos:
    df_filtrado = df_filtrado[df_filtrado['MIN'] >= min_minutos]
if min_partidos is not None:
    df_filtrado = df_filtrado[df_filtrado['GP'] >= min_partidos]
if edad_rango is not None:
    df_filtrado = df_filtrado[(df_filtrado['Edad'] >= edad_rango[0]) & (df_filtrado['Edad'] <= edad_rango[1])]

# VARIABLES NUMÉRICAS PARA CLUSTERING
columnas_excluir = ['#_prom', 'Player', 'Team_prom', '#_adv', 'Team_adv', 'Team_completo', 'Pos']
columnas_numericas = df_filtrado.select_dtypes(include='number').columns
columnas_utiles = [col for col in columnas_numericas if col not in columnas_excluir]

variables = st.sidebar.multiselect("Variables para clustering:", columnas_utiles, default=columnas_utiles[:6])
k = st.sidebar.slider("Número de clusters", 2, 10, 3)

mostrar_radar = st.sidebar.checkbox("Mostrar Radar Charts", True)
mostrar_dendros = st.sidebar.checkbox("Mostrar Dendrogramas", True)
mostrar_similares = st.sidebar.checkbox("Mostrar Jugadores Similares", True)
mostrar_corr = st.sidebar.checkbox("Mostrar Correlaciones", True)

# Validación
if len(variables) < 2:
    st.error("Selecciona al menos 2 variables para continuar.")
    st.stop()

for col in variables:
    df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors='coerce')

X = df_filtrado[variables].replace([np.inf, -np.inf], np.nan).dropna()
if X.empty:
    st.error("No hay datos válidos para clustering con los filtros seleccionados.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clustered = df_filtrado.loc[X.index].copy()
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = X_pca[:, 0]
df_clustered['PCA2'] = X_pca[:, 1]

# --- PESTAÑAS ---
tabs = st.tabs([
    "📊 Clusters",
    "🌳 Dendrogramas",
    "📈 Radar",
    "📏 Distancia al Centroide",
    "🎯 Jugadores Similares",
    "🔥 Correlaciones",
    "📝 Informe Jugador"
])

# TAB 1: Clusters
tabs[0].subheader("Jugadores por Cluster")

# Mejor color para clusters con Plotly
fig = px.scatter(
    df_clustered,
    x='PCA1',
    y='PCA2',
    color=df_clustered['Cluster'].astype(str),
    hover_data=['Player', 'Team_completo', 'Pos'],
    title="PCA 2D - Clustering de Jugadores",
    color_discrete_sequence=px.colors.qualitative.Safe + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
)
fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(legend_title_text='Cluster')
tabs[0].plotly_chart(fig, use_container_width=True)

# Tabla con jugadores y clusters
df_display_cols = ['Player', 'Team_completo', 'Pos'] + variables + ['Cluster']
tabs[0].dataframe(df_clustered[df_display_cols].sort_values('Cluster'))

# TAB 2: Dendrogramas
if mostrar_dendros:
    tabs[1].subheader("Dendrograma de Jugadores")
    if len(df_clustered) > 2:
        X_all = df_clustered[variables].values
        labels_all = df_clustered['Player'].values
        linkage_matrix = linkage(X_all, method='ward')
        fig2, ax2 = plt.subplots(figsize=(20, 7))
        dendrogram(linkage_matrix, labels=labels_all, leaf_rotation=90, leaf_font_size=10, ax=ax2)
        tabs[1].pyplot(fig2)

# TAB 3: Radar Charts
if mostrar_radar:
    tabs[2].subheader("Radar Charts por Cluster")
    colores = plt.cm.get_cmap('tab10', k)

    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        subset = df_clustered[df_clustered['Cluster'] == cluster_id]
        if len(subset) < 2:
            continue

        means = subset[variables].mean()
        scaler_radar = MinMaxScaler((0, 100))
        normalized = pd.Series(
            scaler_radar.fit_transform(means.values.reshape(-1, 1)).flatten(),
            index=means.index
        )

        labels_radar = normalized.index.tolist()
        values = normalized.values.tolist()
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        color = to_hex(colores(cluster_id))
        ax3.plot(angles, values, color=color, linewidth=2)
        ax3.fill(angles, values, color=color, alpha=0.25)
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(labels_radar)
        ax3.set_yticklabels([])
        ax3.set_title(f"Radar (0-100) - Cluster {cluster_id}")
        tabs[2].pyplot(fig3)

# TAB 4: Jugadores más alejados del centroide
tabs[3].subheader("Jugadores más alejados del centroide")
for cluster_id in sorted(df_clustered['Cluster'].unique()):
    subset = df_clustered[df_clustered['Cluster'] == cluster_id]
    if len(subset) <= 2:
        continue
    X_subset = subset[variables].values
    centroid = X_subset.mean(axis=0)
    distances = np.linalg.norm(X_subset - centroid, axis=1)
    subset = subset.copy()
    subset['DistanciaCentroide'] = distances
    top5 = subset.sort_values(by='DistanciaCentroide', ascending=False).head(5)
    tabs[3].write(f"**Cluster {cluster_id}**")
    tabs[3].dataframe(top5[['Player', 'Team_completo', 'Pos', 'DistanciaCentroide'] + variables])

# TAB 5: Jugadores Similares
if mostrar_similares:
    tabs[4].subheader("Buscar jugadores similares")
    jugador_sel = tabs[4].selectbox("Selecciona jugador", df_clustered['Player'].sort_values())
    jugador_data = df_clustered[df_clustered['Player'] == jugador_sel].iloc[0]
    cluster_jugador = jugador_data['Cluster']
    subset = df_clustered[df_clustered['Cluster'] == cluster_jugador]
    if len(subset) > 1:
        distancias = np.linalg.norm(subset[variables].values - jugador_data[variables].values, axis=1)
        subset = subset.copy()
        subset['DistJugador'] = distancias
        similares = subset.sort_values('DistJugador').iloc[1:6]
        tabs[4].write(f"Jugadores similares a **{jugador_sel}** en Cluster {cluster_jugador}:")
        tabs[4].dataframe(similares[['Player', 'Team_completo', 'Pos', 'DistJugador'] + variables])
    else:
        tabs[4].write("No hay suficientes jugadores en el cluster para comparar.")

# TAB 6: Correlaciones
if mostrar_corr:
    tabs[5].subheader("Mapa de calor de correlaciones entre variables seleccionadas")
    corr = df_clustered[variables].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    tabs[5].pyplot(fig_corr)

# TAB 7: Informe jugador automático
tabs[6].subheader("Informe detallado del jugador")
jugador_informe = tabs[6].selectbox("Selecciona jugador para informe", df_clustered['Player'].sort_values())
if jugador_informe:
    data_jugador = df_clustered[df_clustered['Player'] == jugador_informe].iloc[0]
    cluster_j = data_jugador['Cluster']
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_j]

    informe = f"**Informe de {jugador_informe}**\n\n"
    informe += f"- Posición: {data_jugador['Pos']}\n"
    informe += f"- Equipo: {data_jugador['Team_completo']}\n"
    informe += f"- Cluster asignado: {cluster_j}\n"
    informe += f"- Minutos jugados: {data_jugador['MIN']}\n\n"

    informe += "### Variables seleccionadas:\n"
    for var in variables:
        val = data_jugador[var]
        media = cluster_data[var].mean()
        pct = percentileofscore(cluster_data[var], val)
        delta = val - media
        status = "por encima" if delta > 0 else "por debajo"
        informe += f"- **{var}**: {val:.2f} (percentil {pct:.1f}%) — {abs(delta):.2f} {status} del promedio del cluster\n"

    # Fortalezas y debilidades (top 3 y bottom 3)
    percentiles = {var: percentileofscore(cluster_data[var], data_jugador[var]) for var in variables}
    sorted_vars = sorted(percentiles.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_vars[:3]
    bottom3 = sorted_vars[-3:]

    informe += "\n### Fortalezas (variables mejor posicionadas)\n"
    for var, pct in top3:
        informe += f"- {var} (percentil {pct:.1f}%)\n"

    informe += "\n### Áreas de mejora (variables peor posicionadas)\n"
    for var, pct in bottom3:
        informe += f"- {var} (percentil {pct:.1f}%)\n"

    tabs[6].markdown(informe)

    # Botón para descargar informe
    informe_txt = informe.replace("**", "").replace("#", "").replace("\n\n", "\n")
    if tabs[6].button("Descargar informe en texto"):
        buffer = io.StringIO()
        buffer.write(informe_txt)
        buffer.seek(0)
        st.download_button("Descargar Informe", data=buffer, file_name=f"Informe_{jugador_informe}.txt", mime="text/plain")
