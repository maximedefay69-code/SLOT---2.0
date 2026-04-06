import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium

# Configuration de la page
st.set_page_config(page_title="SLOT 2.0 - Map Engine", layout="wide")

st.title("🗺️ SLOT 2.0 - Visualisation Temps Réel")

# --- 1. RECHERCHE D'ADRESSE ---
with st.sidebar:
    st.header("📍 Navigation")
    adresse_input = st.text_input("Entrez une adresse à Paris :", placeholder="ex: 10 rue de Rivoli")
    rayon = st.slider("Rayon d'action (mètres)", 100, 1000, 500)

# Initialisation des coordonnées par défaut (Centre de Paris)
lat_center, lon_center = 48.8566, 2.3522
zoom_level = 13

if adresse_input:
    # Appel à l'API Adresse pour centrer la carte
    url_geo = f"https://api-adresse.data.gouv.fr/search/?q={adresse_input}+Paris&limit=1"
    res = requests.get(url_geo).json()
    
    if res['features']:
        coords = res['features'][0]['geometry']['coordinates']
        lon_center, lat_center = coords[0], coords[1]
        zoom_level = 16
        st.success(f"Ciblage : {res['features'][0]['properties']['label']}")
    else:
        st.error("Adresse introuvable.")

# --- 2. CRÉATION DE LA CARTE ---
# On crée l'objet carte
m = folium.Map(location=[lat_center, lon_center], zoom_start=zoom_level, control_scale=True)

# Ajout d'un marqueur pour Louis
folium.Marker(
    [lat_center, lon_center], 
    popup="Ma position",
    icon=folium.Icon(color="red", icon="info-sign")
).add_to(m)

# Ajout du cercle de 500m (Rayon de recherche)
folium.Circle(
    radius=rayon,
    location=[lat_center, lon_center],
    color="crimson",
    fill=True,
    fill_color="crimson",
    fill_opacity=0.1
).add_to(m)

# --- 3. AFFICHAGE DANS STREAMLIT ---
st_folium(m, width=1200, height=600)

st.info("💡 Prochaine étape : Récupérer les segments de rues dans ce cercle et les colorer via l'IA.")
