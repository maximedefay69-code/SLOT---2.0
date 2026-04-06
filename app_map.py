import streamlit as st
import requests, folium, joblib, pandas as pd, numpy as np
from streamlit_folium import st_folium

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="SLOT 2.0 - Soft Mode", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Style CSS pour personnaliser l'interface (couleurs douces)
st.markdown("""
    <style>
    .stApp { background-color: #fdfaf5; }
    .stTextInput > div > div > input { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CHARGEMENT DES ASSETS (IA) ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("modele_lightgbmDA.pkl")
        prepro = joblib.load("preprocessorDA.pkl")
        return model, prepro
    except:
        return None, None

model, prepro = load_assets()

# --- 3. INTERFACE UTILISATEUR ---
st.title("🌿 SLOT 2.0 - Visualisation Douce")

with st.sidebar:
    st.header("📍 Zone de Recherche")
    adresse_saisie = st.text_input("Adresse à Paris :", "Place de la Concorde, Paris")
    rayon_scan = st.slider("Rayon du scan (mètres)", 100, 1000, 400)
    
    st.divider()
    st.info("Les rues s'affichent en fonction de leur capacité théorique actuelle.")

# --- 4. RÉCUPÉRATION DES COORDONNÉES GPS ---
url_geo = f"https://api-adresse.data.gouv.fr/search/?q={adresse_saisie}&limit=1"
try:
    res_geo = requests.get(url_geo).json()
    if res_geo['features']:
        coords = res_geo['features'][0]['geometry']['coordinates']
        lon, lat = coords[0], coords[1]
        
        # --- 5. CRÉATION DE LA CARTE (STYLE SABLE & BLANC) ---
        # On utilise le fond "HOT" qui est beige/vert d'eau très clair
        m = folium.Map(
            location=[lat, lon], 
            zoom_start=17, 
            tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
            attr="© OpenStreetMap contributors"
        )

        # --- 6. RÉCUPÉRATION DES RUES DANS LE RAYON ---
        url_paris = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {
            "where": f"distance(geom, geom'POINT({lon} {lat})', {rayon_scan}m)",
            "limit": 100
        }
        
        res_paris = requests.get(url_paris, params=params).json()
        
        if 'results' in res_paris:
            for rue in res_paris['results']:
                geom = rue.get('geom', {}).get('geometry', {})
                nom_rue = rue.get('nomvoie', 'Voie sans nom')
                nb_places = rue.get('placal', 0)
                
                # On dessine uniquement les lignes (rues)
                if geom and geom['type'] == 'LineString':
                    # Conversion des points [lon, lat] vers [lat, lon] pour Folium
                    points_gps = [[p[1], p[0]] for p in geom['coordinates']]
                    
                    # LOGIQUE VISUELLE TEMPORAIRE (IA à brancher ici)
                    # Vert si > 5 places, Rouge si < 5
                    couleur_ligne = "#2ecc71" if nb_places > 5 else "#e74c3c"
                    largeur_ligne = 6 if nb_places > 5 else 3
                    
                    folium.PolyLine(
                        points_gps, 
                        color=couleur_ligne, 
                        weight=largeur_ligne, 
                        opacity=0.8,
                        popup=f"<b>{nom_rue}</b><br>Places : {nb_places}"
                    ).add_to(m)

        # Marqueur central (Position de Louis)
        folium.CircleMarker(
            [lat, lon],
            radius=10,
            color="#3498db",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.6,
            popup="Centre du Scan"
        ).add_to(m)

        # --- 7. AFFICHAGE FINAL ---
        st_folium(m, width=1000, height=600, returned_objects=[])
        
    else:
        st.error("Désolé, je ne trouve pas cette adresse à Paris.")

except Exception as e:
    st.error(f"Une erreur est survenue : {e}")

st.caption("Données sources : Open Data Paris | Moteur : SLOT Engine V2")
