import streamlit as st
import requests, folium, joblib, pandas as pd, numpy as np
from streamlit_folium import st_folium

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="SLOT 2.0 - Ultra Clean", 
    layout="wide"
)

# Style CSS pour un fond neutre et propre
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
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
st.title("⚪ SLOT 2.0 - Carte de Précision")

with st.sidebar:
    st.header("⚙️ Paramètres")
    adresse_saisie = st.text_input("Adresse de départ :", "Rue de Rivoli, Paris")
    rayon_scan = st.slider("Rayon du scan (m)", 100, 1000, 400)
    st.write("---")
    st.caption("Mode : Minimaliste (Sans distractions)")

# --- 4. RÉCUPÉRATION DES COORDONNÉES GPS ---
url_geo = f"https://api-adresse.data.gouv.fr/search/?q={adresse_saisie}&limit=1"
try:
    res_geo = requests.get(url_geo).json()
    if res_geo['features']:
        coords = res_geo['features'][0]['geometry']['coordinates']
        lon, lat = coords[0], coords[1]
        
        # --- 5. CRÉATION DE LA CARTE (ULTRA-MINIMALISTE) ---
        # "cartodbpositronnolabels" supprime les noms de lieux, bus, monuments.
        m = folium.Map(
            location=[lat, lon], 
            zoom_start=17, 
            tiles="cartodbpositronnolabels", 
            attr="CartoDB / OpenStreetMap"
        )

        # --- 6. RÉCUPÉRATION DES RUES DANS LE RAYON ---
        url_paris = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {
            "where": f"distance(geom, geom'POINT({lon} {lat})', {rayon_scan}m)",
            "limit": 150
        }
        
        res_paris = requests.get(url_paris, params=params).json()
        
        if 'results' in res_paris:
            for rue in res_paris['results']:
                geom = rue.get('geom', {}).get('geometry', {})
                nom_rue = rue.get('nomvoie', 'Voie sans nom')
                nb_places = rue.get('placal', 0)
                
                if geom and geom['type'] == 'LineString':
                    points_gps = [[p[1], p[0]] for p in geom['coordinates']]
                    
                    # LOGIQUE VISUELLE (Simple Noir & Blanc pour les axes)
                    # On garde le Vert/Rouge car c'est ton code couleur métier
                    couleur_ligne = "#27ae60" if nb_places > 5 else "#c0392b"
                    
                    folium.PolyLine(
                        points_gps, 
                        color=couleur_ligne, 
                        weight=5, 
                        opacity=0.9,
                        popup=f"{nom_rue} : {nb_places} pl."
                    ).add_to(m)

        # Marqueur Louis (Cercle fin)
        folium.CircleMarker(
            [lat, lon],
            radius=6,
            color="#2c3e50",
            fill=True,
            fill_color="#2c3e50",
            fill_opacity=1,
            popup="Moi"
        ).add_to(m)

        # --- 7. AFFICHAGE ---
        st_folium(m, width=1200, height=700, returned_objects=[])
        
    else:
        st.error("Adresse introuvable.")

except Exception as e:
    st.error(f"Erreur : {e}")
