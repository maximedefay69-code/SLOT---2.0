import streamlit as st
import requests, folium, joblib, pandas as pd, numpy as np
from streamlit_folium import st_folium

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="SLOT 2.0 - Equilibre", 
    layout="wide"
)

# Style CSS minimaliste
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
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
st.title("📍 SLOT 2.0 - Navigation Terrain")

with st.sidebar:
    st.header("⚙️ Contrôles")
    adresse_saisie = st.text_input("Localisation :", "Place de la République, Paris")
    rayon_scan = st.slider("Rayon du scan (m)", 100, 1000, 500)
    st.write("---")
    st.caption("Fond de carte : Gris urbain (avec noms)")

# --- 4. GÉOLOCALISATION ---
url_geo = f"https://api-adresse.data.gouv.fr/search/?q={adresse_saisie}&limit=1"
try:
    res_geo = requests.get(url_geo).json()
    if res_geo['features']:
        coords = res_geo['features'][0]['geometry']['coordinates']
        lon, lat = coords[0], coords[1]
        
        # --- 5. CRÉATION DE LA CARTE (LE JUSTE MILIEU) ---
        # "cartodbpositron" affiche les noms des rues/communes en mode discret
        m = folium.Map(
            location=[lat, lon], 
            zoom_start=17, 
            tiles="cartodbpositron", 
            attr="CartoDB / OpenStreetMap"
        )

        # --- 6. RÉCUPÉRATION DES RUES ---
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
                    
                    # Logique couleur : Vert (Libre) / Rouge (Occupé)
                    # Note : On force l'épaisseur pour que ça dépasse bien du fond de carte
                    couleur_ligne = "#27ae60" if nb_places > 5 else "#e74c3c"
                    
                    folium.PolyLine(
                        points_gps, 
                        color=couleur_ligne, 
                        weight=7, 
                        opacity=0.85,
                        popup=f"<b>{nom_rue}</b><br>{nb_places} places théoriques"
                    ).add_to(m)

        # Marqueur Louis (Cercle bleu électrique pour bien se voir)
        folium.CircleMarker(
            [lat, lon],
            radius=8,
            color="#007AFF",
            fill=True,
            fill_color="#007AFF",
            fill_opacity=1,
            popup="Moi"
        ).add_to(m)

        # --- 7. AFFICHAGE ---
        st_folium(m, width=1200, height=700, returned_objects=[])
        
    else:
        st.error("Adresse introuvable.")

except Exception as e:
    st.error(f"Erreur : {e}")

st.caption("Moteur SLOT V2 - Prêt pour l'intégration IA")
