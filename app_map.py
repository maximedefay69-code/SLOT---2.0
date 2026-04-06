import streamlit as st
import requests, folium, joblib, pandas as pd, numpy as np
from streamlit_folium import st_folium

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="SLOT 2.0 - Radar Dynamique", 
    layout="wide"
)

# Style CSS pour l'esthétique
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
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

# --- 3. INTERFACE DE SAISIE (EN HAUT) ---
st.title("🛰️ SLOT 2.0 - Scanner Terrain")

c1, c2, c3 = st.columns([1, 2, 4])

with c1:
    num_voie = st.text_input("N°", placeholder="10")
with c2:
    type_voie = st.selectbox(
        "Type", 
        ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse", "Allée", "Square", "Route", "Voie"]
    )
with c3:
    nom_voie = st.text_input("Nom de la voie", placeholder="Entrez le nom pour zoomer...")

# --- 4. BARRE LATÉRALE ---
with st.sidebar:
    st.header("⚙️ Réglages")
    rayon_scan = st.slider("Rayon du scan (m)", 100, 1000, 400)
    taille_logo = st.slider("Taille du logo SLOT", 30, 100, 60)
    st.write("---")
    if nom_voie:
        st.success("🎯 Cible verrouillée")
    else:
        st.warning("📡 En attente de cible")

# --- 5. LOGIQUE DE ZOOM ET GÉOLOCALISATION ---

# Valeurs par défaut : Vue d'ensemble de Paris
lat_view, lon_view = 48.8566, 2.3522
zoom_level = 12 # Vue large (Arrondissements visibles)

adresse_complete = ""
target_found = False

if nom_voie:
    adresse_complete = f"{num_voie} {type_voie} {nom_voie}, Paris"
    url_geo = f"https://api-adresse.data.gouv.fr/search/?q={adresse_complete}&limit=1"
    
    try:
        res_geo = requests.get(url_geo).json()
        if res_geo['features']:
            f = res_geo['features'][0]
            coords = f['geometry']['coordinates']
            lon_view, lat_view = coords[0], coords[1]
            zoom_level = 18 # Zoom précis (Immeubles visibles)
            target_found = True
    except:
        pass

# --- 6. CRÉATION DE LA CARTE ---
m = folium.Map(
    location=[lat_view, lon_view], 
    zoom_start=zoom_level, 
    tiles="cartodbpositron", 
    attr="CartoDB"
)

# --- 7. RÉCUPÉRATION DES RUES (UNIQUEMENT SI ADRESSE SAISIE) ---
if target_found:
    url_paris = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
    params = {
        "where": f"distance(geom, geom'POINT({lon_view} {lat_view})', {rayon_scan}m)",
        "limit": 150
    }
    
    try:
        res_paris = requests.get(url_paris, params=params).json()
        if 'results' in res_paris:
            for rue in res_paris['results']:
                geom = rue.get('geom', {}).get('geometry', {})
                nom_r = rue.get('nomvoie', 'Voie')
                nb_p = rue.get('placal', 0)
                
                if geom and geom['type'] == 'LineString':
                    points_gps = [[p[1], p[0]] for p in geom['coordinates']]
                    couleur = "#27ae60" if nb_p > 5 else "#e74c3c"
                    
                    folium.PolyLine(
                        points_gps, 
                        color=couleur, 
                        weight=8, 
                        opacity=0.7,
                        popup=f"<b>{nom_r}</b><br>{nb_p} places"
                    ).add_to(m)

        # AJOUT DU LOGO SLOT
        URL_LOGO_OFFICIEL = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
        logo_icon = folium.CustomIcon(URL_LOGO_OFFICIEL, icon_size=(taille_logo, taille_logo))
        folium.Marker([lat_view, lon_view], icon=logo_icon, popup="Cible").add_to(m)
        
    except:
        pass

# --- 8. AFFICHAGE FINAL ---
# returned_objects=[] permet d'éviter de recharger la carte inutilement lors d'un clic
st_folium(m, width=1200, height=700, key="slot_map", returned_objects=[])

if not target_found and nom_voie:
    st.error("⚠️ Adresse introuvable, le radar reste en attente.")
elif not target_found:
    st.info("💡 Entrez une adresse pour que le radar se focalise sur la zone.")
