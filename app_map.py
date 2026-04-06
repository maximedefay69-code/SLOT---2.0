import streamlit as st
import requests, folium, joblib, pandas as pd, numpy as np
from streamlit_folium import st_folium

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="SLOT 2.0 - Scanner Officiel", 
    layout="wide"
)

# Style CSS pour l'esthétique blanche et épurée
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

# Colonnes de saisie pour Louis
c1, c2, c3 = st.columns([1, 2, 4])

with c1:
    num_voie = st.text_input("N°", placeholder="10")

with c2:
    type_voie = st.selectbox(
        "Type", 
        ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse", "Allée", "Square", "Route", "Voie"]
    )

with c3:
    nom_voie = st.text_input("Nom de la voie", placeholder="de Rivoli")

# --- 4. BARRE LATÉRALE (PARAMÈTRES) ---
with st.sidebar:
    st.header("⚙️ Réglages")
    rayon_scan = st.slider("Rayon du scan (m)", 100, 1000, 400)
    taille_logo = st.slider("Taille du logo SLOT", 30, 100, 60)
    st.write("---")
    st.info("Scanner actif : Paris Intra-muros")

# --- 5. LOGIQUE DE GÉOLOCALISATION & CARTE ---
if nom_voie:
    # On construit l'adresse propre
    adresse_complete = f"{num_voie} {type_voie} {nom_voie}, Paris"
    
    url_geo = f"https://api-adresse.data.gouv.fr/search/?q={adresse_complete}&limit=1"
    try:
        res_geo = requests.get(url_geo).json()
        if res_geo['features']:
            f = res_geo['features'][0]
            coords = f['geometry']['coordinates']
            lon, lat = coords[0], coords[1]
            label_trouve = f['properties']['label']
            
            st.success(f"📍 Position confirmée : {label_trouve}")

            # --- CRÉATION DE LA CARTE ---
            m = folium.Map(
                location=[lat, lon], 
                zoom_start=18, 
                tiles="cartodbpositron", 
                attr="CartoDB"
            )

            # --- RÉCUPÉRATION DES SEGMENTS OPEN DATA ---
            url_paris = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
            params = {
                "where": f"distance(geom, geom'POINT({lon} {lat})', {rayon_scan}m)",
                "limit": 150
            }
            
            res_paris = requests.get(url_paris, params=params).json()
            
            if 'results' in res_paris:
                for rue in res_paris['results']:
                    geom = rue.get('geom', {}).get('geometry', {})
                    nom_r = rue.get('nomvoie', 'Voie')
                    nb_p = rue.get('placal', 0)
                    
                    if geom and geom['type'] == 'LineString':
                        points_gps = [[p[1], p[0]] for p in geom['coordinates']]
                        
                        # Code couleur métier (IA Ready)
                        couleur = "#27ae60" if nb_p > 5 else "#e74c3c"
                        
                        folium.PolyLine(
                            points_gps, 
                            color=couleur, 
                            weight=8, 
                            opacity=0.7,
                            popup=f"<b>{nom_r}</b><br>{nb_p} places"
                        ).add_to(m)

            # --- 6. LOGO PERSONNALISÉ (Ton URL GitHub) ---
            URL_LOGO_OFFICIEL = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
            
            try:
                logo_icon = folium.CustomIcon(
                    URL_LOGO_OFFICIEL, 
                    icon_size=(taille_logo, taille_logo)
                )
                folium.Marker(
                    [lat, lon], 
                    icon=logo_icon, 
                    popup="Position Louis"
                ).add_to(m)
            except Exception as e:
                # Secours si l'image ne charge pas
                folium.CircleMarker([lat, lon], radius=10, color="#007AFF", fill=True).add_to(m)

            # --- 7. AFFICHAGE ---
            st_folium(m, width=1200, height=700, returned_objects=[])
            
        else:
            st.warning("⚠️ Adresse non reconnue. Vérifiez l'orthographe.")

    except Exception as e:
        st.error(f"Erreur technique : {e}")
else:
    st.info("👋 En attente d'une adresse pour lancer le scanner...")
