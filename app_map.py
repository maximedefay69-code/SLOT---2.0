import streamlit as st
import requests, folium, joblib, pandas as pd, numpy as np
from streamlit_folium import st_folium

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="SLOT 2.0 - Branded", 
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
    
    st.divider()
    st.write("🔧 Personnalisation :")
    # ✅ RÉGLAGE DE LA TAILLE DU LOGO (Pour Louis)
    taille_logo = st.slider("Taille du logo sur la carte", 30, 80, 50)
    st.caption("Ajustez la taille si le logo cache trop les rues.")

# --- 4. GÉOLOCALISATION ---
url_geo = f"https://api-adresse.data.gouv.fr/search/?q={adresse_saisie}&limit=1"
try:
    res_geo = requests.get(url_geo).json()
    if res_geo['features']:
        coords = res_geo['features'][0]['geometry']['coordinates']
        lon, lat = coords[0], coords[1]
        
        # --- 5. CRÉATION DE LA CARTE ---
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
                    couleur_ligne = "#27ae60" if nb_places > 5 else "#e74c3c"
                    
                    folium.PolyLine(
                        points_gps, 
                        color=couleur_ligne, 
                        weight=7, 
                        opacity=0.85,
                        popup=f"<b>{nom_rue}</b><br>{nb_places} places théoriques"
                    ).add_to(m)

        # --- 7. ✅ INTÉGRATION DU LOGO PERSONNALISÉ ---
        
        # ⚠️ REMPLACE CETTE URL PAR TON URL "RAW" GITHUB ⚠️
        URL_DU_LOGO = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
        
        try:
            # Création de l'icône personnalisée
            logo_slot = folium.CustomIcon(
                URL_DU_LOGO,
                icon_size=(taille_logo, taille_logo) # Taille dynamique via le slider
            )
            
            # Ajout du marqueur avec ton logo au lieu du point bleu
            folium.Marker(
                [lat, lon], 
                icon=logo_slot,
                popup="Moi - SLOT Team"
            ).add_to(m)
            
        except:
            # En cas d'erreur de chargement de l'image, on remet un point rouge de secours
            st.error("Impossible de charger le logo. Vérifiez l'URL RAW.")
            folium.CircleMarker(
                [lat, lon], radius=10, color="red", fill=True, fill_color="red"
            ).add_to(m)

        # --- 8. AFFICHAGE FINAL ---
        st_folium(m, width=1200, height=700, returned_objects=[])
        
    else:
        st.error("Adresse introuvable.")

except Exception as e:
    st.error(f"Erreur : {e}")

st.caption("Moteur SLOT V2 Branded - Prêt pour le terrain")
