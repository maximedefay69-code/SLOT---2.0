import streamlit as st
import requests, folium, joblib, math, pytz, numpy as np, pandas as pd
from streamlit_folium import st_folium
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="SLOT 2.0 - Lignes Épurées", layout="wide")

DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

@st.cache_resource
def load_assets():
    try: return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except: return None, None

model, prepro = load_assets()

# --- 2. FONCTIONS DYNAMIQUES ---
def get_weather(lat, lon):
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code").json()
        c = r['current']['weather_code']
        mto = "Beau" if c in [0,1] else "Nuageux"
        return mto, r['current']['temperature_2m']
    except: return "Beau", 18.0

def predire_dispo_ia(nom_rue, nb_total, arrdt, mto, temp):
    if not model or not prepro: return 0
    now = datetime.now(pytz.timezone('Europe/Paris'))
    socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
    # ... (Logique IA identique à V90)
    return math.floor(nb_total * 0.15) # Simplification pour le bloc, remets ta fonction complète si besoin

# --- 3. INTERFACE ---
st.title("🛰️ SLOT 2.0 - Precision Vector")
c1, c2, c3 = st.columns([1, 2, 4])
with c1: num_v = st.number_input("N°", value=11)
with c2: type_v = st.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place"])
with c3: nom_v = st.text_input("Nom", value="Voltaire")

# --- 4. LOGIQUE DES EXTRÊMES ---
lat_pivot, lon_pivot = 48.8566, 2.3522
pt_MIN, pt_MAX = None, None
total_p, libres = 0, 0
target_found = False

if nom_v:
    # A. PIVOT
    geo_pivot = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
    if geo_pivot['features']:
        lon_pivot, lat_pivot = geo_pivot['features'][0]['geometry']['coordinates']
        arrdt = int(geo_pivot['features'][0]['properties']['postcode'][-2:])
        mto, temp = get_weather(lat_pivot, lon_pivot)
        target_found = True

        # B. RECHERCHE DES BORNES
        nom_api = nom_v.upper()
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {"where": f"suggest(nomvoie, '{nom_api}') AND arrond = {arrdt}", "limit": 100}
        
        res = requests.get(url, params=params).json()
        if 'results' in res and len(res['results']) > 0:
            df = pd.DataFrame(res['results'])
            # Nettoyage colonnes
            col_min = 'nummin' if 'nummin' in df.columns else 'num_min'
            col_max = 'nummax' if 'nummax' in df.columns else 'num_max'
            
            df[col_min] = pd.to_numeric(df[col_min], errors='coerce')
            df[col_max] = pd.to_numeric(df[col_max], errors='coerce')
            
            # Calcul total places
            total_p = df['placal'].sum()
            libres = predire_dispo_ia(nom_v, total_p, arrdt, mto, temp)
            
            # C. EXTRACTION UNIQUE DES EXTRÊMES
            # Le point du numéro le plus bas
            row_min = df.loc[df[col_min].idxmin()]
            if 'geo_point_2d' in row_min:
                pt_MIN = [row_min['geo_point_2d']['lat'], row_min['geo_point_2d']['lon']]
            
            # Le point du numéro le plus haut
            row_max = df.loc[df[col_max].idxmax()]
            if 'geo_point_2d' in row_max:
                pt_MAX = [row_max['geo_point_2d']['lat'], row_max['geo_point_2d']['lon']]

# --- 5. CARTE ---
m = folium.Map(location=[lat_pivot, lon_pivot], zoom_start=18, tiles="cartodbpositron")

if target_found:
    coul = "#e74c3c" if libres <= 1 else ("#f39c12" if libres <= 3 else "#27ae60")
    
    # Segment 1 : Du Min jusqu'au Pivot (Logo)
    if pt_MIN:
        folium.PolyLine([pt_MIN, [lat_pivot, lon_pivot]], color=coul, weight=12, opacity=0.85).add_to(m)
    
    # Segment 2 : Du Pivot (Logo) jusqu'au Max
    if pt_MAX:
        folium.PolyLine([[lat_pivot, lon_pivot], pt_MAX], color=coul, weight=12, opacity=0.85).add_to(m)

# Logo
URL_LOGO = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
folium.Marker([lat_pivot, lon_pivot], icon=folium.CustomIcon(URL_LOGO, icon_size=(60, 60))).add_to(m)

st_folium(m, width=1200, height=750, key="v91_clean")
if target_found:
    st.success(f"✅ Analyse épurée : {total_p} places | IA : **{libres} places libres**.")
