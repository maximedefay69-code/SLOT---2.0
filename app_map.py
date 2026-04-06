import streamlit as st
import requests, folium, joblib, math, pytz, numpy as np, pandas as pd
from streamlit_folium import st_folium
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="SLOT 2.0 - V97 Fix", layout="wide")

DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 11: {"REV_M": 2583, "VEH": 0.24}, # ... Garde tes data complètes ici
}

@st.cache_resource
def load_assets():
    try: return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except: return None, None

model, prepro = load_assets()

# --- 2. INTERFACE ---
st.title("🛰️ SLOT 2.0 - Debug Localisation")
c1, c2, c3 = st.columns([1, 2, 4])
with c1: num_v = st.number_input("N°", value=11)
with c2: type_v = st.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai"])
with c3: nom_v = st.text_input("Nom", value="Voltaire")

# --- 3. LOGIQUE ---
lat_piv, lon_piv = 48.8566, 2.3522
pt_A, pt_C = None, None
total_p, libres = 0, 0
target_found = False

if nom_v:
    # A. PIVOT
    geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
    if geo['features']:
        lon_piv, lat_piv = geo['features'][0]['geometry']['coordinates']
        arrdt = int(geo['features'][0]['properties']['postcode'][-2:])
        target_found = True

        # B. NETTOYAGE DU NOM POUR L'API PARIS (Crucial)
        # On enlève les espaces inutiles et on met en majuscules
        nom_propre = nom_v.strip().upper()
        
        # C. DATA PARIS
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        # On utilise "contains" plutôt que "suggest" pour être plus large
        params = {
            "where": f"nomvoie LIKE '*{nom_propre}*' AND arrond = {arrdt}",
            "limit": 100
        }
        
        try:
            res = requests.get(url, params=params).json()
            if 'results' in res and len(res['results']) > 0:
                df = pd.DataFrame(res['results'])
                
                # Vérification colonnes
                c_min = next((c for c in ['nummin', 'num_min'] if c in df.columns), None)
                c_max = next((c for c in ['nummax', 'num_max'] if c in df.columns), None)

                if c_min and c_max:
                    df[c_min] = pd.to_numeric(df[c_min], errors='coerce')
                    df[c_max] = pd.to_numeric(df[c_max], errors='coerce')
                    df = df.dropna(subset=['geo_point_2d'])
                    
                    total_p = int(df['placal'].sum())
                    
                    # Simulation IA (Remets ta fonction predire_dispo_ia ici)
                    libres = math.ceil(total_p * 0.12) 

                    if not df.empty:
                        row_min = df.loc[df[c_min].idxmin()]
                        pt_A = [row_min['geo_point_2d']['lat'], row_min['geo_point_2d']['lon']]
                        row_max = df.loc[df[c_max].idxmax()]
                        pt_C = [row_max['geo_point_2d']['lat'], row_max['geo_point_2d']['lon']]
            else:
                st.warning(f"⚠️ Aucun tronçon trouvé pour '{nom_propre}' dans le {arrdt}e. Vérifie l'orthographe.")
        except Exception as e:
            st.error(f"Erreur API : {e}")

# --- 4. CARTE ---
m = folium.Map(location=[lat_piv, lon_piv], zoom_start=18, tiles="cartodbpositron")

if target_found:
    coul = "#27ae60" if libres > 2 else "#e74c3c"
    if pt_A: folium.PolyLine([pt_A, [lat_piv, lon_piv]], color=coul, weight=12).add_to(m)
    if pt_C: folium.PolyLine([[lat_piv, lon_piv], pt_C], color=coul, weight=12).add_to(m)

URL_LOGO = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
folium.Marker([lat_piv, lon_piv], icon=folium.CustomIcon(URL_LOGO, icon_size=(60, 60))).add_to(m)

st_folium(m, width=1200, height=750, key="v97")

if target_found and total_p > 0:
    st.success(f"✅ {total_p} places trouvées dans la rue. IA : {libres} libres.")
