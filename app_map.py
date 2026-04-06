import streamlit as st
import requests, folium, joblib, math, pytz, numpy as np, pandas as pd
from streamlit_folium import st_folium
from datetime import datetime

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="SLOT 2.0 - Radar V86", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DONNÉES SOCIO ET ASSETS ---
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

JOURS_FR = {
    "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
    "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
}

@st.cache_resource
def load_assets():
    try:
        m = joblib.load("modele_lightgbmDA.pkl")
        p = joblib.load("preprocessorDA.pkl")
        return m, p
    except: return None, None

model, prepro = load_assets()

# --- 3. FONCTIONS TECHNIQUES ---
def get_weather(lat, lon):
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code").json()
        c = r['current']['weather_code']
        mto = "Beau" if c in [0,1] else ("Nuageux" if c in [2,3,45,48] else "Pluie")
        return mto, r['current']['temperature_2m']
    except: return "Beau", 18.0

def predire_dispo_ia(nom_rue, nb_total, arrdt, mto, temp):
    if not model or not prepro: return 0
    now = datetime.now(pytz.timezone('Europe/Paris'))
    minutes = now.hour * 60 + now.minute
    socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
    
    X_dict = {
        'DATE': now.strftime("%d/%m/%Y"), 'JOUR': JOURS_FR.get(now.strftime("%A")), 'HEURE': now.strftime("%H:%M"),
        'RUE': str(nom_rue).upper(), 'VILLE': "Paris", 'TRAFIC': 0.0, '% PARKING OC': 0.5,
        'NBR PLACES': nb_total, 'REVENUS / H': socio["REV_M"], 'VEHICULES / H': socio["VEH"],
        'MTO': mto, 'TEMPERATURE': temp, 'HEURE_MINUTES': minutes,
        'HEURE_SIN': np.sin(2 * np.pi * minutes / 1440), 'HEURE_COS': np.cos(2 * np.pi * minutes / 1440)
    }
    X_df = pd.DataFrame([X_dict])[['DATE','JOUR','HEURE','RUE','VILLE','TRAFIC','% PARKING OC','NBR PLACES','REVENUS / H','VEHICULES / H','MTO','TEMPERATURE','HEURE_MINUTES','HEURE_SIN','HEURE_COS']]
    try:
        occ = model.predict(prepro.transform(X_df))[0]
        return max(0, math.floor(nb_total * (1 - occ)))
    except: return 0

# --- 4. INTERFACE ---
st.title("🛰️ SLOT 2.0 - Radar Haute Sécurité")

c1, c2, c3 = st.columns([1, 2, 4])
with c1: num_v = st.number_input("N°", value=11, step=1)
with c2: type_v = st.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai"])
with c3: nom_v = st.text_input("Nom de la voie", value="Voltaire")

# --- 5. LOGIQUE DE CALCUL ET GÉOMÉTRIE ---
lat_pivot, lon_pivot = 48.8566, 2.3522
pt_A, pt_C = None, None
total_p = 0
target_found = False

if nom_v:
    # A. GÉOCODAGE DU LOGO (POINT B)
    geo_pivot = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
    if geo_pivot['features']:
        lon_pivot, lat_pivot = geo_pivot['features'][0]['geometry']['coordinates']
        arrdt = int(geo_pivot['features'][0]['properties']['postcode'][-2:])
        mto, temp = get_weather(lat_pivot, lon_pivot)
        target_found = True

        # B. COLLECTE DATA (API PARIS)
        nom_api = nom_v.upper()
        type_api = "BD" if type_v == "Boulevard" else ("AV" if type_v == "Avenue" else "RUE")
        
        url_p = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {"where": f"nomvoie LIKE '{nom_api}' AND typevoie LIKE '{type_api}' AND arrond = {arrdt}", "limit": 100}
        
        try:
            res_p = requests.get(url_p, params=params).json()
            if 'results' in res_p and len(res_p['results']) > 0:
                df = pd.DataFrame(res_p['results'])
                
                # Détection dynamique des colonnes pour éviter le KeyError
                col_min = 'nummin' if 'nummin' in df.columns else ('num_min' if 'num_min' in df.columns else None)
                col_max = 'nummax' if 'nummax' in df.columns else ('num_max' if 'num_max' in df.columns else None)
                col_reg = 'regpri' if 'regpri' in df.columns else ('regime' if 'regime' in df.columns else None)

                if col_reg:
                    mask = df[col_reg].str.upper().str.contains("PAYANT ROTATIF|PAYANT MIXTE|GRATUIT", na=False)
                    df_filtered = df[mask].copy()
                    total_p = df_filtered['placal'].sum()

                    if not df_filtered.empty and col_min and col_max:
                        df_filtered[col_min] = pd.to_numeric(df_filtered[col_min], errors='coerce')
                        df_filtered[col_max] = pd.to_numeric(df_filtered[col_max], errors='coerce')
                        
                        idx_min = df_filtered[col_min].idxmin()
                        idx_max = df_filtered[col_max].idxmax()
                        
                        def extract_latlon(record):
                            g = record.get('geom', {}).get('geometry', {})
                            if g and 'coordinates' in g:
                                pts = g['coordinates']
                                return [pts[0][1], pts[0][0]] if g['type'] == 'LineString' else [pts[0][0][1], pts[0][0][0]]
                            return None

                        pt_A = extract_latlon(df_filtered.loc[idx_min])
                        pt_C = extract_latlon(df_filtered.loc[idx_max])
        except Exception as e:
            st.error(f"Erreur technique : {e}")

# --- 6. RENDU CARTE ---
m = folium.Map(location=[lat_pivot, lon_pivot], zoom_start=17, tiles="cartodbpositron")

if target_found and total_p > 0:
    libres = predire_dispo_ia(nom_v, total_p, arrdt, mto, temp)
    coul = "#e74c3c" if libres <= 1 else ("#f39c12" if libres <= 3 else "#27ae60")

    if pt_A:
        folium.PolyLine([pt_A, [lat_pivot, lon_pivot]], color=coul, weight=12, opacity=0.85).add_to(m)
    if pt_C:
        folium.PolyLine([[lat_pivot, lon_pivot], pt_C], color=coul, weight=12, opacity=0.85).add_to(m)

URL_LOGO = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
folium.Marker([lat_pivot, lon_pivot], icon=folium.CustomIcon(URL_LOGO, icon_size=(60, 60))).add_to(m)

st_folium(m, width=1200, height=750, key="slot_v86_final")
if target_found:
    st.info(f"📊 {total_p} places analysées | 🤖 IA : ~{libres} places libres.")
