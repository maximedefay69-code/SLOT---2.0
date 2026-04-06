import streamlit as st
import requests, folium, joblib, math, pytz, numpy as np, pandas as pd
from streamlit_folium import st_folium
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="SLOT 2.0 - V95 Master", layout="wide")

# Données Socio-économiques par Arrondissement (Features IA)
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

# --- 2. MOTEUR DYNAMIQUE ---
def get_weather(lat, lon):
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code").json()
        c = r['current']['weather_code']
        mto = "Beau" if c in [0,1] else ("Nuageux" if c in [2,3,45,48] else "Pluie")
        return mto, r['current']['temperature_2m']
    except: return "Beau", 18.0

def predire_dispo_ia(nom_rue, nb_total, arrdt, mto, temp):
    if not model or not prepro: return 0
    tz = pytz.timezone('Europe/Paris')
    now = datetime.now(tz)
    minutes = now.hour * 60 + now.minute
    socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
    
    X_dict = {
        'DATE': now.strftime("%d/%m/%Y"),
        'JOUR': JOURS_FR.get(now.strftime("%A")),
        'HEURE': now.strftime("%H:%M"),
        'RUE': str(nom_rue).upper(),
        'VILLE': "Paris",
        'TRAFIC': 0.0,
        '% PARKING OC': 0.5,
        'NBR PLACES': nb_total,
        'REVENUS / H': socio["REV_M"],
        'VEHICULES / H': socio["VEH"],
        'MTO': mto,
        'TEMPERATURE': temp,
        'HEURE_MINUTES': minutes,
        'HEURE_SIN': np.sin(2 * np.pi * minutes / 1440),
        'HEURE_COS': np.cos(2 * np.pi * minutes / 1440)
    }
    
    cols = ['DATE','JOUR','HEURE','RUE','VILLE','TRAFIC','% PARKING OC','NBR PLACES','REVENUS / H','VEHICULES / H','MTO','TEMPERATURE','HEURE_MINUTES','HEURE_SIN','HEURE_COS']
    X_df = pd.DataFrame([X_dict])[cols]
    try:
        occ = model.predict(prepro.transform(X_df))[0]
        return max(0, math.floor(nb_total * (1 - occ)))
    except: return 0

# --- 3. INTERFACE ---
st.title("🛰️ SLOT 2.0 - Radar Master V95")

c1, c2, c3 = st.columns([1, 2, 4])
with c1: num_v = st.number_input("N°", value=11)
with c2: type_v = st.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai"])
with c3: nom_v = st.text_input("Nom de la voie", value="Voltaire")

# --- 4. LOGIQUE GÉOGRAPHIQUE (EXTRACTEUR UNIVERSEL) ---
lat_piv, lon_piv = 48.8566, 2.3522
pt_MIN, pt_MAX = None, None
total_p, libres = 0, 0
target_found = False

if nom_v:
    # A. GÉOCODAGE DU LOGO (POINT PIVOT)
    geo_pivot = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
    if geo_pivot['features']:
        lon_piv, lat_piv = geo_pivot['features'][0]['geometry']['coordinates']
        arrdt = int(geo_pivot['features'][0]['properties']['postcode'][-2:])
        mto, temp = get_weather(lat_piv, lon_piv)
        target_found = True

        # B. COLLECTE DES TRONÇONS (API PARIS)
        url_p = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {"where": f"suggest(nomvoie, '{nom_v.upper()}') AND arrond = {arrdt}", "limit": 100}
        
        try:
            res_p = requests.get(url_p, params=params).json()
            if 'results' in res_p and len(res_p['results']) > 0:
                df = pd.DataFrame(res_p['results'])
                
                # Identification des colonnes
                c_min = next((c for c in ['nummin', 'num_min'] if c in df.columns), None)
                c_max = next((c for c in ['nummax', 'num_max'] if c in df.columns), None)
                c_reg = next((c for c in ['regpri', 'regime'] if c in df.columns), None)

                if c_reg:
                    mask = df[c_reg].str.upper().str.contains("PAYANT|GRATUIT", na=False)
                    df_f = df[mask].copy()
                    total_p = df_f['placal'].sum()
                    libres = predire_dispo_ia(nom_v, total_p, arrdt, mto, temp)

                    # FONCTION D'EXTRACTION GPS TOUT-TERRAIN
                    def get_coords(record):
                        if 'geo_point_2d' in record and isinstance(record['geo_point_2d'], dict):
                            return [record['geo_point_2d']['lat'], record['geo_point_2d']['lon']]
                        if 'geom' in record and 'geometry' in record['geom']:
                            g = record['geom']['geometry']
                            if g['type'] == 'LineString':
                                return [g['coordinates'][0][1], g['coordinates'][0][0]]
                            elif g['type'] == 'MultiLineString':
                                return [g['coordinates'][0][0][1], g['coordinates'][0][0][0]]
                        return None

                    if not df_f.empty and c_min and c_max:
                        df_f[c_min] = pd.to_numeric(df_f[c_min], errors='coerce')
                        df_f[c_max] = pd.to_numeric(df_f[c_max], errors='coerce')
                        
                        row_start = df_f.loc[df_f[c_min].idxmin()]
                        row_end = df_f.loc[df_f[c_max].idxmax()]
                        
                        pt_MIN = get_coords(row_start)
                        pt_MAX = get_coords(row_end)
        except Exception as e:
            st.error(f"Erreur technique : {e}")

# --- 5. RENDU CARTE ---
m = folium.Map(location=[lat_piv, lon_piv], zoom_start=18, tiles="cartodbpositron")

if target_found:
    coul = "#e74c3c" if libres <= 1 else ("#f39c12" if libres <= 3 else "#27ae60")
    
    if pt_MIN:
        folium.PolyLine([pt_MIN, [lat_piv, lon_piv]], color=coul, weight=12, opacity=0.8).add_to(m)
    if pt_MAX:
        folium.PolyLine([[lat_piv, lon_piv], pt_MAX], color=coul, weight=12, opacity=0.8).add_to(m)

URL_LOGO = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
folium.Marker([lat_piv, lon_piv], icon=folium.CustomIcon(URL_LOGO, icon_size=(60, 60))).add_to(m)

st_folium(m, width=1200, height=750, key="v95_master")
if target_found:
    st.info(f"📊 {nom_v.upper()} | {total_p} places | 🌡️ {temp}°C | 🌦️ {mto}")
    st.success(f"🤖 IA SLOT : **{libres} places libres** estimées à {datetime.now(pytz.timezone('Europe/Paris')).strftime('%H:%M')}.")
