import streamlit as st
import requests, folium, joblib, math, pytz, numpy as np, pandas as pd
from streamlit_folium import st_folium
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="SLOT 2.0 - Full Dynamique", layout="wide")

# Données Socio-éco (Features du modèle)
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

# --- 2. MOTEUR DE CALCUL DYNAMIQUE ---
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
st.title("🛰️ SLOT 2.0 - Radar Temps Réel")
c1, c2, c3 = st.columns([1, 2, 4])
with c1: num_v = st.number_input("N°", value=11)
with c2: type_v = st.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai"])
with c3: nom_v = st.text_input("Nom", value="Voltaire")

# --- 4. LOGIQUE GEOGRAPHIQUE ---
lat_pivot, lon_pivot = 48.8566, 2.3522
points_avant, points_apres = [], []
total_p, libres = 0, 0
target_found = False

if nom_v:
    # A. PIVOT (Logo)
    geo_pivot = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
    if geo_pivot['features']:
        lon_pivot, lat_pivot = geo_pivot['features'][0]['geometry']['coordinates']
        arrdt = int(geo_pivot['features'][0]['properties']['postcode'][-2:])
        mto, temp = get_weather(lat_pivot, lon_pivot)
        target_found = True

        # B. RÉCUPÉRATION ET TRI DES POINTS
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {"where": f"suggest(nomvoie, '{nom_v.upper()}') AND arrond = {arrdt}", "limit": 100}
        
        res = requests.get(url, params=params).json()
        if 'results' in res:
            data_list = []
            for r in res['results']:
                # On ne prend que les places valides
                reg = str(r.get('regpri', '')).upper()
                if any(x in reg for x in ["PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT"]):
                    # On utilise la colonne geo_point_2d qui fonctionne !
                    if 'geo_point_2d' in r:
                        lat, lon = r['geo_point_2d'].get('lat'), r['geo_point_2d'].get('lon')
                        n_min = pd.to_numeric(r.get('nummin', 0), errors='coerce')
                        data_list.append({'lat': lat, 'lon': lon, 'num': n_min, 'places': r.get('placal', 0)})

            if data_list:
                df_rue = pd.DataFrame(data_list).sort_values('num')
                total_p = df_rue['places'].sum()
                
                # C. PRÉDICTION IA (Dynamique)
                libres = predire_dispo_ia(nom_v, total_p, arrdt, mto, temp)
                
                # D. SÉPARATION DES SEGMENTS
                points_avant = df_rue[df_rue['num'] <= num_v][['lat', 'lon']].values.tolist()
                points_apres = df_rue[df_rue['num'] >= num_v][['lat', 'lon']].values.tolist()

# --- 5. CARTE ---
m = folium.Map(location=[lat_pivot, lon_pivot], zoom_start=18, tiles="cartodbpositron")

if target_found:
    # Code Couleur IA
    coul = "#e74c3c" if libres <= 1 else ("#f39c12" if libres <= 3 else "#27ae60")
    
    # Tracé 1 : Avant Pivot
    if points_avant:
        folium.PolyLine(points_avant + [[lat_pivot, lon_pivot]], color=coul, weight=12, opacity=0.8).add_to(m)
    # Tracé 2 : Après Pivot
    if points_apres:
        folium.PolyLine([[lat_pivot, lon_pivot]] + points_apres, color=coul, weight=12, opacity=0.8).add_to(m)

# Logo Louis
URL_LOGO = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
folium.Marker([lat_pivot, lon_pivot], icon=folium.CustomIcon(URL_LOGO, icon_size=(60, 60))).add_to(m)

st_folium(m, width=1200, height=750, key="v90_full")

# Affichage des infos dynamiques
if target_found:
    st.info(f"🕒 Heure : {datetime.now(pytz.timezone('Europe/Paris')).strftime('%H:%M')} | 🌡️ {temp}°C | 🌦️ {mto}")
    st.success(f"📊 Rue {nom_v.upper()} : {total_p} places au total. IA : **{libres} places libres** estimées.")
