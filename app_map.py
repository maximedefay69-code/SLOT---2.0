import streamlit as st
import requests, folium, joblib, math, pytz, numpy as np, pandas as pd
from streamlit_folium import st_folium
from datetime import datetime

# --- 1. CONFIGURATION & ASSETS ---
st.set_page_config(page_title="SLOT 2.0 - Scanner Dynamique", layout="wide")

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
        return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except:
        return None, None

model, prepro = load_assets()

# --- 2. FONCTIONS DE CALCUL ---
def get_weather(lat, lon):
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code").json()
        c = r['current']['weather_code']
        m = "Beau" if c in [0,1] else ("Nuageux" if c in [2,3,45,48] else "Pluie")
        return m, r['current']['temperature_2m']
    except: return "Beau", 18.0

def predire_rue_ia(nom_rue, nb_total, arrdt, mto, temp):
    if not model or not prepro: return 0
    now = datetime.now(pytz.timezone('Europe/Paris'))
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
    
    X_df = pd.DataFrame([X_dict])[['DATE','JOUR','HEURE','RUE','VILLE','TRAFIC','% PARKING OC','NBR PLACES','REVENUS / H','VEHICULES / H','MTO','TEMPERATURE','HEURE_MINUTES','HEURE_SIN','HEURE_COS']]
    try:
        occ = model.predict(prepro.transform(X_df))[0]
        return max(0, math.floor(nb_total * (1 - occ)))
    except: return 0

# --- 3. INTERFACE DE SAISIE ---
st.title("🛰️ SLOT 2.0 - Radar Terrain")

c1, c2, c3 = st.columns([1, 2, 4])
with c1: num_v = st.text_input("N°", "11")
with c2: type_v = st.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Voie"])
with c3: nom_v = st.text_input("Nom de la voie", placeholder="ex: Voltaire")

with st.sidebar:
    st.header("⚙️ Réglages")
    rayon_scan = st.slider("Rayon du radar (m)", 100, 1000, 400)
    taille_logo = st.slider("Taille icône SLOT", 30, 100, 60)
    st.write("---")
    st.caption("Données : Open Data Paris (V2.1)")

# --- 4. LOGIQUE DE LOCALISATION ---
lat_v, lon_v, zoom_v = 48.8566, 2.3522, 12
target_found = False
arrdt = 1

if nom_v:
    adresse = f"{num_v} {type_v} {nom_v}, Paris"
    geo_res = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={adresse}&limit=1").json()
    if geo_res['features']:
        lon_v, lat_v = geo_res['features'][0]['geometry']['coordinates']
        arrdt_str = geo_res['features'][0]['properties']['postcode'][-2:]
        arrdt = int(arrdt_str)
        mto, temp = get_weather(lat_v, lon_v)
        zoom_v, target_found = 18, True

# Création de la carte
m = folium.Map(location=[lat_v, lon_v], zoom_start=zoom_v, tiles="cartodbpositron")

# --- 5. RÉCUPÉRATION DYNAMIQUE DES SEGMENTS ---
if target_found:
    # On cherche TOUT dans un rayon autour du point GPS (on ignore les numéros de rue ici)
    url_p = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
    params = {
        "where": f"within_distance(geom, geom'POINT({lon_v} {lat_v})', {rayon_scan}m)",
        "limit": 100
    }
    
    try:
        res_p = requests.get(url_p, params=params).json()
        segments = res_p.get('results', [])
        
        if segments:
            for rue in segments:
                geom = rue.get('geom', {}).get('geometry', {})
                nom_r = rue.get('nomvoie', 'Inconnu')
                nb_t = rue.get('placal', 0)
                
                if geom and geom['type'] == 'LineString':
                    pts = [[p[1], p[0]] for p in geom['coordinates']]
                    
                    # --- IA PRÉDICTION ---
                    libres = predire_rue_ia(nom_r, nb_t, arrdt, mto, temp)
                    
                    # --- LOGIQUE COULEUR DEMANDÉE ---
                    if libres <= 1: couleur = "#e74c3c"   # ROUGE (0-1 place)
                    elif 1 < libres <= 3: couleur = "#f39c12" # ORANGE (2-3 places)
                    else: couleur = "#27ae60"             # VERT (> 3 places)
                    
                    folium.PolyLine(
                        pts, color=couleur, weight=8, opacity=0.8,
                        popup=f"<b>{nom_r}</b><br>Prédit : {libres} places libres"
                    ).add_to(m)
        else:
            st.warning("📡 Radar : Aucun segment de stationnement détecté dans cette zone.")

    except Exception as e:
        st.error(f"Erreur technique radar : {e}")

    # LOGO SLOT SUR TA POSITION
    URL_LOGO = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
    try:
        logo = folium.CustomIcon(URL_LOGO, icon_size=(taille_logo, taille_logo))
        folium.Marker([lat_v, lon_v], icon=logo, popup="Cible").add_to(m)
    except:
        folium.Marker([lat_v, lon_v]).add_to(m)

# --- 6. AFFICHAGE ---
st_folium(m, width=1200, height=750, key="slot_v81", returned_objects=[])
