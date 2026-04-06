import streamlit as st
import requests, folium, joblib, math, pytz, numpy as np, pandas as pd
from streamlit_folium import st_folium
from datetime import datetime

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="SLOT 2.0 - Expert Radar", 
    layout="wide"
)

# Style CSS pour mobile-first
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DONNÉES ET ASSETS ---
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
        model = joblib.load("modele_lightgbmDA.pkl")
        prepro = joblib.load("preprocessorDA.pkl")
        return model, prepro
    except:
        return None, None

model, prepro = load_assets()

# --- 3. FONCTIONS TECHNIQUES ---
def formater_type_voie(type_v):
    """Mappe le choix utilisateur vers les codes de l'API Paris"""
    mapping = {
        "Boulevard": "BD",
        "Avenue": "AV",
        "Rue": "RUE",
        "Place": "PCE",
        "Quai": "QUAI",
        "Impasse": "IMP",
        "Allée": "ALL",
        "Square": "SQ"
    }
    return mapping.get(type_v, "RUE")

def get_weather(lat, lon):
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code").json()
        c = r['current']['weather_code']
        m = "Beau" if c in [0,1] else "Nuageux"
        return m, r['current']['temperature_2m']
    except: return "Beau", 18.0

def predire_dispo_ia(nom_rue, nb_total, arrdt, mto, temp):
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

# --- 4. INTERFACE ---
st.title("🛰️ SLOT 2.0 - Radar de Précision")

c1, c2, c3 = st.columns([1, 2, 4])
with c1: num_v = st.text_input("N°", "11")
with c2: type_v = st.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse", "Square"])
with c3: nom_v = st.text_input("Nom de la voie", placeholder="ex: Voltaire")

# --- 5. LOGIQUE DE CALCUL ET GÉOMÉTRIE ---
lat_map, lon_map, zoom_map = 48.8566, 2.3522, 12
target_found = False
pts_rue = []
total_p = 0
arrdt_final = 1
mto_final, temp_final = "Beau", 18.0

if nom_v:
    # A. Localisation précise du point de recherche (Logo)
    adresse_full = f"{num_v} {type_v} {nom_v} Paris"
    geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={adresse_full}&limit=1").json()
    
    if geo['features']:
        f_geo = geo['features'][0]
        lon_map, lat_map = f_geo['geometry']['coordinates']
        arrdt_final = int(f_geo['properties']['postcode'][-2:])
        mto_final, temp_final = get_weather(lat_map, lon_map)
        zoom_map, target_found = 17, True

        # B. Requête API Paris avec Filtre de Type de Voie
        type_api = formater_type_voie(type_v)
        nom_api = nom_v.upper()
        
        url_p = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        # Filtre SQL strict : Nom exact ET Type exact
        params = {
            "where": f"nomvoie LIKE '{nom_api}' AND typevoie LIKE '{type_api}' AND arrond = {arrdt_final}",
            "limit": 100
        }
        
        try:
            res_p = requests.get(url_p, params=params).json()
            if 'results' in res_p:
                for r in res_p['results']:
                    reg = str(r.get('regpri', '')).upper()
                    if any(x in reg for x in ["PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT"]):
                        total_p += r.get('placal', 0)
                        geom = r.get('geom', {}).get('geometry', {})
                        if geom and geom['type'] == 'LineString':
                            # Extraction de tous les points GPS pour tracer la ligne
                            pts_rue.extend([[p[1], p[0]] for p in geom['coordinates']])
        except: pass

# --- 6. CONSTRUCTION DE LA CARTE ---
m = folium.Map(location=[lat_map, lon_map], zoom_start=zoom_map, tiles="cartodbpositron")

# Affichage de la ligne de rue colorée par l'IA
if pts_rue and total_p > 0:
    # On trie les points par latitude pour définir les extrémités Sud/Nord
    pts_rue.sort(key=lambda x: x[0])
    start_point = pts_rue[0]
    end_point = pts_rue[-1]

    # Calcul IA
    libres = predire_dispo_ia(nom_v, total_p, arrdt_final, mto_final, temp_final)
    
    # Code couleur Louis
    if libres <= 1: coul = "#e74c3c"   # Rouge
    elif 1 < libres <= 3: coul = "#f39c12" # Orange
    else: coul = "#27ae60"             # Vert

    # Tracé de la ligne maîtresse ultra-visible
    folium.PolyLine(
        [start_point, end_point], 
        color=coul, 
        weight=15, 
        opacity=0.85, 
        popup=f"<b>{type_v} {nom_v.upper()}</b><br>Places : {total_p} | IA : {libres} libres"
    ).add_to(m)

# Ajout du logo SLOT
URL_LOGO = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
folium.Marker(
    [lat_map, lon_map], 
    icon=folium.CustomIcon(URL_LOGO, icon_size=(60, 60)),
    popup="Ma cible"
).add_to(m)

# --- 7. RENDU FINAL ---
st_folium(m, width=1200, height=750, key="slot_v83")

if target_found:
    st.success(f"📍 Analyse active : **{type_v.upper()} {nom_v.upper()}** ({total_p} places)")
else:
    st.info("👋 Bonjour Louis ! Entre une adresse pour démarrer le scan.")
