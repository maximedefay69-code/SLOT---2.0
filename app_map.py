import streamlit as st
import requests, folium, joblib, math, pytz, numpy as np, pandas as pd
from streamlit_folium import st_folium
from datetime import datetime

st.set_page_config(page_title="SLOT 2.0 - Debug Bornes", layout="wide")

# --- 1. INTERFACE ---
st.title("📍 SLOT 2.0 - Testeur de Bornes GPS")
c1, c2, c3 = st.columns([1, 2, 4])
with c1: num_v = st.number_input("N°", value=11, step=1)
with c2: type_v = st.selectbox("Type", ["Rue", "Boulevard", "Avenue"])
with c3: nom_v = st.text_input("Nom de la voie", value="Voltaire")

# --- 2. LOGIQUE DE CALCUL ---
lat_pivot, lon_pivot = 48.8566, 2.3522
pt_A = None
target_found = False

if nom_v:
    # A. GÉOCODAGE DU LOGO (POINT B) via API Adresse
    geo_pivot = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
    if geo_pivot['features']:
        lon_pivot, lat_pivot = geo_pivot['features'][0]['geometry']['coordinates']
        arrdt = int(geo_pivot['features'][0]['properties']['postcode'][-2:])
        target_found = True

        # B. RECHERCHE DU POINT MINI (API PARIS)
        nom_api = nom_v.upper()
        type_api = "BD" if type_v == "Boulevard" else ("AV" if type_v == "Avenue" else "RUE")
        
        url_p = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {"where": f"nomvoie LIKE '{nom_api}' AND typevoie LIKE '{type_api}' AND arrond = {arrdt}", "limit": 50}
        
        try:
            res_p = requests.get(url_p, params=params).json()
            if 'results' in res_p and len(res_p['results']) > 0:
                df = pd.DataFrame(res_p['results'])
                
                # Correction des noms de colonnes
                col_min = 'nummin' if 'nummin' in df.columns else 'num_min'
                
                if col_min in df.columns:
                    df[col_min] = pd.to_numeric(df[col_min], errors='coerce')
                    # On prend la ligne qui a le numéro le plus petit
                    row_min = df.loc[df[col_min].idxmin()]
                    
                    # Extraction GPS du premier point de ce segment
                    geom = row_min.get('geom', {}).get('geometry', {})
                    if geom and 'coordinates' in geom:
                        coords = geom['coordinates']
                        # On force l'inversion [Lat, Lon]
                        if geom['type'] == 'LineString':
                            pt_A = [coords[0][1], coords[0][0]]
                        elif geom['type'] == 'MultiLineString':
                            pt_A = [coords[0][0][1], coords[0][0][0]]
        except:
            st.error("Erreur de connexion à l'API Paris")

# --- 3. RENDU CARTE ---
m = folium.Map(location=[lat_pivot, lon_pivot], zoom_start=18, tiles="cartodbpositron")

# Logo (Pivot)
URL_LOGO = "https://raw.githubusercontent.com/maximedefay69-code/SLOT---2.0/refs/heads/main/SLOT_img.png"
folium.Marker([lat_pivot, lon_pivot], icon=folium.CustomIcon(URL_LOGO, icon_size=(50, 50)), popup="MOI").add_to(m)

# Point Vert (Borne Mini trouvée dans l'API)
if pt_A:
    folium.CircleMarker(
        location=pt_A,
        radius=10,
        color="green",
        fill=True,
        fill_color="green",
        popup="Début de la rue (Num Min)"
    ).add_to(m)
    
    # On trace quand même un trait test entre les deux
    folium.PolyLine([pt_A, [lat_pivot, lon_pivot]], color="blue", weight=3, dash_array='5').add_to(m)

st_folium(m, width=1200, height=750, key="debug_v87")

if pt_A:
    st.success(f"✅ Point Mini trouvé ! Coordonnées : {pt_A}")
else:
    st.warning("❌ Aucun point mini détecté dans l'API pour cette rue.")
