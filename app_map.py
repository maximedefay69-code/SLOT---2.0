import streamlit as st
import requests, folium, pandas as pd
from streamlit_folium import st_folium

st.set_page_config(page_title="SLOT 2.0 - Scanner GPS", layout="wide")

# --- 1. INTERFACE ---
st.title("🔍 SLOT 2.0 - Scanner de Coordonnées Paris")
c1, c2, c3 = st.columns([1, 2, 4])
with c1: num_v = st.number_input("N°", value=11)
with c2: type_v = st.selectbox("Type", ["Rue", "Boulevard", "Avenue"])
with c3: nom_v = st.text_input("Nom", value="Voltaire")

# --- 2. LOGIQUE ---
lat_pivot, lon_pivot = 48.8566, 2.3522
debug_data = []

if nom_v:
    # A. LOGO (API Adresse - Toujours OK)
    geo_pivot = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
    if geo_pivot['features']:
        lon_pivot, lat_pivot = geo_pivot['features'][0]['geometry']['coordinates']
        arrdt = int(geo_pivot['features'][0]['properties']['postcode'][-2:])

        # B. SCAN API PARIS
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {
            "where": f"suggest(nomvoie, '{nom_v.upper()}') AND arrond = {arrdt}",
            "limit": 20
        }
        
        try:
            res = requests.get(url, params=params).json()
            if 'results' in res:
                for r in res['results']:
                    # On cherche les coordonnées PARTOUT (geom ou geo_point_2d)
                    lat, lon = None, None
                    
                    # Test 1 : geo_point_2d (Le plus probable pour un point simple)
                    if 'geo_point_2d' in r:
                        lat = r['geo_point_2d'].get('lat')
                        lon = r['geo_point_2d'].get('lon')
                    
                    # Test 2 : geom (Si c'est une géométrie complexe)
                    elif 'geom' in r:
                        g = r['geom'].get('geometry', {})
                        if g.get('type') == 'LineString':
                            lat, lon = g['coordinates'][0][1], g['coordinates'][0][0]
                    
                    if lat and lon:
                        debug_data.append({"lat": lat, "lon": lon, "num": r.get('nummin', '?')})
        except: st.error("Erreur API")

# --- 3. CARTE ---
m = folium.Map(location=[lat_pivot, lon_pivot], zoom_start=18, tiles="cartodbpositron")

# Ton LOGO
folium.Marker([lat_pivot, lon_pivot], icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

# Les points trouvés dans l'API Paris
for p in debug_data:
    folium.CircleMarker(
        location=[p['lat'], p['lon']],
        radius=8,
        color="blue",
        fill=True,
        popup=f"Num: {p['num']}"
    ).add_to(m)

st_folium(m, width=1200, height=600, key="v88")

# --- 4. DEBUG TEXTUEL ---
if debug_data:
    st.write(f"✅ {len(debug_data)} points détectés dans l'API Paris !")
    st.dataframe(pd.DataFrame(debug_data))
else:
    st.warning("⚠️ L'API Paris ne renvoie aucune coordonnée GPS pour cette recherche.")
