
import streamlit as st
import osmnx as ox
import unicodedata
import requests
import folium
import joblib
import math
import pytz
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from streamlit_folium import st_folium
from datetime import datetime


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SLOT 2.0 - Full Dynamique",
    layout="wide"
)

URL_STATIONNEMENT = (
    "https://opendata.paris.fr/api/explore/v2.1/"
    "catalog/datasets/stationnement-sur-voie-publique-emprises/records"
)

URL_LOGO = (
    "https://raw.githubusercontent.com/maximedefay69-code/"
    "SLOT---2.0/refs/heads/main/SLOT_img.png"
)

DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32},
    2: {"REV_M": 2666, "VEH": 0.28},
    3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26},
    5: {"REV_M": 3166, "VEH": 0.35},
    6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42},
    8: {"REV_M": 4000, "VEH": 0.40},
    9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22},
    11: {"REV_M": 2583, "VEH": 0.24},
    12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35},
    14: {"REV_M": 2583, "VEH": 0.36},
    15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45},
    17: {"REV_M": 3083, "VEH": 0.37},
    18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22},
    20: {"REV_M": 1916, "VEH": 0.23},
}

JOURS_FR = {
    "Monday": "Lundi",
    "Tuesday": "Mardi",
    "Wednesday": "Mercredi",
    "Thursday": "Jeudi",
    "Friday": "Vendredi",
    "Saturday": "Samedi",
    "Sunday": "Dimanche",
}


# ============================================================
# 2. CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def load_assets():
    try:
        modele = joblib.load("modele_lightgbmDA.pkl")
        preprocesseur = joblib.load("preprocessorDA.pkl")
        return modele, preprocesseur
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        return None, None


model, prepro = load_assets()


# ============================================================
# 3. OUTILS
# ============================================================

def normaliser_nom_rue(nom):
    """Normalise un nom de rue pour faciliter les comparaisons."""
    nom = str(nom).strip().lower()
    nom = unicodedata.normalize("NFKD", nom)
    nom = "".join(
        c for c in nom
        if not unicodedata.combining(c)
    )
    return " ".join(nom.split())


def couleur_disponibilite(nb_libres):
    """Applique les seuils de couleur de SLOT 2.0."""
    if nb_libres is None:
        return "#95a5a6"

    if nb_libres <= 1:
        return "#e74c3c"

    if nb_libres <= 3:
        return "#f39c12"

    return "#27ae60"


def get_weather(lat, lon):
    """Récupère la météo actuelle via Open-Meteo."""
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code",
            },
            timeout=10,
        )
        response.raise_for_status()

        current = response.json()["current"]
        code = current["weather_code"]

        meteo = (
            "Beau" if code in [0, 1]
            else "Nuageux" if code in [2, 3, 45, 48]
            else "Pluie"
        )

        return meteo, current["temperature_2m"]

    except Exception as e:
        print(f"Erreur météo : {e}")
        return None, None


# ============================================================
# 4. GÉOMÉTRIE DE LA RUE RECHERCHÉE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def obtenir_geometrie_rue(lat, lon, nom_rue, type_rue="Rue"):
    """
    Conserve la recherche OSMnx de la rue principale.
    Retourne les lignes au format [[lat, lon], ...].
    """
    try:
        G = ox.graph.graph_from_point(
            (lat, lon),
            dist=800,
            dist_type="bbox",
            network_type="drive",
            simplify=True,
            retain_all=True,
        )

        _, edges = ox.convert.graph_to_gdfs(G)

        if "name" not in edges.columns:
            return []

        variantes = {
            normaliser_nom_rue(f"{type_rue} {nom_rue}"),
            normaliser_nom_rue(nom_rue),
        }

        def correspond(nom_osm):
            noms = nom_osm if isinstance(nom_osm, list) else [nom_osm]

            for nom in noms:
                if not nom:
                    continue

                nom_norm = normaliser_nom_rue(nom)

                for variante in variantes:
                    if (
                        nom_norm == variante
                        or nom_norm.endswith(variante)
                        or variante.endswith(nom_norm)
                    ):
                        return True

            return False

        rues = edges[edges["name"].apply(correspond)]
        geometries = []

        for geom in rues.geometry:
            if geom is None or geom.is_empty:
                continue

            lignes = (
                list(geom.geoms)
                if geom.geom_type == "MultiLineString"
                else [geom]
            )

            for ligne in lignes:
                if ligne.geom_type != "LineString":
                    continue

                coords = [
                    [lat_pt, lon_pt]
                    for lon_pt, lat_pt in ligne.coords
                ]

                if len(coords) >= 2:
                    geometries.append(coords)

        return geometries

    except Exception as e:
        print(f"Erreur géométrie rue : {type(e).__name__}: {e}")
        return []


# ============================================================
# 5. RÉCUPÉRATION DES TRONÇONS VOISINS
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def obtenir_troncons_voisins(lat, lon, rayon=300):
    """
    Récupère les tronçons routiers nommés dans la zone.
    Les distances sont calculées dans une projection métrique.
    """
    try:
        G = ox.graph.graph_from_point(
            (lat, lon),
            dist=rayon,
            dist_type="bbox",
            network_type="drive",
            simplify=True,
            retain_all=True,
        )

        _, edges = ox.convert.graph_to_gdfs(G)

        if edges.empty or "name" not in edges.columns:
            return []

        # Projection métrique pour mesurer les distances en mètres.
        edges_m = edges.to_crs(epsg=2154)

        centre_geo = gpd.GeoSeries(
            [Point(lon, lat)],
            crs="EPSG:4326",
        )
        centre_m = centre_geo.to_crs(epsg=2154).iloc[0]

        troncons = []

        for _, row in edges_m.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            if geom.distance(centre_m) > rayon:
                continue

            nom = row.get("name")

            if not nom:
                continue

            noms = nom if isinstance(nom, list) else [nom]
            nom_affiche = str(noms[0])

            lignes = (
                list(geom.geoms)
                if geom.geom_type == "MultiLineString"
                else [geom]
            )

            for ligne in lignes:
                if ligne.geom_type != "LineString":
                    continue

                coords = [
                    [lat_pt, lon_pt]
                    for lon_pt, lat_pt in ligne.coords
                ]

                if len(coords) >= 2:
                    troncons.append({
                        "nom": nom_affiche,
                        "coords": coords,
                    })

        return troncons

    except Exception as e:
        print(f"Erreur tronçons voisins : {e}")
        return []


# ============================================================
# 6. DONNÉES DE STATIONNEMENT PAR RUE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def obtenir_places_par_rue(arrdt):
    """
    Récupère les emprises de stationnement d'un arrondissement.
    Utilise la pagination pour parcourir les résultats disponibles.
    """
    totaux = {}
    offset = 0
    limite = 100
    maximum_pages = 50

    for _ in range(maximum_pages):
        params = {
            "where": f"arrond = {arrdt}",
            "limit": limite,
            "offset": offset,
        }

        try:
            response = requests.get(
                URL_STATIONNEMENT,
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            resultat = response.json()

        except Exception as e:
            print(f"Erreur données stationnement : {e}")
            break

        records = resultat.get("results", [])

        if not records:
            break

        for r in records:
            reg = str(r.get("regpri", "")).upper()

            if not any(x in reg for x in [
                "PAYANT ROTATIF",
                "PAYANT MIXTE",
                "GRATUIT",
            ]):
                continue

            nom = r.get("nomvoie")
            places = pd.to_numeric(
                r.get("placal", 0),
                errors="coerce",
            )

            if not nom or pd.isna(places):
                continue

            cle = normaliser_nom_rue(nom)

            totaux[cle] = (
                totaux.get(cle, 0) + max(0, int(places))
            )

        if len(records) < limite:
            break

        offset += limite

    return totaux


def obtenir_places_rue_principale(nom_rue, arrdt):
    """
    Récupère les places de la rue recherchée avec le filtre
    de suggestion utilisé dans le code d'origine.
    """
    params = {
        "where": (
            f"suggest(nomvoie, '{nom_rue.upper()}') "
            f"AND arrond = {arrdt}"
        ),
        "limit": 100,
    }

    try:
        response = requests.get(
            URL_STATIONNEMENT,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        records = response.json().get("results", [])

        data_list = []

        for r in records:
            reg = str(r.get("regpri", "")).upper()

            if not any(x in reg for x in [
                "PAYANT ROTATIF",
                "PAYANT MIXTE",
                "GRATUIT",
            ]):
                continue

            point = r.get("geo_point_2d")

            if not point:
                continue

            lat = point.get("lat")
            lon = point.get("lon")

            if lat is None or lon is None:
                continue

            num = pd.to_numeric(
                r.get("nummin", 0),
                errors="coerce",
            )
            places = pd.to_numeric(
                r.get("placal", 0),
                errors="coerce",
            )

            if pd.isna(places):
                places = 0

            data_list.append({
                "lat": lat,
                "lon": lon,
                "num": num,
                "places": places,
            })

        if not data_list:
            return 0

        df_rue = pd.DataFrame(data_list)
        return max(0, int(df_rue["places"].sum()))

    except Exception as e:
        print(f"Erreur stationnement rue principale : {e}")
        return 0


# ============================================================
# 7. PRÉDICTION IA
# ============================================================

def predire_dispo_ia(nom_rue, nb_total, arrdt, mto, temp):
    """
    Estime les places libres.
    Retourne None si la prédiction échoue, afin de ne pas
    confondre une erreur technique avec zéro place libre.
    """
    if model is None or prepro is None:
        return None

    if nb_total is None or nb_total <= 0:
        return None

    if mto is None or temp is None:
        return None

    tz = pytz.timezone("Europe/Paris")
    now = datetime.now(tz)
    minutes = now.hour * 60 + now.minute

    socio = DATA_ARRDT.get(
        arrdt,
        {"REV_M": 2500, "VEH": 0.30},
    )

    X_dict = {
        "DATE": now.strftime("%d/%m/%Y"),
        "JOUR": JOURS_FR.get(now.strftime("%A")),
        "HEURE": now.strftime("%H:%M"),
        "RUE": str(nom_rue).upper(),
        "VILLE": "Paris",
        "TRAFIC": 0.0,
        "% PARKING OC": 0.5,
        "NBR PLACES": nb_total,
        "REVENUS / H": socio["REV_M"],
        "VEHICULES / H": socio["VEH"],
        "MTO": mto,
        "TEMPERATURE": temp,
        "HEURE_MINUTES": minutes,
        "HEURE_SIN": np.sin(2 * np.pi * minutes / 1440),
        "HEURE_COS": np.cos(2 * np.pi * minutes / 1440),
    }

    cols = [
        "DATE", "JOUR", "HEURE", "RUE", "VILLE",
        "TRAFIC", "% PARKING OC", "NBR PLACES",
        "REVENUS / H", "VEHICULES / H", "MTO",
        "TEMPERATURE", "HEURE_MINUTES",
        "HEURE_SIN", "HEURE_COS",
    ]

    X_df = pd.DataFrame([X_dict])[cols]

    try:
        prediction = model.predict(prepro.transform(X_df))[0]
        prediction = float(prediction)

        if not np.isfinite(prediction):
            return None

        # On conserve la formule du code d'origine.
        libres_estimes = math.floor(
            nb_total * (1 - prediction)
        )

        return max(0, min(int(nb_total), libres_estimes))

    except Exception as e:
        print(f"Erreur prédiction IA pour {nom_rue} : {e}")
        return None


# ============================================================
# 8. INTERFACE
# ============================================================

st.title("🛰️ SLOT 2.0 - Radar Temps Réel")

c1, c2, c3 = st.columns([1, 2, 4])

with c1:
    num_v = st.number_input(
        "N°",
        min_value=1,
        value=11,
        step=1,
    )

with c2:
    type_v = st.selectbox(
        "Type",
        ["Rue", "Boulevard", "Avenue", "Place", "Quai"],
    )

with c3:
    nom_v = st.text_input("Nom", value="Voltaire")

rayon_voisins = st.slider(
    "Rayon des rues voisines (m)",
    min_value=100,
    max_value=500,
    value=300,
    step=50,
)


# ============================================================
# 9. LOGIQUE GÉOGRAPHIQUE
# ============================================================

lat_pivot, lon_pivot = 48.8566, 2.3522
arrdt = None
mto, temp = None, None
total_p = 0
libres = None
target_found = False

if nom_v.strip():
    try:
        adresse = f"{int(num_v)} {type_v} {nom_v} Paris"

        geo_response = requests.get(
            "https://api-adresse.data.gouv.fr/search/",
            params={"q": adresse, "limit": 1},
            timeout=10,
        )
        geo_response.raise_for_status()
        geo_pivot = geo_response.json()

        features = geo_pivot.get("features", [])

        if features:
            feature = features[0]
            lon_pivot, lat_pivot = (
                feature["geometry"]["coordinates"]
            )

            postcode = feature["properties"].get("postcode", "")

            if postcode and postcode[-2:].isdigit():
                arrdt = int(postcode[-2:])

            if arrdt is not None and 1 <= arrdt <= 20:
                target_found = True
                mto, temp = get_weather(lat_pivot, lon_pivot)

                total_p = obtenir_places_rue_principale(
                    nom_v,
                    arrdt,
                )

                libres = predire_dispo_ia(
                    f"{type_v} {nom_v}",
                    total_p,
                    arrdt,
                    mto,
                    temp,
                )

            else:
                st.warning(
                    "L'adresse a été trouvée, mais son "
                    "arrondissement n'a pas pu être identifié."
                )

        else:
            st.warning(
                "Adresse introuvable. Vérifie le numéro et le nom."
            )

    except Exception as e:
        st.error(f"Erreur lors de la recherche de l'adresse : {e}")


# ============================================================
# 10. CARTE
# ============================================================

m = folium.Map(
    location=[lat_pivot, lon_pivot],
    zoom_start=18,
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Tiles &copy; Esri",
    max_zoom=19,
)

if target_found:
    with st.spinner(
        "Recherche des tronçons routiers et des rues voisines..."
    ):
        troncons = obtenir_troncons_voisins(
            lat_pivot,
            lon_pivot,
            rayon_voisins,
        )

        places_rues = obtenir_places_par_rue(arrdt)

    # On utilise la même logique de comparaison des noms
    # que dans la fonction de recherche de rue.
    nom_cible = normaliser_nom_rue(nom_v)
    type_cible = normaliser_nom_rue(f"{type_v} {nom_v}")

    nb_troncons_affiches = 0
    nb_troncons_gris = 0

    for troncon in troncons:
        nom_troncon = troncon["nom"]
        cle = normaliser_nom_rue(nom_troncon)

        # Vérifie si le tronçon correspond à la rue principale.
        est_rue_cible = (
            cle == nom_cible
            or cle == type_cible
            or cle.endswith(nom_cible)
            or nom_cible.endswith(cle)
        )

        if est_rue_cible:
            nb_libres = libres
            couleur = couleur_disponibilite(nb_libres)
            epaisseur = 9

            if nb_libres is None:
                tooltip = (
                    f"{nom_troncon} — "
                    "Données insuffisantes pour la prédiction"
                )
                nb_troncons_gris += 1
            else:
                tooltip = (
                    f"{nom_troncon} — "
                    f"{nb_libres} place(s) libre(s) estimée(s)"
                )

        else:
            # Nombre de places regroupé par nom de rue.
            nb_places = places_rues.get(cle)

            if nb_places is None or nb_places <= 0:
                couleur = "#95a5a6"
                epaisseur = 4
                tooltip = (
                    f"{nom_troncon} — "
                    "Données de stationnement insuffisantes"
                )
                nb_troncons_gris += 1

            else:
                nb_libres = predire_dispo_ia(
                    nom_troncon,
                    nb_places,
                    arrdt,
                    mto,
                    temp,
                )

                couleur = couleur_disponibilite(nb_libres)
                epaisseur = 5

                if nb_libres is None:
                    tooltip = (
                        f"{nom_troncon} — "
                        "Prédiction indisponible"
                    )
                    nb_troncons_gris += 1
                else:
                    tooltip = (
                        f"{nom_troncon} — "
                        f"{nb_libres} place(s) libre(s) estimée(s) "
                        f"sur {nb_places} place(s)"
                    )

        folium.PolyLine(
            locations=troncon["coords"],
            color=couleur,
            weight=epaisseur,
            opacity=0.85,
            tooltip=tooltip,
        ).add_to(m)

        nb_troncons_affiches += 1

    if not troncons:
        st.warning(
            "Aucun tronçon routier nommé n'a été récupéré "
            "autour de cette adresse."
        )

    # Marqueur de l'adresse recherchée
    folium.Marker(
        [lat_pivot, lon_pivot],
        tooltip=f"Adresse recherchée : {int(num_v)} {type_v} {nom_v}",
        icon=folium.CustomIcon(
            URL_LOGO,
            icon_size=(60, 60),
        ),
    ).add_to(m)

    # Légende
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        padding: 10px 12px;
        border: 1px solid #999;
        border-radius: 5px;
        font-size: 13px;
    ">
        <b>Disponibilité estimée</b><br>
        <span style="color:#e74c3c;">━━</span> 0–1 place libre<br>
        <span style="color:#f39c12;">━━</span> 2–3 places libres<br>
        <span style="color:#27ae60;">━━</span> 4 places libres ou plus<br>
        <span style="color:#95a5a6;">━━</span> Données insuffisantes<br>
        <small>Rue recherchée : trait épais</small>
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend_html)
    )

    st_folium(
        m,
        width=1200,
        height=750,
        key="v91_multirues",
    )

    # Informations dynamiques
    if mto is not None and temp is not None:
        heure = datetime.now(
            pytz.timezone("Europe/Paris")
        ).strftime("%H:%M")

        st.info(
            f"🕒 Heure : {heure} | "
            f"🌡️ {temp}°C | 🌦️ {mto}"
        )

    if libres is not None:
        st.success(
            f"📊 Rue {nom_v.upper()} : {total_p} places au total. "
            f"IA : **{libres} places libres estimées**."
        )
    else:
        st.warning(
            f"📊 Rue {nom_v.upper()} : la prédiction principale "
            "n'a pas pu être calculée de manière fiable."
        )

    st.caption(
        f"{nb_troncons_affiches} tronçon(s) affiché(s), "
        f"dont {nb_troncons_gris} en gris faute de données "
        "ou de prédiction exploitable."
    )

else:
    # Carte par défaut en attendant une adresse valide.
    st_folium(
        m,
        width=1200,
        height=750,
        key="v91_multirues",
    )
