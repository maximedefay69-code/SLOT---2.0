import math
import unicodedata
from datetime import datetime

import folium
import joblib
import numpy as np
import osmnx as ox
import pandas as pd
import pytz
import requests
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# 1. CONFIGURATION
# ============================================================
st.set_page_config(page_title="SLOT 2.0 - Radar Temps Réel", layout="wide")

API_ADRESSE = "https://api-adresse.data.gouv.fr/search/"
API_STATIONNEMENT = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "stationnement-sur-voie-publique-emprises/records"
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
        st.error(
            "Impossible de charger le modèle IA ou le préprocesseur. "
            "Vérifie que modele_lightgbmDA.pkl et preprocessorDA.pkl "
            f"sont dans le dossier de l'application. Détail : {e}"
        )
        return None, None


model, prepro = load_assets()


# ============================================================
# 3. OUTILS
# ============================================================
def normaliser_nom_rue(nom):
    """Minuscule, sans accents, espaces normalisés."""
    nom = str(nom or "").strip().lower()
    nom = unicodedata.normalize("NFKD", nom)
    nom = "".join(c for c in nom if not unicodedata.combining(c))
    return " ".join(nom.split())


def obtenir_meteo(lat, lon):
    """Récupère la météo actuelle via Open-Meteo."""
    try:
        reponse = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code",
            },
            timeout=8,
        )
        reponse.raise_for_status()
        courant = reponse.json()["current"]
        code = courant.get("weather_code", 0)
        temperature = float(courant.get("temperature_2m", 18.0))

        if code in [0, 1]:
            meteo = "Beau"
        elif code in [2, 3, 45, 48]:
            meteo = "Nuageux"
        else:
            meteo = "Pluie"

        return meteo, temperature
    except Exception:
        # Valeurs de secours si le service météo est indisponible.
        return "Beau", 18.0


@st.cache_data(ttl=86400, show_spinner=False)
def geocoder_adresse(numero, type_rue, nom_rue):
    """Géocode l'adresse parisienne avec l'API Adresse."""
    requete = f"{numero} {type_rue} {nom_rue} Paris"
    try:
        reponse = requests.get(
            API_ADRESSE,
            params={"q": requete, "limit": 1},
            timeout=10,
        )
        reponse.raise_for_status()
        donnees = reponse.json()
        if not donnees.get("features"):
            return None

        feature = donnees["features"][0]
        lon, lat = feature["geometry"]["coordinates"]
        proprietes = feature.get("properties", {})
        code_postal = str(proprietes.get("postcode", ""))

        # Les codes postaux parisiens sont de type 75001 à 75020.
        arrdt = None
        if len(code_postal) >= 5 and code_postal.startswith("75"):
            arrdt = int(code_postal[-2:])
            if arrdt == 0:
                arrdt = 20

        return {
            "lat": float(lat),
            "lon": float(lon),
            "arrdt": arrdt,
            "adresse": proprietes.get("label", requete),
            "postcode": code_postal,
        }
    except Exception as e:
        st.warning(f"Erreur lors du géocodage : {e}")
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def obtenir_troncons_voisins(lat, lon, rayon=300):
    """
    Télécharge le réseau routier autour de l'adresse et renvoie les
    lignes OSM avec leurs coordonnées Folium [latitude, longitude].
    """
    try:
        graphe = ox.graph.graph_from_point(
            (lat, lon),
            dist=rayon,
            dist_type="bbox",
            network_type="drive",
            simplify=True,
            retain_all=False,
        )
        _, edges = ox.convert.graph_to_gdfs(graphe)

        if edges.empty or "name" not in edges.columns:
            return []

        troncons = []
        for _, ligne in edges.iterrows():
            geometrie = ligne.geometry
            nom = ligne.get("name")

            if geometrie is None or geometrie.is_empty or not nom:
                continue

            noms = nom if isinstance(nom, list) else [nom]
            nom_affiche = str(noms[0])

            lignes = (
                list(geometrie.geoms)
                if geometrie.geom_type == "MultiLineString"
                else [geometrie]
            )

            for segment in lignes:
                if segment.geom_type != "LineString":
                    continue

                coords = [
                    [float(lat_pt), float(lon_pt)]
                    for lon_pt, lat_pt in segment.coords
                ]
                if len(coords) >= 2:
                    troncons.append({"nom": nom_affiche, "coords": coords})

        return troncons

    except Exception as e:
        # Affiché dans les diagnostics Streamlit plus bas.
        return [{"_erreur": f"{type(e).__name__}: {e}"}]


@st.cache_data(ttl=1800, show_spinner=False)
def obtenir_places_par_rue(arrdt):
    """
    Télécharge une page de données de stationnement pour un arrondissement.
    Limite volontairement la requête à 100 résultats pour réduire l'attente.
    Attention : cela peut ne pas couvrir toutes les rues de l'arrondissement.
    """
    if not arrdt:
        return {}

    try:
        reponse = requests.get(
            API_STATIONNEMENT,
            params={"limit": 100, "offset": 0},
            timeout=10,
        )
        reponse.raise_for_status()
        resultat = reponse.json()
        lignes = resultat.get("results", [])

        places_rues = {}
        for record in lignes:
            record_arrdt = str(record.get("arrond", "")).strip()
            # Selon la structure du jeu de données, l'arrondissement peut
            # être un nombre ou une chaîne.
            chiffres = "".join(c for c in record_arrdt if c.isdigit())
            if chiffres and int(chiffres[-2:]) != int(arrdt):
                continue

            nom = record.get("nomvoie")
            if not nom:
                continue

            reg = str(record.get("regpri", "")).upper()
            if not any(
                terme in reg
                for terme in ["PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT"]
            ):
                continue

            nom_norm = normaliser_nom_rue(nom)
            nb = pd.to_numeric(record.get("placal", 0), errors="coerce")
            if pd.isna(nb):
                nb = 0

            places_rues[nom_norm] = places_rues.get(nom_norm, 0) + int(nb)

        return places_rues

    except Exception as e:
        return {"__erreur__": f"{type(e).__name__}: {e}"}


def predire_dispo_ia(nom_rue, nb_total, arrdt, meteo, temperature):
    """Estime le nombre de places libres à partir du modèle existant."""
    if model is None or prepro is None:
        return None, "Modèle IA non chargé."

    tz = pytz.timezone("Europe/Paris")
    maintenant = datetime.now(tz)
    minutes = maintenant.hour * 60 + maintenant.minute
    socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})

    donnees = {
        "DATE": maintenant.strftime("%d/%m/%Y"),
        "JOUR": JOURS_FR.get(maintenant.strftime("%A")),
        "HEURE": maintenant.strftime("%H:%M"),
        "RUE": str(nom_rue).upper(),
        "VILLE": "Paris",
        "TRAFIC": 0.0,
        "% PARKING OC": 0.5,
        "NBR PLACES": nb_total,
        "REVENUS / H": socio["REV_M"],
        "VEHICULES / H": socio["VEH"],
        "MTO": meteo,
        "TEMPERATURE": temperature,
        "HEURE_MINUTES": minutes,
        "HEURE_SIN": np.sin(2 * np.pi * minutes / 1440),
        "HEURE_COS": np.cos(2 * np.pi * minutes / 1440),
    }

    colonnes = [
        "DATE", "JOUR", "HEURE", "RUE", "VILLE", "TRAFIC",
        "% PARKING OC", "NBR PLACES", "REVENUS / H", "VEHICULES / H",
        "MTO", "TEMPERATURE", "HEURE_MINUTES", "HEURE_SIN", "HEURE_COS",
    ]

    try:
        X = pd.DataFrame([donnees])[colonnes]
        occupation = float(model.predict(prepro.transform(X))[0])

        # Si le modèle renvoie une proportion, la conversion est directe.
        # Si la sortie est hors de [0,1], on la borne pour éviter des
        # résultats aberrants dans l'affichage.
        occupation = min(1.0, max(0.0, occupation))
        libres = max(0, min(int(nb_total), math.floor(nb_total * (1 - occupation))))
        return libres, None
    except Exception as e:
        return None, f"Erreur de prédiction IA : {type(e).__name__}: {e}"


def couleur_pour_places(libres):
    """Vert = plus de places estimées, orange = peu, rouge = très peu."""
    if libres is None:
        return "#7f8c8d"
    if libres <= 1:
        return "#e74c3c"
    if libres <= 3:
        return "#f39c12"
    return "#27ae60"


# ============================================================
# 4. INTERFACE
# ============================================================
st.title("🛰️ SLOT 2.0 — Radar de stationnement")
st.caption(
    "Estimation indicative des places libres. Les données disponibles "
    "peuvent être incomplètes et ne constituent pas un comptage en direct."
)

c1, c2, c3 = st.columns([1, 2, 4])
with c1:
    num_v = st.number_input("N°", min_value=1, max_value=999, value=11, step=1)
with c2:
    type_v = st.selectbox(
        "Type de voie",
        ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Passage", "Impasse"],
    )
with c3:
    nom_v = st.text_input("Nom de la voie", value="Voltaire")

c4, c5 = st.columns([1, 3])
with c4:
    rayon = st.select_slider("Rayon des rues voisines", options=[150, 250, 300, 400, 500], value=300)
with c5:
    st.caption(
        "Les données OSM et Open Data sont chargées à la demande et mises en cache "
        "pour accélérer les recherches suivantes."
    )


# ============================================================
# 5. RECHERCHE ET CALCUL
# ============================================================
lat_pivot, lon_pivot = 48.8566, 2.3522
arrdt = None
adresse_trouvee = None
meteo = "Beau"
temperature = 18.0
total_places = 0
libres_rue = None
erreur_ia = None
erreur_troncons = None
troncons = []
places_rues = {}
geometries_rue_principale = []
geocodage_ok = False

if nom_v.strip():
    with st.spinner("Recherche de l'adresse..."):
        resultat_geo = geocoder_adresse(int(num_v), type_v, nom_v.strip())

    if resultat_geo:
        lat_pivot = resultat_geo["lat"]
        lon_pivot = resultat_geo["lon"]
        arrdt = resultat_geo["arrdt"]
        adresse_trouvee = resultat_geo["adresse"]
        geocodage_ok = True

        if arrdt is None:
            st.warning(
                "L'adresse a été trouvée, mais l'arrondissement n'a pas pu être "
                "déduit du code postal. Les statistiques par arrondissement seront limitées."
            )

        with st.spinner("Récupération de la météo..."):
            meteo, temperature = obtenir_meteo(lat_pivot, lon_pivot)

        with st.spinner(f"Recherche des rues dans un rayon de {rayon} m..."):
            resultat_troncons = obtenir_troncons_voisins(lat_pivot, lon_pivot, rayon)

        if resultat_troncons and "_erreur" in resultat_troncons[0]:
            erreur_troncons = resultat_troncons[0]["_erreur"]
            troncons = []
        else:
            troncons = resultat_troncons

        # Récupère une seule page du jeu de données, sans boucle de pagination.
        if arrdt:
            with st.spinner("Récupération des données de stationnement..."):
                places_rues = obtenir_places_par_rue(arrdt)

        # Recherche de la rue principale parmi les tronçons OSM.
        recherche = normaliser_nom_rue(f"{type_v} {nom_v}")
        recherche_simple = normaliser_nom_rue(nom_v)

        for troncon in troncons:
            nom_osm = normaliser_nom_rue(troncon["nom"])
            if (
                nom_osm == recherche
                or nom_osm == recherche_simple
                or nom_osm.endswith(recherche_simple)
                or recherche_simple.endswith(nom_osm)
            ):
                geometries_rue_principale.append(troncon["coords"])

        # Estimation principale : données de places disponibles si la rue
        # correspond à une entrée présente dans la page Open Data récupérée.
        if isinstance(places_rues, dict) and "__erreur__" not in places_rues:
            cle = normaliser_nom_rue(nom_v)
            cle_avec_type = normaliser_nom_rue(f"{type_v} {nom_v}")
            total_places = places_rues.get(cle, places_rues.get(cle_avec_type, 0))

        if total_places > 0 and arrdt:
            libres_rue, erreur_ia = predire_dispo_ia(
                nom_v, total_places, arrdt, meteo, temperature
            )
        elif model is None or prepro is None:
            erreur_ia = "Modèle IA non chargé."
    else:
        st.error(
            "Adresse introuvable. Essaie un nom de voie plus simple, "
            "par exemple « Voltaire », et vérifie le numéro."
        )


# ============================================================
# 6. CARTE
# ============================================================
carte = folium.Map(
    location=[lat_pivot, lon_pivot],
    zoom_start=17,
    tiles="CartoDB positron",
    control_scale=True,
)

# Si la géométrie de la rue principale a été trouvée, on la dessine.
if geometries_rue_principale:
    couleur_principale = couleur_pour_places(libres_rue)
    for coords in geometries_rue_principale:
        folium.PolyLine(
            locations=coords,
            color=couleur_principale,
            weight=8,
            opacity=0.95,
            tooltip=(
                f"{type_v} {nom_v} — "
                + (
                    f"{libres_rue} place(s) libre(s) estimée(s)"
                    if libres_rue is not None
                    else "estimation indisponible"
                )
            ),
        ).add_to(carte)

# Dessine les rues voisines. Faute d'une prédiction fiable par segment,
# la couleur grise signifie « données/prédiction indisponibles ».
for troncon in troncons:
    if "_erreur" in troncon:
        continue

    nom_osm = normaliser_nom_rue(troncon["nom"])
    est_rue_principale = (
        nom_osm == normaliser_nom_rue(nom_v)
        or nom_osm == normaliser_nom_rue(f"{type_v} {nom_v}")
        or nom_osm.endswith(normaliser_nom_rue(nom_v))
    )
    if est_rue_principale:
        continue

    cle = normaliser_nom_rue(troncon["nom"])
    nb_places_voisines = places_rues.get(cle, 0) if isinstance(places_rues, dict) else 0

    # On ne lance pas une prédiction IA pour chaque rue voisine sans données
    # fiables sur son nombre total de places.
    if nb_places_voisines > 0 and arrdt:
        libres_voisines, _ = predire_dispo_ia(
            troncon["nom"], nb_places_voisines, arrdt, meteo, temperature
        )
        couleur = couleur_pour_places(libres_voisines)
        info = f"{troncon['nom']} — {libres_voisines} place(s) estimée(s)"
    else:
        couleur = "#9aa0a6"
        info = f"{troncon['nom']} — données de stationnement non trouvées"

    folium.PolyLine(
        locations=troncon["coords"],
        color=couleur,
        weight=4,
        opacity=0.7,
        tooltip=info,
    ).add_to(carte)

if geocodage_ok:
    folium.Marker(
        [lat_pivot, lon_pivot],
        tooltip=adresse_trouvee or f"{num_v} {type_v} {nom_v}, Paris",
        popup=folium.Popup(
            f"<b>{adresse_trouvee or f'{num_v} {type_v} {nom_v}, Paris'}</b>",
            max_width=300,
        ),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(carte)

folium.LayerControl().add_to(carte)

st.subheader("Carte des rues et du stationnement")
st_folium(carte, width=None, height=700, key="slot20_map")


# ============================================================
# 7. RÉSULTATS ET DIAGNOSTICS
# ============================================================
if geocodage_ok:
    st.markdown("### Résultat")
    st.write(f"**Adresse reconnue :** {adresse_trouvee}")
    st.write(f"**Arrondissement :** {arrdt if arrdt else 'non déterminé'}")
    st.write(f"**Météo :** {meteo}, {temperature} °C")

    if total_places > 0:
        st.write(f"**Places recensées pour la rue :** {total_places}")
        if libres_rue is not None:
            st.success(
                f"Estimation IA : **{libres_rue} place(s) libre(s)** sur "
                f"{total_places} place(s) recensée(s)."
            )
        else:
            st.warning(
                "Les places sont recensées, mais la prédiction IA n'a pas pu "
                "être calculée."
            )
            if erreur_ia:
                st.caption(erreur_ia)
    else:
        st.warning(
            "Aucun total de places n'a été retrouvé pour cette rue dans les "
            "résultats Open Data chargés. Cela ne signifie pas qu'il n'y a "
            "aucune place : la page de résultats peut être incomplète."
        )
        if isinstance(places_rues, dict) and "__erreur__" in places_rues:
            st.caption(f"Erreur Open Data : {places_rues['__erreur__']}")

    with st.expander("Diagnostics — pour comprendre les rues manquantes"):
        st.write("Adresse trouvée :", geocodage_ok)
        st.write("Arrondissement :", arrdt)
        st.write("Nombre de tronçons OSM :", len(troncons))
        st.write(
            "Nombre de rues dans les données de stationnement :",
            len([k for k in places_rues if not k.startswith("__")])
            if isinstance(places_rues, dict)
            else 0,
        )
        st.write("Nombre de géométries reconnues pour la rue principale :", len(geometries_rue_principale))
        if erreur_troncons:
            st.error(f"Erreur OSMnx : {erreur_troncons}")
        if isinstance(places_rues, dict) and "__erreur__" in places_rues:
            st.error(f"Erreur Open Data : {places_rues['__erreur__']}")
        st.caption(
            "La couche grise signifie que les données de stationnement n'ont "
            "pas permis de calculer une estimation pour cette rue."
        )
else:
    st.info("Saisis une adresse parisienne pour afficher la carte.")
