"""
SLOT 2.0 — logique métier, indépendante de Streamlit.

Tout ce qui touche au réseau, aux données et au modèle est ici, afin de
pouvoir être testé sans Streamlit et sans accès réseau (sessions simulées).

Sources de données :
- API Adresse (data.gouv.fr) : géocodage.
- Open-Meteo : météo actuelle.
- Paris Open Data, jeu « stationnement-sur-voie-publique-emprises » :
  emprises de stationnement (polygones) avec le nombre de places
  théoriques (`placal`). Ce sont des places RECENSÉES, pas un comptage
  en direct de l'occupation.
- Overpass (OpenStreetMap), facultatif : tracé des rues voisines en gris.
"""

from __future__ import annotations

import json
import os
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ============================================================
# Constantes
# ============================================================
API_ADRESSE = "https://api-adresse.data.gouv.fr/search/"
API_METEO = "https://api.open-meteo.com/v1/forecast"
API_STATIONNEMENT = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "stationnement-sur-voie-publique-emprises/records"
)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "SLOT-2.0/1.0 (application Streamlit de stationnement a Paris)"

# L'API Explore v2.1 d'Opendatasoft refuse limit > 100 et offset+limit > 10000.
OPENDATA_PAGE = 100
OPENDATA_MAX_RECORDS = 3000

# Régimes de stationnement utiles pour une voiture (même filtre que la v1).
REGIMES_VOITURE = ("PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT")

DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28},
    3: {"REV_M": 2833, "VEH": 0.25}, 4: {"REV_M": 2750, "VEH": 0.26},
    5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40},
    9: {"REV_M": 3000, "VEH": 0.30}, 10: {"REV_M": 2333, "VEH": 0.22},
    11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36},
    15: {"REV_M": 2916, "VEH": 0.38}, 16: {"REV_M": 4583, "VEH": 0.45},
    17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23},
}
SOCIO_DEFAUT = {"REV_M": 2500, "VEH": 0.30}

JOURS_FR = {
    "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
    "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi",
    "Sunday": "Dimanche",
}

# Valeurs constantes envoyées au modèle, inchangées par rapport à la v1 :
# aucune source temps réel n'est branchée pour ces deux variables.
TRAFIC_DEFAUT = 0.0
PARKING_OC_DEFAUT = 0.5

TYPES_VOIE = {
    "rue", "boulevard", "bd", "avenue", "av", "place", "pl", "quai",
    "passage", "impasse", "allee", "cours", "square", "villa", "cite",
    "chaussee", "rond point", "rond-point", "port", "pont", "chemin",
    "sente", "sentier", "voie", "galerie", "parvis", "esplanade",
    "promenade", "carrefour", "porte", "hameau", "cour", "ruelle",
}
ARTICLES = ("de la ", "de l ", "du ", "des ", "de ", "d ", "la ", "le ", "les ", "l ")


# Date du fichier au moment du chargement : permet à app_map.py de détecter
# qu'une version plus récente a été déployée et de recharger ce module.
_MTIME_CHARGE = os.path.getmtime(__file__)


class ApiError(Exception):
    """Erreur d'appel à un service externe, avec un message lisible."""


# ============================================================
# Normalisation des noms de rues
# ============================================================
def normaliser_nom_rue(nom) -> str:
    """Minuscules, sans accents, apostrophes et tirets remplacés, espaces uniques."""
    nom = str(nom or "").strip().lower()
    nom = unicodedata.normalize("NFKD", nom)
    nom = "".join(c for c in nom if not unicodedata.combining(c))
    nom = re.sub(r"[’'`\-_.,/]", " ", nom)
    return " ".join(nom.split())


def cle_rue(nom) -> str:
    """
    Clé de comparaison d'un nom de voie, sans type ni article de tête.
    « Rue Pelouze », « RUE PELOUZE » et « PELOUZE » donnent « pelouze ».
    « Rue du Commandant Lamy » donne « commandant lamy ».
    """
    n = normaliser_nom_rue(nom)
    for t in sorted(TYPES_VOIE, key=len, reverse=True):
        t_norm = normaliser_nom_rue(t)
        if n == t_norm:
            return n  # nom réduit au type : on ne vide pas la clé
        if n.startswith(t_norm + " "):
            n = n[len(t_norm) + 1:]
            break
    for art in ARTICLES:
        if n.startswith(art) and len(n) > len(art):
            n = n[len(art):]
            break
    return n


def libelle_rue(nom, type_voie=None) -> str:
    """Libellé lisible : « RUE PELOUZE » -> « Rue Pelouze »."""
    brut = " ".join(p for p in [str(type_voie or "").strip(), str(nom or "").strip()] if p)
    if not brut:
        return "Voie sans nom"
    mots = brut.lower().split()
    petits = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "aux", "au"}
    return " ".join(m if (i and m in petits) else m[:1].upper() + m[1:] for i, m in enumerate(mots))


# ============================================================
# Outils HTTP
# ============================================================
def _session(session):
    if session is not None:
        return session
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def lire_json(reponse, service: str):
    """Vérifie le statut HTTP et le type de contenu, puis décode le JSON."""
    code = getattr(reponse, "status_code", None)
    if code != 200:
        extrait = (getattr(reponse, "text", "") or "")[:200].replace("\n", " ")
        raise ApiError(f"{service} : HTTP {code}. {extrait}".strip())
    ctype = (reponse.headers.get("Content-Type", "") if getattr(reponse, "headers", None) else "").lower()
    texte = getattr(reponse, "text", "") or ""
    if "html" in ctype or texte.lstrip()[:1] == "<":
        raise ApiError(f"{service} : réponse HTML reçue au lieu de JSON (serveur surchargé ou en erreur).")
    try:
        return reponse.json()
    except ValueError as e:
        raise ApiError(f"{service} : JSON invalide ({e}).") from e


def distance_m(lat1, lon1, lat2, lon2) -> float:
    """Distance haversine en mètres."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ============================================================
# Géocodage
# ============================================================
def geocoder_adresse(numero, type_rue, nom_rue, session=None, timeout=10):
    """
    Géocode une adresse parisienne. Retourne un dict ou None si introuvable.
    Lève ApiError si le service est en erreur.
    """
    requete = f"{numero} {type_rue} {nom_rue}, Paris".strip()
    s = _session(session)
    try:
        r = s.get(API_ADRESSE, params={"q": requete, "limit": 5}, timeout=timeout)
    except requests.RequestException as e:
        raise ApiError(f"API Adresse injoignable : {type(e).__name__}") from e
    donnees = lire_json(r, "API Adresse")
    for feature in donnees.get("features") or []:
        props = feature.get("properties") or {}
        citycode = str(props.get("citycode", ""))
        # Paris : codes INSEE 75101 à 75120 (75056 = Paris entier).
        if not citycode.startswith("751"):
            continue
        try:
            lon, lat = feature["geometry"]["coordinates"][:2]
            arrdt = int(citycode[-2:])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= arrdt <= 20:
            continue
        return {
            "lat": float(lat),
            "lon": float(lon),
            "arrdt": arrdt,
            "adresse": props.get("label", requete),
            "rue": props.get("street") or props.get("name") or f"{type_rue} {nom_rue}",
            "score": props.get("score"),
            "type": props.get("type"),
        }
    return None


API_ADRESSE_INVERSE = "https://api-adresse.data.gouv.fr/reverse/"


def geocoder_inverse(lat, lon, session=None, timeout=10):
    """
    Rue la plus proche d'une position (géolocalisation du téléphone).
    Retourne un dict au même format que geocoder_adresse, mais avec la position
    de l'utilisateur (lat, lon) et non celle de l'adresse trouvée.
    None si la position n'est pas dans Paris. Lève ApiError si le service est en erreur.
    """
    s = _session(session)
    try:
        r = s.get(API_ADRESSE_INVERSE, params={"lat": round(lat, 6), "lon": round(lon, 6), "limit": 5},
                  timeout=timeout)
    except requests.RequestException as e:
        raise ApiError(f"API Adresse injoignable : {type(e).__name__}") from e
    donnees = lire_json(r, "API Adresse")
    for feature in donnees.get("features") or []:
        props = feature.get("properties") or {}
        citycode = str(props.get("citycode", ""))
        if not citycode.startswith("751"):
            continue
        try:
            arrdt = int(citycode[-2:])
        except ValueError:
            continue
        if not 1 <= arrdt <= 20:
            continue
        rue = props.get("street") or props.get("name")
        if not rue:
            continue
        return {
            "lat": float(lat),
            "lon": float(lon),
            "arrdt": arrdt,
            "adresse": props.get("label", rue),
            "rue": rue,
            "score": props.get("score"),
            "type": props.get("type"),
            "distance_adresse_m": props.get("distance"),
        }
    return None


CODES_ERREUR_GEOLOC = {
    0: "Ce navigateur ne permet pas la géolocalisation.",
    1: "Tu as refusé l'accès à ta position.",
    2: "Ta position n'a pas pu être déterminée (signal GPS insuffisant ?).",
    3: "La localisation a pris trop de temps.",
}


def lire_position(valeur):
    """
    Interprète la valeur renvoyée par le composant de géolocalisation.
    Retourne un dict : etat ("attente" | "ok" | "refus" | "erreur"),
    lat, lon, precision (m), message.
    """
    vide = {"etat": "attente", "lat": None, "lon": None, "precision": None, "message": None}
    if isinstance(valeur, dict) and "value" in valeur and "coords" not in valeur and "error" not in valeur:
        valeur = valeur["value"]  # format {value, dataType} de certains composants
    if not isinstance(valeur, dict):
        return vide
    if "error" in valeur:
        err = valeur.get("error") or {}
        code = err.get("code") if isinstance(err, dict) else None
        message = CODES_ERREUR_GEOLOC.get(code, f"Erreur de localisation : {err}")
        return {**vide, "etat": "refus" if code == 1 else "erreur", "message": message}
    coords = valeur.get("coords") or {}
    try:
        lat, lon = float(coords["latitude"]), float(coords["longitude"])
    except (KeyError, TypeError, ValueError):
        return {**vide, "etat": "erreur", "message": "Position reçue illisible."}
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return {**vide, "etat": "erreur", "message": "Position reçue invalide."}
    try:
        precision = float(coords.get("accuracy")) if coords.get("accuracy") is not None else None
    except (TypeError, ValueError):
        precision = None
    return {"etat": "ok", "lat": lat, "lon": lon, "precision": precision, "message": None}


# ============================================================
# Météo
# ============================================================
def code_meteo_vers_mto(code) -> str:
    """Même correspondance que la v1 (catégories vues à l'entraînement)."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = 0
    if code in (0, 1):
        return "Beau"
    if code in (2, 3, 45, 48):
        return "Nuageux"
    return "Pluie"


def obtenir_meteo(lat, lon, session=None, timeout=8):
    """Retourne (mto, température, erreur). Valeurs de repli signalées par `erreur`."""
    s = _session(session)
    try:
        r = s.get(
            API_METEO,
            params={"latitude": round(lat, 4), "longitude": round(lon, 4),
                    "current": "temperature_2m,weather_code", "timezone": "Europe/Paris"},
            timeout=timeout,
        )
        courant = lire_json(r, "Open-Meteo").get("current") or {}
        if "temperature_2m" not in courant:
            raise ApiError("Open-Meteo : champ current.temperature_2m absent.")
        return code_meteo_vers_mto(courant.get("weather_code")), float(courant["temperature_2m"]), None
    except (ApiError, requests.RequestException, ValueError, TypeError) as e:
        return "Beau", 18.0, f"Météo indisponible ({e}) : valeurs par défaut utilisées (Beau, 18 °C)."


# ============================================================
# Open Data : emprises de stationnement
# ============================================================
def _decoder_json_eventuel(valeur):
    if isinstance(valeur, str):
        try:
            return json.loads(valeur)
        except ValueError:
            return None
    return valeur


def geometries_vers_folium(geo) -> list[tuple[str, list[list[float]]]]:
    """
    Convertit une géométrie GeoJSON (ou un Feature) en liste de
    (« polygon » | « line », [[lat, lon], ...]) pour Folium.
    GeoJSON est en (lon, lat), Folium en (lat, lon).
    """
    geo = _decoder_json_eventuel(geo)
    if not isinstance(geo, dict):
        return []
    if geo.get("type") == "Feature":
        geo = geo.get("geometry")
        if not isinstance(geo, dict):
            return []
    t, coords = geo.get("type"), geo.get("coordinates")
    if coords is None:
        return []

    def anneau(points):
        out = []
        for p in points or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    out.append([float(p[1]), float(p[0])])
                except (TypeError, ValueError):
                    return []
        return out

    resultat = []
    try:
        if t == "Polygon":
            ring = anneau(coords[0]) if coords else []
            if len(ring) >= 3:
                resultat.append(("polygon", ring))
        elif t == "MultiPolygon":
            for poly in coords:
                ring = anneau(poly[0]) if poly else []
                if len(ring) >= 3:
                    resultat.append(("polygon", ring))
        elif t == "LineString":
            ligne = anneau(coords)
            if len(ligne) >= 2:
                resultat.append(("line", ligne))
        elif t == "MultiLineString":
            for l in coords:
                ligne = anneau(l)
                if len(ligne) >= 2:
                    resultat.append(("line", ligne))
    except (TypeError, IndexError):
        return []
    return resultat


def _point(record):
    p = _decoder_json_eventuel(record.get("geo_point_2d"))
    try:
        if isinstance(p, dict):
            return float(p["lat"]), float(p["lon"])
        if isinstance(p, (list, tuple)) and len(p) == 2:
            return float(p[0]), float(p[1])
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _entier(valeur):
    n = pd.to_numeric(valeur, errors="coerce")
    if n is None or pd.isna(n):
        return None
    return max(0, int(round(float(n))))


def parser_emprise(record):
    """
    Transforme un enregistrement Open Data en emprise exploitable,
    ou None s'il est inutilisable (pas de nom de voie).
    `places` vaut None si le nombre de places est absent (≠ 0 place).
    """
    if not isinstance(record, dict):
        return None
    nom = record.get("nomvoie")
    if not nom or not str(nom).strip():
        return None
    places = _entier(record.get("placal"))
    if places is None:
        places = _entier(record.get("plarel"))
    return {
        "nom_brut": str(nom).strip(),
        "cle": cle_rue(nom),
        "label": libelle_rue(nom, record.get("typevoie")),
        "regpri": str(record.get("regpri") or "").upper().strip(),
        "places": places,
        "point": _point(record),
        "geometries": geometries_vers_folium(record.get("geo_shape")),
        "arrond": record.get("arrond"),
    }


def est_regime_voiture(regpri: str) -> bool:
    return any(t in (regpri or "").upper() for t in REGIMES_VOITURE)


def _paginer(s, where, timeout, max_records, deadline):
    """Télécharge toutes les pages d'une requête Opendatasoft v2.1."""
    lignes, total, offset = [], None, 0
    while True:
        if time.monotonic() > deadline:
            raise ApiError("Open Data : délai total dépassé pendant la pagination.")
        params = {"where": where, "limit": OPENDATA_PAGE, "offset": offset}
        try:
            r = s.get(API_STATIONNEMENT, params=params, timeout=timeout)
        except requests.RequestException as e:
            raise ApiError(f"Open Data injoignable : {type(e).__name__}") from e
        page = lire_json(r, "Open Data Paris")
        if not isinstance(page, dict) or not isinstance(page.get("results"), list):
            raise ApiError("Open Data Paris : champ « results » absent ou invalide.")
        total = page.get("total_count", total)
        lignes.extend(page["results"])
        offset += OPENDATA_PAGE
        if not page["results"] or len(page["results"]) < OPENDATA_PAGE:
            break
        if total is not None and offset >= total:
            break
        if offset >= min(max_records, 10000):
            break
    return lignes, total


def recuperer_emprises(lat, lon, rayon, arrdt=None, session=None, timeout=15,
                       max_records=OPENDATA_MAX_RECORDS, budget_s=40):
    """
    Récupère les emprises de stationnement dans un rayon autour du point.

    Filtre côté serveur par distance (ODSQL `within_distance`), puis replis :
    `distance(...)`, puis filtre par arrondissement + filtre local par distance.
    Retourne un dict : emprises, total_serveur, recuperes, methode, erreurs, tronque.
    """
    s = _session(session)
    deadline = time.monotonic() + budget_s
    point = f"geom'POINT({lon:.6f} {lat:.6f})'"
    tentatives = [
        ("within_distance", f"within_distance(geo_point_2d, {point}, {int(rayon)}m)"),
        ("distance", f"distance(geo_point_2d, {point}, {int(rayon)}m)"),
    ]
    if arrdt:
        tentatives.append(("arrondissement", f"arrond = {int(arrdt)}"))

    erreurs = []
    for methode, where in tentatives:
        try:
            lignes, total = _paginer(s, where, timeout, max_records, deadline)
        except ApiError as e:
            erreurs.append(f"[{methode}] {e}")
            continue
        emprises = []
        for rec in lignes:
            e = parser_emprise(rec)
            if e is None:
                continue
            if methode == "arrondissement":
                # Filtre local : on garde les emprises dans le rayon.
                if e["point"] is None or distance_m(lat, lon, *e["point"]) > rayon:
                    continue
            emprises.append(e)
        return {
            "emprises": emprises,
            "total_serveur": total,
            "recuperes": len(lignes),
            "methode": methode,
            "erreurs": erreurs,
            "tronque": total is not None and len(lignes) < total,
        }
    return {"emprises": [], "total_serveur": None, "recuperes": 0, "methode": None,
            "erreurs": erreurs, "tronque": False}


def agreger_par_rue(emprises, filtre_regime=True):
    """
    Regroupe les emprises par rue (clé normalisée).
    `places` = somme des places connues ; `places_inconnues` = emprises sans nombre.
    """
    rues = {}
    for e in emprises:
        if filtre_regime and not est_regime_voiture(e["regpri"]):
            continue
        r = rues.setdefault(e["cle"], {"cle": e["cle"], "label": e["label"], "nom_brut": e["nom_brut"],
                                       "places": 0, "places_inconnues": 0, "emprises": []})
        if e["places"] is None:
            r["places_inconnues"] += 1
        else:
            r["places"] += e["places"]
        r["emprises"].append(e)
    return rues


# ============================================================
# Axe d'une rue reconstruit à partir de ses emprises
# ============================================================
def axe_rue(emprises, pas_m=25.0, largeur_max_m=45.0):
    """
    Reconstruit le tracé continu d'une rue à partir de ses emprises de
    stationnement (morceaux des deux côtés), sans service externe.

    Méthode : tous les sommets des polygones sont projetés en mètres, on
    calcule l'axe principal (ACP), on découpe la rue en tranches de `pas_m`
    le long de cet axe et on prend le point moyen de chaque tranche : la
    ligne passe au milieu de la chaussée, d'un bout à l'autre.

    Retourne [[lat, lon], ...] ou [] si la forme n'est pas celle d'une rue
    (place, carrefour : dispersion latérale > `largeur_max_m`) ou si les
    points sont insuffisants.
    """
    points = []
    for e in emprises:
        for _, coords in e.get("geometries") or []:
            points.extend(coords)
        if not e.get("geometries") and e.get("point"):
            points.append(list(e["point"]))
    if len(points) < 2:
        return []

    pts = np.asarray(points, dtype=float)
    lat0, lon0 = pts[:, 0].mean(), pts[:, 1].mean()
    k = math.cos(math.radians(lat0))
    xy = np.column_stack(((pts[:, 1] - lon0) * 111320.0 * k, (pts[:, 0] - lat0) * 110540.0))

    centre = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - centre, full_matrices=False)
    axe, normale = vt[0], np.array([-vt[0][1], vt[0][0]])
    t = (xy - centre) @ axe
    lateral = (xy - centre) @ normale
    if t.max() - t.min() < 10.0:
        return []  # trop court pour être lisible comme une rue

    # Ligne médiane lissée : position latérale ajustée sur TOUS les points
    # (degré 2 pour suivre une rue légèrement courbe). Contrairement à une
    # moyenne par tranche, elle ne zigzague pas quand les morceaux des deux
    # trottoirs sont décalés.
    longueur = t.max() - t.min()
    degre = 2 if longueur > 120 else 1
    poly = np.polyfit(t, lateral, degre)
    residus = lateral - np.polyval(poly, t)
    # Points trop loin de la ligne : place ou carrefour, pas une rue.
    if np.abs(residus).max() > largeur_max_m / 2 + 8:
        return []

    n = max(2, int(math.ceil(longueur / pas_m)) + 1)
    trace = []
    for ti in np.linspace(t.min(), t.max(), n):
        x, y = centre + ti * axe + np.polyval(poly, ti) * normale
        trace.append([float(lat0 + y / 110540.0), float(lon0 + x / (111320.0 * k))])
    return trace


# ============================================================
# Trafic en temps réel (Google Maps)
# ============================================================
API_DISTANCE_MATRIX = "https://maps.googleapis.com/maps/api/distancematrix/json"
API_ROUTES = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Méthode reprise de l'application d'origine (parking-louis) :
# trajet de l'adresse vers un point décalé de +0,005° en latitude et longitude
# (≈ 700 m à vol d'oiseau), écart = durée avec trafic − durée normale.
DECALAGE_TRAJET_DEG = 0.005
SEUIL_TRAFIC_MOYEN_S = 30    # écart < 30 s            -> 0   (fluide)
SEUIL_TRAFIC_FORT_S = 120    # 30 s ≤ écart < 120 s    -> 0,5 (moyen) ; au-delà -> 1 (dense)


def score_trafic(ecart_s) -> float:
    """
    Écart de temps de trajet (secondes) -> valeur TRAFIC du modèle.
    Échelle 0–1, celle de l'entraînement (min 0, max 1, moyenne 0,09) :
    l'ancienne app envoyait 0/50/100, soit 100 fois trop.
    """
    if ecart_s < SEUIL_TRAFIC_MOYEN_S:
        return 0.0
    if ecart_s < SEUIL_TRAFIC_FORT_S:
        return 0.5
    return 1.0


def _secondes(valeur):
    """Durée Routes API « 123s » -> 123."""
    return int(round(float(str(valeur).rstrip("s"))))


# Directions essayées pour le point d'arrivée : celle de l'app d'origine (nord-est)
# d'abord, puis les trois autres si Google ne trouve pas de route (parc, voies ferrées…).
DIRECTIONS_TRAJET = [(1, 1), (-1, -1), (1, -1), (-1, 1)]


class _PasDeTrajet(ApiError):
    """Pas de route vers ce point d'arrivée : on essaie une autre direction."""


def _mesure_distance_matrix(s, lat, lon, dlat, dlon, cle_api, timeout):
    r = s.get(API_DISTANCE_MATRIX, params={
        "origins": f"{lat:.6f},{lon:.6f}", "destinations": f"{dlat:.6f},{dlon:.6f}",
        "departure_time": "now", "traffic_model": "best_guess", "key": cle_api,
    }, timeout=timeout)
    d = lire_json(r, "Google Distance Matrix")
    if d.get("status") != "OK":  # clé refusée, quota, API non activée : inutile d'insister
        raise ApiError(f"Google Distance Matrix : {d.get('status')} {str(d.get('error_message', ''))[:150]}".strip())
    el = d["rows"][0]["elements"][0]
    if el.get("status") != "OK" or "duration_in_traffic" not in el:
        raise _PasDeTrajet(f"Google Distance Matrix : pas de trajet ({el.get('status')}).")
    return int(el["duration_in_traffic"]["value"]) - int(el["duration"]["value"])


def _mesure_routes(s, lat, lon, dlat, dlon, cle_api, timeout):
    corps = {
        "origin": {"location": {"latLng": {"latitude": lat, "longitude": lon}}},
        "destination": {"location": {"latLng": {"latitude": dlat, "longitude": dlon}}},
        "travelMode": "DRIVE", "routingPreference": "TRAFFIC_AWARE",
    }
    r = s.post(API_ROUTES, json=corps, timeout=timeout, headers={
        "X-Goog-Api-Key": cle_api, "X-Goog-FieldMask": "routes.duration,routes.staticDuration",
        "Content-Type": "application/json",
    })
    d = lire_json(r, "Google Routes")  # HTTP 403 (API non activée) -> ApiError : on arrête
    route = (d.get("routes") or [None])[0]
    if not route or "duration" not in route or "staticDuration" not in route:
        raise _PasDeTrajet("Google Routes : pas de trajet.")
    return _secondes(route["duration"]) - _secondes(route["staticDuration"])


def obtenir_trafic(lat, lon, cle_api, session=None, timeout=6):
    """
    Mesure le trafic autour de l'adresse avec Google Maps.
    1) API Distance Matrix (celle de l'app d'origine) ;
    2) à défaut, Routes API (son remplaçant chez Google).
    Pour chaque API, si aucun trajet n'est trouvé vers le point d'arrivée,
    on essaie les autres directions ; une erreur de compte (clé refusée,
    API non activée, quota) arrête aussitôt cette API.
    Retourne un dict : score (0 / 0,5 / 1 ou None), ecart_s, source, erreur.
    La clé n'apparaît jamais dans les messages d'erreur.
    """
    if not cle_api:
        return {"score": None, "ecart_s": None, "source": None,
                "erreur": "Clé GOOGLE_MAPS_API_KEY absente des secrets."}
    s = _session(session)
    erreurs = []
    for nom, mesure in (("Distance Matrix", _mesure_distance_matrix), ("Routes API", _mesure_routes)):
        for sens_lat, sens_lon in DIRECTIONS_TRAJET:
            dlat = lat + sens_lat * DECALAGE_TRAJET_DEG
            dlon = lon + sens_lon * DECALAGE_TRAJET_DEG
            try:
                ecart = mesure(s, lat, lon, dlat, dlon, cle_api, timeout)
                return {"score": score_trafic(ecart), "ecart_s": ecart, "source": nom, "erreur": None}
            except _PasDeTrajet as e:
                erreurs.append(str(e))
                continue
            except ApiError as e:
                erreurs.append(str(e))
            except requests.RequestException as e:
                erreurs.append(f"Google {nom} injoignable : {type(e).__name__}")  # pas d'URL (contient la clé)
            except (KeyError, IndexError, TypeError, ValueError):
                erreurs.append(f"Google {nom} : réponse inattendue.")
            break  # erreur de compte ou de réseau : inutile d'essayer d'autres directions
    # Dédoublonne les messages identiques (même erreur sur plusieurs directions).
    return {"score": None, "ecart_s": None, "source": None, "erreur": " | ".join(dict.fromkeys(erreurs))}


# ============================================================
# Overpass (facultatif) : tracé des rues voisines
# ============================================================
HIGHWAYS_ROUTES = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
    "living_street|primary_link|secondary_link|tertiary_link"
)


def requete_overpass(lat, lon, rayon) -> str:
    return (
        f"[out:json][timeout:20];"
        f'way(around:{int(rayon)},{lat:.6f},{lon:.6f})["highway"~"^({HIGHWAYS_ROUTES})$"]["name"];'
        f"out tags geom;"
    )


def parser_overpass(donnees):
    """Convertit une réponse Overpass JSON en tronçons {nom, cle, coords}."""
    if not isinstance(donnees, dict):
        raise ApiError("Overpass : réponse JSON inattendue.")
    if "elements" not in donnees:
        remarque = str(donnees.get("remark") or donnees.get("error") or "")[:200]
        raise ApiError(f"Overpass : pas de champ « elements » dans la réponse. {remarque}".strip())
    remarque = str(donnees.get("remark") or "")
    if "error" in remarque.lower() and not donnees["elements"]:
        raise ApiError(f"Overpass : {remarque[:200]}")
    troncons = []
    for el in donnees["elements"]:
        if not isinstance(el, dict) or el.get("type") != "way":
            continue
        nom = (el.get("tags") or {}).get("name")
        geom = el.get("geometry") or []
        coords = []
        for p in geom:
            try:
                coords.append([float(p["lat"]), float(p["lon"])])
            except (KeyError, TypeError, ValueError):
                continue
        if nom and len(coords) >= 2:
            troncons.append({"nom": nom, "cle": cle_rue(nom), "coords": coords})
    return troncons


def recuperer_rues_osm(lat, lon, rayon, session=None, endpoints=None, budget_s=20):
    """
    Interroge les serveurs Overpass l'un après l'autre, avec un budget de
    temps total. Retourne (troncons, erreurs). Aucune donnée inventée.
    """
    s = _session(session)
    endpoints = endpoints or OVERPASS_ENDPOINTS
    erreurs = []
    debut = time.monotonic()
    requete = requete_overpass(lat, lon, rayon)
    for url in endpoints:
        restant = budget_s - (time.monotonic() - debut)
        if restant < 3:
            erreurs.append("Budget de temps Overpass épuisé.")
            break
        hote = url.split("/")[2]
        try:
            r = s.post(url, data={"data": requete}, timeout=(4, min(15, restant)),
                       headers={"User-Agent": USER_AGENT})
            if getattr(r, "status_code", None) == 429:
                raise ApiError("HTTP 429 (trop de requêtes, serveur saturé)")
            troncons = parser_overpass(lire_json(r, "Overpass"))
            return troncons, erreurs
        except (ApiError, requests.RequestException) as e:
            erreurs.append(f"{hote} : {type(e).__name__ if not isinstance(e, ApiError) else ''} {e}".strip())
    return [], erreurs


# ============================================================
# Modèle IA
# ============================================================
@dataclass
class Prediction:
    occupation_brute: float
    occupation: float
    libres: int
    places: int
    rue_connue: bool
    hors_plage_places: bool
    bornee: bool
    alertes: list = field(default_factory=list)


class ModeleSlot:
    """
    Enveloppe du modèle LightGBM + préprocesseur (versions DA, v1 et v2).

    Vérifié sur les artefacts (identiques pour les 3 versions sauf mention) :
    - régression LightGBM (objectif L2, métrique de suivi L1), 200 arbres,
      learning_rate 0,05, 31 feuilles, feature_fraction 0,8 ;
    - préprocesseur : StandardScaler sur 8 colonnes numériques +
      OneHotEncoder(handle_unknown="ignore") sur RUE, JOUR, MTO ;
    - v2 : 2 584 observations, 24 catégories RUE (sensibles à la casse,
      ex. « Pelouze », « Moscou »), 43 variables en sortie ;
    - cible : taux d'occupation (moyenne ≈ 0,86) ; la sortie peut dépasser 1
      (jusqu'à ≈ 1,34 en théorie), d'où le bornage signalé.
    """

    def __init__(self, modele, preprocesseur, version="?"):
        self.modele = modele
        self.prepro = preprocesseur
        self.version = version
        self.colonnes = list(getattr(preprocesseur, "feature_names_in_", [])) or [
            "JOUR", "RUE", "VILLE", "TRAFIC", "% PARKING OC", "NBR PLACES",
            "REVENUS / H", "VEHICULES / H", "MTO", "TEMPERATURE", "HEURE_SIN", "HEURE_COS",
        ]
        self.rues_apprises = self._categories("RUE")
        self.mto_appris = self._categories("MTO")
        # Index par clé normalisée. En cas de doublon (« Berne » / « Berne »
        # avec espace final dans la v2), on garde la version sans espace parasite.
        self._index_rues = {}
        for r in sorted(self.rues_apprises, key=lambda c: c != c.strip()):
            self._index_rues.setdefault(cle_rue(r), r)
        self.PLAGE_PLACES = self._plage("NBR PLACES", defaut=(0, 69))

    def _plage(self, colonne, defaut):
        """Plage « vue à l'entraînement » : moyenne ± 3 écarts-types du scaler."""
        try:
            for _, tr, cols in self.prepro.transformers_:
                if hasattr(tr, "mean_") and colonne in list(cols):
                    i = list(cols).index(colonne)
                    m, s = float(tr.mean_[i]), float(tr.scale_[i])
                    return (max(0.0, m - 3 * s), m + 3 * s)
        except Exception:
            pass
        return defaut

    def _categories(self, colonne):
        try:
            for _, tr, cols in self.prepro.transformers_:
                if hasattr(tr, "categories_") and colonne in list(cols):
                    return [str(c) for c in tr.categories_[list(cols).index(colonne)]]
        except Exception:
            pass
        return []

    def categorie_rue(self, nom):
        """
        Retourne la catégorie RUE apprise correspondant au nom (ex. « Boulevard
        Voltaire » -> « Voltaire »), ou None si la rue n'a pas été vue.
        """
        cle = cle_rue(nom)
        if cle in self._index_rues:
            return self._index_rues[cle]
        for cle_apprise, cat in self._index_rues.items():
            # Uniquement pour un nom tronqué à l'entraînement (« Commandant L »
            # ~ « commandant lamy »). Pas de préfixe libre : « italie » ne doit
            # pas capturer « italiens ».
            mots = cle_apprise.split()
            if len(mots) >= 2 and len(mots[-1]) == 1 and cle.startswith(cle_apprise):
                return cat
        return None

    def construire_entrees(self, rues, arrdt, mto, temperature, maintenant, trafic=None):
        minutes = maintenant.hour * 60 + maintenant.minute
        socio = DATA_ARRDT.get(arrdt, SOCIO_DEFAUT)
        lignes = []
        for nom, places in rues:
            cat = self.categorie_rue(nom)
            lignes.append({
                "DATE": maintenant.strftime("%d/%m/%Y"),
                "JOUR": JOURS_FR.get(maintenant.strftime("%A")),
                "HEURE": maintenant.strftime("%H:%M"),
                # Catégorie exacte si la rue est connue ; sinon le nom tel quel,
                # ignoré par l'encodeur (handle_unknown="ignore").
                "RUE": cat if cat else libelle_rue(nom),
                "VILLE": "Paris",
                "TRAFIC": TRAFIC_DEFAUT if trafic is None else float(trafic),
                "% PARKING OC": PARKING_OC_DEFAUT,
                "NBR PLACES": float(places),
                "REVENUS / H": socio["REV_M"],
                "VEHICULES / H": socio["VEH"],
                "MTO": mto,
                "TEMPERATURE": float(temperature),
                "HEURE_MINUTES": minutes,
                "HEURE_SIN": float(np.sin(2 * np.pi * minutes / 1440)),
                "HEURE_COS": float(np.cos(2 * np.pi * minutes / 1440)),
            })
        return pd.DataFrame(lignes)

    def predire(self, rues, arrdt, mto, temperature, maintenant=None, trafic=None):
        """
        rues : liste de (nom, nombre_de_places). Une seule passe du modèle.
        trafic : 0 / 0,5 / 1 mesuré en direct, ou None pour la valeur par défaut (0).
        Retourne une liste de Prediction dans le même ordre.
        """
        if not rues:
            return []
        if maintenant is None:
            maintenant = datetime.now(_tz_paris())
        df = self.construire_entrees(rues, arrdt, mto, temperature, maintenant, trafic)
        X = self.prepro.transform(df[self.colonnes])
        sorties = np.asarray(self.modele.predict(X), dtype=float).ravel()
        resultats = []
        for (nom, places), brute in zip(rues, sorties):
            alertes = []
            occ = float(min(1.0, max(0.0, brute)))
            bornee = occ != float(brute)
            if bornee:
                alertes.append(f"Sortie du modèle {brute:.3f} hors de [0, 1], bornée pour l'affichage.")
            rue_connue = self.categorie_rue(nom) is not None
            hors_plage = not (self.PLAGE_PLACES[0] <= places <= self.PLAGE_PLACES[1])
            # + 1e-9 : évite que 30 × (1 − 0,9) = 2,9999… donne 2 au lieu de 3.
            libres = max(0, min(int(places), math.floor(places * (1 - occ) + 1e-9)))
            resultats.append(Prediction(float(brute), occ, libres, int(places), rue_connue,
                                        hors_plage, bornee, alertes))
        return resultats


def _tz_paris():
    from zoneinfo import ZoneInfo  # bibliothèque standard (remplace pytz)
    return ZoneInfo("Europe/Paris")


# Règle de couleur (à ajuster ici) :
#   rouge  : occupation prédite > 85 %  ET  moins de 3 places libres estimées
#   vert   : occupation prédite < 75 %  ET  au moins 3 places libres estimées
#   orange : tous les autres cas
SEUIL_ROUGE_OCCUPATION = 0.85
SEUIL_VERT_OCCUPATION = 0.75
SEUIL_PLACES_LIBRES = 3

ROUGE, ORANGE, VERT, GRIS = "#e74c3c", "#f39c12", "#27ae60", "#7f8c8d"


def couleur_pour_prediction(occupation, libres):
    """Couleur d'une rue à partir du taux d'occupation prédit et des places libres estimées."""
    if occupation is None or libres is None:
        return GRIS
    if occupation > SEUIL_ROUGE_OCCUPATION and libres < SEUIL_PLACES_LIBRES:
        return ROUGE
    if occupation < SEUIL_VERT_OCCUPATION and libres >= SEUIL_PLACES_LIBRES:
        return VERT
    return ORANGE


def couleur_pour_places(libres):
    """Vert = plus de 3 places estimées, orange = 2 ou 3, rouge = 0 ou 1, gris = inconnu."""
    if libres is None:
        return "#7f8c8d"
    if libres <= 1:
        return "#e74c3c"
    if libres <= 3:
        return "#f39c12"
    return "#27ae60"


# ============================================================
# Podium : meilleures rues autour d'un point
# ============================================================
RAYONS = (300, 500, 800)           # rayon par défaut, puis élargissements proposés si « saturé »
RANG_COULEUR = {VERT: 0, ORANGE: 1, ROUGE: 2}


def distance_rue(lat, lon, rue):
    """
    Distance à vol d'oiseau (m) entre un point et le point le plus proche
    des emprises de stationnement d'une rue. Retourne (distance, [lat, lon]) ou (None, None).
    """
    meilleur, point = None, None
    for e in rue.get("emprises") or []:
        candidats = [c for _, coords in e.get("geometries") or [] for c in coords]
        if e.get("point"):
            candidats.append(list(e["point"]))
        for c in candidats:
            d = distance_m(lat, lon, c[0], c[1])
            if meilleur is None or d < meilleur:
                meilleur, point = d, [float(c[0]), float(c[1])]
    return meilleur, point


def classer_rues(rues, predictions, lat, lon):
    """
    Classe les rues qui ont une estimation :
      1. meilleure couleur (vert, puis orange, puis rouge) ;
      2. puis le plus de places libres estimées ;
      3. à égalité, la plus proche (vol d'oiseau).
    """
    classement = []
    for cle, pred in predictions.items():
        rue = rues.get(cle)
        if rue is None:
            continue
        couleur = couleur_pour_prediction(pred.occupation, pred.libres)
        distance, point = distance_rue(lat, lon, rue)
        classement.append({
            "cle": cle,
            "label": rue["label"],
            "couleur": couleur,
            "libres": pred.libres,
            "places": pred.places,
            "occupation": pred.occupation,
            "pct_libres": 1.0 - pred.occupation,
            "distance": distance,
            "point": point,
            "rue_connue": pred.rue_connue,
        })
    classement.sort(key=lambda r: (RANG_COULEUR.get(r["couleur"], 3), -r["libres"],
                                   r["distance"] if r["distance"] is not None else float("inf")))
    return classement


def est_sature(classement):
    """Saturé : aucune estimation, ou même la meilleure rue est rouge."""
    return not classement or classement[0]["couleur"] == ROUGE


def rayon_suivant(rayon):
    """Rayon d'élargissement proposé (500 puis 800 m), ou None si déjà au maximum."""
    plus_grands = [r for r in RAYONS if r > rayon]
    return plus_grands[0] if plus_grands else None


def arrondir_distance(distance):
    """« à 140 m » : arrondi aux 10 m (au moins 10 m)."""
    if distance is None:
        return None
    return max(10, int(round(distance / 10.0)) * 10)


def liens_itineraire(lat, lon):
    """Liens qui ouvrent directement l'itinéraire en voiture (app si installée, sinon site)."""
    return {
        "waze": f"https://waze.com/ul?ll={lat:.6f},{lon:.6f}&navigate=yes",
        "maps": f"https://www.google.com/maps/dir/?api=1&destination={lat:.6f},{lon:.6f}&travelmode=driving",
    }
