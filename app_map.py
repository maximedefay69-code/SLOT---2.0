"""
SLOT 2.0 — Radar de stationnement (point d'entrée Streamlit).

Lancer en local :  streamlit run app_map.py
"""

import base64
import os
import time
from datetime import datetime

import folium
import joblib
import streamlit as st
from streamlit_folium import st_folium

import importlib

import slot_core as core

# Streamlit Cloud relance app_map.py après une mise à jour sur GitHub, mais peut garder
# l'ancien slot_core.py en mémoire (d'où des AttributeError). On le recharge si le fichier
# sur disque est plus récent que la version en mémoire (ou si celle-ci est une ancienne
# version sans date de chargement).
if getattr(core, "_MTIME_CHARGE", None) != os.path.getmtime(core.__file__):
    core = importlib.reload(core)

_LOGO_ICONE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SLOT_img.png")
st.set_page_config(page_title="SLOT", page_icon=_LOGO_ICONE if os.path.exists(_LOGO_ICONE) else "🅿️",
                   layout="wide", initial_sidebar_state="collapsed")

DOSSIER = os.path.dirname(os.path.abspath(__file__))
# Versions du modèle, par ordre de préférence : (nom affiché, modèle, préprocesseur).
# Le modèle et son préprocesseur doivent TOUJOURS être de la même version.
VERSIONS_MODELE = [
    ("SLOT v2", "modele_SLOT_v2.pkl", "prepr_SLOT_v2.pkl"),
    ("DA (ancienne version)", "modele_lightgbmDA.pkl", "preprocessorDA.pkl"),
]
FICHIER_LOGO = os.path.join(DOSSIER, "SLOT_img.png")


# ============================================================
# 1. CHARGEMENT DU MODÈLE (une seule fois par serveur)
# ============================================================
@st.cache_resource(show_spinner=False)
def charger_modele():
    """Charge la version la plus récente disponible ; signale les échecs."""
    erreurs = []
    for nom, f_modele, f_prepro in VERSIONS_MODELE:
        try:
            modele = joblib.load(os.path.join(DOSSIER, f_modele))
            prepro = joblib.load(os.path.join(DOSSIER, f_prepro))
            m = core.ModeleSlot(modele, prepro)
            # Attribut posé après coup : fonctionne même si Streamlit Cloud a gardé
            # en mémoire une ancienne version de slot_core.py (sans `version`).
            m.version = nom
            if erreurs:
                m.avertissement = "Version préférée indisponible : " + " | ".join(erreurs)
            return m, None
        except Exception as e:  # fichier absent, version de bibliothèque incompatible...
            erreurs.append(f"{nom} : {type(e).__name__}: {e}")
    return None, " | ".join(erreurs)


# ============================================================
# 2. APPELS RÉSEAU MIS EN CACHE
#    Les coordonnées sont arrondies (~1 m) pour réutiliser le cache.
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def c_geocoder(numero, type_v, nom_v):
    try:
        return core.geocoder_adresse(numero, type_v, nom_v), None
    except core.ApiError as e:
        return None, str(e)


NOMS_CLE_GOOGLE = ("GOOGLE_MAPS_API_KEY", "google_maps_api_key", "GOOGLE_API_KEY", "google_api_key",
                   "GMAPS_API_KEY", "MAPS_API_KEY")


def lire_cle_google():
    """
    Clé Google Maps depuis les secrets Streamlit (nom exact ou variante courante,
    au premier niveau ou dans une section comme [google]), sinon variable d'environnement.
    Retourne (clé ou None, emplacement trouvé ou None).
    """
    try:
        secrets = dict(st.secrets)
    except Exception:  # pas de fichier secrets en local
        secrets = {}
    for nom in NOMS_CLE_GOOGLE:
        if secrets.get(nom):
            return str(secrets[nom]).strip(), nom
    for section, contenu in secrets.items():
        if hasattr(contenu, "items"):
            for nom in NOMS_CLE_GOOGLE:
                if contenu.get(nom):
                    return str(contenu[nom]).strip(), f"[{section}] {nom}"
    for nom in NOMS_CLE_GOOGLE:
        if os.environ.get(nom):
            return os.environ[nom].strip(), f"variable d'environnement {nom}"
    return None, None


@st.cache_data(ttl=300, show_spinner=False)
def c_trafic(lat, lon, _cle):
    # Coordonnées arrondies à ~100 m : même résultat pendant 5 min pour le quartier,
    # ce qui limite les appels facturés par Google. `_cle` est exclue du cache (préfixe _).
    return core.obtenir_trafic(lat, lon, _cle)


@st.cache_data(ttl=600, show_spinner=False)
def c_meteo(lat, lon):
    return core.obtenir_meteo(lat, lon)


@st.cache_data(ttl=3600, show_spinner=False)
def c_emprises(lat, lon, rayon, arrdt):
    return core.recuperer_emprises(lat, lon, rayon, arrdt)


@st.cache_data(ttl=1800, show_spinner=False)
def c_rues_osm(lat, lon, rayon):
    return core.recuperer_rues_osm(lat, lon, rayon)


# ============================================================
# 3. PARCOURS EN TROIS ÉCRANS : intro -> recherche -> résultats
# ============================================================
MARRON = "#603818"        # couleur du logo
CREME = "#FBF7F2"
DUREE_INTRO_S = 2.6
SLOGAN = "Le Fast Food du stationnement"
TYPES_VOIE = ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Passage", "Impasse", "Allée", "Cours", "Square"]


@st.cache_resource(show_spinner=False)
def logo_base64():
    try:
        with open(FICHIER_LOGO, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except OSError:
        return None


def aller_a(ecran):
    st.session_state["ecran"] = ecran


# Allure « application mobile » : sans en-tête ni pied de page Streamlit, marges réduites.
st.markdown(f"""<style>
header[data-testid="stHeader"], footer, #MainMenu {{ display: none !important; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
.slot-entete {{ text-align: center; margin: .4rem 0 1.2rem; }}
.slot-entete img {{ width: 64px; height: 64px; border-radius: 16px; }}
.slot-entete .nom {{ font-size: 2rem; font-weight: 800; letter-spacing: .18em; color: {MARRON}; margin-top: .3rem; }}
.slot-entete .slogan {{ color: #8a7a6c; font-size: .95rem; }}
div[data-testid="stFormSubmitButton"] button {{ min-height: 3rem; font-size: 1.05rem; font-weight: 700; }}
</style>""", unsafe_allow_html=True)

ecran = st.session_state.setdefault("ecran", "intro")

# ---------- Écran 1 : intro (une fois par ouverture de l'app) ----------
if ecran == "intro":
    img = logo_base64()
    logo_html = f'<img src="data:image/png;base64,{img}" alt="SLOT">' if img else ""
    zone = st.empty()
    zone.markdown(f"""
<style>
.slot-intro {{ position: fixed; inset: 0; z-index: 1000000; background: {CREME};
  display: flex; flex-direction: column; align-items: center; justify-content: center; }}
.slot-intro img {{ width: 128px; height: 128px; border-radius: 30px;
  box-shadow: 0 12px 30px rgba(96,56,24,.25); animation: slot-pop .7s cubic-bezier(.2,1.4,.4,1) both; }}
.slot-intro .nom {{ margin-top: 22px; font-size: 3rem; font-weight: 800; letter-spacing: .25em;
  padding-left: .25em; color: {MARRON}; animation: slot-monte .6s .25s both; }}
.slot-intro .slogan {{ margin-top: 6px; color: #8a7a6c; font-size: 1.05rem; animation: slot-monte .6s .45s both; }}
.slot-intro .piste {{ margin-top: 42px; width: 160px; height: 6px; border-radius: 3px;
  background: #eadfd4; overflow: hidden; }}
.slot-intro .barre {{ height: 100%; width: 0; border-radius: 3px; background: {MARRON};
  animation: slot-charge {DUREE_INTRO_S - 0.2}s .3s ease-in-out forwards; }}
@keyframes slot-pop {{ from {{ transform: scale(.6); opacity: 0; }} to {{ transform: scale(1); opacity: 1; }} }}
@keyframes slot-monte {{ from {{ transform: translateY(12px); opacity: 0; }} to {{ transform: none; opacity: 1; }} }}
@keyframes slot-charge {{ from {{ width: 0; }} to {{ width: 100%; }} }}
</style>
<div class="slot-intro">
  {logo_html}
  <div class="nom">SLOT</div>
  <div class="slogan">{SLOGAN}</div>
  <div class="piste"><div class="barre"></div></div>
</div>""", unsafe_allow_html=True)
    # Le vrai chargement (modèle IA) se fait pendant l'animation.
    debut = time.monotonic()
    charger_modele()
    reste = DUREE_INTRO_S - (time.monotonic() - debut)
    if reste > 0:
        time.sleep(reste)
    st.session_state["ecran"] = "recherche"
    st.rerun()

modele, erreur_modele = charger_modele()


def entete_slot():
    img = logo_base64()
    logo_html = f'<img src="data:image/png;base64,{img}" alt="SLOT">' if img else ""
    st.markdown(f'<div class="slot-entete">{logo_html}<div class="nom">SLOT</div>'
                f'<div class="slogan">{SLOGAN}</div></div>', unsafe_allow_html=True)


# ---------- Écran 2 : recherche ----------
if ecran == "recherche":
    _, centre, _ = st.columns([1, 2, 1])  # colonne centrale sur ordinateur, pleine largeur sur téléphone
    with centre:
        entete_slot()
        if erreur_modele:
            st.error("Le modèle IA n'a pas pu être chargé : seules les places recensées seront affichées. "
                     f"Détail : {erreur_modele}")
        prec = st.session_state.get("derniere_recherche") or {}
        with st.form("formulaire_adresse", border=False):
            c1, c2 = st.columns([1, 2])
            with c1:
                num_v = st.number_input("N°", min_value=1, max_value=999, value=int(prec.get("num", 6)), step=1)
            with c2:
                type_prec = prec.get("type", "Rue")
                type_v = st.selectbox("Type de voie", TYPES_VOIE,
                                      index=TYPES_VOIE.index(type_prec) if type_prec in TYPES_VOIE else 0)
            nom_v = st.text_input("Nom de la voie", value=prec.get("nom", ""), placeholder="ex. Pelouze")
            with st.expander("Options"):
                rayon = st.select_slider("Rayon de recherche (m)", options=[150, 250, 300, 400, 500],
                                         value=int(prec.get("rayon", 300)))
                afficher_osm = st.checkbox(
                    "Tracer aussi les rues sans stationnement (OSM, plus lent)", value=bool(prec.get("osm", False)),
                    help="Ajoute en gris les rues sans place voiture recensée, via OpenStreetMap/Overpass "
                         "(serveurs publics souvent saturés : jusqu'à 20 s d'attente). "
                         "N'a aucun effet sur les estimations du modèle.",
                )
            lancer = st.form_submit_button("🔎  Rechercher", type="primary", use_container_width=True)
        st.caption("Estimation indicative par un modèle d'IA, à partir des places recensées par la "
                   "Ville de Paris. Ce n'est pas un comptage en direct."
                   + (f" Modèle : {modele.version}." if modele else ""))

    if lancer:
        if not nom_v.strip():
            with centre:
                st.warning("Indique le nom de la voie.")
            st.stop()
        st.session_state["derniere_recherche"] = {
            "num": int(num_v), "type": type_v, "nom": nom_v.strip(), "rayon": int(rayon), "osm": afficher_osm,
        }
        st.session_state["ecran"] = "resultats"
        st.rerun()
    st.stop()

# ---------- Écran 3 : résultats (carte + prédictions) ----------
recherche = st.session_state.get("derniere_recherche")
if not recherche or not recherche.get("nom"):
    st.session_state["ecran"] = "recherche"
    st.rerun()

st.button("←  Nouvelle recherche", on_click=aller_a, args=("recherche",))


# ============================================================
# 4. RECHERCHE ET CALCUL
# ============================================================
with st.status("Recherche en cours…", expanded=False) as statut:
    statut.update(label="Géocodage de l'adresse…")
    geo, err_geo = c_geocoder(recherche["num"], recherche["type"], recherche["nom"])
    if err_geo:
        statut.update(label="Erreur de géocodage", state="error")
        st.error(f"Le service de géocodage ne répond pas correctement : {err_geo}")
        st.stop()
    if not geo:
        statut.update(label="Adresse introuvable", state="error")
        st.error(
            "Adresse introuvable dans Paris. Vérifie le numéro et le nom de la voie "
            "(par exemple « Voltaire » avec le type « Boulevard »)."
        )
        st.stop()

    lat, lon, arrdt = round(geo["lat"], 5), round(geo["lon"], 5), geo["arrdt"]

    statut.update(label="Météo…")
    mto, temperature, err_meteo = c_meteo(lat, lon)

    statut.update(label="Trafic en temps réel (Google Maps)…")
    cle_google, emplacement_cle = lire_cle_google()
    trafic = c_trafic(round(lat, 3), round(lon, 3), cle_google)

    statut.update(label="Places de stationnement (Paris Open Data)…")
    res_od = c_emprises(lat, lon, recherche["rayon"], arrdt)

    troncons_osm, erreurs_osm = [], []
    if recherche["osm"]:
        statut.update(label="Tracé des rues (OpenStreetMap)…")
        troncons_osm, erreurs_osm = c_rues_osm(lat, lon, recherche["rayon"])

    rues = core.agreger_par_rue(res_od["emprises"])
    cle_principale = core.cle_rue(geo["rue"])

    # Prédiction IA : une seule passe pour toutes les rues avec des places connues.
    predictions = {}
    erreur_ia = None
    a_predire = [(r["cle"], r) for r in rues.values() if r["places"] > 0]
    if modele and a_predire:
        statut.update(label="Estimation IA…")
        try:
            sorties = modele.predire(
                [(r["label"], r["places"]) for _, r in a_predire], arrdt, mto, temperature,
                trafic=trafic["score"],
            )
            predictions = {cle: p for (cle, _), p in zip(a_predire, sorties)}
        except Exception as e:
            erreur_ia = f"{type(e).__name__}: {e}"
    statut.update(label="Recherche terminée", state="complete")

rue_principale = rues.get(cle_principale)
pred_principale = predictions.get(cle_principale)


# ============================================================
# 5. CARTE
# ============================================================
carte = folium.Map(location=[lat, lon], zoom_start=16, max_zoom=19, tiles=None, control_scale=True)

# Fonds de carte épurés, sans clé API (CARTO, Stadia, Mapbox en exigent une).
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
ATTRIBUTION_ESRI = "Fond © Esri, HERE, Garmin, © contributeurs OpenStreetMap"

# Secours : OpenStreetMap passé en gris clair par un filtre CSS (le fond OSM est vérifié).
folium.TileLayer("OpenStreetMap", name="Plan OpenStreetMap épuré (gris)", max_zoom=19, show=False).add_to(carte)
carte.get_root().header.add_child(folium.Element(
    "<style>img.leaflet-tile[src*='tile.openstreetmap.org']"
    "{filter:grayscale(1) brightness(1.12) contrast(.78) opacity(.85);}</style>"
))
# Par défaut (ajouté en dernier) : Esri Light Gray Canvas, très épuré.
# Ses tuiles existent jusqu'au zoom 16 ; au-delà, Leaflet les agrandit (max_native_zoom).
folium.TileLayer(
    tiles=ESRI.format("World_Light_Gray_Base"), attr=ATTRIBUTION_ESRI,
    name="Plan épuré (gris clair)", max_native_zoom=16, max_zoom=19, show=True,
).add_to(carte)
# Noms de rues AU-DESSUS des traits colorés (calque dédié qui laisse passer les clics),
# pour qu'ils restent lisibles sur les rues coloriées.
folium.map.CustomPane("noms_rues", z_index=650, pointer_events=False).add_to(carte)
folium.TileLayer(
    tiles=ESRI.format("World_Light_Gray_Reference"), attr=ATTRIBUTION_ESRI,
    name="Noms des rues", overlay=True, control=True, max_native_zoom=16, max_zoom=19,
    pane="noms_rues",
).add_to(carte)

COULEUR_SANS_DONNEE = "#9aa0a6"


def info_rue(cle, nom_defaut):
    """Texte d'infobulle d'une rue (identique pour tous ses tronçons et emprises)."""
    rue, pred = rues.get(cle), predictions.get(cle)
    titre = f"<b>{rue['label'] if rue else nom_defaut}</b>" + (" (rue recherchée)" if cle == cle_principale else "")
    if pred:
        texte = (f"{titre}<br>{pred.libres} place(s) libre(s) estimée(s) sur {rue['places']} recensée(s) "
                 f"dans le rayon<br>Occupation estimée : {pred.occupation:.0%}")
        if not pred.rue_connue:
            texte += "<br><i>Rue absente de l'entraînement : estimation générique.</i>"
        return texte
    if rue:
        return f"{titre}<br>{rue['places']} place(s) recensée(s) — estimation indisponible"
    return f"{titre}<br>Aucun stationnement voiture recensé dans le rayon"


def couleur_rue(cle):
    pred = predictions.get(cle)
    if pred:
        return core.couleur_pour_prediction(pred.occupation, pred.libres)
    return core.couleur_pour_places(None) if cle in rues else COULEUR_SANS_DONNEE


# Rues coloriées d'un seul tenant, en trait continu et épais.
# Tracé utilisé, par ordre de préférence :
#   1. tronçons OpenStreetMap (si Overpass a répondu) ;
#   2. axe reconstruit à partir des emprises de la Ville (toujours disponible) ;
#   3. à défaut (place, carrefour), les emprises elles-mêmes.
traces = [(t["cle"], t["nom"], t["coords"]) for t in troncons_osm]
cles_tracees = {cle for cle, _, _ in traces if cle in rues}
for cle, rue in rues.items():
    if cle not in cles_tracees:
        axe = core.axe_rue(rue["emprises"])
        if axe:
            traces.append((cle, rue["label"], axe))
            cles_tracees.add(cle)
nb_formes = 0

couche_rues = folium.FeatureGroup(name="Rues (estimation IA)", show=True)
# Les rues sans stationnement d'abord (en dessous), la rue recherchée en dernier (au-dessus).
for cle, nom, coords in sorted(traces, key=lambda t: (t[0] in rues, t[0] == cle_principale)):
    avec_donnees = cle in rues
    if cle == cle_principale:
        # Liseré noir sous le trait coloré : la rue ressort sans perdre sa couleur.
        folium.PolyLine(coords, color="#111111", weight=16, opacity=0.9, line_cap="round").add_to(couche_rues)
    elif avec_donnees:
        # Liseré blanc : le trait reste lisible sur le fond de carte.
        folium.PolyLine(coords, color="#ffffff", weight=13, opacity=0.9, line_cap="round").add_to(couche_rues)
    folium.PolyLine(
        coords,
        color=couleur_rue(cle),
        weight=10 if avec_donnees else 3,
        opacity=1.0 if avec_donnees else 0.6,
        line_cap="round",
        line_join="round",
        tooltip=folium.Tooltip(info_rue(cle, nom)),
    ).add_to(couche_rues)
    nb_formes += 1 if avec_donnees else 0

# Emprises de stationnement (les morceaux de la Ville de Paris) :
# - rue déjà tracée en entier -> couche « détail », masquée par défaut ;
# - rue sans tracé OSM (ou Overpass indisponible) -> affichées, pour ne rien perdre.
couche_detail = folium.FeatureGroup(name="Détail des emplacements (Ville de Paris)", show=False)
couche_seules = folium.FeatureGroup(name="Stationnement sans tracé de rue", show=True)
for cle, rue in rues.items():
    couleur = couleur_rue(cle)
    cible = couche_detail if cle in cles_tracees else couche_seules
    info = info_rue(cle, rue["label"])
    for emprise in rue["emprises"]:
        details = folium.Tooltip(
            f"{info}<br><small>Emprise : {emprise['regpri'].title() or 'régime inconnu'}, "
            f"{emprise['places'] if emprise['places'] is not None else '?'} place(s)</small>")
        for genre, coords in emprise["geometries"]:
            if genre == "polygon":
                folium.Polygon(coords, color=couleur, weight=2, fill=True, fill_color=couleur,
                               fill_opacity=0.8, tooltip=details).add_to(cible)
            else:
                folium.PolyLine(coords, color=couleur, weight=6, tooltip=details).add_to(cible)
            nb_formes += cible is couche_seules
        if not emprise["geometries"] and emprise["point"]:
            # Pas de polygone : on montre au moins le point de l'emprise.
            folium.CircleMarker(emprise["point"], radius=5, color=couleur, fill=True, fill_opacity=0.9,
                                tooltip=details).add_to(cible)
            nb_formes += cible is couche_seules

couche_detail.add_to(carte)
couche_seules.add_to(carte)
couche_rues.add_to(carte)

folium.Marker(
    [lat, lon],
    tooltip=geo["adresse"],
    popup=folium.Popup(f"<b>{geo['adresse']}</b>", max_width=300),
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(carte)
folium.LayerControl(collapsed=True).add_to(carte)

st.subheader(f"📍 {geo['adresse']}")
st_folium(carte, height=650, use_container_width=True, returned_objects=[], key="carte_slot")
st.markdown(
    f"<span style='color:{core.VERT}'>■</span> occupation < {core.SEUIL_VERT_OCCUPATION:.0%} "
    f"et au moins {core.SEUIL_PLACES_LIBRES} places libres &nbsp; "
    f"<span style='color:{core.ROUGE}'>■</span> occupation > {core.SEUIL_ROUGE_OCCUPATION:.0%} "
    f"et moins de {core.SEUIL_PLACES_LIBRES} places libres &nbsp; "
    f"<span style='color:{core.ORANGE}'>■</span> autres cas &nbsp; "
    "<span style='color:#7f8c8d'>■</span> estimation indisponible &nbsp; "
    "— liseré noir : rue recherchée ; traits gris fins : rues sans stationnement voiture recensé. "
    "Le détail des emplacements de la Ville s'active via le sélecteur de couches (en haut à droite).",
    unsafe_allow_html=True,
)
if nb_formes == 0:
    st.warning(
        "Aucune emprise de stationnement n'a pu être dessinée dans ce rayon. "
        "Voir les diagnostics ci-dessous : la carte n'est pas complète."
    )


# ============================================================
# 6. RÉSULTATS
# ============================================================
st.markdown("### Résultat")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Arrondissement", f"{arrdt}ᵉ")
c2.metric("Météo", f"{mto}, {temperature:.0f} °C")
c3.metric("Heure", datetime.now(core._tz_paris()).strftime("%H:%M"))
NIVEAUX_TRAFIC = {0.0: "Fluide", 0.5: "Moyen", 1.0: "Dense"}
if trafic["score"] is None:
    c4.metric("Trafic", "Non mesuré", "fluide supposé (0)", delta_color="off")
else:
    c4.metric("Trafic", NIVEAUX_TRAFIC[trafic["score"]], f"{trafic['ecart_s']:+d} s", delta_color="inverse")
if err_meteo:
    st.caption("⚠️ " + err_meteo)
if trafic["score"] is None:
    st.caption(
        "⚠️ Trafic non mesuré : l'estimation suppose une circulation fluide (valeur par défaut). "
        f"Raison : {trafic['erreur']}"
    )

if res_od["methode"] is None:
    st.error("Les données de stationnement de Paris Open Data n'ont pas pu être récupérées. "
             "Aucune estimation n'est possible pour l'instant.")
elif rue_principale is None:
    st.warning(
        f"Aucune emprise de stationnement voiture n'a été trouvée pour **{geo['rue']}** dans un rayon de "
        f"{recherche['rayon']} m. Cela peut signifier qu'il n'y a pas de place autorisée sur cette rue, "
        "ou que la donnée de la Ville est incomplète."
    )
elif pred_principale:
    st.success(
        f"**{geo['rue']}** : environ **{pred_principale.libres} place(s) libre(s)** sur "
        f"{rue_principale['places']} place(s) recensée(s) dans un rayon de {recherche['rayon']} m "
        f"(occupation estimée {pred_principale.occupation:.0%})."
    )
    if not pred_principale.rue_connue:
        st.caption(
            f"ℹ️ Cette rue ne fait pas partie des {len(modele.rues_apprises)} rues observées pendant "
            "l'entraînement. L'estimation repose sur l'heure, le jour, la météo et le nombre de places, "
            "pas sur l'historique propre à la rue."
        )
else:
    st.warning(f"{rue_principale['places']} place(s) recensée(s) pour {geo['rue']}, "
               "mais l'estimation IA n'est pas disponible.")

if predictions:
    st.markdown("#### Rues voisines")
    tableau = sorted(
        ({"Rue": rues[c]["label"], "Places recensées": p.places, "Libres estimées": p.libres,
          "Occupation estimée": f"{p.occupation:.0%}", "Rue connue du modèle": "oui" if p.rue_connue else "non"}
         for c, p in predictions.items()),
        key=lambda d: -d["Libres estimées"],
    )
    st.dataframe(tableau, use_container_width=True, hide_index=True)


# ============================================================
# 7. DIAGNOSTICS
# ============================================================
with st.expander("Diagnostics"):
    st.write(f"**Adresse géocodée :** {geo['adresse']} (score {geo.get('score')}) — rue : {geo['rue']}")
    st.write(f"**Coordonnées :** {lat}, {lon} — arrondissement {arrdt}")

    st.write("**Paris Open Data (emprises de stationnement)**")
    st.write(
        f"- méthode de filtre : {res_od['methode'] or 'échec'} ; enregistrements récupérés : "
        f"{res_od['recuperes']} / total serveur : {res_od['total_serveur']}"
    )
    st.write(f"- emprises lues : {len(res_od['emprises'])} ; rues avec places voiture : {len(rues)}")
    if res_od["tronque"]:
        st.warning("Résultats tronqués : toutes les emprises n'ont pas été téléchargées.")
    for e in res_od["erreurs"]:
        st.caption(f"Erreur Open Data : {e}")
    inconnues = sum(r["places_inconnues"] for r in rues.values())
    if inconnues:
        st.caption(f"{inconnues} emprise(s) sans nombre de places renseigné (non comptées comme 0).")

    st.write("**OpenStreetMap / Overpass**")
    if not recherche["osm"]:
        st.write("- désactivé")
    else:
        st.write(f"- tronçons de rues récupérés : {len(troncons_osm)}")
        for e in erreurs_osm:
            st.caption(f"Erreur Overpass : {e}")
        if not troncons_osm:
            st.caption("Le tracé gris des rues est absent ; les emprises de stationnement restent affichées.")

    st.write("**Modèle IA**")
    if erreur_modele:
        st.write(f"- non chargé : {erreur_modele}")
    else:
        st.write(f"- version utilisée : **{modele.version}**")
        if getattr(modele, "avertissement", None):
            st.warning(modele.avertissement)
        st.write(f"- rues apprises ({len(modele.rues_apprises)}) : "
                 f"{', '.join(dict.fromkeys(r.strip() for r in modele.rues_apprises)) or 'inconnues'}")
        st.write(f"- prédictions calculées : {len(predictions)}")
        if trafic["score"] is None:
            st.write(f"- TRAFIC : indisponible, valeur par défaut {core.TRAFIC_DEFAUT} utilisée")
            st.caption(f"Erreur trafic : {trafic['erreur']}")
            if not cle_google:
                try:
                    noms = ", ".join(sorted(dict(st.secrets).keys())) or "aucun"
                except Exception:
                    noms = "aucun fichier de secrets"
                # Seuls les NOMS des secrets sont affichés, jamais leurs valeurs.
                st.caption(f"Secrets visibles par cette app : {noms}")
        else:
            st.caption(f"Clé Google lue depuis : {emplacement_cle}")
            st.write(f"- TRAFIC : **{trafic['score']}** (écart {trafic['ecart_s']:+d} s, source Google "
                     f"{trafic['source']} ; seuils {core.SEUIL_TRAFIC_MOYEN_S} s / {core.SEUIL_TRAFIC_FORT_S} s)")
        st.write(f"- entrée fixe faute de source temps réel : % PARKING OC = {core.PARKING_OC_DEFAUT}")
        alertes = [a for p in predictions.values() for a in p.alertes]
        hors_plage = [rues[c]["label"] for c, p in predictions.items() if p.hors_plage_places]
        if hors_plage:
            st.caption("Nombre de places hors de la plage vue à l'entraînement pour : " + ", ".join(hors_plage))
        for a in alertes:
            st.caption(a)
    if erreur_ia:
        st.error(f"Erreur de prédiction IA : {erreur_ia}")
