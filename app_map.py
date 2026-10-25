"""
SLOT 2.0 — Radar de stationnement (point d'entrée Streamlit).

Lancer en local :  streamlit run app_map.py
"""

import base64
import os
import tempfile
import time
from datetime import datetime

import folium
import joblib
import streamlit as st
import streamlit.components.v1 as components
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


@st.cache_data(ttl=3600, show_spinner=False)
def c_geocoder_inverse(lat, lon):
    try:
        return core.geocoder_inverse(lat, lon), None
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
# 3. PARCOURS : intro -> choix -> (géolocalisation | adresse) -> résultats
# ============================================================
MARRON = "#603818"        # couleur du logo
CREME = "#FBF7F2"
OR_JACKPOT = "#F2B705"
DUREE_INTRO_S = 3.2
SLOGAN = "Le Jackpot du stationnement !"
TYPES_VOIE = ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Passage", "Impasse", "Allée", "Cours", "Square"]

# Composant de géolocalisation maison, sans dépendance : son HTML est intégré ici et
# écrit au démarrage dans un dossier temporaire (aucun fichier à ajouter sur GitHub).
HTML_COMPOSANT_GEOLOC = """<!doctype html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0">
<script>
  // Composant Streamlit minimal (sans dépendance) : demande la position GPS
  // au navigateur une seule fois, puis la renvoie à Python.
  // Valeur renvoyée : {coords: {latitude, longitude, accuracy}, timestamp}
  //               ou {error: {code, message}}  (1 = refus, 2 = indisponible, 3 = délai dépassé)

  function envoyer(type, donnees) {
    window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, donnees), "*");
  }

  var dejaDemande = false;

  function demanderPosition() {
    if (dejaDemande) return;
    dejaDemande = true;
    if (!navigator.geolocation) {
      envoyer("streamlit:setComponentValue",
              {value: {error: {code: 0, message: "geolocation non supportée"}}, dataType: "json"});
      return;
    }
    navigator.geolocation.getCurrentPosition(
      function (pos) {
        envoyer("streamlit:setComponentValue", {
          value: {
            coords: {
              latitude: pos.coords.latitude,
              longitude: pos.coords.longitude,
              accuracy: pos.coords.accuracy
            },
            timestamp: pos.timestamp
          },
          dataType: "json"
        });
      },
      function (err) {
        envoyer("streamlit:setComponentValue",
                {value: {error: {code: err.code, message: err.message}}, dataType: "json"});
      },
      {enableHighAccuracy: true, timeout: 15000, maximumAge: 0}
    );
  }

  window.addEventListener("message", function (event) {
    if (event.data && event.data.type === "streamlit:render") demanderPosition();
  });

  window.addEventListener("load", function () {
    envoyer("streamlit:componentReady", {apiVersion: 1});
    envoyer("streamlit:setFrameHeight", {height: 0});
  });
</script>
</body>
</html>
"""


@st.cache_resource(show_spinner=False)
def dossier_composant_geoloc():
    dossier = os.path.join(tempfile.gettempdir(), "slot_composant_geoloc")
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, "index.html")
    try:
        with open(chemin, encoding="utf-8") as f:
            a_jour = f.read() == HTML_COMPOSANT_GEOLOC
    except OSError:
        a_jour = False
    if not a_jour:
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(HTML_COMPOSANT_GEOLOC)
    return dossier


_composant_geoloc = components.declare_component("slot_geoloc", path=dossier_composant_geoloc())


@st.cache_resource(show_spinner=False)
def logo_base64():
    try:
        with open(FICHIER_LOGO, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except OSError:
        return None


def aller_a(ecran):
    st.session_state["ecran"] = ecran


def lancer_geolocalisation():
    """Nouvelle demande de position (nouvelle clé = le composant redemande au téléphone)."""
    st.session_state["n_geoloc"] = st.session_state.get("n_geoloc", 0) + 1
    st.session_state["ecran"] = "geoloc"


def elargir(rayon):
    rech = dict(st.session_state.get("derniere_recherche") or {})
    rech["rayon"] = rayon
    st.session_state["derniere_recherche"] = rech


# Allure « application mobile » : sans en-tête ni pied de page Streamlit, marges réduites.
st.markdown(f"""<style>
header[data-testid="stHeader"], footer, #MainMenu {{ display: none !important; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
.slot-entete {{ text-align: center; margin: .4rem 0 1.2rem; }}
.slot-entete img {{ width: 64px; height: 64px; border-radius: 16px; }}
.slot-entete .nom {{ font-size: 2rem; font-weight: 800; letter-spacing: .18em; color: {MARRON}; margin-top: .3rem; }}
.slot-entete .slogan {{ color: #8a7a6c; font-size: .95rem; }}
div[data-testid="stFormSubmitButton"] button {{ min-height: 3rem; font-size: 1.05rem; font-weight: 700; }}
.st-key-choix_geo button, .st-key-choix_adresse button {{
  min-height: 4.6rem; border-radius: 18px; font-size: 1.12rem; font-weight: 700; }}
.st-key-choix_geo button {{ background: {MARRON}; border-color: {MARRON}; color: #fff; }}
.st-key-choix_geo button:hover {{ background: #4a2b12; border-color: #4a2b12; color: #fff; }}
.slot-carte {{ border-radius: 16px; padding: .9rem 1rem; margin: .5rem 0; background: #fff;
  border: 1px solid #eadfd4; box-shadow: 0 2px 10px rgba(96,56,24,.06); }}
.slot-carte .etiquette {{ font-size: .72rem; font-weight: 800; letter-spacing: .12em; color: #8a7a6c; }}
.slot-carte .rue {{ font-size: 1.15rem; font-weight: 800; color: #2b1a0c; margin-top: .1rem; }}
.slot-carte .detail {{ color: #5b4a3c; margin-top: .15rem; }}
.slot-pastille {{ display: inline-block; width: .8rem; height: .8rem; border-radius: 50%;
  vertical-align: -1px; margin-right: .35rem; }}
.slot-podium {{ border-radius: 16px; padding: .8rem 1rem; margin: .6rem 0 .3rem; }}
.slot-podium .ligne1 {{ font-weight: 800; font-size: 1.05rem; color: #2b1a0c; }}
.slot-podium .ligne2 {{ color: #4a3a2c; margin-top: .15rem; }}
.slot-podium .tag {{ font-size: .75rem; font-weight: 700; background: rgba(0,0,0,.08);
  border-radius: 8px; padding: 1px 6px; margin-left: .3rem; }}
</style>""", unsafe_allow_html=True)

ecran = st.session_state.setdefault("ecran", "intro")

# Mode debug : ajouter ?debug=1 à l'adresse de l'app pour voir les chiffres et les diagnostics.
try:
    if st.query_params.get("debug") == "1":
        st.session_state["debug"] = True
except Exception:
    pass
DEBUG = bool(st.session_state.get("debug"))


def _pixels(rects, couleur):
    return "".join(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{couleur}"/>' for x, y, w, h in rects)


def svg_voiture(carrosserie, vitres="#3b2a1e"):
    """Voiture vue de dessus en pixel art (64 × 28, pixels de 4), avant à droite."""
    return (
        _pixels([(8, -2, 12, 4), (44, -2, 12, 4), (8, 26, 12, 4), (44, 26, 12, 4)], "#1b1b1b")  # roues
        + _pixels([(4, 0, 56, 28), (0, 4, 64, 20)], carrosserie)                                  # caisse
        + _pixels([(40, 4, 8, 20), (12, 4, 6, 20)], vitres)                                     # pare-brise / lunette
        + _pixels([(18, 6, 22, 16)], "rgba(255,255,255,.22)")                                   # toit
        + _pixels([(60, 4, 4, 4), (60, 20, 4, 4)], "#fff6c8")                                   # phares
        + _pixels([(0, 4, 4, 4), (0, 20, 4, 4)], "#d64541")                                     # feux arrière
    )


def html_intro():
    img = logo_base64()
    logo_html = f'<img class="logo" src="data:image/png;base64,{img}" alt="SLOT">' if img else ""
    d = DUREE_INTRO_S - 0.4          # durée de la manœuvre (synchronisée avec la barre)
    tirets = "".join(f'<rect x="{x}" y="34" width="10" height="3" fill="#fbf7f2"/>' for x in range(4, 240, 22))
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
.slot-intro {{ position: fixed; inset: 0; z-index: 1000000; background: {CREME};
  display: flex; flex-direction: column; align-items: center; justify-content: center; }}
.slot-intro .logo {{ width: 96px; height: 96px; border-radius: 24px;
  box-shadow: 0 12px 30px rgba(96,56,24,.25); animation: slot-pop .7s cubic-bezier(.2,1.4,.4,1) both; }}
.slot-intro .nom {{ margin-top: 16px; font-size: 2.8rem; font-weight: 800; letter-spacing: .25em;
  padding-left: .25em; color: {MARRON}; animation: slot-monte .6s .2s both; }}
.slot-intro .slogan {{ margin-top: 4px; color: #8a7a6c; font-size: 1.05rem; animation: slot-monte .6s .35s both; }}
.slot-intro svg.scene {{ margin-top: 18px; width: 264px; height: auto; overflow: visible;
  image-rendering: pixelated; }}
.slot-intro .voiture {{ transform-box: fill-box; transform-origin: center;
  animation: slot-creneau {d}s .2s ease-in-out both; }}
.slot-intro .place {{ opacity: 0; animation: slot-place .5s {0.2 + 0.86 * d:.2f}s both; }}
.slot-intro .jackpot {{ font-family: 'Press Start 2P', ui-monospace, monospace; font-size: 11px;
  fill: {OR_JACKPOT}; stroke: {MARRON}; stroke-width: .6px; paint-order: stroke;
  opacity: 0; transform-box: fill-box; transform-origin: center;
  animation: slot-jackpot .55s {0.2 + 0.88 * d:.2f}s cubic-bezier(.2,1.6,.4,1) both; }}
.slot-intro .piste {{ margin-top: 18px; width: 180px; height: 6px; border-radius: 3px;
  background: #eadfd4; overflow: hidden; }}
.slot-intro .barre {{ height: 100%; width: 0; border-radius: 3px; background: {MARRON};
  animation: slot-charge {d}s .2s ease-in-out forwards; }}
@keyframes slot-pop {{ from {{ transform: scale(.6); opacity: 0; }} to {{ transform: scale(1); opacity: 1; }} }}
@keyframes slot-monte {{ from {{ transform: translateY(12px); opacity: 0; }} to {{ transform: none; opacity: 1; }} }}
@keyframes slot-charge {{ from {{ width: 0; }} to {{ width: 100%; }} }}
@keyframes slot-creneau {{
  0%   {{ transform: translate(-90px, 18px) rotate(0deg); }}
  36%  {{ transform: translate(146px, 18px) rotate(0deg); }}
  42%  {{ transform: translate(146px, 18px) rotate(0deg); }}
  70%  {{ transform: translate(104px, 42px) rotate(-20deg); }}
  86%  {{ transform: translate(89px, 66px) rotate(0deg); }}
  100% {{ transform: translate(89px, 66px) rotate(0deg); }}
}}
@keyframes slot-place {{ from {{ opacity: 0; }} to {{ opacity: .55; }} }}
@keyframes slot-jackpot {{ from {{ opacity: 0; transform: scale(.4); }} to {{ opacity: 1; transform: scale(1); }} }}
</style>
<div class="slot-intro">
  {logo_html}
  <div class="nom">SLOT</div>
  <div class="slogan">{SLOGAN}</div>
  <svg class="scene" viewBox="0 -22 240 134" shape-rendering="crispEdges" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="12" width="240" height="48" fill="#cfc4b8"/>
    {tirets}
    <rect x="0" y="60" width="240" height="40" fill="#e6dcd1"/>
    <rect x="0" y="100" width="240" height="12" fill="#b9ab9c"/>
    <rect class="place" x="82" y="63" width="78" height="34" fill="#27ae60"/>
    <g transform="translate(12 66)">{svg_voiture("#7a5a3e")}</g>
    <g transform="translate(166 66)">{svg_voiture("#8f7e6e")}</g>
    <g class="voiture">{svg_voiture(OR_JACKPOT, "#5a3d14")}</g>
    <text class="jackpot" x="120" y="-4" text-anchor="middle">JACKPOT !</text>
  </svg>
  <div class="piste"><div class="barre"></div></div>
</div>"""


def entete_slot():
    img = logo_base64()
    logo_html = f'<img src="data:image/png;base64,{img}" alt="SLOT">' if img else ""
    st.markdown(f'<div class="slot-entete">{logo_html}<div class="nom">SLOT</div>'
                f'<div class="slogan">{SLOGAN}</div></div>', unsafe_allow_html=True)


# ---------- Écran 1 : intro (une fois par ouverture de l'app) ----------
if ecran == "intro":
    zone = st.empty()
    zone.markdown(html_intro(), unsafe_allow_html=True)
    # Le vrai chargement (modèle IA) se fait pendant l'animation.
    debut = time.monotonic()
    charger_modele()
    reste = DUREE_INTRO_S - (time.monotonic() - debut)
    if reste > 0:
        time.sleep(reste)
    st.session_state["ecran"] = "choix"
    st.rerun()

modele, erreur_modele = charger_modele()

# ---------- Écran 2 : choix ----------
if ecran == "choix":
    _, centre, _ = st.columns([1, 2, 1])  # colonne centrale sur ordinateur, pleine largeur sur téléphone
    with centre:
        entete_slot()
        message = st.session_state.pop("message_choix", None)
        if message:
            st.warning(message)
        st.button(f"📍  Me stationner autour de moi", key="choix_geo", type="primary",
                  use_container_width=True, on_click=lancer_geolocalisation)
        st.caption(f"Option 1 · à partir de ta position, dans un rayon de {core.RAYONS[0]} m autour de toi.")
        st.write("")
        st.button("✏️  Trouver du stationnement à partir d'une adresse", key="choix_adresse",
                  use_container_width=True, on_click=aller_a, args=("recherche",))
        st.caption("Option 2 · saisis une adresse à Paris.")
        st.write("")
        st.caption("Estimation indicative par un modèle d'IA, à partir des places recensées par la "
                   "Ville de Paris. Ce n'est pas un comptage en direct. Ta position n'est jamais enregistrée."
                   + (f" Modèle : {modele.version}." if modele else ""))
    st.stop()

# ---------- Écran 2a : géolocalisation ----------
if ecran == "geoloc":
    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        entete_slot()
        valeur = _composant_geoloc(key=f"geoloc_{st.session_state.get('n_geoloc', 0)}", default=None)
        position = core.lire_position(valeur)
        if position["etat"] == "ok":
            st.session_state["derniere_recherche"] = {
                "mode": "geo", "lat": position["lat"], "lon": position["lon"],
                "precision": position["precision"], "rayon": core.RAYONS[0], "osm": False,
            }
            st.session_state["ecran"] = "resultats"
            st.rerun()
        if position["etat"] == "refus":
            st.session_state["message_recherche"] = (
                "Localisation refusée : saisis une adresse. Pour l'activer plus tard, autorise la "
                "localisation pour ce site dans les réglages de ton navigateur.")
            st.session_state["ecran"] = "recherche"
            st.rerun()
        if position["etat"] == "erreur":
            st.error(position["message"])
            st.button("↻  Réessayer", use_container_width=True, type="primary", on_click=lancer_geolocalisation)
        else:
            st.info("📍 Localisation en cours… Autorise l'accès à ta position quand ton téléphone te le demande.")
        st.button("✏️  Saisir une adresse à la place", use_container_width=True,
                  on_click=aller_a, args=("recherche",))
        st.button("←  Retour", on_click=aller_a, args=("choix",))
    st.stop()

# ---------- Écran 2b : recherche par adresse ----------
if ecran == "recherche":
    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        st.button("←  Retour", on_click=aller_a, args=("choix",))
        entete_slot()
        message = st.session_state.pop("message_recherche", None)
        if message:
            st.info(message)
        if erreur_modele:
            st.error("Le modèle IA n'a pas pu être chargé : seules les places recensées seront affichées. "
                     f"Détail : {erreur_modele}")
        prec = st.session_state.get("derniere_recherche") or {}
        if prec.get("mode") != "adresse":
            prec = {}
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
                rayon = st.select_slider("Rayon de recherche (m)", options=[150, 250, 300, 500, 800],
                                         value=int(prec.get("rayon", core.RAYONS[0])))
                afficher_osm = st.checkbox(
                    "Tracer aussi les rues sans stationnement (OSM, plus lent)", value=bool(prec.get("osm", False)),
                    help="Ajoute en gris les rues sans place voiture recensée, via OpenStreetMap/Overpass "
                         "(serveurs publics souvent saturés : jusqu'à 20 s d'attente). "
                         "N'a aucun effet sur les estimations du modèle.",
                )
            lancer = st.form_submit_button("🔎  Rechercher", type="primary", use_container_width=True)

    if lancer:
        if not nom_v.strip():
            with centre:
                st.warning("Indique le nom de la voie.")
            st.stop()
        st.session_state["derniere_recherche"] = {
            "mode": "adresse", "num": int(num_v), "type": type_v, "nom": nom_v.strip(),
            "rayon": int(rayon), "osm": afficher_osm,
        }
        st.session_state["ecran"] = "resultats"
        st.rerun()
    st.stop()

# ---------- Écran 3 : résultats (carte + prédictions) ----------
recherche = st.session_state.get("derniere_recherche")
if not recherche or (recherche.get("mode", "adresse") == "adresse" and not recherche.get("nom")):
    st.session_state["ecran"] = "choix"
    st.rerun()
mode_geo = recherche.get("mode") == "geo"

b1, b2 = st.columns(2)
with b1:
    st.button("←  Accueil", use_container_width=True, on_click=aller_a, args=("choix",))
with b2:
    if mode_geo:
        st.button("↻  Actualiser ma position", use_container_width=True, on_click=lancer_geolocalisation)
    else:
        st.button("✏️  Nouvelle adresse", use_container_width=True, on_click=aller_a, args=("recherche",))


# ============================================================
# 4. RECHERCHE ET CALCUL
# ============================================================
with st.status("Recherche en cours…", expanded=False) as statut:
    if mode_geo:
        statut.update(label="Recherche de ta rue…")
        geo, err_geo = c_geocoder_inverse(round(recherche["lat"], 5), round(recherche["lon"], 5))
    else:
        statut.update(label="Géocodage de l'adresse…")
        geo, err_geo = c_geocoder(recherche["num"], recherche["type"], recherche["nom"])
    if err_geo:
        statut.update(label="Erreur de géocodage", state="error")
        st.error(f"Le service de géocodage ne répond pas correctement : {err_geo}")
        st.stop()
    if not geo:
        statut.update(label="Hors de Paris" if mode_geo else "Adresse introuvable", state="error")
        if mode_geo:
            st.error("Ta position n'est pas dans Paris : SLOT ne couvre que Paris pour l'instant.")
        else:
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
    if recherche.get("osm"):
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

# Podium : meilleure couleur, puis le plus de places libres, puis la plus proche du point de départ.
classement = core.classer_rues(rues, predictions, lat, lon)
podium = classement[:3]
estimation_ok = res_od["methode"] is not None and modele is not None and erreur_ia is None
aucune_place = estimation_ok and not rues                    # aucune place voiture dans le rayon
sature = estimation_ok and bool(classement) and core.est_sature(classement)


# ============================================================
# 5. CARTE
# ============================================================
carte = folium.Map(location=[lat, lon], zoom_start={300: 16, 500: 15, 800: 15}.get(recherche["rayon"], 16), max_zoom=19, tiles=None, control_scale=True)

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


def pastille(couleur):
    return f'<span class="slot-pastille" style="background:{couleur}"></span>'


# Ce que voit l'utilisateur : une couleur et un mot, pas de chiffres de places.
LIBELLES_COULEUR = {core.VERT: "Bonnes chances", core.ORANGE: "Ça peut passer",
                    core.ROUGE: "Difficile", core.GRIS: "Pas d'estimation"}


def badge(couleur):
    return f'{pastille(couleur)}<b>{LIBELLES_COULEUR.get(couleur, LIBELLES_COULEUR[core.GRIS])}</b>'


def info_rue(cle, nom_defaut):
    """Texte d'infobulle d'une rue (identique pour tous ses tronçons et emprises)."""
    rue, pred = rues.get(cle), predictions.get(cle)
    titre = f"<b>{rue['label'] if rue else nom_defaut}</b>" + ((" (ta rue)" if mode_geo else " (rue recherchée)") if cle == cle_principale else "")
    if pred:
        return f"{titre}<br>{LIBELLES_COULEUR[core.couleur_pour_prediction(pred.occupation, pred.libres)]}"
    if rue:
        return f"{titre}<br>Pas d'estimation"
    return f"{titre}<br>Pas de stationnement voiture"


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
        details = folium.Tooltip(info)
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

# Zone de recherche : cercle en pointillés.
folium.Circle([lat, lon], radius=recherche["rayon"], color=MARRON, weight=2, dash_array="6 8",
              fill=False, interactive=False).add_to(carte)

if mode_geo:
    # « Toi » : point bleu façon Plans, avec le halo de précision du GPS.
    if recherche.get("precision"):
        folium.Circle([lat, lon], radius=min(float(recherche["precision"]), 300), color="#007AFF", weight=1,
                      fill=True, fill_color="#007AFF", fill_opacity=0.12, interactive=False).add_to(carte)
    folium.CircleMarker([lat, lon], radius=9, color="#ffffff", weight=3, fill=True, fill_color="#007AFF",
                        fill_opacity=1, tooltip="Tu es ici").add_to(carte)
else:
    folium.Marker(
        [lat, lon],
        tooltip=geo["adresse"],
        popup=folium.Popup(f"<b>{geo['adresse']}</b>", max_width=300),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(carte)

# Médailles du podium sur la carte.
MEDAILLES = ["🥇", "🥈", "🥉"]
for medaille, r in zip(MEDAILLES, podium):
    if r["point"]:
        folium.Marker(
            r["point"], tooltip=f"{medaille} {r['label']}",
            icon=folium.DivIcon(html=f'<div style="font-size:26px;line-height:26px;'
                                     f'filter:drop-shadow(0 1px 2px rgba(0,0,0,.35))">{medaille}</div>',
                                icon_size=(26, 26), icon_anchor=(13, 13)),
        ).add_to(carte)
folium.LayerControl(collapsed=True).add_to(carte)


# ---------- En-tête des résultats + carte « ta rue » ----------
if mode_geo:
    st.markdown(f"#### 📍 Dans un rayon de {recherche['rayon']} m autour de toi")
    if recherche.get("precision") and recherche["precision"] > 50:
        st.caption(f"Position précise à ±{core.arrondir_distance(recherche['precision'])} m.")
else:
    st.markdown(f"#### 📍 Dans un rayon de {recherche['rayon']} m autour de {geo['adresse']}")

etiquette = "TA RUE" if mode_geo else "RUE RECHERCHÉE"
if res_od["methode"] is None:
    detail_rue = "Données de stationnement indisponibles pour l'instant."
    couleur_principale = core.GRIS
elif pred_principale:
    couleur_principale = core.couleur_pour_prediction(pred_principale.occupation, pred_principale.libres)
    detail_rue = badge(couleur_principale)
    if podium and podium[0]["cle"] == cle_principale and not sature:
        detail_rue += "<br>✅ Tu es déjà dans la meilleure rue du coin !"
elif rue_principale:
    couleur_principale = core.GRIS
    detail_rue = badge(core.GRIS)
else:
    couleur_principale = core.GRIS
    detail_rue = "Pas de stationnement voiture recensé sur cette rue. Regarde le podium plus bas 👇"
st.markdown(f'<div class="slot-carte" style="border-left:6px solid {couleur_principale}">'
            f'<div class="etiquette">{etiquette}</div><div class="rue">{geo["rue"]}</div>'
            f'<div class="detail">{detail_rue}</div></div>', unsafe_allow_html=True)

# ---------- Saturé (ou aucune place) : proposer d'élargir ----------
if sature or aucune_place:
    suivant = core.rayon_suivant(recherche["rayon"])
    constat = ("Tout est saturé" if sature else "Aucune place voiture recensée")
    if suivant:
        st.warning(f"😬 {constat} dans un rayon de {recherche['rayon']} m.")
        st.button(f"🔍  Élargir à {suivant} m", type="primary", use_container_width=True,
                  on_click=elargir, args=(suivant,))
    else:
        st.warning(f"😬 Même à {recherche['rayon']} m : {constat.lower()}. Retente dans quelques minutes.")

st_folium(carte, height=520, use_container_width=True, returned_objects=[], key="carte_slot")


# ============================================================
# 6. PODIUM ET RÉSULTATS
# ============================================================
STYLES_PODIUM = [
    ("🥇", "OR", "linear-gradient(135deg,#fff3c4,#f7d86b)", "#c99a06"),
    ("🥈", "ARGENT", "linear-gradient(135deg,#f6f7f9,#d9dce1)", "#9aa0a8"),
    ("🥉", "BRONZE", "linear-gradient(135deg,#fbeadb,#e7b98d)", "#b07a45"),
]

if res_od["methode"] is None:
    st.error("Les données de stationnement de Paris Open Data n'ont pas pu être récupérées. "
             "Aucune estimation n'est possible pour l'instant.")
elif not podium:
    st.info("Aucune estimation disponible dans ce rayon : pas de place voiture recensée, "
            "ou modèle IA indisponible.")
else:
    st.markdown(f"#### 🎰 Le podium à moins de {recherche['rayon']} m")
    for (medaille, nom_rang, fond, bord), r in zip(STYLES_PODIUM, podium):
        distance = core.arrondir_distance(r["distance"])
        a_distance = "ici" if r["cle"] == cle_principale and mode_geo else (
            f"à {distance} m" if distance is not None else "")
        tag = '<span class="tag">📍 ta rue</span>' if r["cle"] == cle_principale and mode_geo else (
            '<span class="tag">rue recherchée</span>' if r["cle"] == cle_principale else "")
        st.markdown(
            f'<div class="slot-podium" style="background:{fond};border:1px solid {bord}">'
            f'<div class="ligne1">{medaille} {r["label"]} · {a_distance}{tag}</div>'
            f'<div class="ligne2">{badge(r["couleur"])}</div></div>',
            unsafe_allow_html=True)
        if r["point"]:
            liens = core.liens_itineraire(*r["point"])
            l1, l2 = st.columns(2)
            with l1:
                st.link_button("🚙  Waze", liens["waze"], use_container_width=True)
            with l2:
                st.link_button("🗺️  Google Maps", liens["maps"], use_container_width=True)

st.markdown(
    f"<small>{badge(core.VERT)} &nbsp; {badge(core.ORANGE)} &nbsp; {badge(core.ROUGE)}"
    " &nbsp;·&nbsp; distances à vol d'oiseau.</small>",
    unsafe_allow_html=True,
)
if nb_formes == 0 and res_od["methode"] is not None:
    st.warning("Aucune emprise de stationnement n'a pu être dessinée dans ce rayon : la carte n'est pas complète.")

st.markdown("#### Conditions en ce moment")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Arrondissement", f"{arrdt}ᵉ")
c2.metric("Météo", f"{mto}, {temperature:.0f} °C")
c3.metric("Heure", datetime.now(core._tz_paris()).strftime("%H:%M"))
NIVEAUX_TRAFIC = {0.0: "Fluide", 0.5: "Moyen", 1.0: "Dense"}
if trafic["score"] is None:
    c4.metric("Trafic", "Non mesuré", "fluide supposé (0)", delta_color="off")
else:
    c4.metric("Trafic", NIVEAUX_TRAFIC[trafic["score"]], f"{trafic['ecart_s']:+d} s", delta_color="inverse")
if DEBUG and err_meteo:
    st.caption("⚠️ " + err_meteo)
if DEBUG and trafic["score"] is None:
    st.caption(
        "⚠️ Trafic non mesuré : l'estimation suppose une circulation fluide (valeur par défaut). "
        f"Raison : {trafic['erreur']}"
    )
if DEBUG and pred_principale and not pred_principale.rue_connue:
    st.caption(
        f"ℹ️ {geo['rue']} ne fait pas partie des {len(modele.rues_apprises)} rues observées pendant "
        "l'entraînement : l'estimation repose sur l'heure, le jour, la météo et le nombre de places."
    )

if DEBUG and classement:
    with st.expander(f"Toutes les rues du rayon ({len(classement)})"):
        st.dataframe(
            [{"Rang": i, "Rue": r["label"], "Distance (m)": core.arrondir_distance(r["distance"]),
              "% libres": f"{r['pct_libres']:.0%}", "Libres estimées": r["libres"],
              "Places recensées": r["places"]} for i, r in enumerate(classement, 1)],
            use_container_width=True, hide_index=True)


# ============================================================
# 7. DIAGNOSTICS
# ============================================================
if not DEBUG:
    st.stop()  # détails techniques réservés au mode debug (ajouter ?debug=1 à l'adresse de l'app)

with st.expander("Diagnostics", expanded=True):
    if mode_geo:
        st.write(f"**Position (géolocalisation) :** {lat}, {lon} — précision "
                 f"{recherche.get('precision') or '?'} m — adresse la plus proche : {geo['adresse']} "
                 f"(à {geo.get('distance_adresse_m', '?')} m) — rue : {geo['rue']}")
    else:
        st.write(f"**Adresse géocodée :** {geo['adresse']} (score {geo.get('score')}) — rue : {geo['rue']}")
    st.write(f"**Coordonnées :** {lat}, {lon} — arrondissement {arrdt} — rayon {recherche['rayon']} m")

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
    if not recherche.get("osm"):
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
