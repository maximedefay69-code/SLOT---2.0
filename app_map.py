"""
SLOT 2.0 — Radar de stationnement (point d'entrée Streamlit).

Lancer en local :  streamlit run app_map.py
"""

import os
from datetime import datetime

import folium
import joblib
import streamlit as st
from streamlit_folium import st_folium

import slot_core as core

st.set_page_config(page_title="SLOT 2.0 - Radar de stationnement", layout="wide")

DOSSIER = os.path.dirname(os.path.abspath(__file__))
FICHIER_MODELE = os.path.join(DOSSIER, "modele_lightgbmDA.pkl")
FICHIER_PREPRO = os.path.join(DOSSIER, "preprocessorDA.pkl")
FICHIER_LOGO = os.path.join(DOSSIER, "SLOT_img.png")


# ============================================================
# 1. CHARGEMENT DU MODÈLE (une seule fois par serveur)
# ============================================================
@st.cache_resource(show_spinner=False)
def charger_modele():
    try:
        modele = joblib.load(FICHIER_MODELE)
        prepro = joblib.load(FICHIER_PREPRO)
        return core.ModeleSlot(modele, prepro), None
    except Exception as e:  # fichier absent, version de bibliothèque incompatible...
        return None, f"{type(e).__name__}: {e}"


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


@st.cache_data(ttl=600, show_spinner=False)
def c_meteo(lat, lon):
    return core.obtenir_meteo(lat, lon)


@st.cache_data(ttl=3600, show_spinner=False)
def c_emprises(lat, lon, rayon, arrdt):
    return core.recuperer_emprises(lat, lon, rayon, arrdt)


@st.cache_data(ttl=1800, show_spinner=False)
def c_rues_osm(lat, lon, rayon):
    return core.recuperer_rues_osm(lat, lon, rayon)


modele, erreur_modele = charger_modele()


# ============================================================
# 3. INTERFACE DE SAISIE (formulaire : aucune requête avant validation)
# ============================================================
col_logo, col_titre = st.columns([1, 12])
with col_logo:
    if os.path.exists(FICHIER_LOGO):
        st.image(FICHIER_LOGO, width=64)
with col_titre:
    st.title("SLOT 2.0 — Radar de stationnement")
st.caption(
    "Estimation indicative des places libres par un modèle d'IA, à partir des "
    "places recensées par la Ville de Paris. Ce n'est pas un comptage en direct."
)

if erreur_modele:
    st.error(
        "Le modèle IA n'a pas pu être chargé : la carte affichera les places "
        f"recensées, sans estimation. Détail : {erreur_modele}"
    )

with st.form("formulaire_adresse"):
    c1, c2, c3, c4 = st.columns([1, 2, 4, 2])
    with c1:
        num_v = st.number_input("N°", min_value=1, max_value=999, value=6, step=1)
    with c2:
        type_v = st.selectbox(
            "Type de voie",
            ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Passage", "Impasse", "Allée", "Cours", "Square"],
        )
    with c3:
        nom_v = st.text_input("Nom de la voie", value="Pelouze")
    with c4:
        rayon = st.select_slider("Rayon (m)", options=[150, 250, 300, 400, 500], value=300)
    c5, c6 = st.columns([2, 5])
    with c5:
        afficher_osm = st.checkbox(
            "Tracer aussi les rues OSM (plus lent)", value=True,
            help="Ajoute en gris le tracé des rues via OpenStreetMap/Overpass. "
                 "Si ce service est indisponible, la carte reste utilisable.",
        )
    lancer = st.form_submit_button("Rechercher", type="primary")

if lancer:
    st.session_state["derniere_recherche"] = {
        "num": int(num_v), "type": type_v, "nom": nom_v.strip(), "rayon": int(rayon), "osm": afficher_osm,
    }

recherche = st.session_state.get("derniere_recherche")
if not recherche or not recherche["nom"]:
    st.info("Saisis une adresse parisienne puis clique sur « Rechercher ».")
    st.stop()


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
                [(r["label"], r["places"]) for _, r in a_predire], arrdt, mto, temperature
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
carte = folium.Map(location=[lat, lon], zoom_start=17, tiles=None, control_scale=True)
# Fond OpenStreetMap seul : accessible sans clé API (CARTO en exige une désormais).
# Légèrement atténué pour faire ressortir les couleurs du stationnement.
folium.TileLayer("OpenStreetMap", name="Plan OpenStreetMap", opacity=0.75).add_to(carte)

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
        return core.couleur_pour_places(pred.libres)
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
    "<span style='color:#27ae60'>■</span> plus de 3 places libres estimées &nbsp; "
    "<span style='color:#f39c12'>■</span> 2 ou 3 &nbsp; "
    "<span style='color:#e74c3c'>■</span> 0 ou 1 &nbsp; "
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
c1, c2, c3 = st.columns(3)
c1.metric("Arrondissement", f"{arrdt}ᵉ")
c2.metric("Météo", f"{mto}, {temperature:.0f} °C")
c3.metric("Heure", datetime.now(core._tz_paris()).strftime("%H:%M"))
if err_meteo:
    st.caption("⚠️ " + err_meteo)

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
            "ℹ️ Le modèle a été entraîné sur 3 rues seulement (Voltaire, Menat, Commandant L…). "
            "Pour cette rue, l'estimation repose sur l'heure, le jour, la météo et le nombre de places, "
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
        st.write(f"- rues apprises : {', '.join(modele.rues_apprises) or 'inconnues'}")
        st.write(f"- prédictions calculées : {len(predictions)}")
        st.write(
            f"- entrées fixes faute de source temps réel : TRAFIC = {core.TRAFIC_DEFAUT}, "
            f"% PARKING OC = {core.PARKING_OC_DEFAUT}"
        )
        alertes = [a for p in predictions.values() for a in p.alertes]
        hors_plage = [rues[c]["label"] for c, p in predictions.items() if p.hors_plage_places]
        if hors_plage:
            st.caption("Nombre de places hors de la plage vue à l'entraînement pour : " + ", ".join(hors_plage))
        for a in alertes:
            st.caption(a)
    if erreur_ia:
        st.error(f"Erreur de prédiction IA : {erreur_ia}")
