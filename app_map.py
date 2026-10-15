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
folium.TileLayer("OpenStreetMap", name="Plan OpenStreetMap").add_to(carte)
folium.TileLayer("CartoDB positron", name="Plan clair (CARTO)").add_to(carte)

couche_osm = folium.FeatureGroup(name="Rues (OpenStreetMap)", show=True)
for t in troncons_osm:
    principale = t["cle"] == cle_principale
    folium.PolyLine(
        t["coords"],
        color="#1f4e79" if principale else "#9aa0a6",
        weight=6 if principale else 3,
        opacity=0.8 if principale else 0.6,
        tooltip=t["nom"] + (" (rue recherchée)" if principale else ""),
    ).add_to(couche_osm)
couche_osm.add_to(carte)

couche_places = folium.FeatureGroup(name="Stationnement (estimation IA)", show=True)
nb_formes = 0
for cle, rue in rues.items():
    pred = predictions.get(cle)
    couleur = core.couleur_pour_places(pred.libres if pred else None)
    principale = cle == cle_principale
    if pred:
        info = (f"<b>{rue['label']}</b>{' (rue recherchée)' if principale else ''}<br>"
                f"{pred.libres} place(s) libre(s) estimée(s) sur {rue['places']} recensée(s) "
                f"dans le rayon<br>Occupation estimée : {pred.occupation:.0%}")
        if not pred.rue_connue:
            info += "<br><i>Rue absente de l'entraînement : estimation générique.</i>"
    else:
        info = f"<b>{rue['label']}</b><br>{rue['places']} place(s) recensée(s) — estimation indisponible"
    for emprise in rue["emprises"]:
        details = (f"{info}<br><small>Emprise : {emprise['regpri'].title() or 'régime inconnu'}, "
                   f"{emprise['places'] if emprise['places'] is not None else '?'} place(s)</small>")
        for genre, coords in emprise["geometries"]:
            options = dict(color="#111111" if principale else couleur, weight=3 if principale else 2,
                           fill=True, fill_color=couleur, fill_opacity=0.75,
                           tooltip=folium.Tooltip(details))
            if genre == "polygon":
                folium.Polygon(coords, **options).add_to(couche_places)
            else:
                folium.PolyLine(coords, color=couleur, weight=6, tooltip=folium.Tooltip(details)).add_to(couche_places)
            nb_formes += 1
        if not emprise["geometries"] and emprise["point"]:
            # Pas de polygone : on montre au moins le point de l'emprise.
            folium.CircleMarker(emprise["point"], radius=5, color=couleur, fill=True, fill_opacity=0.9,
                                tooltip=folium.Tooltip(details)).add_to(couche_places)
            nb_formes += 1
couche_places.add_to(carte)

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
    "— contour noir : rue recherchée ; traits gris : rues OSM sans emprise de stationnement.",
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
