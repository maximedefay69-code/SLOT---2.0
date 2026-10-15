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
