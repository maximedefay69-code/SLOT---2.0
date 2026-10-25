# SLOT 2.0 — Radar de stationnement à Paris

Application Streamlit : on saisit une adresse parisienne, l'application affiche
les zones de stationnement autour, coloriées selon le nombre de places libres
estimé par un modèle d'IA (LightGBM).

## Fichiers

| Fichier | Rôle |
|---|---|
| `app_map.py` | **Point d'entrée Streamlit** (interface, carte) |
| `slot_core.py` | Logique : géocodage (et inverse), météo, Open Data, Overpass, modèle, podium |
| `composants/geoloc/` | Composant de géolocalisation du navigateur (HTML, sans dépendance) |
| `modele_SLOT_v2.pkl`, `prepr_SLOT_v2.pkl` | Modèle IA v2 et son préprocesseur (utilisés en priorité) |
| `modele_lightgbmDA.pkl`, `preprocessorDA.pkl` | Ancien modèle, utilisé seulement si la v2 est absente |
| `tests/` | Tests automatisés (sans réseau) |
| `app_SLOT_2_0.py`, `app_SLOT_2_0_corrige.py` | Anciennes versions, conservées pour mémoire, non utilisées |

## Parcours dans l'app

1. **Intro** (à chaque ouverture) : logo, « SLOT », « Le Jackpot du stationnement ! », créneau animé
   en pixel art et barre de chargement (≈ 3,2 s, pendant lesquelles le modèle IA est chargé).
2. **Choix** :
   - **Option 1 — Me stationner autour de moi** : le téléphone demande l'accès à la position
     (composant maison `composants/geoloc/index.html`, sans dépendance), puis la rue est retrouvée
     par géocodage inverse (API Adresse). En cas de refus, bascule sur l'option 2.
   - **Option 2 — Trouver du stationnement à partir d'une adresse** : formulaire (rayon et tracé OSM
     dans « Options »).
3. **Résultats** : carte « ta rue » (ou rue recherchée), carte avec cercle du rayon, point bleu
   (option 1) et médailles, puis **podium 🥇🥈🥉** avec boutons **Waze** et **Google Maps**.
   - Classement : meilleure couleur (vert > orange > rouge), puis le plus de places libres estimées,
     puis la plus proche (vol d'oiseau).
   - Si tout est rouge (ou aucune place) : bouton « Élargir à 500 m », puis 800 m.

Réglages dans `app_map.py` : `DUREE_INTRO_S`, `SLOGAN`, `MARRON`, `CREME`, `OR_JACKPOT` ;
dans `slot_core.py` : `RAYONS` (300, 500, 800) et les seuils de couleur.

## Sources de données

- **API Adresse** (data.gouv.fr) : géocodage et géocodage inverse (position → rue). Sans clé.
- **Open-Meteo** : météo actuelle. Sans clé.
- **Paris Open Data**, jeu *stationnement-sur-voie-publique-emprises* : polygones des
  zones de stationnement et nombre de places **recensées** (`placal`). Sans clé.
  Ce n'est **pas** un comptage en direct.
- **Overpass / OpenStreetMap** (facultatif) : tracé gris des rues. Sans clé.
  Si tous les serveurs sont indisponibles, la carte reste utilisable.

### Secret (facultatif) : trafic en temps réel

| Nom | Où | Rôle |
|---|---|---|
| `GOOGLE_MAPS_API_KEY` | Streamlit Cloud → Settings → Secrets (ou `.streamlit/secrets.toml` en local, à ne jamais pousser sur GitHub) | Mesure du trafic via Google Maps |

```toml
GOOGLE_MAPS_API_KEY = "ta_clé"
```

La clé doit avoir l'API **Distance Matrix** activée (ou, à défaut, **Routes API**, utilisée en repli).
Méthode reprise de l'app d'origine : trajet de l'adresse vers un point décalé de +0,005°,
écart = durée avec trafic − durée normale ; < 30 s → 0 (fluide), < 120 s → 0,5, sinon 1 (dense).
Mesure mise en cache 5 min par quartier pour limiter la facturation Google.
Sans clé, l'app fonctionne avec TRAFIC = 0.

## Lancer en local

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -q
streamlit run app_map.py
```

## Déployer sur Streamlit Community Cloud

1. Pousser le dépôt sur GitHub (branche `main`).
2. Sur share.streamlit.io : **Create app** → choisir le dépôt, la branche `main`,
   fichier principal `app_map.py`.
3. **Advanced settings → Python version : 3.12** (important : `numpy<2` n'a pas de
   version précompilée pour Python 3.13).
4. (Facultatif) Coller le secret `GOOGLE_MAPS_API_KEY` (voir plus haut). Déployer.

Si l'application était déjà déployée avec une autre version de Python, il faut la
supprimer puis la recréer pour changer de version.

## Modèle IA (v2)

- LightGBM en régression, 200 arbres, learning_rate 0,05, 31 feuilles, feature_fraction 0,8.
- Entraîné sur **2 584 relevés** et **24 rues** (surtout quartier Europe / Batignolles : Moscou,
  Turin, Pelouze, Berne, Madrid, Naples, Clapeyron…), de 6 à 85 places par relevé, températures
  de −8,5 à 39,7 °C.
- Cible : **taux d'occupation** (moyenne ≈ 0,86). Places libres = places recensées × (1 − occupation).
  La sortie peut dépasser 1 : elle est alors bornée et signalée dans les diagnostics.
- Variables les plus utilisées : heure (sin/cos), température, nombre de places, `% PARKING OC`, trafic.

## Limites connues

- Pour une rue hors des 24 rues apprises, l'estimation repose sur l'heure, le jour, la météo,
  l'arrondissement et le nombre de places, sans historique propre à la rue.
- `% PARKING OC` (0,5) est une constante faute de source temps réel.
- `TRAFIC` est sur une échelle 0–1 à l'entraînement (moyenne 0,09) : l'app envoie 0 / 0,5 / 1, jamais 0 / 50 / 100.
- `VEHICULES / H` va de 0,22 à 1,30 à l'entraînement, alors que l'app envoie la valeur fixe
  par arrondissement (0,20 à 0,45) : la source exacte de cette variable reste à confirmer.
- Le dépôt ne contient pas le script d'entraînement : la méthode de validation n'est pas vérifiable.
