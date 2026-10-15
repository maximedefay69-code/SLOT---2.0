# SLOT 2.0 — Radar de stationnement à Paris

Application Streamlit : on saisit une adresse parisienne, l'application affiche
les zones de stationnement autour, coloriées selon le nombre de places libres
estimé par un modèle d'IA (LightGBM).

## Fichiers

| Fichier | Rôle |
|---|---|
| `app_map.py` | **Point d'entrée Streamlit** (interface, carte) |
| `slot_core.py` | Logique : géocodage, météo, Open Data, Overpass, modèle |
| `modele_SLOT_v2.pkl`, `prepr_SLOT_v2.pkl` | Modèle IA v2 et son préprocesseur (utilisés en priorité) |
| `modele_lightgbmDA.pkl`, `preprocessorDA.pkl` | Ancien modèle, utilisé seulement si la v2 est absente |
| `tests/` | Tests automatisés (sans réseau) |
| `app_SLOT_2_0.py`, `app_SLOT_2_0_corrige.py` | Anciennes versions, conservées pour mémoire, non utilisées |

## Sources de données

- **API Adresse** (data.gouv.fr) : géocodage. Sans clé.
- **Open-Meteo** : météo actuelle. Sans clé.
- **Paris Open Data**, jeu *stationnement-sur-voie-publique-emprises* : polygones des
  zones de stationnement et nombre de places **recensées** (`placal`). Sans clé.
  Ce n'est **pas** un comptage en direct.
- **Overpass / OpenStreetMap** (facultatif) : tracé gris des rues. Sans clé.
  Si tous les serveurs sont indisponibles, la carte reste utilisable.

Aucun secret ni variable d'environnement n'est nécessaire.

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
4. Aucun secret à renseigner. Déployer.

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
- `TRAFIC` (0,0) et `% PARKING OC` (0,5) sont des constantes faute de source temps réel.
  À l'entraînement, `TRAFIC` est sur une échelle 0–1 (moyenne 0,09) : ne pas y envoyer un score 0–100.
- `VEHICULES / H` va de 0,22 à 1,30 à l'entraînement, alors que l'app envoie la valeur fixe
  par arrondissement (0,20 à 0,45) : la source exacte de cette variable reste à confirmer.
- Le dépôt ne contient pas le script d'entraînement : la méthode de validation n'est pas vérifiable.
