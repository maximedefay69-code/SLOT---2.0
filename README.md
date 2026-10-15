# SLOT 2.0 — Radar de stationnement à Paris

Application Streamlit : on saisit une adresse parisienne, l'application affiche
les zones de stationnement autour, coloriées selon le nombre de places libres
estimé par un modèle d'IA (LightGBM).

## Fichiers

| Fichier | Rôle |
|---|---|
| `app_map.py` | **Point d'entrée Streamlit** (interface, carte) |
| `slot_core.py` | Logique : géocodage, météo, Open Data, Overpass, modèle |
| `modele_lightgbmDA.pkl`, `preprocessorDA.pkl` | Modèle IA et préprocesseur |
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

## Limites connues du modèle

- Entraîné sur **3 rues** seulement (Voltaire, Menat, Commandant L…). Pour les autres
  rues, l'estimation repose sur l'heure, le jour, la météo, l'arrondissement et le
  nombre de places, sans historique propre à la rue.
- Données d'entraînement plutôt hivernales (température moyenne ≈ 7 °C).
- Les variables `TRAFIC` (0,0) et `% PARKING OC` (0,5) sont des constantes, faute de
  source temps réel.
- La sortie est un taux d'occupation (≈ 0,75–0,99 en pratique) : places libres =
  places recensées × (1 − occupation).
