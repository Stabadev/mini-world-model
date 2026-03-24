# 📍 Point de reprise — Acte 4

## État actuel du projet

- ✅ Acte 1 : environnement GridWorld 10×10, dataset 10 000 transitions
- ✅ Acte 2 : WorldModel entraîné (latent_dim=32, 50 époques, grille 10×10)
- ✅ Acte 3 : PCA latent + probing documenté
- 🔲 Acte 4 : planning + détection d'anomalies

## Prochaine étape

Ouvrir `notebooks/04_planning.ipynb` et implémenter :

1. **Planning par recherche exhaustive**
   - générer toutes les séquences d'actions jusqu'à longueur H
   - simuler chaque séquence avec le modèle
   - choisir la séquence qui minimise la distance latente à l'objectif

2. **Planning CEM** (Cross-Entropy Method)
   - version plus réaliste, fidèle au papier LeWM

3. **Détection d'anomalies**
   - introduire des événements impossibles (téléportation, traversée de mur)
   - mesurer la surprise du modèle (erreur de prédiction)
   - visualiser le pic de surprise

## Paramètres importants à garder en tête

- Grille : 10×10
- latent_dim : 32
- Modèle sauvegardé : `data/worldmodel.pt`
- Dataset : `data/observations.npy`, `data/actions.npy`, `data/next_obs.npy`

## Rappel architecture

- `environment.py` : classe GridWorld
- `model.py` : Encoder, ActionEncoder, Predictor, WorldModel
- `notebooks/01_environment.ipynb` : environnement + dataset
- `notebooks/02_training.ipynb` : entraînement
- `notebooks/03_understanding.ipynb` : PCA + probing
