# Mini World Model

Une implémentation pédagogique d'un **world model** (architecture JEPA) sur un environnement GridWorld 10×10, inspirée des travaux de Yann LeCun et du papier [LeWM](research/references/leWorldModel.pdf).

→ **[Voir le site du projet](https://worldmodel.rogues.fr)**

## Structure

```
research/    ← code, notebooks, données, références académiques
site/        ← site web de présentation (portfolio)
```

## Notebooks (dans l'ordre)

| Notebook | Contenu |
|----------|---------|
| [01 — Environnement](research/notebooks/01_environment.ipynb) | GridWorld + génération du dataset |
| [02 — Entraînement](research/notebooks/02_training.ipynb) | Entraînement du WorldModel |
| [03 — Compréhension](research/notebooks/03_understanding.ipynb) | PCA + probing de l'espace latent |
| [04 — Planning](research/notebooks/04_planning.ipynb) | Rollout, planning, détection d'anomalies |
