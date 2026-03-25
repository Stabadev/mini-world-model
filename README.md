# Mini World Model

Mon fil d'actu s'est mis à parler d'AMI Labs et de Yann LeCun. J'ai voulu comprendre.
Ce dépôt, c'est le résultat : quelques jours de lecture, d'échanges avec Claude et ChatGPT,
et un projet de Data Science pour ancrer tout ça dans quelque chose de concret.

**→ [worldmodel.rogues.fr](https://worldmodel.rogues.fr)**

---

L'idée : entraîner un **world model** (architecture JEPA) sur un mini-jeu —
une grille 10×10 avec un agent, une boîte, une cible.
Le modèle n'apprend pas à gagner. Il apprend à comprendre le monde.
À le simuler dans sa tête. À prédire ce qui va se passer.

Et ensuite on essaie de lire ce qu'il a compris. C'est là que c'est fascinant.

---

## Le site

Le projet est raconté sous forme de site web, acte par acte :

| | |
|---|---|
| [Acte 1](https://worldmodel.rogues.fr/acte1.html) | L'environnement — un simulateur jouable au clavier |
| [Acte 2](https://worldmodel.rogues.fr/acte2.html) | L'entraînement — courbes animées, architecture détaillée |
| [Acte 3](https://worldmodel.rogues.fr/acte3.html) | La représentation — PCA, probing, ce que le modèle a retenu |
| [Acte 4](https://worldmodel.rogues.fr/acte4.html) | Les limites — rollout stable, planning qui échoue, pourquoi |
| [Conclusion](https://worldmodel.rogues.fr/conclusion.html) | Ce que j'ai appris — et pourquoi ça valait le coup |

## Les notebooks

Pour ceux qui veulent aller plus loin, tout le code est là :

| Notebook | Contenu |
|----------|---------|
| [01 — Environnement](research/notebooks/01_environment.ipynb) | GridWorld + génération du dataset dirigé |
| [02 — Entraînement](research/notebooks/02_training.ipynb) | Architecture WorldModel + training JEPA |
| [03 — Compréhension](research/notebooks/03_understanding.ipynb) | PCA + probing linéaire de l'espace latent |
| [04 — Planning](research/notebooks/04_planning.ipynb) | Rollout multi-step + décodeur de position + limites |

## Structure du dépôt

```
research/    ← notebooks, code Python, assets, papiers de référence
site/        ← site web statique (HTML/CSS/JS, zéro framework)
```

---

*Projet personnel — Alexandre Rogues · [alexandre.rogues.fr](https://alexandre.rogues.fr)*
