# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python research project implementing a simplified **world model** (inspired by LeWM — Latent Effort World Models) for a 10×10 GridWorld environment. The project is structured as a progressive series of Jupyter notebooks (Acts 1–4).

**GridWorld**: 10×10 grid with an agent, a movable box, fixed walls, and a target cell. The agent can push the box; the box locks in place once it reaches the target.

## Development Commands

```bash
# Install dependencies (no requirements.txt — install manually)
pip install torch numpy matplotlib scikit-learn jupyter

# Launch Jupyter to work on notebooks
jupyter notebook

# Run a notebook non-interactively
jupyter nbconvert --to notebook --execute notebooks/02_training.ipynb
```

Notebooks must be run in order: `01 → 02 → 03 → 04`, as each produces artifacts consumed by the next.

## Architecture

### Core modules

- **`environment.py`** — `GridWorld` class: 10×10 physics simulation. `render()` returns a `float32` numpy array (1×10×10). Key rules: walls block movement, agent can push box if the cell behind it is free, box becomes immobile once on the target.

- **`model.py`** — Four PyTorch classes:
  - `Encoder`: CNN (2× Conv2d + BN + ReLU, then FC) → 32D latent vector `z`
  - `ActionEncoder`: `nn.Embedding(4, action_dim)` for discrete actions {up, down, left, right}
  - `Predictor`: MLP `(z_t ⊕ a_t) → z_{t+1}`
  - `WorldModel`: composes all three; `forward(obs, action) → (z_t, z_next_pred, z_next_real)`

### Training approach (JEPA-style)

No image reconstruction. The model predicts in **latent space**:

```
Loss = MSE(z_predicted, z_real) + λ * covariance_regularization
```

`λ = 0.1`, `latent_dim = 32`, batch size 64, Adam lr=0.001, 50 epochs. Converges to ~0.0001.

### Data artifacts (`data/` — git-ignored)

| File | Description |
|------|-------------|
| `observations_v3.npy` | 20 000 grid images (float32, shape N×1×10×10) |
| `actions_v3.npy` | Action indices (int) |
| `next_obs_v3.npy` | Next-step observations |
| `worldmodel_v3.pt` | Trained model weights |

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_environment.ipynb` | GridWorld demo + directed dataset generation |
| `02_training.ipynb` | WorldModel training + loss curves |
| `03_understanding.ipynb` | PCA + linear probing of the latent space |
| `04_planning.ipynb` | Rollout evaluation, box-position decoder, planning |

### Known latent-space properties (from probing)

- Box position: R² ≈ 0.68 — well encoded
- Agent position: R² ≈ 0.40–0.61 — partially encoded
- Agent–box adjacency: accuracy ≈ 0.94 — strongly encoded
- Box-on-target: accuracy ≈ 0.59 — weakly encoded (planning limitation)

## Current Status

- Acts 1–3: complete
- Act 4 (in progress): exhaustive-search planning, CEM planning, anomaly detection (teleportation / wall-clipping events measured via prediction surprise)
