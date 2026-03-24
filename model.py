import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encode une image 10×10 en vecteur latent de dimension latent_dim.
    
    Architecture :
    - Conv1 : détecte les motifs locaux (objets, murs)
    - Conv2 : combine les motifs en features de haut niveau
    - FC    : projette vers l'espace latent
    - BN    : normalise le latent pour stabiliser l'entraînement
    """

    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 10 * 10, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        """
        x : (batch, 1, 10, 10) — batch d'images
        retourne : (batch, latent_dim) — batch de vecteurs latents
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


class ActionEncoder(nn.Module):
    """
    Transforme une action discrète (0-3) en vecteur de dimension latent_dim.
    """

    def __init__(self, n_actions=4, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_actions, latent_dim)

    def forward(self, a):
        """
        a : (batch,) — batch d'indices d'actions (entiers 0-3)
        retourne : (batch, latent_dim) — batch de vecteurs d'action
        """
        return self.embedding(a)


class Predictor(nn.Module):
    """
    Prédit le vecteur latent suivant à partir du latent actuel et de l'action.
    
    Entrée  : z_t (latent_dim) + a_t (latent_dim) → concaténés (2 * latent_dim)
    Sortie  : z_t+1 prédit (latent_dim)
    """

    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, z, a):
        """
        z : (batch, latent_dim) — vecteur latent actuel
        a : (batch, latent_dim) — vecteur d'action encodé
        retourne : (batch, latent_dim) — vecteur latent prédit
        """
        x = torch.cat([z, a], dim=1)
        return self.net(x)


class WorldModel(nn.Module):
    """
    World model complet : encode les observations et prédit
    l'évolution du monde dans l'espace latent.
    
    Pipeline :
        image  → Encoder       → z_t
        action → ActionEncoder → a_t
        z_t, a_t → Predictor   → z_t+1 prédit
    """

    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder        = Encoder(latent_dim)
        self.action_encoder = ActionEncoder(latent_dim=latent_dim)
        self.predictor      = Predictor(latent_dim)

    def forward(self, obs, action):
        """
        obs    : (batch, 1, 10, 10) — image actuelle
        action : (batch,)           — action (entier 0-3)
        
        retourne :
            z_t   : (batch, latent_dim) — latent actuel
            z_pred: (batch, latent_dim) — latent suivant prédit
        """
        z_t    = self.encoder(obs)
        a_t    = self.action_encoder(action)
        z_pred = self.predictor(z_t, a_t)
        return z_t, z_pred