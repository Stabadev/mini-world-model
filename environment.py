import numpy as np

class GridWorld:
    """
    Environnement 2D minimaliste pour le world model.
    
    Le monde est une grille 7×7 avec des murs sur les bords.
    Trois objets évoluent à l'intérieur : un agent (contrôlé),
    une boîte (poussable), et une cible (fixe).
    
    Chaque état est représenté par une image 7×7 où chaque pixel
    encode la présence d'un objet via une valeur flottante.
    """

    # --- Valeurs des pixels ---
    EMPTY           = 0.0  # case vide
    WALL            = 0.3  # mur (bord de la grille)
    AGENT_ON_TARGET = 0.6  # agent sur la cible
    AGENT_ON_BOX    = 0.7  # agent sur la boîte (ne devrait pas arriver)
    BOX             = 0.8  # boîte seule
    BOX_ON_TARGET   = 0.9  # boîte sur la cible → objectif atteint
    TARGET          = 1.0  # cible seule
    AGENT           = 0.5  # agent seul

    # --- Actions disponibles ---
    ACTIONS = {
        0: (-1,  0),  # haut
        1: ( 1,  0),  # bas
        2: ( 0, -1),  # gauche
        3: ( 0,  1),  # droite
    }

    def __init__(self):
        self.size = 7
        self.reset()

    def reset(self):
        """
        Initialise le monde avec des positions aléatoires.
        La boîte est placée dans l'intérieur profond (loin des murs).
        """
        deep_interior = [
            (r, c)
            for r in range(2, self.size - 2)
            for c in range(2, self.size - 2)
        ]
        interior = [
            (r, c)
            for r in range(1, self.size - 1)
            for c in range(1, self.size - 1)
        ]
        box_idx  = np.random.randint(len(deep_interior))
        self.box = deep_interior[box_idx]
        remaining = [p for p in interior if p != self.box]
        indices   = np.random.choice(len(remaining), size=2, replace=False)
        self.agent  = remaining[indices[0]]
        self.target = remaining[indices[1]]

    def render(self):
        """Retourne l'état courant sous forme d'image 7×7."""
        grid = np.full((self.size, self.size), self.EMPTY)
        grid[0, :]  = self.WALL
        grid[-1, :] = self.WALL
        grid[:, 0]  = self.WALL
        grid[:, -1] = self.WALL
        grid[self.target] = self.TARGET
        if self.box == self.target:
            grid[self.box] = self.BOX_ON_TARGET
        else:
            grid[self.box] = self.BOX
        if self.agent == self.target:
            grid[self.agent] = self.AGENT_ON_TARGET
        elif self.agent == self.box:
            grid[self.agent] = self.AGENT_ON_BOX
        else:
            grid[self.agent] = self.AGENT
        return grid

    def step(self, action):
        """Applique une action et retourne le nouvel état."""
        dr, dc = self.ACTIONS[action]
        ar, ac = self.agent
        nr, nc = ar + dr, ac + dc
        if not self._is_interior(nr, nc):
            return self.render()
        if (nr, nc) == self.box:
            br, bc = nr + dr, nc + dc
            if not self._is_interior(br, bc):
                return self.render()
            self.box = (br, bc)
        self.agent = (nr, nc)
        return self.render()

    def _is_interior(self, r, c):
        """Vérifie qu'une position est dans l'intérieur de la grille."""
        return 1 <= r <= self.size - 2 and 1 <= c <= self.size - 2