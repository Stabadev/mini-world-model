import numpy as np


class GridWorld:
    """
    Monde en grille simple pour world model.

    Règles v3 :
    - l'agent se déplace librement sur une case vide ;
    - l'agent est bloqué par les murs ;
    - si l'agent pousse la boîte et que la case derrière est libre, la boîte avance ;
    - si la boîte est contre un mur, elle ne bouge pas ;
    - si la boîte est sur la cible, elle est bloquée ;
    - la cible est visible dans l'état.
    """

    # Valeurs de rendu
    EMPTY = 0.0
    WALL = 0.2
    AGENT = 0.5
    BOX = 0.8
    TARGET = 1.0
    BOX_ON_TARGET = 0.9
    AGENT_ON_TARGET = 0.6

    # Actions : 0=haut, 1=bas, 2=gauche, 3=droite
    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    ACTION_NAMES = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
    }

    def __init__(self, size: int = 10):
        if size < 5:
            raise ValueError("La taille minimale recommandée est 5.")
        self.size = size
        self.reset()

    def _is_interior(self, r: int, c: int) -> bool:
        """Vrai si la case n'est pas un mur extérieur."""
        return 1 <= r < self.size - 1 and 1 <= c < self.size - 1

    def _all_interior_cells(self):
        return [
            (r, c)
            for r in range(1, self.size - 1)
            for c in range(1, self.size - 1)
        ]

    def clone_state(self):
        """Retourne l'état symbolique courant."""
        return {
            "agent": self.agent,
            "box": self.box,
            "target": self.target,
        }

    def set_state(self, agent, box, target):
        """
        Fixe explicitement un état.
        Très utile pour générer des cas dirigés.
        """
        agent = tuple(agent)
        box = tuple(box)
        target = tuple(target)

        for name, pos in [("agent", agent), ("box", box), ("target", target)]:
            if not self._is_interior(*pos):
                raise ValueError(f"{name} hors zone intérieure : {pos}")

        if agent == box:
            raise ValueError("agent et box ne peuvent pas être sur la même case")

        # target peut être sous la boîte : c'est précisément BOX_ON_TARGET
        self.agent = agent
        self.box = box
        self.target = target

    def reset(self):
        """
        Reset aléatoire simple.
        Pour le dataset dirigé, on utilisera plutôt set_state(...).
        """
        cells = self._all_interior_cells()
        idx = np.random.choice(len(cells), size=3, replace=False)
        self.agent = cells[idx[0]]
        self.box = cells[idx[1]]
        self.target = cells[idx[2]]

    def render(self) -> np.ndarray:
        """
        Retourne une grille 2D float32.
        """
        grid = np.full((self.size, self.size), self.EMPTY, dtype=np.float32)

        # Murs extérieurs
        grid[0, :] = self.WALL
        grid[-1, :] = self.WALL
        grid[:, 0] = self.WALL
        grid[:, -1] = self.WALL

        # Cible
        grid[self.target] = self.TARGET

        # Boîte
        if self.box == self.target:
            grid[self.box] = self.BOX_ON_TARGET
        else:
            grid[self.box] = self.BOX

        # Agent
        if self.agent == self.target:
            grid[self.agent] = self.AGENT_ON_TARGET
        else:
            grid[self.agent] = self.AGENT

        return grid

    def step(self, action: int) -> np.ndarray:
        """
        Applique une action et retourne l'observation suivante.

        Règles :
        1. Si l'agent tente d'aller dans un mur -> rien ne bouge.
        2. Si l'agent va sur une case libre (ou cible) -> il se déplace.
        3. Si l'agent pousse la boîte :
           - si la boîte est sur la cible -> rien ne bouge
           - si derrière la boîte il y a un mur -> rien ne bouge
           - sinon la boîte avance et l'agent prend sa place
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Action invalide: {action}")

        dr, dc = self.ACTIONS[action]

        ar, ac = self.agent
        next_agent = (ar + dr, ac + dc)

        # Cas 1 : mur
        if not self._is_interior(*next_agent):
            return self.render()

        # Cas 2 : déplacement libre
        if next_agent != self.box:
            self.agent = next_agent
            return self.render()

        # Cas 3 : l'agent pousse la boîte
        # Si la boîte est déjà sur la cible, elle est bloquée
        if self.box == self.target:
            return self.render()

        br, bc = self.box
        next_box = (br + dr, bc + dc)

        # Si derrière la boîte il y a un mur, on ne bouge pas
        if not self._is_interior(*next_box):
            return self.render()

        # Ici, monde simple : une seule boîte, pas d'autre obstacle dynamique
        self.box = next_box
        self.agent = next_agent
        return self.render()