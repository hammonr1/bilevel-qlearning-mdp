"""Model parameters of Sections 5.1.1-5.1.3 (base case) with per-experiment overrides."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

# Class-level actions, lexicographic order; the first entry is the arg-max tie-break (DECISIONS D10).
ACTIONS: tuple[tuple[int, int], ...] = tuple((h, l) for h in range(3) for l in range(3))
NO_ACTION = (0, 0)


@dataclass(frozen=True)
class Params:
    # engagement structure
    T: int = 8                      # salvos
    M: int = 64                     # attacker initial AM inventory
    B: int = 6                      # SAM batteries (len(sites) must equal B)
    N_IM: int = 10                  # IMs per battery
    omega: float = 0.65             # P(one IM neutralizes its AM)
    # class-level action realization (Sec. 5.1.2)
    K_H: tuple = (0, 2, 4)          # max selected high-importance targets per attack level
    K_L: tuple = (0, 4, 8)
    eta_H: tuple = (0, 1, 2)        # AMs per selected high-importance target
    eta_L: tuple = (0, 1, 1)
    L_H: tuple = (0, 2, 4)          # max defended threatened high-importance assets per defense level
    L_L: tuple = (0, 4, 8)
    rho_H: tuple = (0, 1, 2)        # IMs per incoming AM
    rho_L: tuple = (0, 1, 2)
    # per-salvo caps; None = uncapped environment
    LAM: Optional[int] = None
    LIM: Optional[int] = None
    # learning (Sec. 5.1.3)
    gamma: float = 0.95
    alpha_A: float = 0.10
    alpha_D: float = 0.10
    eps0: float = 0.60
    eps_min: float = 0.02
    # arg-max tie-break when several actions share the maximal Q (incl. unvisited states): DECISIONS D10
    tie_break: str = "aggressive"   # "aggressive" (primary) | "random" | "lexicographic"

    @property
    def capped(self) -> bool:
        return self.LAM is not None or self.LIM is not None

    def with_caps(self, LAM: Optional[int], LIM: Optional[int]) -> "Params":
        return replace(self, LAM=LAM, LIM=LIM)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def epsilon(self, episode: int, n_train: int) -> float:
        """Linear decay from eps0 (episode 1) to eps_min (episode n_train); DECISIONS D10."""
        if n_train <= 1:
            return self.eps_min
        return self.eps0 - (self.eps0 - self.eps_min) * (episode - 1) / (n_train - 1)

    def label(self) -> str:
        cap = f"LAM{self.LAM}_LIM{self.LIM}" if self.capped else "uncapped"
        return f"T{self.T}_M{self.M}_B{self.B}_NIM{self.N_IM}_{cap}"


BASE = Params()
CAPPED_MAIN = BASE.with_caps(10, 12)          # Sections 5.2, 5.4, 5.6
