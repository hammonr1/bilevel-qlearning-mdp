"""Section 5.1.1 base-case instance generation and the Section 5.4 static allocation rules.

Generation order (found by trial so that Table 1's "Initially covered assets" column and the
Section 5.4 static rules are reproduced exactly; see DECISIONS.md, D1-D2):

    r = random.Random(7)
    for each asset i in 0..29:   x_i = r.uniform(0, 100); y_i = r.uniform(0, 100)
    I_H = sorted(r.sample(range(30), 8))
    for each asset i in 0..29:   v_i = r.uniform(80,120) if i in I_H else r.uniform(20,60)
                                 p_i = r.uniform(0.75,0.95) if i in I_H else r.uniform(0.55,0.80)
    c_ji = 1  iff  euclidean(j, i) <= R = 40
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Candidate SAM-allocation plans of Tables 1-2 (Section 5.1.3) and the Section 5.2 refined plan.
CANDIDATES = {
    0: (0, 1, 2, 3, 8, 19),
    1: (9, 14, 19, 22, 26, 27),
    2: (9, 19, 20, 22, 26, 27),
    3: (7, 18, 19, 21, 23, 24),
    4: (2, 4, 12, 21, 22, 26),
}
REFINED_PLAN = (3, 7, 12, 18, 19, 23)

# "Initially covered assets" reported in Table 1 / Section 5.2.
TABLE1_COVERED = {0: 30, 1: 27, 2: 27, 3: 30, 4: 28, "refined": 28}


@dataclass
class Instance:
    seed: int
    R: float
    locs: np.ndarray            # (n, 2)
    is_high: np.ndarray         # (n,) bool
    v: np.ndarray               # (n,) asset value
    p: np.ndarray               # (n,) damage probability if an AM is not neutralized
    C: np.ndarray               # (n, n) int, C[j, i] = c_ji
    high: tuple = field(init=False)
    low: tuple = field(init=False)

    def __post_init__(self):
        self.high = tuple(int(i) for i in np.flatnonzero(self.is_high))
        self.low = tuple(int(i) for i in np.flatnonzero(~self.is_high))

    @property
    def n(self) -> int:
        return len(self.v)

    def covered_count(self, sites) -> int:
        """Number of assets covered by at least one SAM site in `sites`."""
        return int(self.C[list(sites)].max(axis=0).sum())

    def covered_by(self, sites) -> np.ndarray:
        return self.C[list(sites)].max(axis=0).astype(bool)


def generate_instance(seed: int = 7, n: int = 30, n_high: int = 8, R: float = 40.0,
                      area: float = 100.0) -> Instance:
    r = random.Random(seed)
    locs = np.array([(r.uniform(0, area), r.uniform(0, area)) for _ in range(n)])
    high = sorted(r.sample(range(n), n_high))
    is_high = np.zeros(n, dtype=bool)
    is_high[high] = True
    v = np.zeros(n)
    p = np.zeros(n)
    for i in range(n):
        if is_high[i]:
            v[i] = r.uniform(80, 120)
            p[i] = r.uniform(0.75, 0.95)
        else:
            v[i] = r.uniform(20, 60)
            p[i] = r.uniform(0.55, 0.80)
    d = np.sqrt(((locs[:, None, :] - locs[None, :, :]) ** 2).sum(-1))
    C = (d <= R).astype(int)
    return Instance(seed=seed, R=R, locs=locs, is_high=is_high, v=v, p=p, C=C)


# --------------------------------------------------------------------------------------
# Section 5.4 static allocation rules (Panel B benchmarks)
# --------------------------------------------------------------------------------------
def greedy_set_cover(inst: Instance, B: int = 6) -> tuple:
    """Greedy incremental set cover: each step picks the location covering the most currently
    uncovered assets; ties broken by value-weighted uncovered coverage, then by lowest index."""
    uncovered = set(range(inst.n))
    chosen: list[int] = []
    for _ in range(B):
        def score(j):
            new = [i for i in uncovered if inst.C[j, i]]
            return (len(new), float(sum(inst.v[i] for i in new)), -j)
        best = max((j for j in range(inst.n) if j not in chosen), key=score)
        chosen.append(best)
        uncovered -= {i for i in range(inst.n) if inst.C[best, i]}
    return tuple(sorted(chosen))


def value_weighted_rule(inst: Instance, B: int = 6) -> tuple:
    """Static value-weighted coverage rule: H_j = sum_i v_i c_ji, take the B largest."""
    H = inst.C @ inst.v
    order = sorted(range(inst.n), key=lambda j: (-H[j], j))
    return tuple(sorted(order[:B]))


def coverage_count_rule(inst: Instance, B: int = 6) -> tuple:
    """Static coverage-count rule: G_j = sum_i c_ji, take the B largest.
    Ties are broken by descending index (the np.argsort(G)[::-1] idiom); see DECISIONS.md D2."""
    G = inst.C.sum(axis=1)
    order = np.argsort(G, kind="stable")[::-1]
    return tuple(sorted(int(j) for j in order[:B]))


def write_instance_csv(inst: Instance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset", "x", "y", "class", "value", "damage_prob", "coverage_count_as_SAM_site"])
        G = inst.C.sum(axis=1)
        for i in range(inst.n):
            w.writerow([i, f"{inst.locs[i,0]:.6f}", f"{inst.locs[i,1]:.6f}",
                        "H" if inst.is_high[i] else "L", f"{inst.v[i]:.6f}", f"{inst.p[i]:.6f}", int(G[i])])
    with open(path.with_name("coverage_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sam_site_j \\ asset_i"] + list(range(inst.n)))
        for j in range(inst.n):
            w.writerow([j] + inst.C[j].tolist())
