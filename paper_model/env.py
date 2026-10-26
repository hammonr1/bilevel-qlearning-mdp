"""Engagement environment: Sections 4.2-4.4 of the manuscript.

For a fixed SAM-allocation plan (the tuple of allocated sites), the environment
  * realizes a class-level attack action into y_it (Eqs. 13-18) with trimming,
  * realizes a class-level defense action into u_it (Eqs. 30-39) with trimming,
  * sources the IMs from SAM sites, w*_jit (Eqs. 57-62, no kappa term),
  * computes survival probabilities and rewards (Eqs. 47-56),
  * samples the Bernoulli transition and updates (z, m, q) (Eqs. 63-66).
All ambiguities are logged in DECISIONS.md (D4-D13).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pulp

from .instance import Instance
from .params import ACTIONS, Params


# --------------------------------------------------------------------------------------
# exact sourcing-feasibility test (max-flow on the SAM -> asset bipartite network); D7
# --------------------------------------------------------------------------------------
def sourcing_feasible(sites, q, demand, C) -> bool:
    """True iff there is an integer w with sum_j w_ji = demand_i, sum_i w_ji <= q_j, w_ji <= c_ji q_j."""
    need = sum(demand.values())
    if need == 0:
        return True
    if need > sum(q):
        return False
    # quick necessary check: each demanded asset must be reachable from a stocked site
    for i in demand:
        if not any(C[j, i] and q[k] > 0 for k, j in enumerate(sites)):
            return False
    # Ford-Fulkerson with DFS on a tiny graph: source -> site k (cap q_k) -> asset i (inf) -> sink (cap demand_i)
    assets = list(demand)
    cap_site = list(q)                      # residual source->site
    cap_sink = dict(demand)                 # residual asset->sink
    flow = {}                               # (k, i) -> flow on site->asset arc (residual back-capacity)
    total = 0

    def dfs(k, seen_sites, seen_assets):
        # from site k try to push one unit to the sink
        j = sites[k]
        for i in assets:
            if i in seen_assets or not C[j, i]:
                continue
            seen_assets.add(i)
            if cap_sink[i] > 0:
                cap_sink[i] -= 1
                flow[(k, i)] = flow.get((k, i), 0) + 1
                return True
            # try to reroute a unit currently flowing from another site k2 into asset i
            for k2 in range(len(sites)):
                if k2 != k and flow.get((k2, i), 0) > 0 and k2 not in seen_sites:
                    seen_sites.add(k2)
                    if dfs(k2, seen_sites, seen_assets):
                        flow[(k2, i)] -= 1
                        flow[(k, i)] = flow.get((k, i), 0) + 1
                        return True
        return False

    for _ in range(need):
        pushed = False
        for k in range(len(sites)):
            if cap_site[k] > 0 and dfs(k, {k}, set()):
                cap_site[k] -= 1
                total += 1
                pushed = True
                break
        if not pushed:
            return False
    return total == need


@dataclass
class AttackRealization:
    a: tuple
    Lambda: int
    selected: list            # [(R_i, i, eta)] in selection order, before trimming
    trimmed: list             # [(R_i, i, eta)] removed, in removal order
    y: dict                   # asset -> AMs (only positive entries)

    @property
    def total(self) -> int:
        return sum(self.y.values())


@dataclass
class DefenseRealization:
    d: tuple
    Lambda: int
    selected: list            # [(H_i, i, u_i)] before trimming
    trimmed: list             # [(H_i, i, u_i, reason)] removed
    u: dict                   # asset -> IMs (only positive entries)
    w: dict = field(default_factory=dict)   # (site j, asset i) -> IMs fired

    @property
    def total(self) -> int:
        return sum(self.u.values())


@dataclass
class StepResult:
    attack: AttackRealization
    defense: DefenseRealization
    P_S: dict                 # asset -> survival probability for attacked assets
    rh: dict                  # asset -> (r_it, h_it)
    D_t: float                # expected damage (Eq. 53) = attacker reward
    S_t: float                # expected saved value (Eq. 54) = defender reward
    xi: dict                  # asset -> Bernoulli draw for attacked assets
    z_next: tuple
    m_next: int
    q_next: tuple
    realized_damage: float    # value of assets that actually flipped to damaged in this salvo


class Environment:
    def __init__(self, inst: Instance, params: Params, sites, N_IM: Optional[int] = None,
                 sourcing_solver: str = "PULP_CBC_CMD"):
        self.inst = inst
        self.p = params
        self.sites = tuple(sorted(int(j) for j in sites))
        if len(self.sites) != params.B:
            raise ValueError(f"len(sites)={len(self.sites)} != B={params.B}")
        self.N_IM = params.N_IM if N_IM is None else N_IM
        self.n = inst.n
        self.C = inst.C
        self.v = [float(x) for x in inst.v]        # plain floats (keys/prints stay numpy-free)
        self.pdmg = [float(x) for x in inst.p]
        self.high = inst.high
        self.low = inst.low
        # value-at-risk score for surviving assets (Eq. 13 with z=1); ranking order fixed per asset
        self.R = [float(self.v[i] * self.pdmg[i]) for i in range(self.n)]
        self.sourcing_solver_name = sourcing_solver
        if sourcing_solver == "PULP_CBC_CMD":
            self._solver = pulp.PULP_CBC_CMD(msg=False)
        elif sourcing_solver == "HiGHS":
            self._solver = pulp.HiGHS(msg=False)
        else:
            self._solver = pulp.getSolver(sourcing_solver, msg=False)
        self._w_cache: dict = {}
        self.n_lp_solves = 0

    # ----------------------------------------------------------------------------- state
    def initial_state(self):
        z = tuple([1] * self.n)
        q = tuple([self.N_IM] * len(self.sites))
        return z, self.p.M, q

    @staticmethod
    def zmask(z) -> int:
        m = 0
        for i, zi in enumerate(z):
            if zi:
                m |= 1 << i
        return m

    def attacker_key(self, z, m, q, t):
        return (self.zmask(z), m, q, t)

    # --------------------------------------------------------------------- Eqs. 13-18
    def Lambda_A(self, m: int) -> int:
        return m if self.p.LAM is None else min(m, self.p.LAM)

    def realize_attack(self, z, m, a) -> AttackRealization:
        aH, aL = a
        Lam = self.Lambda_A(m)
        selected = []
        for cls, K, eta in ((self.high, self.p.K_H[aH], self.p.eta_H[aH]),
                            (self.low, self.p.K_L[aL], self.p.eta_L[aL])):
            if K == 0 or eta == 0:
                continue
            surv = [i for i in cls if z[i]]
            surv.sort(key=lambda i: (-self.R[i], i))          # largest R first (Eqs. 14-15)
            selected.extend((self.R[i], i, eta) for i in surv[:K])
        total = sum(e for _, _, e in selected)
        kept = list(selected)
        trimmed = []
        while total > Lam:                                     # Eq. 17-18 trimming (D5, D6)
            idx = min(range(len(kept)), key=lambda k: (kept[k][0], -kept[k][1]))
            rem = kept.pop(idx)
            trimmed.append(rem)
            total -= rem[2]
        y = {i: e for _, i, e in kept}
        return AttackRealization(a=a, Lambda=Lam, selected=selected, trimmed=trimmed, y=y)

    # --------------------------------------------------------------------- Eqs. 30-40
    def Lambda_D(self, q) -> int:
        tot = sum(q)
        return tot if self.p.LIM is None else min(tot, self.p.LIM)

    def realize_defense(self, z, q, y: dict, d) -> DefenseRealization:
        dH, dL = d
        Lam = self.Lambda_D(q)
        H = {i: self.v[i] * self.pdmg[i] * y[i] for i in y}  # Eq. 30 (z_it = 1 for attacked assets)
        selected = []
        for cls, L, rho in ((self.high, self.p.L_H[dH], self.p.rho_H[dH]),
                            (self.low, self.p.L_L[dL], self.p.rho_L[dL])):
            if L == 0 or rho == 0:
                continue
            thr = [i for i in cls if i in y]
            thr.sort(key=lambda i: (-H[i], i))                # Eqs. 31-32
            selected.extend((H[i], i, rho * y[i]) for i in thr[:L])   # Eq. 33
        kept = list(selected)
        trimmed = []
        while True:
            u = {i: ui for _, i, ui in kept}
            tot = sum(u.values())
            if tot <= Lam and sourcing_feasible(self.sites, q, u, self.C):
                break
            reason = "cap" if tot > Lam else "sourcing"
            idx = min(range(len(kept)), key=lambda k: (kept[k][0], -kept[k][1]))
            rem = kept.pop(idx)
            trimmed.append((*rem, reason))
        return DefenseRealization(d=d, Lambda=Lam, selected=selected, trimmed=trimmed, u=u)

    # --------------------------------------------------------------------- Eqs. 57-62
    def source(self, q, u: dict, Lam: int) -> dict:
        """Integer sourcing w*_jit via PuLP, zero objective (no kappa term; D7-D8). Cached on (q, u)."""
        if not u:
            return {}
        key = (q, tuple(sorted(u.items())))
        w = self._w_cache.get(key)
        if w is not None:
            return w
        prob = pulp.LpProblem("IM_sourcing", pulp.LpMinimize)
        var = {}
        for k, j in enumerate(self.sites):
            if q[k] <= 0:
                continue
            for i in u:
                if self.C[j, i]:
                    var[(j, i)] = pulp.LpVariable(f"w_{j}_{i}", lowBound=0, upBound=min(q[k], u[i]),
                                                  cat="Integer")                      # Eq. 62
        prob += pulp.lpSum([])                                                         # Eq. 57 without kappa
        for i in u:                                                                    # Eq. 58
            prob += pulp.lpSum(var[(j, i)] for j in self.sites if (j, i) in var) == u[i]
        for k, j in enumerate(self.sites):                                             # Eq. 59 (60 via bounds)
            terms = [var[(j, i)] for i in u if (j, i) in var]
            if terms:
                prob += pulp.lpSum(terms) <= q[k]
        prob += pulp.lpSum(var.values()) <= Lam                                        # Eq. 61
        status = prob.solve(self._solver)
        self.n_lp_solves += 1
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"sourcing ILP infeasible after trimming: q={q} u={u} status={pulp.LpStatus[status]}")
        w = {ji: int(round(x.value())) for ji, x in var.items() if x.value() and round(x.value()) > 0}
        self._w_cache[key] = w
        return w

    # --------------------------------------------------------------------- Eqs. 47-54
    def survival(self, y_i: int, u_i: int, p_i: float):
        if y_i == 0:
            return 1.0, (0, 0)
        r = u_i // y_i                                                                 # Eq. 48
        h = u_i - r * y_i                                                              # Eq. 49
        om = 1.0 - self.p.omega
        P = (1.0 - p_i * om ** (r + 1)) ** h * (1.0 - p_i * om ** r) ** (y_i - h)      # Eq. 51
        return P, (r, h)

    # --------------------------------------------------------------------- one salvo
    def step(self, z, m, q, a, d, rng) -> StepResult:
        atk = self.realize_attack(z, m, a)
        dfn = self.realize_defense(z, q, atk.y, d)
        dfn.w = self.source(q, dfn.u, dfn.Lambda)
        P_S, rh = {}, {}
        D_t = S_t = 0.0
        for i, yi in atk.y.items():
            P, r_h = self.survival(yi, dfn.u.get(i, 0), self.pdmg[i])
            P_S[i] = P
            rh[i] = r_h
            D_t += self.v[i] * (1.0 - P)                                               # Eq. 53 (z_it = 1)
        total_value_alive = sum(self.v[i] for i in range(self.n) if z[i])
        S_t = total_value_alive - D_t                                                  # Eq. 54
        # Eqs. 63-64
        z_next = list(z)
        xi = {}
        realized = 0.0
        for i in atk.y:
            draw = 1 if rng.random() < P_S[i] else 0
            xi[i] = draw
            if draw == 0:
                z_next[i] = 0
                realized += self.v[i]
        z_next = tuple(z_next)
        m_next = m - atk.total                                                         # Eq. 65
        fired = [0] * len(self.sites)
        for (j, i), wji in dfn.w.items():
            fired[self.sites.index(j)] += wji
        q_next = tuple(z_next[j] * (q[k] - fired[k]) for k, j in enumerate(self.sites))   # Eq. 66
        return StepResult(attack=atk, defense=dfn, P_S=P_S, rh=rh, D_t=D_t, S_t=S_t, xi=xi,
                          z_next=z_next, m_next=m_next, q_next=q_next, realized_damage=realized)


__all__ = ["Environment", "StepResult", "AttackRealization", "DefenseRealization", "sourcing_feasible", "ACTIONS"]
