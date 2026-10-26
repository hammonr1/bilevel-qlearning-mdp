"""Hierarchical Stackelberg Q-learning (Algorithm 1, Eqs. 69-78) with dict-based Q-tables."""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

from .env import Environment, StepResult
from .params import ACTIONS, Params

TIE_BREAK_RULES = ("aggressive", "random", "lexicographic")
# aggressiveness order: total intensity, then high-importance intensity (most aggressive first)
AGGRESSIVE_ORDER = tuple(sorted(ACTIONS, key=lambda a: (-(a[0] + a[1]), -a[0])))


def argmax_action(Q: dict, key, tie_break: str, rng: Optional[random.Random] = None):
    """arg-max over the 9 class-level actions; ties (incl. the all-zero row of an unvisited state)
    are broken per `tie_break` (DECISIONS D10)."""
    best_q = max(Q.get((key, a), 0.0) for a in ACTIONS)
    if tie_break == "aggressive":
        order = AGGRESSIVE_ORDER
    elif tie_break == "lexicographic":
        order = ACTIONS
    elif tie_break == "random":
        ties = [a for a in ACTIONS if Q.get((key, a), 0.0) == best_q]
        return ties[0] if len(ties) == 1 else rng.choice(ties)
    else:
        raise ValueError(f"unknown tie_break {tie_break!r}; use one of {TIE_BREAK_RULES}")
    for a in order:
        if Q.get((key, a), 0.0) == best_q:
            return a


def max_q(Q: dict, key) -> float:
    return max(Q.get((key, a), 0.0) for a in ACTIONS)


# ------------------------------------------------------------------------------ attacker policies
class AttackerPolicy:
    """Interface: action(key, eps, rng) for acting, probs(key, eps) for the Eq. 72 continuation (D12)."""
    def action(self, key, eps, rng):
        raise NotImplementedError

    def probs(self, key, eps):
        raise NotImplementedError


class QAttacker(AttackerPolicy):
    def __init__(self, QA: dict, tie_break: str, rng: Optional[random.Random] = None):
        self.Q = QA
        self.tie_break = tie_break
        self.rng = rng

    def greedy(self, key):
        return argmax_action(self.Q, key, self.tie_break, self.rng)

    def action(self, key, eps, rng):
        if eps > 0 and rng.random() < eps:
            return rng.choice(ACTIONS)
        return self.greedy(key)

    def probs(self, key, eps):
        """eps-greedy distribution; with random tie-breaking the greedy mass is split over the ties."""
        base = eps / len(ACTIONS)
        if self.tie_break == "random":
            best_q = max(self.Q.get((key, a), 0.0) for a in ACTIONS)
            ties = [a for a in ACTIONS if self.Q.get((key, a), 0.0) == best_q]
            return {a: base + ((1.0 - eps) / len(ties) if a in ties else 0.0) for a in ACTIONS}
        g = self.greedy(key)
        return {a: base + (1.0 - eps if a == g else 0.0) for a in ACTIONS}


class FixedAttacker(AttackerPolicy):
    """Non-learning doctrine: always the same class-level action (Table 5 Panel A)."""
    def __init__(self, action):
        self.fixed = tuple(action)

    def action(self, key, eps, rng):
        return self.fixed

    def probs(self, key, eps):
        return {a: (1.0 if a == self.fixed else 0.0) for a in ACTIONS}


# ------------------------------------------------------------------------------ episode runner
@dataclass
class EpisodeRecord:
    discounted_D: float = 0.0        # sum_t gamma^(t-1) D_t  (Eq. 77)
    undiscounted_D: float = 0.0
    realized_damage: float = 0.0     # value of assets actually destroyed
    # per-salvo (discounted D_t, attacker state visited in training?, defender state visited?)
    salvos: list = field(default_factory=list)


def run_episode(env: Environment, params: Params, QA: dict, QD: dict, attacker: AttackerPolicy,
                learn_A: bool, learn_D: bool, eps: float, rng: random.Random,
                trace: Optional[Callable[[dict], None]] = None,
                visited_A: Optional[set] = None, visited_D: Optional[set] = None,
                train_visited: Optional[tuple] = None) -> EpisodeRecord:
    """One engagement of T salvos following Algorithm 1. With learn_*=False nothing is written to
    that agent's Q-table. eps=0 gives the greedy policies of Eqs. 75-76. `train_visited=(setA, setD)`
    switches on the evaluation diagnostic (was this state visited during training?)."""
    z, m, q = env.initial_state()
    rec = EpisodeRecord()
    g = params.gamma
    for t in range(1, params.T + 1):
        kA = env.attacker_key(z, m, q, t)                              # Eq. 9  (x constant, D9)
        if visited_A is not None:
            visited_A.add(kA)
        a = attacker.action(kA, eps, rng)                              # Eq. 70
        kD = (kA, a)                                                   # Eq. 26
        if visited_D is not None:
            visited_D.add(kD)
        if eps > 0 and rng.random() < eps:                             # Eq. 71
            d = rng.choice(ACTIONS)
        else:
            d = argmax_action(QD, kD, params.tie_break, rng)
        res: StepResult = env.step(z, m, q, a, d, rng)
        disc = g ** (t - 1)
        rec.discounted_D += disc * res.D_t
        rec.undiscounted_D += res.D_t
        rec.realized_damage += res.realized_damage
        if train_visited is not None:
            rec.salvos.append((disc * res.D_t, kA in train_visited[0], kD in train_visited[1]))

        terminal = t == params.T
        kA_next = None if terminal else env.attacker_key(res.z_next, res.m_next, res.q_next, t + 1)

        upd = {}
        if learn_D:                                                    # Eqs. 72-73
            if terminal:
                Vbar, comps = 0.0, {}
            else:
                pa = attacker.probs(kA_next, eps)
                comps = {ab: (pa[ab], max_q(QD, (kA_next, ab))) for ab in ACTIONS if pa[ab] > 0}
                Vbar = sum(pr * mq for pr, mq in comps.values())
            old = QD.get((kD, d), 0.0)
            target_D = res.S_t + g * Vbar
            QD[(kD, d)] = (1 - params.alpha_D) * old + params.alpha_D * target_D
            upd["D"] = (old, res.S_t, Vbar, target_D, QD[(kD, d)], comps)
        if learn_A:                                                    # Eq. 74
            maxA = 0.0 if terminal else max_q(QA, kA_next)
            old = QA.get((kA, a), 0.0)
            target_A = res.D_t + g * maxA
            QA[(kA, a)] = (1 - params.alpha_A) * old + params.alpha_A * target_A
            upd["A"] = (old, res.D_t, maxA, target_A, QA[(kA, a)])

        if trace is not None:
            trace(dict(t=t, z=z, m=m, q=q, kA=kA, a=a, d=d, eps=eps, res=res, upd=upd, kA_next=kA_next))
        z, m, q = res.z_next, res.m_next, res.q_next
    return rec


# ------------------------------------------------------------------------------ training / evaluation
@dataclass
class RunResult:
    QA: dict
    QD: dict
    train_losses: list            # per training episode discounted D
    eval_losses: list             # per evaluation episode discounted D
    eval_undiscounted: list
    eval_realized: list
    visited_A: set                # attacker states visited in training
    visited_D: set                # defender states visited in training
    n_lp_solves: int
    eval_salvos: list = field(default_factory=list)   # flat list of (disc D_t, visitedA, visitedD)
    eval_episode_flags: list = field(default_factory=list)  # per episode (any unvisited A, any unvisited D)

    @property
    def loss(self) -> float:      # Eq. 78
        return sum(self.eval_losses) / len(self.eval_losses)

    def summary(self) -> dict:
        n_ep = len(self.eval_losses)
        n_s = len(self.eval_salvos)
        vis_A = sum(1 for _, vA, _ in self.eval_salvos if vA)
        vis_D = sum(1 for _, _, vD in self.eval_salvos if vD)
        d_visA = sum(x for x, vA, _ in self.eval_salvos if vA) / n_ep
        d_unvA = sum(x for x, vA, _ in self.eval_salvos if not vA) / n_ep
        d_visD = sum(x for x, _, vD in self.eval_salvos if vD) / n_ep
        d_unvD = sum(x for x, _, vD in self.eval_salvos if not vD) / n_ep
        return dict(
            loss=self.loss,
            loss_sd_episodes=statistics.pstdev(self.eval_losses) if n_ep > 1 else 0.0,
            undiscounted=sum(self.eval_undiscounted) / n_ep,
            realized=sum(self.eval_realized) / n_ep,
            train_last100=sum(self.train_losses[-100:]) / len(self.train_losses[-100:]),
            unique_attacker_states=len(self.visited_A), unique_defender_states=len(self.visited_D),
            QA_entries=len(self.QA), QD_entries=len(self.QD), n_lp_solves=self.n_lp_solves,
            # unvisited-state diagnostics (fractions)
            unvisited_A_salvo_frac=1 - vis_A / n_s, unvisited_D_salvo_frac=1 - vis_D / n_s,
            unvisited_A_episode_frac=sum(1 for fA, _ in self.eval_episode_flags if fA) / n_ep,
            unvisited_D_episode_frac=sum(1 for _, fD in self.eval_episode_flags if fD) / n_ep,
            # loss decomposition (per-episode mean of discounted D_t accumulated in each salvo class)
            loss_visited_A=d_visA, loss_unvisited_A=d_unvA, loss_visited_D=d_visD, loss_unvisited_D=d_unvD,
            n_eval_episodes=n_ep, n_train_episodes=len(self.train_losses))


def train_and_evaluate(env: Environment, params: Params, n_train: int, n_eval: int, seed: int,
                       attacker: Optional[AttackerPolicy] = None, QA: Optional[dict] = None,
                       QD: Optional[dict] = None, learn_A: bool = True, learn_D: bool = True,
                       eval_attacker: Optional[AttackerPolicy] = None,
                       trace_episode: Optional[int] = None, trace_fn=None,
                       progress: Optional[Callable[[int, float], None]] = None) -> RunResult:
    """Algorithm 1 Steps 1-5. `attacker=None` -> learning Q-attacker on QA. Pass a frozen QD with
    learn_D=False (stress test) or a FixedAttacker with learn_A=False (threat forecast)."""
    rng = random.Random(seed)                                           # D11
    QA = {} if QA is None else QA
    QD = {} if QD is None else QD
    if attacker is None:
        attacker = QAttacker(QA, params.tie_break, rng)
    visited_A, visited_D = set(), set()
    train_losses = []
    for ep in range(1, n_train + 1):
        eps = params.epsilon(ep, n_train)
        tr = trace_fn if (trace_fn is not None and ep == trace_episode) else None
        rec = run_episode(env, params, QA, QD, attacker, learn_A, learn_D, eps, rng, trace=tr,
                          visited_A=visited_A, visited_D=visited_D)
        train_losses.append(rec.discounted_D)
        if progress is not None and (ep % 500 == 0 or ep == n_train):
            progress(ep, sum(train_losses[-100:]) / len(train_losses[-100:]))
    # Steps 3-5: greedy evaluation, no Q writes (Eqs. 75-78)
    if eval_attacker is not None:
        ev_att = eval_attacker
    elif isinstance(attacker, QAttacker):
        ev_att = QAttacker(QA, params.tie_break, rng)
    else:
        ev_att = attacker
    ev, evu, evr, salvos, flags = [], [], [], [], []
    for _ in range(n_eval):
        rec = run_episode(env, params, QA, QD, ev_att, False, False, 0.0, rng,
                          train_visited=(visited_A, visited_D))
        ev.append(rec.discounted_D)
        evu.append(rec.undiscounted_D)
        evr.append(rec.realized_damage)
        salvos.extend(rec.salvos)
        flags.append((any(not vA for _, vA, _ in rec.salvos), any(not vD for _, _, vD in rec.salvos)))
    return RunResult(QA=QA, QD=QD, train_losses=train_losses, eval_losses=ev, eval_undiscounted=evu,
                     eval_realized=evr, visited_A=visited_A, visited_D=visited_D,
                     n_lp_solves=env.n_lp_solves, eval_salvos=salvos, eval_episode_flags=flags)
