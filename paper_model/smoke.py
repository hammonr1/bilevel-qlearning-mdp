"""Stage (b): 50-episode smoke run on Candidate 3, capped (LAM=10, LIM=12), seed 7.

1. cross-checks the max-flow sourcing-feasibility test against a PuLP feasibility ILP,
2. benchmarks the sourcing solver backends,
3. trains 50 episodes, writes a per-salvo trace of one training episode for hand-checking
   against Eqs. 47-68 (and the Q-updates of Eqs. 72-74), then evaluates greedily.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import pulp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_model.env import Environment, sourcing_feasible
from paper_model.instance import CANDIDATES, generate_instance
from paper_model.learner import train_and_evaluate
from paper_model.params import ACTIONS, CAPPED_MAIN

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
inst = generate_instance(7)
SITES = CANDIDATES[3]
TRACE_EPISODE = int(sys.argv[1]) if len(sys.argv) > 1 else 50


# ----------------------------------------------------------------------------- 1. feasibility cross-check
def pulp_feasible(sites, q, u, C):
    if not u:
        return True
    prob = pulp.LpProblem("f", pulp.LpMinimize)
    var = {(j, i): pulp.LpVariable(f"w_{j}_{i}", 0, None, "Integer")
           for k, j in enumerate(sites) for i in u if C[j, i] and q[k] > 0}
    prob += pulp.lpSum([])
    for i in u:
        prob += pulp.lpSum(var[(j, i)] for j in sites if (j, i) in var) == u[i]
    for k, j in enumerate(sites):
        terms = [var[(j, i)] for i in u if (j, i) in var]
        if terms:
            prob += pulp.lpSum(terms) <= q[k]
    return pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=False))] == "Optimal"


rng = random.Random(123)
agree = n_feas = 0
N_CHECK = 300
for _ in range(N_CHECK):
    sites = tuple(sorted(rng.sample(range(30), rng.choice([3, 6, 9]))))
    q = tuple(rng.choice([0, 0, 1, 2, 3, 5, 10]) for _ in sites)
    assets = rng.sample(range(30), rng.randint(1, 8))
    u = {i: rng.randint(1, 4) for i in assets}
    f1, f2 = sourcing_feasible(sites, q, u, inst.C), pulp_feasible(sites, q, u, inst.C)
    agree += f1 == f2
    n_feas += f2
print(f"[1] max-flow feasibility vs PuLP ILP on {N_CHECK} random (sites,q,u) cases: "
      f"{agree}/{N_CHECK} agree ({n_feas} feasible, {N_CHECK - n_feas} infeasible)")
assert agree == N_CHECK, "feasibility test disagrees with PuLP"

# ----------------------------------------------------------------------------- 2. solver benchmark
z, m, q = Environment(inst, CAPPED_MAIN, SITES).initial_state()
bench = {}
for name in ["PULP_CBC_CMD", "HiGHS"]:
    try:
        env_b = Environment(inst, CAPPED_MAIN, SITES, sourcing_solver=name)
    except Exception as e:  # solver missing
        print(f"[2] {name}: unavailable ({e})")
        continue
    t0 = time.perf_counter()
    n = 0
    for a in ACTIONS:
        for d in ACTIONS:
            atk = env_b.realize_attack(z, m, a)
            dfn = env_b.realize_defense(z, q, atk.y, d)
            env_b._w_cache.clear()
            env_b.source(q, dfn.u, dfn.Lambda)
            n += 1
    bench[name] = (time.perf_counter() - t0) / n * 1000
    print(f"[2] sourcing solve with {name}: {bench[name]:.1f} ms per solve (incl. empty-u cases)")

# ----------------------------------------------------------------------------- 3. smoke run with trace
SOLVER = "PULP_CBC_CMD"
env = Environment(inst, CAPPED_MAIN, SITES, sourcing_solver=SOLVER)
lines = []


def P(s=""):
    lines.append(s)


def trace(ev):
    t, a, d, res, upd = ev["t"], ev["a"], ev["d"], ev["res"], ev["upd"]
    z, m, q = ev["z"], ev["m"], ev["q"]
    atk, dfn = res.attack, res.defense
    aliveH = [i for i in inst.high if z[i]]
    aliveL = [i for i in inst.low if z[i]]
    P(f"\n{'=' * 100}\nSALVO t={t}   (eps={ev['eps']:.4f})\n{'=' * 100}")
    P(f"State s^A_t: m_t={m}   q_t={{{', '.join(f'{j}:{qk}' for j, qk in zip(env.sites, q))}}}   "
      f"sum q={sum(q)}   surviving H={aliveH}   surviving L={aliveL}")
    P(f"\n-- ATTACK (Eqs. 13-18) --  a_t=(a^H,a^L)={a}   K^H={env.p.K_H[a[0]]} eta^H={env.p.eta_H[a[0]]}   "
      f"K^L={env.p.K_L[a[1]]} eta^L={env.p.eta_L[a[1]]}   Lambda^A=min(m,LAM)=min({m},{env.p.LAM})={atk.Lambda}")
    rankH = sorted(aliveH, key=lambda i: (-env.R[i], i))
    rankL = sorted(aliveL, key=lambda i: (-env.R[i], i))
    P("   R_it=v_i*p_i ranking H: " + ", ".join(f"{i}:{env.R[i]:.2f}" for i in rankH))
    P("   R_it=v_i*p_i ranking L: " + ", ".join(f"{i}:{env.R[i]:.2f}" for i in rankL))
    P(f"   preliminary y~: {{{', '.join(f'{i}:{e}' for _, i, e in atk.selected)}}}  sum={sum(e for _, _, e in atk.selected)}")
    P(f"   trimmed (lowest R first): {[(i, e, round(R, 2)) for R, i, e in atk.trimmed] or 'none'}")
    P(f"   realized y_it: {atk.y}   sum y={atk.total} <= Lambda^A={atk.Lambda}")
    P(f"\n-- DEFENSE (Eqs. 30-40) --  d_t=(d^H,d^L)={d}   L^H={env.p.L_H[d[0]]} rho^H={env.p.rho_H[d[0]]}   "
      f"L^L={env.p.L_L[d[1]]} rho^L={env.p.rho_L[d[1]]}   Lambda^D=min(sum q,LIM)=min({sum(q)},{env.p.LIM})={dfn.Lambda}")
    H = {i: env.v[i] * env.pdmg[i] * atk.y[i] for i in atk.y}
    thrH = sorted([i for i in inst.high if i in atk.y], key=lambda i: (-H[i], i))
    thrL = sorted([i for i in inst.low if i in atk.y], key=lambda i: (-H[i], i))
    P("   H_it=v*p*y threatened H: " + (", ".join(f"{i}:{H[i]:.2f}" for i in thrH) or "none"))
    P("   H_it=v*p*y threatened L: " + (", ".join(f"{i}:{H[i]:.2f}" for i in thrL) or "none"))
    P(f"   preliminary u~ (rho*y): {{{', '.join(f'{i}:{ui}' for _, i, ui in dfn.selected)}}}  sum={sum(ui for _, _, ui in dfn.selected)}")
    P(f"   trimmed (lowest H first; reason): {[(i, ui, round(Hi, 2), why) for Hi, i, ui, why in dfn.trimmed] or 'none'}")
    P(f"   realized u_it: {dfn.u}   sum u={dfn.total} <= Lambda^D={dfn.Lambda}")
    P(f"   sourcing w*_jit (site->asset): {dfn.w}")
    fired = {j: sum(w for (jj, _), w in dfn.w.items() if jj == j) for j in env.sites}
    P(f"   IMs fired per site: {fired}")
    P(f"\n-- ENGAGEMENT (Eqs. 47-54) --  omega={env.p.omega}")
    P(f"   {'i':>3s} {'cls':>3s} {'v_i':>8s} {'p_i':>7s} {'y':>2s} {'u':>2s} {'r':>2s} {'h':>2s} {'P^S_it':>8s} {'v(1-P)':>8s}")
    for i in sorted(atk.y):
        r, h = res.rh[i]
        P(f"   {i:3d} {'H' if inst.is_high[i] else 'L':>3s} {env.v[i]:8.3f} {env.pdmg[i]:7.4f} {atk.y[i]:2d} {dfn.u.get(i, 0):2d} "
          f"{r:2d} {h:2d} {res.P_S[i]:8.5f} {env.v[i] * (1 - res.P_S[i]):8.3f}")
    alive_val = sum(env.v[i] for i in range(env.n) if z[i])
    P(f"   D_t = sum v_i(1-P^S) = {res.D_t:.4f}   (attacker reward r^A, Eq. 55)")
    P(f"   S_t = sum_alive v_i - D_t = {alive_val:.4f} - {res.D_t:.4f} = {res.S_t:.4f}   (defender reward r^D, Eq. 56)")
    P(f"\n-- TRANSITION (Eqs. 63-66) --")
    P(f"   xi_it draws (1=survives): {res.xi}   destroyed this salvo: {[i for i, x in res.xi.items() if x == 0]}   realized value lost={res.realized_damage:.3f}")
    P(f"   m_(t+1) = {m} - {atk.total} = {res.m_next}")
    P(f"   q_(t+1) = z_j,t+1 * (q_jt - sum_i w_jit): {{{', '.join(f'{j}:{res.z_next[j]}*({q[k]}-{fired[j]})={res.q_next[k]}' for k, j in enumerate(env.sites))}}}")
    P(f"   surviving after salvo: H={[i for i in inst.high if res.z_next[i]]}  L={[i for i in inst.low if res.z_next[i]]}")
    P(f"\n-- Q-UPDATES --  gamma={env.p.gamma}  alpha_A={env.p.alpha_A}  alpha_D={env.p.alpha_D}")
    if "D" in upd:
        old, rD, Vbar, tgt, new, comps = upd["D"]
        P(f"   Eq.72  Vbar^D(s^A_t+1) = sum_a pi^A(a|s') max_d Q^D((s',a),d) = {Vbar:.4f}")
        if comps:
            P("          components a:(pi^A, max_d Q^D) = " + ", ".join(f"{ab}:({pr:.4f},{mq:.3f})" for ab, (pr, mq) in comps.items()))
        else:
            P("          terminal salvo -> 0")
        P(f"   Eq.73  Q^D <- (1-{env.p.alpha_D})*{old:.4f} + {env.p.alpha_D}*[{rD:.4f} + {env.p.gamma}*{Vbar:.4f}] = {new:.4f}   (target={tgt:.4f})")
    if "A" in upd:
        old, rA, maxA, tgt, new = upd["A"]
        P(f"   Eq.74  max_a Q^A(s^A_t+1,a) = {maxA:.4f}")
        P(f"          Q^A <- (1-{env.p.alpha_A})*{old:.4f} + {env.p.alpha_A}*[{rA:.4f} + {env.p.gamma}*{maxA:.4f}] = {new:.4f}   (target={tgt:.4f})")


P(f"SMOKE RUN: Candidate 3 sites={SITES}, capped LAM={CAPPED_MAIN.LAM} LIM={CAPPED_MAIN.LIM}, T={CAPPED_MAIN.T}, "
  f"M={CAPPED_MAIN.M}, N_IM={CAPPED_MAIN.N_IM}, omega={CAPPED_MAIN.omega}, seed=7, 50 training episodes, "
  f"traced episode = {TRACE_EPISODE}, sourcing solver = {SOLVER}")
P("Asset table: i cls v p R=v*p   (H = high importance)")
for i in range(inst.n):
    P(f"   {i:2d} {'H' if inst.is_high[i] else 'L'} v={inst.v[i]:8.3f} p={inst.p[i]:.4f} R={env.R[i]:8.3f}")
P("Coverage of Candidate 3 sites (site: covered assets):")
for j in env.sites:
    P(f"   {j:2d}: {[i for i in range(inst.n) if inst.C[j, i]]}")

t0 = time.perf_counter()
res = train_and_evaluate(env, CAPPED_MAIN, n_train=50, n_eval=20, seed=7,
                         trace_episode=TRACE_EPISODE, trace_fn=trace)
elapsed = time.perf_counter() - t0
s = res.summary()
P(f"\n{'=' * 100}\nSUMMARY after 50 training episodes + 20 greedy evaluation episodes ({elapsed:.1f}s wall)")
P(f"   training discounted D per episode (first 5): {[round(x, 1) for x in res.train_losses[:5]]}  ... last 5: {[round(x, 1) for x in res.train_losses[-5:]]}")
P(f"   greedy evaluation loss L(x) = {s['loss']:.2f}  (episode SD {s['loss_sd_episodes']:.2f}; undiscounted {s['undiscounted']:.2f}; realized value destroyed {s['realized']:.2f})")
P(f"   unique attacker states {s['unique_attacker_states']}, defender states {s['unique_defender_states']}, "
  f"Q^A entries {s['QA_entries']}, Q^D entries {s['QD_entries']}, LP solves {s['n_lp_solves']}")
(OUT / "smoke_trace_candidate3_capped.txt").write_text("\n".join(lines))
print("\n".join(lines[-6:]))
print(f"\ntrace written to {OUT / 'smoke_trace_candidate3_capped.txt'} ({len(lines)} lines)")
print(f"TIMING: {elapsed:.1f}s for 70 episodes -> {elapsed / 70 * 1000:.0f} ms/episode with {SOLVER}; "
      f"LP solves={s['n_lp_solves']}, cache size={len(env._w_cache)}")
