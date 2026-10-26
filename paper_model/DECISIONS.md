# Modelling decisions log — `paper_model/`

Every place where the manuscript (RESS revision, "Dynamic Resilience Assessment and Adaptive
Resource Allocation for Networked Protective Systems Using Hierarchical Stackelberg RL") is
ambiguous, or where the author's rebuild instructions differ from the manuscript text, is logged
here with the choice taken. Items marked **[AUTHORS: CONFIRM]** need the authors' sign-off before
the regenerated numbers are used in the paper.

## Stage (a): instance generation

**D1. Random-number stream and generation order (Sec. 5.1.1).**
The manuscript gives seed 7 but not the generator or the draw order. Searched all of:
NumPy legacy `np.random.seed(7)`, `np.random.RandomState(7)`, `np.random.default_rng(7)` and the
Python standard library `random.Random(7)`, with the 60 location coordinates laid out either
`(30, 2)` row-major or x-block-then-y-block, and with 0–299 doubles consumed before the location
block. Exactly one combination reproduces the "Initially covered assets" column of Table 1 for all
five candidates **and** the refined plan of Sec. 5.2 (28 covered):

```
r = random.Random(7)                                   # Python stdlib, not NumPy
locs  = [(r.uniform(0,100), r.uniform(0,100)) for i in range(30)]   # x then y, per asset, index order
I_H   = sorted(r.sample(range(30), 8))                 # -> {10, 11, 15, 19, 22, 26, 27, 28}
for i in range(30):                                    # per asset, index order, value then probability
    v_i = r.uniform(80,120) if i in I_H else r.uniform(20,60)
    p_i = r.uniform(0.75,0.95) if i in I_H else r.uniform(0.55,0.80)
c_ji  = 1 iff dist(j, i) <= 40
```
The class assignment and value/probability order were then fixed by requiring the Sec. 5.4 static
rules to reproduce Candidates 0 and 1 (both depend on asset values). Among 28 tested
(class-assignment × draw-order) variants, only `random.sample` for I_H with per-asset
value-then-probability draws reproduces both. Rejected: I_H = first 8 / last 8 / shuffled indices;
block-wise draws (all values then all probabilities, or by class).
Implemented in `instance.py`; gated by `check_instance.py` (non-zero exit on any mismatch).

**D2. Tie-break in the static coverage-count rule (Sec. 5.4).**
Locations 2, 14, 15 and 26 all cover 13 assets. The paper's Candidate 2 includes 26 and excludes 2.
Breaking ties by ascending index gives (2, 9, 19, 20, 22, 27) ≠ Candidate 2; by value-weighted
coverage gives (9, 14, 19, 20, 22, 27) ≠ Candidate 2. Two tie-breaks reproduce Candidate 2:
descending index (the `np.argsort(G)[::-1]` idiom) and the candidate location's own asset value.
Implemented: descending index (simplest). Affects only the Panel B benchmark label, not any
learning result.

**D3. Greedy set-cover rule tie-break (Sec. 5.4).** Implemented exactly as the text describes:
maximise newly covered assets, ties by value-weighted uncovered coverage, then lowest index.
Reproduces Candidate 0 without further assumptions.

## Stage (b): environment and learner

**D4. Over-cap actions: trimming (manuscript) vs. filter-before-selection (considered, rejected).**
The manuscript text (Sec. 4.2 after Eq. 17; Sec. 4.3 after Eq. 39; Sec. 4.4 after Eq. 62; Sec. 5.1.2)
says the environment *removes the lowest-ranked selected targets / defended assets until the
realized vector satisfies the cap and sourcing constraints*. Filter-before-selection (drop the
class-level action from the feasible set instead) was considered because the author's first rebuild
brief asked for it, and was **rejected** by the author on review of stage (a): the manuscript's
trimming mechanism is authoritative. Implemented:
- every class-level action in {0,1,2}^2 is selectable in every state, so A^A(s) and A^D(s) are
  always the full 9-action grids (Eqs. 19 and 40 are then satisfied by construction);
- the environment realizes the action and trims: attacker removes the selected target with the
  smallest R_it (Eq. 13) until sum_i y_it <= Lambda^A_t (Eq. 17-18); defender removes the defended
  asset with the smallest H_it (Eq. 30) until sum_i u_it <= Lambda^D_t and the sourcing system
  (34)-(37) is feasible (Eqs. 38-40);
- a defense trimmed to zero IMs is identical to the no-defense action (its Q-entry is still keyed
  by the class-level action that was chosen, as the manuscript's Q^D(s^D, d_t) requires).

**D5. "Lowest-ranked" is taken across both importance classes**, i.e. by the raw score R_it
(attacker) or H_it (defender), not class-by-class. Because high-importance scores dominate
(v_i p_i ~ 70-110 vs. 11-48), low-importance targets are trimmed first in practice. Ties (measure
zero) go to the higher asset index.

**D6. Trimming removes a whole selected target** (all eta AMs or all rho*y IMs assigned to it),
never a fraction of them, since the manuscript trims *targets/assets*, not missiles. Leftover
capacity after a removal is not redistributed.

**D7. Sourcing feasibility during trimming is checked exactly with a max-flow test** on the
bipartite SAM-to-asset network (supplies q_jt, demands u_it, arcs where c_ji = 1), which is
equivalent to the existence of an integer w satisfying (34)-(37) by total unimodularity; the total
cap (39) is checked separately. The realized w* is then obtained from the PuLP integer program
(58)-(62) with **no kappa cost term** (zero objective, any feasible solution), as instructed.
**[AUTHORS: CONFIRM]** With a zero objective the particular feasible w* (hence which SAM site is
depleted, and hence SAM-vulnerability effects) is left to the solver and is deterministic but
arbitrary. Eq. (57)'s kappa_ji (e.g. kappa = distance) would make this choice principled; the
manuscript gives no kappa value.

**D8. Sourcing solver.** PuLP 2.9.0 with the bundled CBC (`PULP_CBC_CMD`), fixed for all reported
runs and recorded in the results metadata (`sourcing_solver`). Stage (b) benchmark on Candidate 3:
CBC 5.2 ms and HiGHS 1.5 ms per sourcing solve; a full 1,000-train/500-eval run takes about 5 s with
CBC (the (q, u) -> w* cache leaves only ~500 distinct solves per run), so speed did not force the
choice. Because the zero-objective solution is solver-dependent (D7), switching backends changes the
realized SAM depletion and therefore the numbers (stage (b) single-seed check: capped 952 vs 970,
uncapped 1000 vs 895 for CBC vs HiGHS).

**D9. Q-table keys.** The allocation vector x is constant within a learning run and is therefore not
stored inside each key; attacker key = (z as a 30-bit mask, m_t, q_t restricted to the allocated SAM
sites in ascending site order, t); defender key = (attacker key, a_t). Unique-state counts for
Table 7 are counts of these keys.

**D10. Exploration schedule and arg-max tie-break.**
eps_k = 0.60 - (0.60 - 0.02) (k-1)/(N_train-1) for training episode k (linear, same for both
agents, stretched to whatever N_train a run uses); eps = 0 in evaluation. The manuscript does not
say how arg-max ties are broken, and this matters: the greedy evaluation trajectory visits states
that were never visited in training in a large share of salvos (Bernoulli outcomes keep producing
new (z, m, q) combinations), and in such a state every Q-entry is 0, so the "arg-max" is entirely
the tie-break rule.

**Primary rule (author decision): `aggressive`** — ties go to the most aggressive action, ordered by
total intensity then high-importance intensity, so (2,2) first, applied identically to the attacker
and the defender, in training and in evaluation. Rationale: consistency with the conservative
high-information adversary assumption of Section 3.2 — an attacker that has not learned a state
should be assumed to press the attack, not to stand down, so that the defender's resilience is not
overstated; the same rule for the defender means an unlearned defender response is maximal defence,
which is the symmetric conservative default for the operational layer. The rule is *not* chosen to
match the manuscript's numbers. Two alternatives are carried through stage (c) as a robustness
study: `random` (uniform over the tied actions, using the run's RNG) and `lexicographic` (first
action in (0,0), (0,1), ... order, i.e. no attack / no defence). Their effect is reported in
`results/stage_c/REPORT.md`.

**D11. Randomness.** One `random.Random(seed)` stream per learning run drives both agents'
eps-greedy draws and the Bernoulli survival draws (Eq. 63), in the order they occur in Algorithm 1;
evaluation episodes continue the same stream.

**D12. Defender continuation value (Eq. 72).** pi^A(a|s^A_{t+1}) is the attacker's *current*
eps-greedy policy: eps/9 on every action plus (1-eps) on arg-max_a Q^A(s^A_{t+1}, a); unseen
Q-entries read as 0. Under a fixed (non-learning) attacker doctrine the policy puts mass 1 on the
doctrine's action.

**D13. Rewards and loss.** r^A_t = D_t (Eq. 53, expected damage) and r^D_t = S_t (Eq. 54), both
computed from the survival probabilities, not from the realized Bernoulli outcome. The reported
post-engagement loss is the mean over evaluation episodes of sum_t gamma^(t-1) D_t (Eqs. 77-78),
where D_t is evaluated along the realized trajectory. Undiscounted sums are also stored for reference.

## Stage (c): robustness study

**D14. Swap refinement acceptance (Algorithm 2, Step 6).** The manuscript says "if there exists a
neighbouring allocation with lower objective, set x_inc = x'". Implemented as best-improvement: all
144 one-battery swap neighbours are evaluated with the same 3-seed protocol and the incumbent moves
to the neighbour with the lowest mean loss if that is strictly lower (Eq. 88); the search stops at the
first iteration with no improving neighbour (at most 10 iterations). The Sec. 5.2 trace (one swap
per iteration, each the best found) is consistent with this reading.

**D15. Parallel execution.** Runs are independent learning problems executed in a process pool (8
workers on 4 cores to overlap the CBC subprocess latency); each run's RNG is seeded by its own seed,
so results do not depend on scheduling. Completed runs are cached as JSON and never recomputed.
