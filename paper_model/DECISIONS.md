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

## Flagged for the authors before later stages run

**D4. Over-cap actions: filter-before-selection vs. trimming. [AUTHORS: CONFIRM]**
The manuscript text says the environment *removes the lowest-ranked selected targets / defended
assets until the realized vector satisfies the cap* (Sec. 4.2 after Eq. 17; Sec. 4.3 after Eq. 39;
Sec. 4.4 after Eq. 62; Sec. 5.1.2). The rebuild instructions say instead: *keep an action in the
feasible set only if its realized vector already satisfies the cap and (for defence) can be sourced;
filter-before-selection; no trimming.* The rebuild follows the instructions. Note that under
filtering, Eqs. (19) and (40) are the operative definitions and the "remove lowest-ranked" sentences
in Secs. 4.2–4.4 and 5.1.2 no longer describe the code; the manuscript text should be aligned with
whichever mechanism is kept. Both mechanisms give the same non-binding thresholds used as the Table 3
sanity check (LAM ≥ 16 and LIM ≥ 32).
