"""Stage (c) robustness study (runs + CSVs). The narrative report and figure are produced by
stage_c_report.py from the CSVs written here.

1. Tables 1-2 under three arg-max tie-break rules x two environments, seeds 7/17/27, 1000/500.
2. One-battery swap refinement (Algorithm 2, Steps 3-7) under each rule from that rule's capped best.
3. Unvisited-state diagnostics for every run (stored per run; summarised in the CSVs).
4. Training-length sweep, Candidate 3 capped, aggressive rule.
5. Loss decomposition visited vs unvisited salvos (stored per run; summarised in the CSVs).
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_model.experiments import make_spec, run_many
from paper_model.instance import CANDIDATES, REFINED_PLAN, generate_instance
from paper_model.learner import TIE_BREAK_RULES
from paper_model.params import BASE, CAPPED_MAIN

OUT = Path(__file__).resolve().parent / "results" / "stage_c"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (7, 17, 27)
N_TRAIN, N_EVAL = 1000, 500
ENVS = {"capped": CAPPED_MAIN, "uncapped": BASE}
RULES = TIE_BREAK_RULES          # ("aggressive", "random", "lexicographic")
inst = generate_instance(7)

DIAG_KEYS = ["unvisited_A_salvo_frac", "unvisited_D_salvo_frac", "unvisited_A_episode_frac",
             "unvisited_D_episode_frac", "loss_visited_A", "loss_unvisited_A", "loss_visited_D",
             "loss_unvisited_D", "unique_attacker_states", "unique_defender_states", "QA_entries", "QD_entries"]


def write_csv(path: Path, rows: list, fields: list):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def per_seed_row(r: dict, **extra) -> dict:
    row = dict(extra)
    row.update(sites="-".join(map(str, r["sites"])), covered=r["covered"], seed=r["seed"], n_train=r["n_train"],
               loss=r["loss"], undiscounted=r["undiscounted"], realized=r["realized"], wall_s=r.get("wall_s"))
    row.update({k: r[k] for k in DIAG_KEYS})
    return row


def mean_sd(xs):
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)


# ============================================================================ 1. Tables 1-2 x rules
def phase_tables():
    specs = [make_spec("tables12", ENVS[env].with_(tie_break=rule), CANDIDATES[c], seed, N_TRAIN, N_EVAL,
                       rule=rule, env=env, candidate=c)
             for rule in RULES for env in ENVS for c in CANDIDATES for seed in SEEDS]
    res = run_many(specs, label="tables 1-2")
    per_seed, summary = [], []
    for rule in RULES:
        for env in ENVS:
            rows = []
            for c in CANDIDATES:
                rs = [r for r in res if r["rule"] == rule and r["env"] == env and r["candidate"] == c]
                for r in rs:
                    per_seed.append(per_seed_row(r, rule=rule, env=env, candidate=c))
                m, sd = mean_sd([r["loss"] for r in rs])
                row = dict(rule=rule, env=env, candidate=c, sites=", ".join(map(str, CANDIDATES[c])),
                           covered=rs[0]["covered"], mean_loss=m, sd_seeds=sd,
                           per_seed=" / ".join(f"{r['loss']:.2f}" for r in sorted(rs, key=lambda r: r["seed"])))
                for k in DIAG_KEYS:
                    row[k] = st.mean(r[k] for r in rs)
                rows.append(row)
            rows.sort(key=lambda r: r["mean_loss"])
            for rank, row in enumerate(rows, 1):
                row["rank"] = rank
            summary.extend(rows)
    write_csv(OUT / "tables_1_2_per_seed.csv", per_seed,
              ["rule", "env", "candidate", "sites", "covered", "seed", "n_train", "loss", "undiscounted", "realized", "wall_s"] + DIAG_KEYS)
    write_csv(OUT / "tables_1_2_summary.csv", summary,
              ["rule", "env", "rank", "candidate", "sites", "covered", "mean_loss", "sd_seeds", "per_seed"] + DIAG_KEYS)
    return summary


# ============================================================================ 2. swap refinement x rules
def neighbors(x: tuple):
    xs = set(x)
    for j in x:
        for k in range(inst.n):
            if k not in xs:
                yield j, k, tuple(sorted((xs - {j}) | {k}))


def evaluate_plan(rule: str, sites: tuple, exp: str):
    params = CAPPED_MAIN.with_(tie_break=rule)
    specs = [make_spec(exp, params, sites, seed, N_TRAIN, N_EVAL, rule=rule, env="capped") for seed in SEEDS]
    return specs


def phase_refinement(summary):
    trace_rows, final_rows = [], []
    for rule in RULES:
        best = min((r for r in summary if r["rule"] == rule and r["env"] == "capped"), key=lambda r: r["mean_loss"])
        inc = CANDIDATES[best["candidate"]]
        inc_mean, inc_sd = best["mean_loss"], best["sd_seeds"]
        trace_rows.append(dict(rule=rule, iteration=0, incumbent="-".join(map(str, inc)), swap_out="", swap_in="",
                               candidate="-".join(map(str, inc)), covered=inst.covered_count(inc), mean_loss=inc_mean,
                               sd_seeds=inc_sd, accepted="start", note=f"capped best = Candidate {best['candidate']}"))
        print(f"[refinement/{rule}] start from Candidate {best['candidate']} {inc} mean={inc_mean:.2f}", file=sys.stderr, flush=True)
        for it in range(1, 11):
            nbrs = list(neighbors(inc))
            specs = [s for _, _, x in nbrs for s in evaluate_plan(rule, x, "refine")]
            res = run_many(specs, label=f"refine/{rule}/iter{it} ({len(nbrs)} neighbours)")
            by_name = {r["name"]: r for r in res}
            scored = []
            for j, k, x in nbrs:
                rs = [by_name[s["name"]] for s in evaluate_plan(rule, x, "refine")]
                m, sd = mean_sd([r["loss"] for r in rs])
                scored.append((m, sd, j, k, x))
            scored.sort(key=lambda s: (s[0], s[4]))
            for m, sd, j, k, x in scored:
                trace_rows.append(dict(rule=rule, iteration=it, incumbent="-".join(map(str, inc)), swap_out=j, swap_in=k,
                                       candidate="-".join(map(str, x)), covered=inst.covered_count(x), mean_loss=m,
                                       sd_seeds=sd, accepted="", note=""))
            m, sd, j, k, x = scored[0]
            if m < inc_mean:                                   # Eq. 88, best-improvement (D15)
                trace_rows[-len(scored)]["accepted"] = "accepted"
                print(f"[refinement/{rule}] iter {it}: swap {j}->{k} gives {x} mean={m:.2f} (was {inc_mean:.2f})", file=sys.stderr, flush=True)
                inc, inc_mean, inc_sd = x, m, sd
            else:
                trace_rows[-len(scored)]["accepted"] = "rejected (no improving neighbour) -> stop"
                print(f"[refinement/{rule}] iter {it}: best neighbour {x} mean={m:.2f} >= {inc_mean:.2f}; stop", file=sys.stderr, flush=True)
                break
        final_rows.append(dict(rule=rule, start_candidate=best["candidate"], final="-".join(map(str, inc)),
                               final_covered=inst.covered_count(inc), final_mean_loss=inc_mean, final_sd_seeds=inc_sd,
                               start_mean_loss=best["mean_loss"],
                               reduction_pct=100 * (best["mean_loss"] - inc_mean) / best["mean_loss"],
                               reaches_paper_plan=(inc == REFINED_PLAN), paper_plan="-".join(map(str, REFINED_PLAN))))
    write_csv(OUT / "refinement_trace.csv", trace_rows,
              ["rule", "iteration", "incumbent", "swap_out", "swap_in", "candidate", "covered", "mean_loss", "sd_seeds", "accepted", "note"])
    write_csv(OUT / "refinement_summary.csv", final_rows,
              ["rule", "start_candidate", "start_mean_loss", "final", "final_covered", "final_mean_loss", "final_sd_seeds",
               "reduction_pct", "reaches_paper_plan", "paper_plan"])


# ============================================================================ 4. training-length sweep
def phase_training_length():
    lengths = (1000, 2500, 5000, 10000, 20000)
    params = CAPPED_MAIN.with_(tie_break="aggressive")
    specs = [make_spec("trainlen", params, CANDIDATES[3], seed, n, N_EVAL, rule="aggressive", env="capped", candidate=3)
             for n in lengths for seed in SEEDS]
    res = run_many(specs, label="training length")
    per_seed = [per_seed_row(r, rule="aggressive", env="capped", candidate=3) for r in res]
    summary = []
    for n in lengths:
        rs = [r for r in res if r["n_train"] == n]
        m, sd = mean_sd([r["loss"] for r in rs])
        row = dict(n_train=n, mean_loss=m, sd_seeds=sd, per_seed=" / ".join(f"{r['loss']:.2f}" for r in sorted(rs, key=lambda r: r["seed"])),
                   mean_wall_s=st.mean(r["wall_s"] for r in rs))
        for k in DIAG_KEYS:
            row[k] = st.mean(r[k] for r in rs)
        summary.append(row)
    write_csv(OUT / "training_length_per_seed.csv", per_seed,
              ["rule", "env", "candidate", "sites", "seed", "n_train", "loss", "undiscounted", "realized", "wall_s"] + DIAG_KEYS)
    write_csv(OUT / "training_length_summary.csv", summary, ["n_train", "mean_loss", "sd_seeds", "per_seed", "mean_wall_s"] + DIAG_KEYS)


if __name__ == "__main__":
    phases = sys.argv[1:] or ["tables", "trainlen", "refine"]
    summary = phase_tables()
    if "trainlen" in phases:
        phase_training_length()
    if "refine" in phases:
        phase_refinement(summary)
    print("stage (c) runs complete", file=sys.stderr)
