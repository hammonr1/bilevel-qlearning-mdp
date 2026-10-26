"""Stage (c) narrative report and figure, generated from the CSVs written by stage_c.py."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_model.instance import CANDIDATES, REFINED_PLAN

OUT = Path(__file__).resolve().parent / "results" / "stage_c"
RULES = ("aggressive", "random", "lexicographic")
RULE_LABEL = {"aggressive": "aggressive (primary; ties -> (2,2))", "random": "random (uniform over ties)",
              "lexicographic": "lexicographic (ties -> (0,0))"}
PAPER = {"capped": {3: 1363.04, 4: 1399.72, 0: 1401.34, 2: 1419.94, 1: 1438.17},
         "uncapped": {0: 1346.07, 3: 1363.67, 1: 1406.98, 4: 1407.77, 2: 1441.00}}
PAPER_REFINE = [("21->12", 1320.85), ("24->3", 1318.71)]


def read(name):
    with open(OUT / name) as f:
        return [dict(r) for r in csv.DictReader(f)]


def fl(x):
    return float(x)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x):
    return f"{100 * fl(x):.1f}%"


# ------------------------------------------------------------------------------------- per-seed refinement CSV
# (raw/ is git-ignored; this keeps every refinement run's per-seed result in the committed outputs)
import json
RAW = OUT.parent / "raw"
_ref_runs = sorted(RAW.glob("refine__*.json"))
if _ref_runs:
    with open(OUT / "refinement_per_seed.csv", "w", newline="") as f:
        w = None
        for p in _ref_runs:
            r = json.load(open(p))
            row = dict(rule=r["rule"], env=r["env"], sites="-".join(map(str, r["sites"])), covered=r["covered"],
                       seed=r["seed"], n_train=r["n_train"], loss=r["loss"], undiscounted=r["undiscounted"],
                       realized=r["realized"], unvisited_A_salvo_frac=r["unvisited_A_salvo_frac"],
                       unvisited_D_salvo_frac=r["unvisited_D_salvo_frac"], unvisited_A_episode_frac=r["unvisited_A_episode_frac"],
                       unvisited_D_episode_frac=r["unvisited_D_episode_frac"], QA_entries=r["QA_entries"], QD_entries=r["QD_entries"])
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)

# ------------------------------------------------------------------------------------- load
summary = read("tables_1_2_summary.csv")
per_seed = read("tables_1_2_per_seed.csv")
trainlen = read("training_length_summary.csv") if (OUT / "training_length_summary.csv").exists() else []
trainlen_seed = read("training_length_per_seed.csv") if (OUT / "training_length_per_seed.csv").exists() else []
ref_trace = read("refinement_trace.csv") if (OUT / "refinement_trace.csv").exists() else []
ref_sum = read("refinement_summary.csv") if (OUT / "refinement_summary.csv").exists() else []


def best(rule, env):
    rows = [r for r in summary if r["rule"] == rule and r["env"] == env]
    return min(rows, key=lambda r: fl(r["mean_loss"]))


def ranking(rule, env):
    rows = sorted((r for r in summary if r["rule"] == rule and r["env"] == env), key=lambda r: fl(r["mean_loss"]))
    return [int(r["candidate"]) for r in rows]


# ------------------------------------------------------------------------------------- figure (item 4)
fig_path = None
if trainlen:
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
    xs = [int(r["n_train"]) for r in trainlen]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:,}" for x in xs])
        ax.grid(True, axis="y", color=GRID, linewidth=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("training episodes (same ε schedule, stretched)", color=MUTED, fontsize=9)
        ax.minorticks_off()
    ax1.errorbar(xs, [fl(r["mean_loss"]) for r in trainlen], yerr=[fl(r["sd_seeds"]) for r in trainlen],
                 color=BLUE, linewidth=2, marker="o", markersize=7, capsize=4, ecolor=BLUE)
    ax1.set_title("Post-engagement loss  L(x), mean ± SD over seeds 7/17/27", loc="left", fontsize=10, color=INK)
    ax1.set_ylabel("discounted expected damage", color=MUTED, fontsize=9)
    for series, col, key in (("attacker state", BLUE, "unvisited_A_salvo_frac"), ("defender state", ORANGE, "unvisited_D_salvo_frac")):
        ys = [100 * fl(r[key]) for r in trainlen]
        ax2.plot(xs, ys, color=col, linewidth=2, marker="o", markersize=7, label=series,
                 markeredgecolor="#fcfcfb", markeredgewidth=1.5)   # surface ring: the two series overlap
    ax2.set_ylim(0, 100)
    ax2.set_title("Evaluation salvos in states never visited in training", loc="left", fontsize=10, color=INK)
    ax2.set_ylabel("% of evaluation salvos", color=MUTED, fontsize=9)
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    ax2.set_xlim(xs[0] * 0.8, xs[-1] * 1.25)
    fig.suptitle("Candidate 3, capped (LAM=10, LIM=12), aggressive tie-break", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig_path = OUT / "training_length.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

# ------------------------------------------------------------------------------------- report
L = []
P = L.append
P("# Stage (c) robustness study — Tables 1–2, refinement, unvisited states, training length, loss decomposition\n")
P("All runs: 30-asset instance (seed 7), T = 8, M = 64, B = 6, N_IM = 10, ω = 0.65, γ = 0.95, α = 0.10, "
  "ε linear 0.60 → 0.02, 1,000 training / 500 evaluation episodes unless stated, seeds 7, 17, 27; "
  "capped = LAM 10 / LIM 12. Trimming per Secs. 4.2–4.4; sourcing by PuLP/CBC with zero objective (DECISIONS D7–D8). "
  "The one free choice studied here is the arg-max tie-break (DECISIONS D10): **aggressive** is the primary rule; "
  "**random** and **lexicographic** are the alternatives. \"Unvisited\" means the evaluation state has no entry in "
  "the training-visited set, so every Q-value in its row is 0 and the tie-break alone decides the action.\n")

# ---- 1
P("## 1. Tables 1–2 under the three tie-break rules\n")
flags = {}
for rule in RULES:
    P(f"### Rule: {RULE_LABEL[rule]}\n")
    for env in ("uncapped", "capped"):
        rows = sorted((r for r in summary if r["rule"] == rule and r["env"] == env), key=lambda r: fl(r["mean_loss"]))
        title = "Table 1 (uncapped)" if env == "uncapped" else "Table 2 (capped, LAM 10 / LIM 12)"
        P(f"**{title}**\n")
        P(md_table(["Rank", "Candidate", "SAM sites", "Covered", "Mean loss", "SD (seeds)", "Per seed 7 / 17 / 27", "Paper mean", "Paper rank"],
                   [[r["rank"], r["candidate"], r["sites"], r["covered"], f"{fl(r['mean_loss']):.2f}", f"{fl(r['sd_seeds']):.2f}",
                     r["per_seed"], f"{PAPER[env][int(r['candidate'])]:.2f}",
                     sorted(PAPER[env], key=PAPER[env].get).index(int(r["candidate"])) + 1] for r in rows]))
        P("")
    bu, bc = best(rule, "uncapped"), best(rule, "capped")
    a = int(bu["candidate"]) == 0
    b = int(bc["candidate"]) == 3
    c = int(bu["candidate"]) != int(bc["candidate"])
    flags[rule] = dict(a=a, b=b, c=c, best_unc=int(bu["candidate"]), best_cap=int(bc["candidate"]),
                       rank_unc=ranking(rule, "uncapped"), rank_cap=ranking(rule, "capped"))
    P(f"- (a) Candidate 0 best uncapped: **{'YES' if a else 'NO'}** (best = Candidate {bu['candidate']}, {fl(bu['mean_loss']):.2f}).")
    P(f"- (b) Candidate 3 best capped: **{'YES' if b else 'NO'}** (best = Candidate {bc['candidate']}, {fl(bc['mean_loss']):.2f}).")
    P(f"- (c) Ranking flip between environments: **{'YES' if c else 'NO'}** "
      f"(uncapped order {flags[rule]['rank_unc']}, capped order {flags[rule]['rank_cap']}; paper: uncapped [0, 3, 1, 4, 2], capped [3, 4, 0, 2, 1]).")
    # gaps vs SD
    gap_u = fl(sorted((r for r in summary if r["rule"] == rule and r["env"] == "uncapped"), key=lambda r: fl(r["mean_loss"]))[1]["mean_loss"]) - fl(bu["mean_loss"])
    gap_c = fl(sorted((r for r in summary if r["rule"] == rule and r["env"] == "capped"), key=lambda r: fl(r["mean_loss"]))[1]["mean_loss"]) - fl(bc["mean_loss"])
    P(f"- Margin of the best over the runner-up: uncapped {gap_u:.2f} (best's seed SD {fl(bu['sd_seeds']):.2f}), capped {gap_c:.2f} (seed SD {fl(bc['sd_seeds']):.2f}).\n")

# ---- 2
P("## 2. Swap-neighbourhood refinement (Algorithm 2) under each rule\n")
P("Start = that rule's capped best. Each iteration evaluates all 144 one-battery swap neighbours with the same "
  "3-seed protocol and moves to the best neighbour if its mean loss is strictly lower (best-improvement, DECISIONS D14); "
  "it stops when no neighbour improves. Paper (Sec. 5.2): Candidate 3 → swap 21→12 (1363.04 → 1320.85) → swap 24→3 "
  "(→ 1318.71) → final {3, 7, 12, 18, 19, 23}, 28 assets covered, 3.25% reduction.\n")
if ref_sum:
    for rule in RULES:
        rs = [r for r in ref_sum if r["rule"] == rule]
        if not rs:
            P(f"### Rule: {RULE_LABEL[rule]} — not run\n")
            continue
        r = rs[0]
        P(f"### Rule: {RULE_LABEL[rule]}\n")
        acc = [t for t in ref_trace if t["rule"] == rule and t["accepted"] in ("start", "accepted") or
               (t["rule"] == rule and t["accepted"].startswith("rejected"))]
        P(md_table(["Iteration", "Incumbent", "Swap", "Best neighbour", "Covered", "Mean loss", "SD", "Outcome"],
                   [[t["iteration"], "{" + t["incumbent"].replace("-", ", ") + "}",
                     (f"{t['swap_out']}→{t['swap_in']}" if t["swap_out"] else "—"),
                     "{" + t["candidate"].replace("-", ", ") + "}", t["covered"], f"{fl(t['mean_loss']):.2f}",
                     f"{fl(t['sd_seeds']):.2f}", t["accepted"] or t["note"]] for t in acc]))
        P("")
        P(f"- Final allocation {{{r['final'].replace('-', ', ')}}} covering {r['final_covered']} assets, mean loss {fl(r['final_mean_loss']):.2f} "
          f"(start {fl(r['start_mean_loss']):.2f}, reduction {fl(r['reduction_pct']):.2f}%). "
          f"Reaches the paper's {{3, 7, 12, 18, 19, 23}}: **{'YES' if r['reaches_paper_plan'] == 'True' else 'NO'}**.")
        # runner-up neighbours at each iteration for context
        for it in sorted({int(t["iteration"]) for t in ref_trace if t["rule"] == rule and int(t["iteration"]) > 0}):
            top = sorted((t for t in ref_trace if t["rule"] == rule and int(t["iteration"]) == it), key=lambda t: fl(t["mean_loss"]))[:3]
            P(f"  - iteration {it}, three best neighbours: " + "; ".join(
                f"{t['swap_out']}→{t['swap_in']} = {fl(t['mean_loss']):.2f} ± {fl(t['sd_seeds']):.2f}" for t in top))
        P("")
else:
    P("_Refinement CSVs not present._\n")

# ---- 3
P("## 3. Unvisited-state diagnostic\n")
P("Fractions over the 500 evaluation episodes (4,000 salvos) of each run, averaged over the three seeds. "
  "\"Salvo\" = share of evaluation salvos whose state was never visited in training; \"episode\" = share of evaluation "
  "episodes with at least one such salvo. The defender state is (attacker state, a_t), so it is unvisited whenever the "
  "attacker state is, and sometimes also when it is not.\n")
P(md_table(["Rule", "Env", "Candidate", "Attacker: salvo", "Defender: salvo", "Attacker: episode", "Defender: episode", "Unique attacker states", "Q^A entries", "Q^D entries"],
           [[r["rule"], r["env"], r["candidate"], pct(r["unvisited_A_salvo_frac"]), pct(r["unvisited_D_salvo_frac"]),
             pct(r["unvisited_A_episode_frac"]), pct(r["unvisited_D_episode_frac"]), f"{fl(r['unique_attacker_states']):.0f}",
             f"{fl(r['QA_entries']):.0f}", f"{fl(r['QD_entries']):.0f}"]
            for rule in RULES for env in ("uncapped", "capped")
            for r in sorted((r for r in summary if r["rule"] == rule and r["env"] == env), key=lambda r: int(r["candidate"]))]))
P("")
if ref_sum:
    P("Refinement runs (all neighbours, all rules) show the same pattern; their per-run values are in `refinement_trace.csv` "
      "and the per-seed JSON files under `results/raw/refine__*`.\n")

# ---- 4
P("## 4. Training-length sweep (Candidate 3, capped, aggressive rule)\n")
if trainlen:
    P(md_table(["Training episodes", "Mean loss", "SD (seeds)", "Per seed", "Unvisited attacker (salvo)", "Unvisited defender (salvo)",
                "Unvisited attacker (episode)", "Unvisited defender (episode)", "Unique attacker states", "Q^A entries", "Q^D entries", "Wall s/run"],
               [[f"{int(r['n_train']):,}", f"{fl(r['mean_loss']):.2f}", f"{fl(r['sd_seeds']):.2f}", r["per_seed"],
                 pct(r["unvisited_A_salvo_frac"]), pct(r["unvisited_D_salvo_frac"]), pct(r["unvisited_A_episode_frac"]),
                 pct(r["unvisited_D_episode_frac"]), f"{fl(r['unique_attacker_states']):.0f}", f"{fl(r['QA_entries']):.0f}",
                 f"{fl(r['QD_entries']):.0f}", f"{fl(r['mean_wall_s']):.0f}"] for r in trainlen]))
    P("")
    P(f"![training length]({fig_path.name})\n")
else:
    P("_Training-length CSVs not present._\n")

# ---- 5
P("## 5. Loss decomposition: visited vs unvisited salvos\n")
P("Per evaluation episode, the discounted damage Σ γ^(t−1) D_t is split by whether the salvo's attacker state (and, "
  "separately, defender state) was visited in training; values are per-episode means. Candidate 3, capped, 1,000 training episodes.\n")
rows = []
for rule in RULES:
    rs = sorted((r for r in per_seed if r["rule"] == rule and r["env"] == "capped" and int(r["candidate"]) == 3), key=lambda r: int(r["seed"]))
    for r in rs:
        tot = fl(r["loss"])
        rows.append([rule, r["seed"], f"{tot:.2f}", f"{fl(r['loss_visited_A']):.2f}", f"{fl(r['loss_unvisited_A']):.2f}",
                     f"{100 * fl(r['loss_unvisited_A']) / tot:.1f}%", f"{fl(r['loss_visited_D']):.2f}", f"{fl(r['loss_unvisited_D']):.2f}",
                     f"{100 * fl(r['loss_unvisited_D']) / tot:.1f}%"])
    if rs:
        n = len(rs)
        m = lambda k: sum(fl(r[k]) for r in rs) / n
        rows.append([rule, "mean", f"{m('loss'):.2f}", f"{m('loss_visited_A'):.2f}", f"{m('loss_unvisited_A'):.2f}",
                     f"{100 * m('loss_unvisited_A') / m('loss'):.1f}%", f"{m('loss_visited_D'):.2f}", f"{m('loss_unvisited_D'):.2f}",
                     f"{100 * m('loss_unvisited_D') / m('loss'):.1f}%"])
P(md_table(["Rule", "Seed", "Total loss", "Visited-A salvos", "Unvisited-A salvos", "Share unvisited-A", "Visited-D salvos", "Unvisited-D salvos", "Share unvisited-D"], rows))
P("")
if trainlen:
    P("Same decomposition along the training-length sweep (aggressive rule, means over seeds):\n")
    P(md_table(["Training episodes", "Total", "Unvisited-A salvos", "Share", "Unvisited-D salvos", "Share"],
               [[f"{int(r['n_train']):,}", f"{fl(r['mean_loss']):.2f}", f"{fl(r['loss_unvisited_A']):.2f}",
                 f"{100 * fl(r['loss_unvisited_A']) / fl(r['mean_loss']):.1f}%", f"{fl(r['loss_unvisited_D']):.2f}",
                 f"{100 * fl(r['loss_unvisited_D']) / fl(r['mean_loss']):.1f}%"] for r in trainlen]))
    P("")

# ---- what changes
P("## What changes if the rule changes\n")
conclusions = []
def held(fn):
    return {rule: fn(rule) for rule in RULES}
conclusions.append(("Candidate 0 is the best uncapped allocation (paper Table 1)", held(lambda r: flags[r]["a"])))
conclusions.append(("Candidate 3 is the best capped allocation (paper Table 2)", held(lambda r: flags[r]["b"])))
conclusions.append(("The best allocation differs between the uncapped and capped environments (ranking flip)", held(lambda r: flags[r]["c"])))
conclusions.append(("Candidates 0 and 3 (the two full-coverage plans) are the two best uncapped allocations (paper Table 1)",
                    held(lambda r: set(flags[r]["rank_unc"][:2]) == {0, 3})))
conclusions.append(("Candidate 4 is the capped runner-up (paper Table 2)", held(lambda r: flags[r]["rank_cap"][1] == 4)))
def margin_ok(r):
    rows = sorted((x for x in summary if x["rule"] == r and x["env"] == "capped"), key=lambda x: fl(x["mean_loss"]))
    return fl(rows[1]["mean_loss"]) - fl(rows[0]["mean_loss"]) > fl(rows[0]["sd_seeds"])
conclusions.append(("Capped best beats the runner-up by more than one seed SD (ranking statistically resolvable at 3 seeds)", held(margin_ok)))
conclusions.append(("Loss magnitude within 5% of the paper's best-capped value 1363.04",
                    held(lambda r: abs(fl(best(r, "capped")["mean_loss"]) - 1363.04) / 1363.04 <= 0.05)))
if ref_sum:
    rs = {r["rule"]: r for r in ref_sum}
    conclusions.append(("Swap refinement improves on the capped best (at least one accepted swap)",
                        held(lambda r: r in rs and fl(rs[r]["reduction_pct"]) > 0)))
    conclusions.append(("Refinement reaches the paper's final plan {3, 7, 12, 18, 19, 23}",
                        held(lambda r: r in rs and rs[r]["reaches_paper_plan"] == "True")))
    conclusions.append(("The refined plan covers fewer than 30 assets (coverage alone does not rank allocations)",
                        held(lambda r: r in rs and int(rs[r]["final_covered"]) < 30)))
P(md_table(["Paper conclusion", "aggressive", "random", "lexicographic", "Verdict"],
           [[c, *("held" if v[r] else "NOT held" for r in RULES),
             "robust to the rule" if all(v.values()) else ("fails under every rule" if not any(v.values()) else "depends on the rule")]
            for c, v in conclusions]))
P("")
P("Held under all three rules: " + ("; ".join(c for c, v in conclusions if all(v.values())) or "none") + ".\n")
P("Not held under at least one rule: " + ("; ".join(f"{c} (held only under: {', '.join(r for r in RULES if v[r]) or 'none'})" for c, v in conclusions if not all(v.values())) or "none") + ".\n")

(OUT / "REPORT.md").write_text("\n".join(L))
print(f"wrote {OUT / 'REPORT.md'}" + (f" and {fig_path}" if fig_path else ""))
