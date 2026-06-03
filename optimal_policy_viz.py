import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import ast
import os

os.makedirs("outputs", exist_ok=True)

# ── Load and derive optimal policies from Q-matrices ─────────────────────────
df = pd.read_csv("outputs/q_matrices_by_stage.csv")
df['asset_status'] = df['asset_status'].apply(ast.literal_eval)
df['attack'] = df['attack'].apply(ast.literal_eval)
df['defense'] = df['defense'].apply(lambda x: ast.literal_eval(x) if x != '-' else None)
df['q_value'] = df['q_value'].astype(float)

def status_label(s):
    return {(1,1):'both intact', (1,0):'asset 0 only', (0,1):'asset 1 only', (0,0):'both destroyed'}.get(tuple(s), str(s))

def action_str(a):
    return f"({a[0]},{a[1]})" if a is not None else "(0,0)"

num_stages = df['stage'].max()

# Attacker optimal: argmax_action Q for each (stage, status)
att_df = df[df['level'] == 'attacker'].copy()
att_df['status_lbl'] = att_df['asset_status'].apply(status_label)
att_opt = (att_df.sort_values('q_value', ascending=False)
               .groupby(['stage', 'status_lbl'])
               .first()
               .reset_index()[['stage', 'status_lbl', 'attack', 'q_value']])
att_opt['action_lbl'] = att_opt['attack'].apply(action_str)

# Defender optimal: argmax_action Q for each (stage, status, attack)
def_df = df[df['level'] == 'defender'].copy()
def_df['status_lbl'] = def_df['asset_status'].apply(status_label)
def_df['attack_lbl'] = def_df['attack'].apply(action_str)
def_opt = (def_df.sort_values('q_value', ascending=False)
               .groupby(['stage', 'status_lbl', 'attack_lbl'])
               .first()
               .reset_index()[['stage', 'status_lbl', 'attack_lbl', 'defense', 'q_value']])
def_opt['action_lbl'] = def_opt['defense'].apply(action_str)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Optimal policies learned by bi-level Q-learning", fontsize=15, fontweight='500', y=1.02)

STATUS_ORDER = ['both intact', 'asset 0 only', 'asset 1 only', 'both destroyed']
stages = list(range(1, num_stages + 1))

coral_cmap = plt.cm.get_cmap('YlOrBr')
teal_cmap  = plt.cm.get_cmap('YlGn')

# ── LEFT: Attacker optimal policy ────────────────────────────────────────────
ax = axes[0]
att_pivot_action = att_opt.pivot(index='status_lbl', columns='stage', values='action_lbl').reindex(STATUS_ORDER)
att_pivot_q      = att_opt.pivot(index='status_lbl', columns='stage', values='q_value').reindex(STATUS_ORDER)

qmin, qmax = att_pivot_q.min().min(), att_pivot_q.max().max()
norm_att = Normalize(vmin=qmin, vmax=qmax)

for r, status in enumerate(STATUS_ORDER):
    for c, stage in enumerate(stages):
        try:
            action = att_pivot_action.loc[status, stage]
            q      = att_pivot_q.loc[status, stage]
        except KeyError:
            action, q = '—', 0.0

        if pd.isna(action) or action is None:
            action, q = '—', 0.0

        color = coral_cmap(norm_att(q)) if action != '—' else '#f5f5f5'
        rect = mpatches.FancyBboxPatch(
            (c + 0.05, r + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='white', linewidth=1.5,
            transform=ax.transData
        )
        ax.add_patch(rect)

        text_color = '#4A1B0C' if norm_att(q) > 0.3 else '#888780'
        ax.text(c + 0.5, r + 0.58, action,
                ha='center', va='center', fontsize=11, fontweight='500', color=text_color)
        ax.text(c + 0.5, r + 0.28, f"Q={q:.1f}",
                ha='center', va='center', fontsize=8, color=text_color)

ax.set_xlim(0, num_stages)
ax.set_ylim(0, len(STATUS_ORDER))
ax.set_xticks([s - 0.5 for s in stages])
ax.set_xticklabels([f"Stage {s}" for s in stages], fontsize=10)
ax.set_yticks([i + 0.5 for i in range(len(STATUS_ORDER))])
ax.set_yticklabels(STATUS_ORDER, fontsize=10)
ax.set_title("Attacker optimal policy\n(AMs on asset 0, AMs on asset 1)", fontsize=11, pad=10)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

sm_att = ScalarMappable(cmap=coral_cmap, norm=norm_att)
sm_att.set_array([])
plt.colorbar(sm_att, ax=ax, shrink=0.6, label='Q-value', pad=0.02)

# ── RIGHT: Defender optimal policy (best defense given best attack) ───────────
ax = axes[1]

# For each (stage, status) find the defender's best response to the attacker's best action
merged = att_opt[['stage', 'status_lbl', 'action_lbl']].rename(columns={'action_lbl': 'attack_lbl'})
def_best = def_opt.merge(merged, on=['stage', 'status_lbl', 'attack_lbl'], how='inner')

def_pivot_action = def_best.pivot(index='status_lbl', columns='stage', values='action_lbl').reindex(STATUS_ORDER)
def_pivot_q      = def_best.pivot(index='status_lbl', columns='stage', values='q_value').reindex(STATUS_ORDER)
def_pivot_atk    = def_best.pivot(index='status_lbl', columns='stage', values='attack_lbl').reindex(STATUS_ORDER)

qmin2, qmax2 = def_pivot_q.min().min(), def_pivot_q.max().max()
norm_def = Normalize(vmin=qmin2, vmax=qmax2)

for r, status in enumerate(STATUS_ORDER):
    for c, stage in enumerate(stages):
        try:
            defense = def_pivot_action.loc[status, stage]
            q       = def_pivot_q.loc[status, stage]
            attack  = def_pivot_atk.loc[status, stage]
        except KeyError:
            defense, q, attack = '—', 0.0, '—'

        if pd.isna(defense) or defense is None:
            defense, q, attack = '—', 0.0, '—'

        color = teal_cmap(norm_def(q)) if defense != '—' else '#f5f5f5'
        rect = mpatches.FancyBboxPatch(
            (c + 0.05, r + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='white', linewidth=1.5,
            transform=ax.transData
        )
        ax.add_patch(rect)

        text_color = '#04342C' if norm_def(q) > 0.3 else '#888780'
        ax.text(c + 0.5, r + 0.65, defense,
                ha='center', va='center', fontsize=11, fontweight='500', color=text_color)
        ax.text(c + 0.5, r + 0.42, f"Q={q:.1f}",
                ha='center', va='center', fontsize=8, color=text_color)
        ax.text(c + 0.5, r + 0.2, f"atk={attack}",
                ha='center', va='center', fontsize=7,
                color='#888780' if text_color == '#04342C' else '#aaaaaa')

ax.set_xlim(0, num_stages)
ax.set_ylim(0, len(STATUS_ORDER))
ax.set_xticks([s - 0.5 for s in stages])
ax.set_xticklabels([f"Stage {s}" for s in stages], fontsize=10)
ax.set_yticks([i + 0.5 for i in range(len(STATUS_ORDER))])
ax.set_yticklabels(STATUS_ORDER, fontsize=10)
ax.set_title("Defender optimal policy\n(IMs on asset 0, IMs on asset 1) | vs. best attacker", fontsize=11, pad=10)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

sm_def = ScalarMappable(cmap=teal_cmap, norm=norm_def)
sm_def.set_array([])
plt.colorbar(sm_def, ax=ax, shrink=0.6, label='Q-value', pad=0.02)

plt.tight_layout()
plt.savefig("outputs/optimal_policy_viz.pdf", bbox_inches='tight', dpi=150)
plt.savefig("outputs/optimal_policy_viz.png", bbox_inches='tight', dpi=150)
print("Saved outputs/optimal_policy_viz.pdf and .png")