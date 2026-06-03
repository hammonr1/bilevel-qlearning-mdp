import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import ast

df = pd.read_csv("q_matrices_by_stage.csv")

df['asset_status'] = df['asset_status'].apply(ast.literal_eval)
df['attack'] = df['attack'].apply(ast.literal_eval)
df['defense'] = df['defense'].apply(lambda x: ast.literal_eval(x) if x != '-' else None)

def status_label(s):
    names = {(1,1): 'both intact', (1,0): 'asset 0 only', (0,1): 'asset 1 only', (0,0): 'both destroyed'}
    return names.get(tuple(s), str(s))

def action_label(a):
    return f"({a[0]},{a[1]})"

num_stages = df['stage'].max()
teal_cmap = LinearSegmentedColormap.from_list(
    'teal_custom', ['#f0f9f6', '#9FE1CB', '#1D9E75', '#085041'], N=256
)
coral_cmap = LinearSegmentedColormap.from_list(
    'coral_custom', ['#fdf3f0', '#F5C4B3', '#D85A30', '#712B13'], N=256
)

fig = plt.figure(figsize=(20, 6 * num_stages), constrained_layout=True)
fig.suptitle("Bi-level Q-learning: Q-value matrices by stage", fontsize=16, fontweight='500', y=1.01)

outer = gridspec.GridSpec(num_stages, 2, figure=fig, hspace=0.5, wspace=0.35)

for stage in range(1, num_stages + 1):

    # ── ATTACKER ──────────────────────────────────────────────────────────────
    att = df[(df['stage'] == stage) & (df['level'] == 'attacker')].copy()
    att['state'] = att['asset_status'].apply(status_label)
    att['action'] = att['attack'].apply(action_label)
    att_pivot = att.groupby(['state', 'action'])['q_value'].max().unstack(fill_value=0)

    ax_att = fig.add_subplot(outer[stage - 1, 0])
    sns.heatmap(
        att_pivot,
        ax=ax_att,
        cmap=coral_cmap,
        annot=True,
        fmt='.1f',
        linewidths=0.4,
        linecolor='#f0ece8',
        cbar_kws={'shrink': 0.8, 'label': 'Q-value'},
        annot_kws={'size': 9},
    )
    ax_att.set_title(f"Stage {stage} — attacker Q-matrix", fontsize=12, fontweight='500', pad=8)
    ax_att.set_xlabel("Attack action (AM asset0, AM asset1)", fontsize=9)
    ax_att.set_ylabel("Asset status", fontsize=9)
    ax_att.tick_params(axis='x', labelsize=8, rotation=30)
    ax_att.tick_params(axis='y', labelsize=8, rotation=0)

    # ── DEFENDER ──────────────────────────────────────────────────────────────
    dfd = df[(df['stage'] == stage) & (df['level'] == 'defender')].copy()
    dfd['state'] = dfd['asset_status'].apply(status_label)
    dfd['attack_lbl'] = dfd['attack'].apply(action_label)
    dfd['defense_lbl'] = dfd['defense'].apply(lambda x: action_label(x) if x is not None else '(0,0)')
    dfd['row'] = dfd['state'] + '\natk=' + dfd['attack_lbl']
    dfd_pivot = dfd.groupby(['row', 'defense_lbl'])['q_value'].max().unstack(fill_value=0)

    ax_dfd = fig.add_subplot(outer[stage - 1, 1])
    sns.heatmap(
        dfd_pivot,
        ax=ax_dfd,
        cmap=teal_cmap,
        annot=True,
        fmt='.1f',
        linewidths=0.4,
        linecolor='#f0f5f0',
        cbar_kws={'shrink': 0.8, 'label': 'Q-value'},
        annot_kws={'size': 8},
    )
    ax_dfd.set_title(f"Stage {stage} — defender Q-matrix", fontsize=12, fontweight='500', pad=8)
    ax_dfd.set_xlabel("Defense action (IM asset0, IM asset1)", fontsize=9)
    ax_dfd.set_ylabel("Asset status + attack", fontsize=9)
    ax_dfd.tick_params(axis='x', labelsize=8, rotation=30)
    ax_dfd.tick_params(axis='y', labelsize=7, rotation=0)

plt.savefig("q_value_heatmaps.pdf", bbox_inches='tight', dpi=150)
plt.savefig("q_value_heatmaps.png", bbox_inches='tight', dpi=150)
print("Saved q_value_heatmaps.pdf and q_value_heatmaps.png")