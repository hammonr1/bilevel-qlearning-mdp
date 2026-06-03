import numpy as np
import random

# ============================================================================
# PROBLEM PARAMETERS  (
# ============================================================================

TAM        = 20   # Total Attacking Missiles  (TAN=20 from notes)
TIM        = 15   # Total Intercepting Missiles (TIM=15 from notes; TIM/4 = 3.75 per stage)
num_stages = 5    # Number of sequential salvos (T=4 from Colab)
TAM_per_stage = TAM / num_stages   # 5 AMs per stage
TIM_per_stage = TIM / num_stages   # ~3.75 IMs per stage

# Network structure
num_nodes     = 5
all_nodes     = list(range(1, num_nodes + 1))  # [1, 2, 3, 4, 5]
SAM_positions = [1, 4]  # SAM batteries placed at nodes 1 and 4 (D=2 batteries)

coverage_matrix = {
    1: {1: 1, 2: 1, 3: 1, 4: 0, 5: 0},  # Node 1 covers {1,2,3}
    2: {1: 1, 2: 1, 3: 1, 4: 1, 5: 0},  # Node 2 covers {1,2,3,4}
    3: {1: 1, 2: 1, 3: 1, 4: 1, 5: 0},  # Node 3 covers {1,2,3,4}
    4: {1: 0, 2: 1, 3: 1, 4: 1, 5: 1},  # Node 4 covers {2,3,4,5}
    5: {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}   # Node 5 covers {4,5}
}

# Asset parameters
# Maps asset index (0-based) → network node number
# To add/remove assets, only edit ASSET_NODE
ASSET_NODE = {
    0: 3,   # asset 0 lives at node 3
    1: 4,   # asset 1 lives at node 4
}

num_assets    = len(ASSET_NODE)           # derived automatically
asset_indices = list(ASSET_NODE.keys())   # [0, 1]
asset_value   = 8                         # value of each intact asset

assert num_assets == len(ASSET_NODE), "num_assets must match ASSET_NODE entries"

# Probability parameters (from Colab: P=0.85, P_tilde=0.80)
P_AM_HIT  = 0.85   # p  — probability AM destroys asset if not intercepted
P_IM_KILL = 0.80   # p̃  — probability IM destroys an AM

# Initial state
initial_asset_status = [1, 1]   # Both assets intact at start
initial_IM_inventory = {
    1: 8,   # alpha=8 IMs loaded at SAM node 1
    2: 0,
    3: 0,
    4: 8,   # alpha=8 IMs loaded at SAM node 4
    5: 0
}
initial_state = {
    'asset_status': initial_asset_status,
    'IM_inventory': initial_IM_inventory
}

# Learning parameters — defined once here, passed into functions as arguments
# (Q-tables are created inside train_bilevel_qlearning, not here)
gamma         = 0.9    # Discount factor
epsilon_outer = 1.0    # Exploration rate for attacker
epsilon_inner = 1.0    # Exploration rate for defender

epsilon_decay = 0.995
epsilon_min   = 0.01

PRESET_ATTACK_STRATEGIES = [
    (0, 0),  # No attack
    (2, 0),  # 2 AMs on asset 0, 0 AMs on asset 1
    (5, 0),  # 5 AMs on asset 0, 0 AMs on asset 1
    (0, 2),  # 0 AMs on asset 0, 2 AMs on asset 1
    (2, 2),  # 2 AMs on asset 0, 2 AMs on asset 1
    (5, 2),  # 5 AMs on asset 0, 2 AMs on asset 1
    (0, 5),  # 0 AMs on asset 0, 5 AMs on asset 1
    (2, 5),  # 2 AMs on asset 0, 5 AMs on asset 1
    (5, 5),  # 5 AMs on asset 0, 5 AMs on asset 1
]

PRESET_DEFENSE_STRATEGIES = [
    (0, 0),  # No defense
    (1, 0),  # 1 IM on asset 0, 0 IMs on asset 1
    (3, 0),  # 3 IMs on asset 0, 0 IMs on asset 1
    (0, 1),  # 0 IMs on asset 0, 1 IM on asset 1
    (1, 1),  # 1 IM on asset 0, 1 IM on asset 1
    (3, 1),  # 3 IMs on asset 0, 1 IM on asset 1
    (0, 3),  # 0 IMs on asset 0, 3 IMs on asset 1
    (1, 3),  # 1 IM on asset 0, 3 IMs on asset 1
    (3, 3),  # 3 IMs on asset 0, 3 IMs on asset 1
]
