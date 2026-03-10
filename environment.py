import random
from config import (
    num_assets, asset_value,
    P_AM_HIT, P_IM_KILL,
    PRESET_ATTACK_STRATEGIES, PRESET_DEFENSE_STRATEGIES
)

def generate_attack_strategies(state, TAM_this_stage):
    """
    Helper for STEP 1: Generate feasible attack strategies

    Args:
        state: Current state of assets, e.g., [1, 1] where 1=intact, 0=destroyed
        TAM_this_stage: Total Attack Modules available (used as constraint check)

    Returns: List of attack actions
    Each attack action is a distribution of TAM_this_stage across assets
    """
    n_assets = len(state)
    feasible_strategies = []

    for attack_strategy in PRESET_ATTACK_STRATEGIES:
        # Check if strategy is valid given current state
        is_valid = True
        total_AMs = 0

        for i in range(n_assets):
            # Can only attack intact assets (state[i] == 1)
            if attack_strategy[i] > 0 and state[i] == 0:
                is_valid = False
                break
            total_AMs += attack_strategy[i]

        # Check if total AMs used doesn't exceed TAM available
        if is_valid and total_AMs <= TAM_this_stage:
            feasible_strategies.append(attack_strategy)

    return feasible_strategies


def generate_defend_strategies(inner_state, TIM_this_stage, attack_action):
    """
    Helper for STEP 3: Generate feasible defend strategies

    Returns: List of defend actions
    Each defend action is a distribution of TIM_this_stage across threatened assets
    """
    n_assets = len(inner_state)
    feasible_strategies = []

    # Identify which assets are being attacked
    threatened_indices = [i for i in range(n_assets) if attack_action[i] > 0]

    # If no attack, return "no defense"
    if not threatened_indices:
        return [tuple([0] * n_assets)]

    for defense_strategy in PRESET_DEFENSE_STRATEGIES:
        is_valid = True
        total_IMs = 0

        for i in range(n_assets):
            # Check various validity conditions:

            # 1. Can only defend assets that are being attacked
            if defense_strategy[i] > 0 and attack_action[i] == 0:
                is_valid = False
                break

            # 2. Optional: Don't over-defend (defense <= attack)
            # Comment this out if your model allows over-defense
            if defense_strategy[i] > attack_action[i]:
                is_valid = False
                break

            total_IMs += defense_strategy[i]

        # 3. Check if total IMs used doesn't exceed TIM available
        if is_valid and total_IMs <= TIM_this_stage:
            feasible_strategies.append(defense_strategy)

    # If no feasible strategies, at least return "no defense"
    if not feasible_strategies:
        feasible_strategies.append(tuple([0] * n_assets))

    return feasible_strategies

def epsilon_greedy(Q_table, state_key, feasible_actions, epsilon):
    """
    Helper for STEP 1 and STEP 3: Epsilon-greedy action selection
    """
    if random.random() < epsilon or not feasible_actions:
        return random.choice(feasible_actions)

    # Find action with highest Q-value (default 0.0 for unseen state-action pairs)
    best_action = feasible_actions[0]
    best_q      = Q_table.get((state_key, best_action), 0.0)
    for action in feasible_actions[1:]:
        q = Q_table.get((state_key, action), 0.0)
        if q > best_q:
            best_q      = q
            best_action = action
    return best_action

def state_to_key(state:dict):
    """
    Convert mutable state dict to a hashable key for Q-tables.
    """
    status = tuple(state['asset_status'])
    inventory = tuple(state['IM_inventory'][n] for n in sorted(state['IM_inventory']))
    return (status, inventory)

def calculate_saving_probability(num_AMs, num_IMs, p, p_tilde, epsilon: float = 1e-6):
    """
    Helper for STEP 5: Calculate probability asset survives

    From notes: Uses formula involving (1-P(Loss|s))^(z_i)
    From paper: P_save ≈ (1 - (1-p_d)^(num_IMs/num_AMs) * p_s)^num_AMs
    """
    if num_AMs == 0:
        return 1.0
    if num_IMs == 0:
        return (1.0 - p) ** num_AMs

    ratio   = num_IMs / (num_AMs + epsilon)
    p_save  = (1.0 - p * (1.0 - p_tilde) ** ratio) ** num_AMs
    return max(0.0, min(1.0, p_save))   # clamp to [0, 1]