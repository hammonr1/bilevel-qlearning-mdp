import random
from config import (TAM, TIM, num_stages, num_assets, asset_value,
                    SAM_positions, coverage_matrix, initial_asset_status,
                    initial_IM_inventory, gamma, epsilon_outer, epsilon_inner,
                    P_AM_HIT, P_IM_KILL, ASSET_NODE,
                    PRESET_ATTACK_STRATEGIES, PRESET_DEFENSE_STRATEGIES)

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

def generate_defend_strategies(inner_state, TIM_this_stage, attack_action, 
                                IM_inventory, coverage_matrix, ASSET_NODE):
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

    # Pre-compute reachable supply per asset given current inventory and coverage.
    # Derived from Model (5-10) constraints (6) and (7):
    #   Constraint (6): v[n,m] <= M * beta[n,m]  — IMs can only flow along covered edges
    #   Constraint (7): sum_m v[n,m] <= x_n^t    — supply limited by node inventory
    # Together these bound the max deliverable IMs to asset j as:
    #   reachable[j] = sum_{i: beta[i][j]=1} x_i^t
    node_indices = list(IM_inventory.keys())
    reachable = {}
    for j in range(n_assets):
        reachable[j] = sum(
            IM_inventory[i] for i in node_indices
            if coverage_matrix.get(i, {}).get(ASSET_NODE[j], 0) == 1
        )

    print(f"DEBUG reachable: {reachable}")
    print(f"DEBUG IM_inventory: {IM_inventory}")
    print(f"DEBUG attack_action: {attack_action}")
    print(f"DEBUG TIM_this_stage: {TIM_this_stage}")

    for defense_strategy in PRESET_DEFENSE_STRATEGIES:
        is_valid = True
        total_IMs = 0

        for i in range(n_assets):
            # VALIDITY CHECK 1: Can only defend assets that are being attacked.
            # From inner MDP feasibility condition: z_n^t = 0 if y_n^t = 0
            # (Section 3.1.2: "IMs can only be assigned to assets threatened by AMs")
            if defense_strategy[i] > 0 and attack_action[i] == 0:
                is_valid = False
                break

            # VALIDITY CHECK 2: Don't over-defend (defense <= attack).
            # From Proposition 2: when p̃ = 1, number of IMs should be <= number of AMs.
            # We apply this as a general upper bound to avoid wasteful allocations.
            if defense_strategy[i] > attack_action[i]:
                is_valid = False
                break

            # VALIDITY CHECK 3: IMs requested cannot exceed physically reachable supply.
            # This is a precondition for Model (5-10) constraint (8) to be feasible:
            #   Constraint (8): sum_i v[i,j] >= z_j
            # If reachable[j] < z_j, constraint (8) can never be satisfied and the LP
            # will be infeasible. We filter these strategies out here rather than
            # letting the LP discover the contradiction after the fact.
            if defense_strategy[i] > reachable[i]:
                is_valid = False
                break

            total_IMs += defense_strategy[i]

        # VALIDITY CHECK 4: Total IMs used cannot exceed budget for this stage.
        # From problem description: N_D = (alpha * B) / |T|
        # The defender distributes at most N_D IMs per salvo across all assets.
        if is_valid and total_IMs <= TIM_this_stage:
            feasible_strategies.append(defense_strategy)

    # If no feasible strategies remain, fall back to "no defense"
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

def state_to_key(state: dict, stage: int = None):
    """
    Convert mutable state dict to a hashable key for Q-tables.
    Stage is included so Q-tables are naturally partitioned per stage,
    enabling extraction of 5 separate attacker/defender matrices after training.
    """
    status = tuple(state['asset_status'])
    inventory = tuple(
        int(round(state['IM_inventory'][n])) 
        for n in sorted(state['IM_inventory'])
    )
    if stage is not None:
        return (stage, status, inventory)
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