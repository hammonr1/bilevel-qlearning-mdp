from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value
from config import asset_value, coverage_matrix

def solve_IM_allocation_model(current_state, attack_action, defend_action,
                               SAM_positions, coverage_matrix,
                               p=0.85, p_tilde=0.8, epsilon=0.01, M=100,
                               verbose=False):
    """
    Helper for STEP 4: Solve optimization model to allocate IMs optimally

    This is the "Inner MDP" optimization problem that determines:
    - w_ij: How many IMs to send from SAM at location i to defend asset j

    Args:
        current_state: Dict with 'asset_status' and 'IM_inventory'
                      e.g., {'asset_status': [1, 1], 'IM_inventory': {1: 8, 2: 0, 3: 0, 4: 8, 5: 0}}
        attack_action: Tuple of AMs allocated to each asset, e.g., (5, 2)
        defend_action: Tuple of IMs allocated to each asset, e.g., (1, 1)
        SAM_positions: List of node indices with SAM batteries, e.g., [1, 4]
        coverage_matrix: Dict beta[sam_node][asset_node] = 1 if can cover, 0 otherwise
        p: Probability of successful interception (default: 0.85)
        p_tilde: Alternative probability parameter β̃ (default: 0.8)
        epsilon: Small value to avoid division by zero (default: 0.01)
        M: Big-M constant for coverage constraints (default: 100)
        verbose: If True, print detailed results (default: False)

    Returns:
        allocation: Dict {(sam_node, asset_node): num_IMs}
                   e.g., {(1, 0): 1, (1, 1): 0, (4, 0): 0, (4, 1): 1}
        next_state: Updated state after IM allocation
        objective_value: The optimal objective function value
    """
    ## SET-UP
    num_assets = len(current_state['asset_status'])
    asset_indices = list(range(num_assets))

    # Create node indices from SAM positions
    all_nodes = list(current_state['IM_inventory'].keys())
    node_indices = all_nodes

    # Get current IM inventory at each SAM location
    x_current = current_state['IM_inventory']  # Dict {node: num_IMs}

    # Asset value: uses global asset_value (8 = intact, 0 = destroyed)
    s = {idx: asset_value if current_state['asset_status'][idx] == 1 else 0
         for idx in asset_indices}

    # Map defend_action and attack_action to dictionaries
    z = {asset_idx: defend_action[asset_idx] for asset_idx in asset_indices}
    y = {asset_idx: attack_action[asset_idx] for asset_idx in asset_indices}

    if all(y[m] == 0 for m in asset_indices):
    # No attack — skip LP entirely, no IMs needed
      return {(i,j): 0 for i in node_indices for j in asset_indices}, current_state, 0.0

    ## OPTIMIZATION MODEL
    problem = LpProblem(name="IM_Allocation", sense=LpMaximize)

    indices = [(i, j) for i in node_indices for j in asset_indices]
    v = LpVariable.dicts("v", indices, lowBound=0, cat='Continuous')
    x = LpVariable.dicts("x", node_indices, lowBound=0, cat='Continuous')

    # Pre-compute survival probability for each asset (these are just numbers, not variables)
    prob_save = {}
    for m in asset_indices:
        if y[m] > 0:
            ratio = z[m] / (y[m] + epsilon)
            prob_save[m] = (1 - p * (1 - p_tilde)**ratio)**y[m]
        else:
            prob_save[m] = 1.0  # no attack = asset survives

    # Objective: maximize expected saved asset value (now fully linear in v)
    problem += lpSum(
        v[(i, j)] * prob_save[j] * s[j]
        for i in node_indices
        for j in asset_indices
        if coverage_matrix.get(i, {}).get(j, 0) == 1 and y[j] > 0
)

    # 1. Coverage constraints: Can only send IMs to assets within coverage
    for i in node_indices:
        for j in asset_indices:
            beta_ij = coverage_matrix.get(i, {}).get(j, 0)
            problem += v[(i, j)] <= M * beta_ij

    # 2. Row sum constraints (supply): Total IMs sent from node i cannot exceed inventory
    for i in node_indices:
        problem += lpSum([v[(i, j)] for j in asset_indices]) <= x_current[i]

    # 3. Column sum constraints (demand): Total IMs received at asset j must meet defense strategy
    for j in asset_indices:
        problem += lpSum([v[(i, j)] for i in node_indices]) >= z[j]

    # 4. Slack variable definitions: Remaining IMs after allocation
    for i in node_indices:
        problem += x[i] == x_current[i] - lpSum([v[(i, j)] for j in asset_indices])

    ## SOLVE
    problem.solve()

    if verbose:
        print(f"\nOptimization Status: {LpStatus[problem.status]}")
        print(f"Objective Value: {problem.objective.value()}")

    ## EXTRACT RESULTS
    allocation = {}
    for i in node_indices:
        for j in asset_indices:
            allocation[(i, j)] = v[(i, j)].varValue if v[(i, j)].varValue else 0

    # Update IM inventory for next state
    next_IM_inventory = {}
    for n in node_indices:
        next_IM_inventory[n] = x[n].varValue if x[n].varValue else 0

    # Next state keeps same asset status (actual damage happens in Step 8)
    next_state = {
        'asset_status': current_state['asset_status'].copy(),
        'IM_inventory': next_IM_inventory
    }

    objective_value = problem.objective.value() if problem.objective.value() else 0

    ## RETURN RESULTS  ← CRITICAL!
    return allocation, next_state, objective_value

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

