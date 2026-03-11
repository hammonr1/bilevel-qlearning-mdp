from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value
from config import asset_value, coverage_matrix, ASSET_NODE

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

    ## Survival probability for each SAM node m (based on what's being defended at that node)
    # Nodes not under attack survive with probability 1
    # Needed because prob_save is keyed by asset index, but objective needs node keys
    prob_save_node = {}
    for m in node_indices:
        asset_at_node = [j for j in asset_indices if ASSET_NODE[j] == m]
        if asset_at_node and y[asset_at_node[0]] > 0:
            prob_save_node[m] = prob_save[asset_at_node[0]]
        else:
            prob_save_node[m] = 1.0

    # Objective: maximize expected future coverage using REMAINING IMs after allocation
    # From Model (5): max sum_n sum_m x[m] * prob_save[m] * prob_save[n] * beta[m][n] * theta[n]
    # x[m] = remaining IMs at node m after this stage (defined in constraint 4)
    # This naturally discourages over-allocation — sending more IMs reduces x[m]
    problem += lpSum(
        x[m] * prob_save_node[m] * prob_save[j] *
        coverage_matrix.get(m, {}).get(ASSET_NODE[j], 0) * s[j]
        for m in node_indices
        for j in asset_indices
    )

    # 1. Coverage constraints: Can only send IMs to assets within coverage
    for i in node_indices:
        for j in asset_indices:
            beta_ij = coverage_matrix.get(i, {}).get(ASSET_NODE[j], 0)
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

    # status = LpStatus[problem.status]
    # if status != 'Optimal':
    #     print(f"\n⚠️  INFEASIBLE LP DETECTED")
    #     print(f"   attack_action: {attack_action}")
    #     print(f"   defend_action: {defend_action}")
    #     print(f"   asset_status:  {current_state['asset_status']}")
    #     print(f"   IM_inventory:  {current_state['IM_inventory']}")
    #     print(f"   z (IMs requested per asset): {z}")
    #     print(f"   y (AMs per asset):           {y}")
    #     print(f"   Supply available: { {i: x_current[i] for i in node_indices} }")
    #     print(f"   Demand required:  { {j: z[j] for j in asset_indices} }")
    #     # Check if demand can physically be met given coverage
    #     for j in asset_indices:
    #         reachable = sum(x_current[i] for i in node_indices 
    #                     if coverage_matrix.get(i, {}).get(ASSET_NODE[j], 0) == 1)
    #         print(f"   Asset {j}: needs {z[j]} IMs, reachable supply = {reachable}")
        
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

