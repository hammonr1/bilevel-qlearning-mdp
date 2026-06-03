"""
THE 9 STEPS (executed each stage):

    OUTER LEVEL (Attacker):
      Step 1: Select an attack strategy
      Step 2: Identify reward of this attack (requires inner level)

    INNER LEVEL (Defender):
      Step 3: Select a defend strategy
      Step 4: Calculate optimal IM allocation
      Step 5: Calculate reward for defender (expected saved assets)
      Step 5b: Speculatively compute next state (shared by Steps 6, 8, 9)
      Step 6: Update inner level Q-value (defender anticipates worst-case attacker)

    BACK TO OUTER LEVEL:
      Step 7: Use inner level result as outer level reward
      Step 8: Observe next state (reuse speculative transition from Step 5b)
      Step 9: Update outer level Q-value using Bellman equation
"""

import random
from config import (TAM, TIM, num_stages, num_assets, asset_value,
    SAM_positions, coverage_matrix, initial_asset_status,
    initial_IM_inventory, gamma, epsilon_outer, epsilon_inner,
    P_AM_HIT, P_IM_KILL, ASSET_NODE)

from environment import (
    generate_attack_strategies, generate_defend_strategies,
    calculate_saving_probability, state_to_key, epsilon_greedy
)
from optimization import solve_IM_allocation_model

# ============================================================================
# MAIN TRAINING LOOP - ONE EPISODE
# ============================================================================

def run_one_episode(Q_outer, Q_inner, N_outer, N_inner, gamma, epsilon_outer, epsilon_inner, verbose: bool = True):
    """
    Execute one complete episode through all stages.
    Each stage goes through all 9 steps.

    Key structural fix vs previous version:
      - Step 5b computes the speculative next state BEFORE Step 6.
      - Step 6 uses that look-ahead to update Q_inner with worst-case
        attacker anticipation (defender-optimal Bellman target).
      - Steps 8 and 9 reuse the same speculative transition so all three
        updates are consistent with a single stochastic draw.
    """

    # Initialize episode
    current_state = {
        'asset_status': initial_asset_status.copy(),
        'IM_inventory': dict(initial_IM_inventory)
    }

    TAM_remaining = TAM
    TIM_remaining = TIM
    episode_damage = 0.0

    for stage in range(1, num_stages + 1):

        # Calculate resources for this stage
        stages_left    = num_stages - stage + 1
        TAM_this_stage = TAM_remaining / stages_left
        TIM_this_stage = TIM_remaining / stages_left

        if verbose:
            print(f"\n{'='*60}")
            print(f"STAGE {stage}: TAM={TAM_this_stage}, TIM={TIM_this_stage}")
            print(f"Current State: {current_state}")
            print(f"{'='*60}\n")

        state_key = state_to_key(current_state, stage)

        # ====================================================================
        # OUTER LEVEL (ATTACKER) - PART 1
        # ====================================================================

        # --------------------------------------------------------------------
        # STEP 1: SELECT AN ATTACK STRATEGY
        # --------------------------------------------------------------------
        if verbose: print("STEP 1: Select an attack strategy (outer level)")

        feasible_attacks = generate_attack_strategies(current_state['asset_status'], TAM_this_stage)

        for a in feasible_attacks:
            if (state_key, a) not in Q_outer:
                Q_outer[(state_key, a)] = 0.0
            if (state_key, a) not in N_outer:
                N_outer[(state_key, a)] = 0

        attack_action = epsilon_greedy(Q_outer, state_key, feasible_attacks, epsilon_outer)
        N_outer[(state_key, attack_action)] += 1

        if verbose:
            print(f"   Selected attack: {attack_action}")
            print(f"   (This determines how {TAM_this_stage} AMs are distributed across assets)")


        # --------------------------------------------------------------------
        # STEP 2: IDENTIFY THE REWARD OF THIS ATTACK (resolved after inner level)
        # --------------------------------------------------------------------


        # ====================================================================
        # INNER LEVEL (DEFENDER)
        # ====================================================================

        # Construct inner level state key (includes attack information)
        inner_state_key = (state_key, attack_action)

        # --------------------------------------------------------------------
        # STEP 3: SELECT A DEFEND STRATEGY
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 3: Select a defend strategy (inner level)")

        feasible_defends = generate_defend_strategies(
            inner_state    = current_state['asset_status'],
            TIM_this_stage = TIM_this_stage,
            attack_action  = attack_action,
            IM_inventory   = current_state['IM_inventory'],
            coverage_matrix = coverage_matrix,
            ASSET_NODE     = ASSET_NODE
        )

        for d in feasible_defends:
            if (inner_state_key, d) not in Q_inner:
                Q_inner[(inner_state_key, d)] = 0.0
            if (inner_state_key, d) not in N_inner:
                N_inner[(inner_state_key, d)] = 0

        defend_action = epsilon_greedy(Q_inner, inner_state_key, feasible_defends, epsilon_inner)
        N_inner[(inner_state_key, defend_action)] += 1

        if verbose:
            print(f"   Selected defense: {defend_action}")
            print(f"   (This determines how {TIM_this_stage} IMs are allocated to threatened assets)")


        # --------------------------------------------------------------------
        # STEP 4: CALCULATE OPTIMAL IM ALLOCATION
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 4: Calculate optimal IM allocation from SAM batteries to assets")

        allocation, state_after_alloc, _ = solve_IM_allocation_model(
            current_state,
            attack_action,
            defend_action,
            SAM_positions,
            coverage_matrix,
            verbose=verbose
        )

        if verbose:
            print(f"   IM allocation: {allocation}")
            print(f"   (Maps SAM batteries to threatened assets)")


        # --------------------------------------------------------------------
        # STEP 5: CALCULATE THE REWARD FOR DEFENDER (expected saved assets)
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 5: Calculate the reward for defender (expected saved assets)")

        reward_inner = 0.0
        p_save_per_asset = []
        for i in range(num_assets):
            if current_state['asset_status'][i] == 1:
                ams = attack_action[i]
                ims = defend_action[i]
                p_save = calculate_saving_probability(ams, ims, P_AM_HIT, P_IM_KILL)
                reward_inner += asset_value * p_save
                p_save_per_asset.append(p_save)
            else:
                p_save_per_asset.append(0.0)   # already destroyed

        if verbose:
            print(f"   Expected saved assets (defender reward): {reward_inner:.4f}")


        # --------------------------------------------------------------------
        # STEP 5b: SPECULATIVELY COMPUTE NEXT STATE
        # This single stochastic draw is shared by Steps 6, 8, and 9 so
        # all three Q-updates are consistent with the same transition outcome.
        # --------------------------------------------------------------------
        speculative_asset_status = list(current_state['asset_status'])
        for i in range(num_assets):
            if current_state['asset_status'][i] == 1:
                if random.random() >= p_save_per_asset[i]:   # asset destroyed
                    speculative_asset_status[i] = 0

        speculative_next_state = {
            'asset_status': speculative_asset_status,
            'IM_inventory': state_after_alloc['IM_inventory']
        }
        next_state_key = state_to_key(speculative_next_state, stage + 1)

        # Pre-compute next-stage resources (needed in Steps 6 and 9)
        next_stages_left   = stages_left - 1
        next_TAM_remaining = TAM_remaining - TAM_this_stage
        next_TAM_per_stage = next_TAM_remaining / max(next_stages_left, 1)

        # Next feasible attacks (empty list signals terminal stage)
        if stage < num_stages:
            next_feasible_attacks = generate_attack_strategies(
                speculative_asset_status, next_TAM_per_stage
            )
        else:
            next_feasible_attacks = []


        # --------------------------------------------------------------------
        # STEP 6: UPDATE INNER LEVEL Q-VALUE
        # Defender is the primary optimizer: Q_inner Bellman target uses
        # worst-case (min over attacker) best-response (max over defender)
        # look-ahead so the defender learns to plan against the worst attacker.
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 6: Update inner level Q-value")

        alpha_inner = 1.0 / N_inner[(inner_state_key, defend_action)]

        if stage < num_stages and next_feasible_attacks:
            next_inner_vals = []
            for a in next_feasible_attacks:
                next_inner_key = (next_state_key, a)
                # Best defender response to this next attack
                next_feasible_defs = generate_defend_strategies(
                    inner_state     = speculative_asset_status,
                    TIM_this_stage  = next_TAM_per_stage,
                    attack_action   = a,
                    IM_inventory    = speculative_next_state['IM_inventory'],
                    coverage_matrix = coverage_matrix,
                    ASSET_NODE      = ASSET_NODE
                )
                best_def_q = max(
                    Q_inner.get(((next_inner_key, d)), 0.0)
                    for d in next_feasible_defs
                )
                next_inner_vals.append(best_def_q)

            # Worst-case: attacker picks the move that minimises defender's best Q
            worst_case_next = min(next_inner_vals)
        else:
            worst_case_next = 0.0   # terminal stage

        Q_inner[(inner_state_key, defend_action)] += alpha_inner * (
            reward_inner + gamma * worst_case_next
            - Q_inner[(inner_state_key, defend_action)]
        )

        if verbose:
            print(f"   worst_case_next={worst_case_next:.4f}")
            print(f"   Updated Q_inner[{inner_state_key}, {defend_action}] = "
                  f"{Q_inner[(inner_state_key, defend_action)]:.4f}")


        # ====================================================================
        # BACK TO OUTER LEVEL (ATTACKER) - PART 2
        # ====================================================================

        # --------------------------------------------------------------------
        # STEP 7: USE INNER LEVEL RESULT AS OUTER LEVEL REWARD
        # Attacker reward = total possible value − expected saved value (= damage)
        # --------------------------------------------------------------------
        if verbose:
            print("\n" + "-"*60)
            print("STEP 7: Use inner level result as outer level reward")

        reward_outer = (sum(current_state['asset_status']) * asset_value) - reward_inner

        if verbose:
            print(f"   Outer reward (attacker damage): {reward_outer:.4f}")


        # --------------------------------------------------------------------
        # STEP 8: OBSERVE NEXT STATE
        # Reuse the speculative transition drawn in Step 5b (no new random draw).
        # Count damage from assets that flipped 0 in that draw.
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 8: Observe next state (reusing Step 5b transition)")

        for i in range(num_assets):
            if current_state['asset_status'][i] == 1 and speculative_asset_status[i] == 0:
                episode_damage += asset_value

        next_state = speculative_next_state   # already computed above

        if verbose:
            print(f"   Next state: assets={next_state['asset_status']}  "
                  f"IMs={next_state['IM_inventory']}")


        # --------------------------------------------------------------------
        # STEP 9: UPDATE OUTER LEVEL Q-VALUE USING BELLMAN EQUATION
        # Reuse next_feasible_attacks computed in Step 5b.
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 9: Update outer level Q-value using Bellman equation")

        alpha_outer = 1.0 / N_outer[(state_key, attack_action)]

        if stage < num_stages and next_feasible_attacks:
            max_Q_next_outer = max(
                Q_outer.get((next_state_key, a), 0.0) for a in next_feasible_attacks
            )
        else:
            max_Q_next_outer = 0.0   # terminal stage

        Q_outer[(state_key, attack_action)] += alpha_outer * (
            reward_outer
            + gamma * max_Q_next_outer
            - Q_outer[(state_key, attack_action)]
        )

        if verbose:
            print(f"   Q_outer updated → {Q_outer[(state_key, attack_action)]:.4f}")


        # ====================================================================
        # PREPARE FOR NEXT STAGE
        # ====================================================================
        if verbose:
            print(f"\n{'='*60}")
            print(f"END OF STAGE {stage}")
            print(f"{'='*60}\n")

        current_state  = next_state
        TAM_remaining -= TAM_this_stage
        TIM_remaining -= TIM_this_stage

    # End of episode
    if verbose:
        print("\nEPISODE COMPLETE")
        print(f"All {num_stages} stages executed")
        print(f"Both Q_outer and Q_inner have been updated throughout")

    return episode_damage


# ============================================================================
# COMPLETE TRAINING LOOP
# ============================================================================

def train_bilevel_qlearning(num_episodes, verbose_every: int = 100):
    """
    Train the bi-level Q-learning system.
    Each episode runs through all stages; each stage executes all 9 steps.

    Q-tables and visit counts are initialized here (not globally) so there
    is exactly one set of tables in use at a time.
    """

    print("="*70)
    print("BI-LEVEL Q-LEARNING TRAINING")
    print("="*70)

    Q_outer = {}   # Q_outer[(state_key, attack_action)]      = expected attacker reward
    Q_inner = {}   # Q_inner[(inner_state_key, defend_action)] = expected defender reward
    N_outer = {}   # Visit counts for outer-level learning rate
    N_inner = {}   # Visit counts for inner-level learning rate

    damage_history = []

    epsilon_outer = 1.0
    epsilon_inner = 1.0
    epsilon_decay = 0.995
    epsilon_min   = 0.01

    for episode in range(1, num_episodes + 1):
        epsilon_outer = max(epsilon_min, epsilon_outer * epsilon_decay)
        epsilon_inner = max(epsilon_min, epsilon_inner * epsilon_decay)

        verbose = (episode == 1 or episode % verbose_every == 0)
        if verbose:
            print(f"\n{'#'*70}")
            print(f"EPISODE {episode} / {num_episodes}")
            print(f"{'#'*70}")

        ep_damage = run_one_episode(
            Q_outer, Q_inner, N_outer, N_inner,
            gamma, epsilon_outer, epsilon_inner,
            verbose=verbose
        )
        damage_history.append(ep_damage)

        if verbose:
            avg = sum(damage_history[-50:]) / len(damage_history[-50:])
            print(f"\n  → Episode damage: {ep_damage:.1f}  |  "
                  f"Last-50 avg: {avg:.2f}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    return Q_outer, Q_inner, damage_history