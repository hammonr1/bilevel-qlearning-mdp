"""
THE 9 STEPS (executed each stage):

    OUTER LEVEL (Attacker):
      Step 1: Select an attack strategy
      Step 2: Identify reward of this attack (requires inner level)

    INNER LEVEL (Defender):
      Step 3: Select a defend strategy
      Step 4: Calculate optimal IM allocation
      Step 5: Calculate reward for defender (expected saved assets)
      Step 6: Update inner level Q-value

    BACK TO OUTER LEVEL:
      Step 7: Use inner level result as outer level reward
      Step 8: Observe next state (stochastic transition)
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
    Execute one complete episode through all stages
    Each stage goes through all 9 steps
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
        stages_left      = num_stages - stage + 1
        TAM_this_stage   = TAM_remaining / stages_left
        TIM_this_stage   = TIM_remaining / stages_left

        if verbose:
            print(f"\n{'='*60}")
            print(f"STAGE {stage}: TAM={TAM_this_stage}, TIM={TIM_this_stage}")
            print(f"Current State: {current_state}")
            print(f"{'='*60}\n")

        state_key = state_to_key(current_state)

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
        # STEP 2: IDENTIFY THE REWARD OF THIS ATTACK
        # --------------------------------------------------------------------


        # ====================================================================
        # INNER LEVEL (DEFENDER)
        # ====================================================================

        # Construct inner level state (includes attack information)
        inner_state_key = (state_key, attack_action)

        # --------------------------------------------------------------------
        # STEP 3: SELECT A DEFEND STRATEGY
        # --------------------------------------------------------------------

        # Generate all feasible defend actions given attack_action
        feasible_defends = generate_defend_strategies(
                                inner_state      = current_state['asset_status'],
                                TIM_this_stage   = TIM_this_stage,
                                attack_action    = attack_action,
                                IM_inventory     = current_state['IM_inventory'],
                                coverage_matrix  = coverage_matrix,
                                ASSET_NODE       = ASSET_NODE
                            )
        # Select defend action using epsilon-greedy on Q_inner
        for d in feasible_defends:
            if (inner_state_key, d) not in Q_inner:
                Q_inner[(inner_state_key, d)] = 0.0
            if (inner_state_key, d) not in N_inner:
                N_inner[(inner_state_key, d)] = 0

        defend_action = epsilon_greedy(Q_inner, inner_state_key,feasible_defends, epsilon_inner)

        # Increment visit count
        N_inner[(inner_state_key, defend_action)] += 1

        if verbose:
            print(f"   Selected defense: {defend_action}")
            print(f"   (This determines how {TIM_this_stage} IMs are allocated to threatened assets)")


        # --------------------------------------------------------------------
        # STEP 4: CALCULATE OPTIMAL IM ALLOCATION
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 4: Calculate optimal IM allocation from SAM batteries to assets")

        # Solve optimization model (5-10) to allocate IMs from SAM batteries
        # This determines w_ij (IMs from SAM at i to defend asset j)

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
        # STEP 5: CALCULATE THE REWARD FOR DEFENDER
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 5: Calculate the reward for defender (expected saved assets)")

        # Calculate expected saved assets using the formula from notes:

        reward_inner = 0
        p_save_per_asset = []
        for i in range(num_assets):
            if current_state['asset_status'][i] == 1:   # only intact assets
                ams = attack_action[i]
                ims = defend_action[i]

                # Calculate saving probability
                p_save = calculate_saving_probability(ams, ims, P_AM_HIT, P_IM_KILL)
                reward_inner += asset_value * p_save
                p_save_per_asset.append(p_save)
            else:
                p_save_per_asset.append(0.0)   # already destroyed

        if verbose: print(f"   Expected saved assets (defender reward): {reward_inner}")


        # --------------------------------------------------------------------
        # STEP 6: UPDATE INNER LEVEL Q-VALUE
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 6: Update inner level Q-value")

        # Calculate learning rate for inner level
        alpha_inner = 1.0 / N_inner[(inner_state_key, defend_action)]

        # Inner level is single-stage from defender's perspective within
        # this salvo → no future inner Q to look up; target = reward_inner

        Q_inner[(inner_state_key, defend_action)] += alpha_inner * (
            reward_inner - Q_inner[(inner_state_key, defend_action)]
        )

        if verbose:
            print(f"   Updated Q_inner[{inner_state_key}, {defend_action}]")

        # ====================================================================
        # BACK TO OUTER LEVEL (ATTACKER) - PART 2
        # ====================================================================

        # --------------------------------------------------------------------
        # STEP 7: USE INNER LEVEL Q-VALUE AS OUTER LEVEL REWARD
        # --------------------------------------------------------------------
        if verbose:
            print("\n" + "-"*60)
            print("STEP 7: Use inner level result as outer level reward")

        # attacker should be rewarded for damage caused, not assets saved
        reward_outer = (sum(current_state['asset_status']) * asset_value) - reward_inner


        if verbose:
            print(f"\nSTEP 7 — Outer reward (attacker, = damage): {reward_outer:.4f}")


        # --------------------------------------------------------------------
        # STEP 8: OBSERVE NEXT STATE
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 8: Observe next state (stochastic transition)")

        new_asset_status = list(current_state['asset_status'])
        for i in range(num_assets):
            if current_state['asset_status'][i] == 1:
                ams = attack_action[i]
                ims = defend_action[i]
                p_save = calculate_saving_probability(ams, ims, P_AM_HIT, P_IM_KILL)
                if random.random() >= p_save:        # asset destroyed
                    new_asset_status[i] = 0
                    episode_damage += asset_value

        next_state = {
            'asset_status': new_asset_status,
            'IM_inventory': state_after_alloc['IM_inventory']   # from LP
        }

        if verbose:
            print(f"STEP 8 — Next state: assets={next_state['asset_status']}  "
                  f"IMs={next_state['IM_inventory']}")

        # --------------------------------------------------------------------
        # STEP 9: UPDATE OUTER LEVEL Q-VALUE USING BELLMAN EQUATION
        # --------------------------------------------------------------------
        if verbose: print("\nSTEP 9: Update outer level Q-value using Bellman equation")

        # Calculate learning rate for outer level
        alpha_outer = 1.0 / N_outer[(state_key, attack_action)]

        # Find maximum Q-value for next state (Bellman look-ahead)
        if stage < num_stages:
            next_state_key = state_to_key(next_state)
            next_stages_left = stages_left - 1
            next_TAM_per_stage = (TAM_remaining - TAM_this_stage) / max(next_stages_left, 1)
            next_feasible_attacks = generate_attack_strategies(next_state['asset_status'], next_TAM_per_stage)
            max_Q_next_outer = max(
                Q_outer.get((next_state_key, a), 0.0) for a in next_feasible_attacks
            )
        else:
            max_Q_next_outer = 0.0   # terminal stage

        # Update Q_outer using Bellman equation
        Q_outer[(state_key, attack_action)] += alpha_outer * (
            reward_outer
            + gamma * max_Q_next_outer
            - Q_outer[(state_key, attack_action)]
        )

        if verbose:
            print(f"STEP 9 — Q_outer updated → "
                  f"{Q_outer[(state_key, attack_action)]:.4f}")


        # ====================================================================
        # PREPARE FOR NEXT STAGE
        # ====================================================================
        if verbose:
            print(f"\n{'='*60}")
            print(f"END OF STAGE {stage}")
            print(f"{'='*60}\n")

        # Update state for next stage
        current_state = next_state

        # Update remaining resources (from notes: "TAM = TAM - [TIM/stage]")
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

    # Q-tables and visit counts — initialized here, passed into run_one_episode
    Q_outer = {}   # Q_outer[(state_key, attack_action)]  = expected attacker reward
    Q_inner = {}   # Q_inner[(inner_state_key, defend_action)] = expected defender reward
    N_outer = {}   # Visit counts for outer-level learning rate
    N_inner = {}   # Visit counts for inner-level learning rate

    damage_history = []

    for episode in range(1, num_episodes + 1):

        verbose = (episode == 1 or episode % verbose_every == 0)
        if verbose:
            print(f"\n{'#'*70}")
            print(f"EPISODE {episode} / {num_episodes}")
            print(f"{'#'*70}")

        # Run one complete episode through all stages (executes all 9 steps per stage)
        ep_damage = run_one_episode(
            Q_outer, Q_inner, N_outer, N_inner,
            gamma, epsilon_outer, epsilon_inner,
            verbose=verbose
        )
        damage_history.append(ep_damage)

        # Optional: Track and report convergence metrics
        # TODO: Check if Q-values are converging
        # TODO: Optionally decay epsilon over time
        if verbose:
            avg = sum(damage_history[-50:]) / len(damage_history[-50:])
            print(f"\n  → Episode damage: {ep_damage:.1f}  |  "
                  f"Last-50 avg: {avg:.2f}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    return Q_outer, Q_inner, damage_history