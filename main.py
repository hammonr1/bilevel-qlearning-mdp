import random
import numpy as np
from config import gamma, epsilon_outer, epsilon_inner
from agent import run_one_episode

# ============================================================================
# GREEDY EVALUATION (epsilon = 0)
# ============================================================================

def evaluate_policy(Q_outer: dict, Q_inner: dict, n_eval: int = 100) -> float:
    """Run the learned policy greedily and report average attacker damage."""
    total_damage = 0.0
    for _ in range(n_eval):
        total_damage += run_one_episode(
            Q_outer, Q_inner, {}, {},
            gamma=gamma,
            epsilon_outer=0.0,   # pure exploitation
            epsilon_inner=0.0,
            verbose=False
        )
    avg = total_damage / n_eval
    print(f"\nGREEDY EVALUATION ({n_eval} runs): avg attacker damage = {avg:.2f}")
    return avg

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("""
    ============================================================================
    BI-LEVEL Q-LEARNING FOR MISSILE DEFENSE
    ============================================================================

    THE 9 STEPS (executed each stage):

    OUTER LEVEL (Attacker):
      Step 1: Select an attack strategy
      Step 2: Identify reward of this attack (requires inner level)

    INNER LEVEL (Defender):
      Step 3: Select a defend strategy
      Step 4: Calculate optimal IM allocation (Model 5-10)
      Step 5: Calculate reward for defender (expected saved assets)
      Step 6: Update inner level Q-value

    BACK TO OUTER LEVEL:
      Step 7: Use inner level result as outer level reward
      Step 8: Observe next state (stochastic transition)
      Step 9: Update outer level Q-value using Bellman equation

    ============================================================================
    """)

    random.seed(42)
    np.random.seed(42)

    # Train — all parameters are set at the top of the file
    Q_outer, Q_inner, damage_history = train_bilevel_qlearning(
        num_episodes=300,
        verbose_every=100
    )

    # Evaluate learned policies greedily (epsilon=0)
    evaluate_policy(Q_outer, Q_inner, n_eval=200)

    # Summary of learned Q-tables
    print(f"\nQ_outer entries: {len(Q_outer)}")
    print(f"Q_inner entries: {len(Q_inner)}")

    # Show top attacker strategies (highest Q → most damage)
    top_attacks = sorted(Q_outer.items(), key=lambda x: -x[1])[:5]
    print("\nTop 5 attacker strategies (by Q-value):")
    for (s_key, action), q_val in top_attacks:
        print(f"  state={s_key[0]}  attack={action}  Q={q_val:.4f}")
        
        
