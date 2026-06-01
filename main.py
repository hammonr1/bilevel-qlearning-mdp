import random
import numpy as np
from config import gamma, epsilon_outer, epsilon_inner
from agent import run_one_episode, train_bilevel_qlearning
import os
import matplotlib.pyplot as plt
import pandas as pd

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
        num_episodes=10000,
        verbose_every=1000
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
    
    # ==========================================================================
    # DIAGNOSTICS: Reconstruct epsilon decay
    # ==========================================================================
    # Since epsilon is not returned from train_bilevel_qlearning, we estimate it
    # Adjust these values to match your actual decay in agent.py/config.py
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995  # ADJUST THIS to match your actual decay rate
    
    episode_numbers = list(range(len(damage_history)))
    episode_epsilon = [
        max(epsilon_min, epsilon_start * (epsilon_decay ** ep))
        for ep in episode_numbers
    ]
    
    # ==========================================================================
    # PLOTS - WITH EPISODE NUMBERS IN FILENAMES
    # ==========================================================================
    # Create outputs folder if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Get total number of episodes for filename
    num_episodes = len(damage_history)
    
    # --- Plot 1: Raw damage per episode ---
    plt.figure(figsize=(10, 4))
    plt.plot(damage_history, alpha=0.4, color='steelblue', label='Episode damage')
    plt.xlabel('Episode')
    plt.ylabel('Damage')
    plt.title(f'Attacker Damage per Episode (Total: {num_episodes})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # CHANGED: Added episode number to filename
    plt.savefig(f"outputs/damage_per_episode_ep{num_episodes:05d}.png", dpi=150)
    plt.close()
    
    # --- Plot 2: Rolling average with enhanced diagnostics ---
    window = 50
    rolling_avg = [
        sum(damage_history[max(0, i-window):i+1]) / len(damage_history[max(0, i-window):i+1])
        for i in range(len(damage_history))
    ]
    
    # Create a 2-subplot figure
    plt.figure(figsize=(15, 5))
    
    # Left: Original rolling average
    plt.subplot(1, 2, 1)
    plt.plot(rolling_avg, color='steelblue', linewidth=2, label=f'{window}-episode rolling avg')
    plt.xlabel('Episode')
    plt.ylabel('Avg Damage')
    plt.title('Convergence — Rolling Average Attacker Damage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right: NEW - Damage vs Epsilon overlay
    plt.subplot(1, 2, 2)
    ax1 = plt.gca()
    ax1.plot(episode_numbers, rolling_avg, 'b-', linewidth=2, label='Damage (rolling avg)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Damage', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Twin axis for epsilon
    ax2 = ax1.twinx()
    ax2.plot(episode_numbers, episode_epsilon, 'r--', alpha=0.6, linewidth=2, label='Epsilon (exploration)')
    ax2.set_ylabel('Epsilon (Exploration Rate)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Damage vs Exploration Rate')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    # CHANGED: Added episode number to filename
    plt.savefig(f"outputs/convergence_with_epsilon_ep{num_episodes:05d}.png", dpi=150)
    plt.close()
    
    # ==========================================================================
    # DIP ANALYSIS
    # ==========================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC ANALYSIS: Identifying Performance Dips")
    print("="*70)
    
    rolling_series = pd.Series(rolling_avg)
    changes = rolling_series.diff()
    
    # Detect significant drops (adjust threshold as needed)
    dip_threshold = -2.0
    big_dips = changes < dip_threshold
    
    if big_dips.sum() >= 5:
        dip_episodes = [episode_numbers[i] for i, is_dip in enumerate(big_dips) if is_dip]
        dip_epsilons = [episode_epsilon[i] for i, is_dip in enumerate(big_dips) if is_dip]
        dip_magnitudes = [changes[i] for i, is_dip in enumerate(big_dips) if is_dip]
        
        print(f"\n📉 Found {len(dip_episodes)} significant dips (drops > {dip_threshold})")
        print(f"\nDip Statistics:")
        print(f"  Average epsilon during dips:     {np.mean(dip_epsilons):.4f}")
        print(f"  Average epsilon overall:         {np.mean(episode_epsilon):.4f}")
        print(f"  Average magnitude of dips:       {np.mean(dip_magnitudes):.2f}")
        print(f"  Episodes with dips: {dip_episodes[:10]}{'...' if len(dip_episodes) > 10 else ''}")
        
        # Correlation analysis
        epsilon_ratio = np.mean(dip_epsilons) / (np.mean(episode_epsilon) + 1e-6)
        
        print("\n" + "-"*70)
        print("DIAGNOSIS:")
        print("-"*70)
        
        if epsilon_ratio > 1.15:  # Dips occur when epsilon is 15% higher than average
            print("✓ CONFIRMED: Dips strongly correlate with HIGH epsilon values")
            print("  → Dips are caused by EXPLORATION (epsilon-greedy random actions)")
            print("\n💡 RECOMMENDATIONS:")
            print("  1. These dips are NORMAL - they represent exploration")
            print("  2. To reduce dips, try:")
            print(f"     - Faster decay: epsilon_decay = 0.99 (currently ~{epsilon_decay})")
            print(f"     - Lower minimum: epsilon_min = 0.005 (currently {epsilon_min})")
            print("  3. Or accept them - exploration is necessary for learning!")
        elif epsilon_ratio > 1.05:
            print("⚠️  MODERATE correlation with epsilon")
            print("  → Dips partially caused by exploration, but other factors present")
            print("  → Consider: state transitions, resource depletion, or solver issues")
        else:
            print("⚠️  LOW correlation with epsilon")
            print("  → Dips NOT primarily caused by exploration")
            print("\n🔍 Investigate:")
            print("  1. State space transitions (entering under-explored regions)")
            print("  2. Resource depletion patterns (interceptors running out)")
            print("  3. Inner MDP solver failures")
            print("  4. Stochastic state transitions")
            print("\n  → Use full diagnostics in missile_defense_diagnostics.py")
    else:
        print("\n✓ No significant dips detected!")
        print("  Your learning curve is smooth - good convergence.")
    
    # ==========================================================================
    # EXPLORATION ANALYSIS
    # ==========================================================================
    print("\n" + "="*70)
    print("EXPLORATION SCHEDULE ANALYSIS")
    print("="*70)
    
    # Find when epsilon drops below key thresholds
    milestones = [0.5, 0.2, 0.1, 0.05, epsilon_min]
    print("\nEpsilon decay milestones:")
    for threshold in milestones:
        episodes_below = [ep for ep, eps in enumerate(episode_epsilon) if eps <= threshold]
        if episodes_below:
            first_ep = episodes_below[0]
            print(f"  Epsilon < {threshold:4.2f}: Episode {first_ep:3d} ({first_ep/len(episode_epsilon)*100:.1f}% through training)")
    
    # Calculate approximate exploration rate over time
    exploration_bins = 5
    bin_size = len(episode_epsilon) // exploration_bins
    print(f"\nAverage epsilon by training phase ({exploration_bins} bins):")
    for i in range(exploration_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, len(episode_epsilon))
        avg_eps = np.mean(episode_epsilon[start:end])
        avg_dmg = np.mean(rolling_avg[start:end])
        print(f"  Episodes {start:3d}-{end:3d}: ε={avg_eps:.4f}  →  Avg Damage={avg_dmg:.2f}")
    
    print("\n" + "="*70)
    print("Plots saved to outputs/")
    print(f"  - damage_per_episode_ep{num_episodes:05d}.png")
    print(f"  - convergence_with_epsilon_ep{num_episodes:05d}.png")
    print("="*70)