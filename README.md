# bilevel-qlearning-mdp

## File Structure

`config.py` — all problem parameters in one place, edit this to change the scenario:
* `TAM`, `TIM`, `num_stages` — missile counts and salvo structure
* `coverage_matrix` — which SAM nodes can defend which assets
* `initial_asset_status`, `initial_IM_inventory` — starting state
* `PRESET_ATTACK_STRATEGIES`, `PRESET_DEFENSE_STRATEGIES` — discrete action spaces
* `gamma`, `epsilon_outer`, `epsilon_inner` — Q-learning hyperparameters

`optimization.py` — the LP model (Model 5-10) that allocates interceptor missiles each stage:
* `solve_IM_allocation_model()` — maximizes expected saved asset value by optimally routing IMs from SAM batteries to threatened assets, subject to coverage and inventory constraints

`environment.py` — the world the agents operate in:
* `generate_attack_strategies()` — filters preset attacks to those feasible given current asset status and available AMs
* `generate_defend_strategies()` — filters preset defenses to those feasible given the current attack and available IMs
* `calculate_saving_probability()` — computes probability an asset survives given AMs attacking and IMs defending
* `state_to_key()` — converts mutable state dict to a hashable tuple for Q-table indexing
* `epsilon_greedy()` — selects actions with epsilon-greedy exploration over feasible actions

`agent.py` — the bi-level Q-learning algorithm:
* `run_one_episode()` — executes all 9 steps of the algorithm across all stages, updating both Q-tables
* `train_bilevel_qlearning()` — runs the full training loop, initializes Q-tables, tracks damage history
* `evaluate_policy()` — runs the learned policy with epsilon=0 (pure exploitation) to report final attacker damage

`main.py` — entry point:
* Sets random seeds for reproducibility
* Calls `train_bilevel_qlearning()` and `evaluate_policy()`
* Prints top 5 attacker strategies by Q-value
* Saves convergence plots to `outputs/`
