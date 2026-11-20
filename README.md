# Multi-Agent Bandits with Shared, Depleting Resources

A simulation study of how classic multi-armed bandit algorithms behave when several agents compete for the same limited resources, developed as the first assignment of a Reinforcement Learning course. Five bandit algorithms are implemented from scratch, adapted to an environment in which arms permanently deactivate once their capacity is exhausted, and compared statistically across heterogeneous and homogeneous agent populations.

**Institution:** University of Tehran, School of Electrical and Computer Engineering
**Course:** Reinforcement Learning
**Author:** Babak Hosseini Mohtasham
**Assignment specification:** [`Assignment 1.pdf`](./Assignment%201.pdf) (Persian)

---

## Overview

In a standard N-armed bandit, one agent pulls arms from a fixed, unlimited set. Here, a population of agents pulls from the *same* set of arms, and every arm has a finite capacity: each successful pull consumes one unit, and once the capacity reaches zero the arm becomes unavailable to everyone. Good arms are therefore depleted first, and the quality of the remaining options degrades over time. This breaks the usual assumption that maximizing individual reward is the same as maximizing collective reward.

**Environment (per simulation run):**

| Parameter | Distribution |
|---|---|
| Reward on success, `r_max` | Uniform(0, 10) |
| Reward on failure, `r_min` | Uniform(−10, 0) |
| Success probability, `p` | Uniform(0.2, 0.8) |
| Capacity (successes before deactivation), `z` | Discrete uniform {80, …, 150} |
| Population / arms / horizon | 10 agents, 5 arms, 100 rounds |

**Objectives:**

1. Implement Epsilon-Greedy, Boltzmann (softmax), UCB, Thompson Sampling, and Gradient Bandit, and modify each so that it only selects among currently active arms.
2. Simulate a heterogeneous population (two agents per algorithm) and analyze how population reward, arm availability, and demand across arms evolve over time.
3. Define a *social regret* for the shared-resource setting and discuss why single-agent regret is inadequate.
4. Compare algorithms in homogeneous populations with statistically sound tests, and measure each algorithm's sensitivity to its main hyperparameter.
5. Study how the results change with population size, capacity range, and success-probability range.

## Methodology

| Component | Description |
|---|---|
| **Algorithms** | Epsilon-Greedy, Boltzmann, UCB, Thompson Sampling (Gaussian posterior with running mean/variance via Welford updates), and Gradient Bandit, all implemented in NumPy with a shared `select` / `update` interface |
| **Handling arm deactivation** | Selection is restricted to the active arms while estimates for inactive arms are retained. The Gradient Bandit baseline is modified to average rewards over only the arms that were active at the current decision step |
| **Agents** | Each agent keeps its own learning state and shares no information with the others; agent order is re-randomized every round to avoid a fixed-priority advantage |
| **Reproducibility** | Every run is seeded through `numpy.random.SeedSequence`, which spawns independent streams for the environment, the agent ordering, and each agent |
| **Hyperparameter selection** | 100 candidate values per algorithm, each evaluated over 5 runs; the best mean final reward is chosen using a separate seed range from the final experiments, so evaluation runs never influence tuning (selected: ε = 0.10, τ ≈ 0.131, α = 0.19; UCB's constant is set to the maximum reward scale, c = 10) |
| **Uncertainty** | 100 independent runs per configuration; mean curves with 95% confidence intervals based on the Student's *t* distribution |
| **Statistical testing** | Paired *t*-tests over matched seeds, Holm–Bonferroni correction for multiple comparisons, and Cohen's *d_z* effect sizes |
| **Social regret** | Defined as the gap between the expected cumulative reward of an oracle joint policy and the realized population reward; the oracle needs each arm's expected reward and remaining capacity, and fills arms greedily in order of expected value |

## Repository Structure

| Path | Description |
|---|---|
| [`CA1.ipynb`](./CA1.ipynb) | Complete implementation: environment, five algorithms, experiment runner, hyperparameter search, all experiments, statistical tests, and written answers to the assignment questions |
| [`Assignment 1.pdf`](./Assignment%201.pdf) | Original assignment specification (Persian) |

## Key Results

**Population reward rises, then collapses.** In the heterogeneous population, average per-round reward climbs for roughly the first 15 rounds, then declines and turns negative around round 60; cumulative population reward peaks and then falls. The cause is the shared capacity constraint: the highest-value arm is exhausted first (the first arm deactivates between rounds ~20 and ~55 depending on the run), and demand then shifts progressively down the ranking until agents are forced onto arms with negative expected reward.

**Individual performance and collective welfare are different things.** An agent that exploits the best arm aggressively earns more for itself but depletes that arm faster for everyone else. This motivates the social-regret definition, since single-agent regret measures performance against an always-best-arm benchmark that is unavailable when capacity is shared.

**Algorithm ranking in the heterogeneous population** (cumulative reward per agent at round 100, read from the plots): Boltzmann highest (~28), then Gradient Bandit (~26), Epsilon-Greedy and Thompson Sampling similar (~14), and UCB lowest (~8).

**Homogeneous populations: the differences are statistically significant.** Final cumulative reward per agent, 100 paired runs, Holm-corrected:

| Comparison | Mean difference | Cohen's *d_z* | Adjusted *p* |
|---|---|---|---|
| Epsilon-Greedy vs. UCB | +6.97 | 0.52 | 4 × 10⁻⁶ |
| UCB vs. Gradient Bandit | −3.53 | −0.31 | 0.005 |
| Epsilon-Greedy vs. Gradient Bandit | +3.44 | 0.23 | 0.023 |

**Sensitivity to hyperparameters.** Doubling each algorithm's main hyperparameter:

| Algorithm | Original mean | Doubled mean | Difference | Adjusted *p* |
|---|---|---|---|---|
| Epsilon-Greedy (ε: 0.10 → 0.20) | 18.79 | 14.86 | −3.93 | 0.005 |
| UCB (c: 10 → 20) | 11.81 | 1.88 | **−9.93** | 1 × 10⁻⁹ |
| Gradient Bandit (α: 0.19 → 0.38) | 15.35 | 14.03 | −1.32 | 0.37 (n.s.) |

UCB degrades most when its exploration bonus is increased, while the Gradient Bandit difference is not statistically significant. This holds only for the specific values tested.

*(Full plots (mean reward, active arms, remaining capacity, pull share per ranked arm), the complete answers to the conceptual questions, and the population-size and capacity-range sensitivity runs are in the notebook.)*

## Reproducing the Results

1. Install dependencies: `pip install numpy scipy pandas matplotlib statsmodels`.
2. Open [`CA1.ipynb`](./CA1.ipynb) and run all cells top to bottom. No external data is required, as the environment is simulated. The default configuration is 100 runs of 100 rounds each.
3. Seeds are fixed (`BASE_SEED = 0` for evaluation, a separate offset for tuning), so results are reproducible. The constants at the top of the notebook (`NUM_AGENTS`, `NUM_ACTIONS`, `HORIZON`, `N_TRIALS`, and the reward, capacity, and probability ranges) control the experiment.

## Notes on Scope

- **Tuning was single-agent.** Hyperparameters were selected by running one agent alone, then used unchanged in 10-agent competitive populations, where the optimal exploration level may differ. UCB's constant was set from the reward scale rather than tuned, so the "UCB performs worst" finding is conditional on that choice (and the doubling experiment suggests a smaller constant could do considerably better).
- **Social regret is defined, not measured.** The notebook gives the definition and the information an oracle would need, but does not compute social regret numerically.
- **Thompson Sampling and Boltzmann** are included in the heterogeneous comparison but not in the homogeneous significance tests and sensitivity analysis, which cover Epsilon-Greedy, UCB, and Gradient Bandit.
- The assignment PDF is in Persian; the environment parameters above are taken from the notebook's implementation.
