import numpy as np
import itertools
from scipy.special import expit
import matplotlib.pyplot as plt

class MOMDP:
    def __init__(self, d, m):
        self.d = d
        self.m = m

    def evaluate_policy(self, xi):
        xi = np.array(xi)
        if np.any(xi < 0) or np.any(xi > 1) or np.sum(xi) > 1:
            return np.full(self.m + 1, -np.inf)
        f = 1 - np.sum(xi ** 2)
        return np.concatenate([xi, [f]])

    def optimal_policy(self, w_star):
        w_last = w_star[-1]
        w_front = w_star[:-1]
        if w_last <= 0:
            return self.evaluate_policy(np.zeros(self.m))
        xi_opt = w_front / (2 * w_last)
        if np.sum(xi_opt) > 1:
            xi_opt /= np.sum(xi_opt)
        xi_opt = np.clip(xi_opt, 0, 1)
        return self.evaluate_policy(xi_opt)

def create_initial_grid(m, rho):
    grid_1d = np.linspace(0, 1, int(1 / rho) + 1)
    full_grid = np.array(list(itertools.product(grid_1d, repeat=m)))
    feasible = np.sum(full_grid, axis=1) <= 1.0 + 1e-10
    return full_grid[feasible]

def refine_grid(region, m, rho):
    if len(region) == 0:
        return np.empty((0, m))
    min_vals, max_vals = np.min(region, axis=0), np.max(region, axis=0)
    refined = [np.clip(np.arange(min_vals[i], max_vals[i] + rho / 2, rho), 0, 1) for i in range(m)]
    if any(len(x) == 0 for x in refined):
        return np.empty((0, m))
    grid = np.array(list(itertools.product(*refined)))
    return grid[np.sum(grid, axis=1) <= 1.0 + 1e-10]

def find_adjacent_pairs(grid, rho):
    pairs = []
    n = len(grid)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.abs(grid[i] - grid[j])
            if np.max(diff) <= rho + 1e-10 and np.sum(diff > 1e-10) == 1:
                pairs.append((i, j))
    return pairs

def AdaptiveGridPreferenceLearning(epsilon, delta, alpha, Delta_min, momdp, random_state,
                                   ablation=None):
    rho, m = 1.0, momdp.m
    w_star = random_state.dirichlet(np.ones(m + 1))

    def oracle(V1, V2):
        delta_val = np.dot(w_star, V1 - V2)
        if ablation == "no_oracle_noise":
            return delta_val > 0
        prob = expit(2 * alpha * delta_val)
        return random_state.random() < prob

    W = create_initial_grid(m, rho)
    total_queries = 0

    while rho > epsilon:
        if len(W) <= 1:
            break

        if ablation == "no_majority_vote":
            tau = 1
        else:
            tau = int(np.ceil((2 / (alpha ** 2 * rho ** 2)) * np.log(4 * (rho ** (-m)) / delta)))

        pairs = find_adjacent_pairs(W, rho)
        if not pairs:
            break

        keep = np.ones(len(W), dtype=bool)

        if ablation == "fixed_grid":
            rho /= 2
            if rho > epsilon:
                W = refine_grid(W, m, rho)
            continue

        for i, j in pairs:
            if not (keep[i] and keep[j]):
                continue
            xi_i, xi_j = W[i], W[j]
            V_i, V_j = momdp.evaluate_policy(xi_i), momdp.evaluate_policy(xi_j)
            if np.any(V_i == -np.inf) or np.any(V_j == -np.inf):
                continue
            wins_i = sum(oracle(V_i, V_j) for _ in range(tau))
            total_queries += tau
            if wins_i > tau // 2:
                keep[j] = False
            else:
                keep[i] = False

        W = W[keep]
        rho /= 2
        if rho > epsilon and ablation != "no_refinement":
            W = refine_grid(W, m, rho)

    final_xi = np.zeros(m) if len(W) == 0 else W[0]
    return momdp.evaluate_policy(final_xi), w_star, total_queries

def run_ablation_experiment():
    epsilon = 0.250
    delta = 0.05
    alpha = 2.0
    Delta_min = 0.30
    d = 4
    m = 3
    num_trials = 10

    ablations = {
        "no_refinement": "No Refinement",
        "no_majority_vote": "No Majority Vote",
        "no_oracle_noise": "No Oracle Noise",
        "fixed_grid": "Fixed Grid Only"
    }

    results = []

    master_rng = np.random.default_rng(123)

    for ablation_key, label in ablations.items():
        regrets, queries = [], []

        for trial in range(num_trials):
            rng = np.random.default_rng(master_rng.integers(1, 2**32 - 1))
            momdp = MOMDP(d, m)

            V_hat, w_star, q = AdaptiveGridPreferenceLearning(
                epsilon=epsilon,
                delta=delta,
                alpha=alpha,
                Delta_min=Delta_min,
                momdp=momdp,
                random_state=rng,
                ablation=ablation_key
            )

            V_opt = momdp.optimal_policy(w_star)
            regret = max(0, np.dot(w_star, V_opt - V_hat))
            regrets.append(regret)
            queries.append(q)

        mean_regret = np.mean(regrets)
        ci_regret = 1.96 * np.std(regrets, ddof=1) / np.sqrt(num_trials)
        mean_queries = np.mean(queries)
        ci_queries = 1.96 * np.std(queries, ddof=1) / np.sqrt(num_trials)

        print(f"{label}:")
        print(f"  Regret: {mean_regret:.6f} ± {ci_regret:.6f}")
        print(f"  Queries: {mean_queries:.2f} ± {ci_queries:.2f}")
        results.append((label, mean_regret, ci_regret, mean_queries, ci_queries))

if __name__ == "__main__":
    run_ablation_experiment()
