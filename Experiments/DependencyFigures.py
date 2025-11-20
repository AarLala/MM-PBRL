import numpy as np
import itertools
from scipy.special import expit
import matplotlib.pyplot as plt
import os

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
            xi_opt = xi_opt / np.sum(xi_opt)
        xi_opt = np.clip(xi_opt, 0, 1)
        return self.evaluate_policy(xi_opt)

def create_initial_grid(m, rho):
    grid_1d = np.linspace(0, 1, int(1 / rho) + 1)
    full_grid = np.array(list(itertools.product(grid_1d, repeat=m)))
    feasible = np.sum(full_grid, axis=1) <= 1.0 + 1e-10
    return full_grid[feasible]

def refine_grid(region, m, rho):
    if len(region) == 0:
        return np.array([]).reshape(0, m)
    min_vals, max_vals = np.min(region, axis=0), np.max(region, axis=0)
    refined = [np.clip(np.arange(min_vals[i], max_vals[i] + rho / 2, rho), 0, 1) for i in range(m)]
    if any(len(x) == 0 for x in refined):
        return np.array([]).reshape(0, m)
    grid = np.array(list(itertools.product(*refined)))
    return grid[np.sum(grid, axis=1) <= 1.0 + 1e-10]

def find_adjacent_pairs(grid, rho):
    pairs, n = [], len(grid)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.abs(grid[i] - grid[j])
            if np.max(diff) <= rho + 1e-10 and np.sum(diff > 1e-10) == 1:
                pairs.append((i, j))
    return pairs

def AdaptiveGridPreferenceLearning(epsilon, delta, alpha, Delta_min, momdp, random_state):
    rho, m = 1.0, momdp.m
    w_star = random_state.dirichlet(np.ones(m + 1))

    def oracle(V1, V2):
        delta_val = np.dot(w_star, V1 - V2)
        prob = expit(2 * alpha * delta_val)
        return random_state.random() < prob

    W = create_initial_grid(m, rho)
    total_queries = 0

    while rho > epsilon:
        if len(W) <= 1:
            break
        tau = int(np.ceil((2 / (alpha ** 2 * rho ** 2)) * np.log(4 * (rho ** (-m)) / delta)))
        pairs = find_adjacent_pairs(W, rho)
        if not pairs:
            break
        keep = np.ones(len(W), dtype=bool)
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
        rho = rho / 2
        if rho > epsilon:
            W = refine_grid(W, m, rho)

    final_xi = np.zeros(m) if len(W) == 0 else W[0]
    return momdp.evaluate_policy(final_xi), w_star, total_queries

def run_full_hyperparam_analysis():
    epsilon_values = [0.5, 0.45, 0.35, 0.25, 0.15, 0.125]
    alpha_values = [0.5, 1.0, 1.5, 2.0]
    delta_values = [0.05, 0.1, 0.15, 0.17, 0.2]
    Delta_min_values = [0.1, 0.2, 0.3]

    d, m = 4, 3
    num_trials = 10
    trial_seeds = [383329928, 3324115916, 2811363264, 1884968545, 1859786276,
                   3687649985, 369133709, 2995172877, 865305067, 404488629]

    default_params = {"epsilon": 0.25, "delta": 0.1, "alpha": 1.0, "Delta_min": 0.2}

    os.makedirs("figures", exist_ok=True)

    def evaluate_setting(epsilon, delta, alpha, Delta_min):
        momdp = MOMDP(d, m)
        regrets, queries = [], []
        for trial in range(num_trials):
            rng = np.random.default_rng(trial_seeds[trial])
            V_hat, w_star, q = AdaptiveGridPreferenceLearning(
                epsilon=epsilon, delta=delta, alpha=alpha, Delta_min=Delta_min, momdp=momdp, random_state=rng
            )
            V_opt = momdp.optimal_policy(w_star)
            regrets.append(max(0, np.dot(w_star, V_opt - V_hat)))
            queries.append(q)
        mean_regret = np.mean(regrets)
        ci_regret = 1.96 * np.std(regrets, ddof=1) / np.sqrt(num_trials)
        mean_queries = np.mean(queries)
        ci_queries = 1.96 * np.std(queries, ddof=1) / np.sqrt(num_trials)
        return mean_regret, ci_regret, mean_queries, ci_queries

    def sweep_and_plot(x_param_name, x_values, line_param_name, line_param_values):
        plt.figure(figsize=(8,5))
        for line_val in line_param_values:
            means, cis = [], []
            for x_val in x_values:
                params = default_params.copy()
                params[x_param_name] = x_val
                params[line_param_name] = line_val
                mean, ci, _, _ = evaluate_setting(**params)
                means.append(mean)
                cis.append(ci)
            means, cis = np.array(means), np.array(cis)
            plt.plot(x_values, means, '-o', label=f'{line_param_name}={line_val}')
            plt.fill_between(x_values, means-cis, means+cis, alpha=0.2)
        plt.xlabel(x_param_name)
        plt.ylabel("Mean Regret")
        plt.title(f"Effect of {x_param_name} with varying {line_param_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"figures/regret_{x_param_name}_vs_{line_param_name}.png", dpi=300)
        plt.show()

        plt.figure(figsize=(8,5))
        for line_val in line_param_values:
            queries_means, queries_cis = [], []
            for x_val in x_values:
                params = default_params.copy()
                params[x_param_name] = x_val
                params[line_param_name] = line_val
                _, _, q_mean, q_ci = evaluate_setting(**params)
                queries_means.append(q_mean)
                queries_cis.append(q_ci)
            queries_means, queries_cis = np.array(queries_means), np.array(queries_cis)
            plt.plot(x_values, queries_means, '-s', label=f'{line_param_name}={line_val}')
            plt.fill_between(x_values, queries_means-queries_cis, queries_means+queries_cis, alpha=0.2)
        plt.xlabel(x_param_name)
        plt.ylabel("Mean Queries")
        plt.title(f"Queries vs {x_param_name} with varying {line_param_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"figures/queries_{x_param_name}_vs_{line_param_name}.png", dpi=300)
        plt.show()

    sweep_and_plot("epsilon", epsilon_values, "alpha", alpha_values)
    sweep_and_plot("alpha", alpha_values, "epsilon", epsilon_values)
    sweep_and_plot("delta", delta_values, "epsilon", epsilon_values)
    sweep_and_plot("Delta_min", Delta_min_values, "epsilon", epsilon_values)

if __name__ == "__main__":
    run_full_hyperparam_analysis()
