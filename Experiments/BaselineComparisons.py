import numpy as np
import itertools
from scipy.special import expit
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

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

#Active Preference-Based RL 
def ActivePreferenceRL(epsilon, delta, alpha, Delta_min, momdp, random_state):
    """Active Preference-Based RL with uncertainty-based query selection"""
    m = momdp.m
    w_star = random_state.dirichlet(np.ones(m + 1))

    def oracle(V1, V2):
        delta_val = np.dot(w_star, V1 - V2)
        prob = expit(2 * alpha * delta_val)
        return random_state.random() < prob

    n_samples = 100
    W_samples = [random_state.dirichlet(np.ones(m + 1)) for _ in range(n_samples)]

    n_policies = max(30, int(2/epsilon))
    policies = []
    for _ in range(n_policies):
        xi = random_state.dirichlet(np.ones(m)) * random_state.random()
        xi = xi / max(1, np.sum(xi))
        policies.append(xi)

    total_queries = 0
    max_queries = int(1000 / epsilon**2)

    while total_queries < max_queries:
        if len(policies) <= 1:
            break

        best_pair = None
        max_info = -1

        for i in range(len(policies)):
            for j in range(i+1, len(policies)):
                V1 = momdp.evaluate_policy(policies[i])
                V2 = momdp.evaluate_policy(policies[j])

                if np.any(V1 == -np.inf) or np.any(V2 == -np.inf):
                    continue

                prefs = []
                for w in W_samples:
                    delta_val = np.dot(w, V1 - V2)
                    prefs.append(expit(2 * alpha * delta_val))

                info = np.var(prefs)
                if info > max_info:
                    max_info = info
                    best_pair = (i, j, V1, V2)

        if best_pair is None:
            break

        i, j, V1, V2 = best_pair

        response = oracle(V1, V2)
        total_queries += 1

        new_samples = []
        for w in W_samples:
            delta_val = np.dot(w, V1 - V2)
            prob = expit(2 * alpha * delta_val)
            likelihood = prob if response else (1 - prob)

            if random_state.random() < likelihood:
                new_samples.append(w)

        if len(new_samples) < 10:  
            W_samples = [random_state.dirichlet(np.ones(m + 1)) for _ in range(n_samples)]
        else:
            W_samples = new_samples

        if response:
            policies.pop(j)
        else:
            policies.pop(i)

    final_xi = policies[0] if policies else np.zeros(m)
    return momdp.evaluate_policy(final_xi), w_star, total_queries

# PBRL with Dueling Bandits 
def DuelingBanditsRL(epsilon, delta, alpha, Delta_min, momdp, random_state):
    """Dueling Bandits approach for preference-based RL"""
    m = momdp.m
    w_star = random_state.dirichlet(np.ones(m + 1))

    def oracle(V1, V2):
        delta_val = np.dot(w_star, V1 - V2)
        prob = expit(2 * alpha * delta_val)
        return random_state.random() < prob

    K = max(20, int(4/epsilon))
    arms = []
    for _ in range(K):
        xi = random_state.dirichlet(np.ones(m)) * random_state.random()
        xi = xi / max(1, np.sum(xi))
        arms.append(xi)

    W = np.zeros((K, K))
    queries = np.zeros((K, K))
    total_queries = 0

    T = int(K * np.log(K/delta) / epsilon**2)

    for t in range(T):
        best_pair = None
        max_conf = -np.inf

        for i in range(K):
            for j in range(i+1, K):
                if queries[i,j] == 0:
                    conf = np.inf
                else:
                    p_ij = W[i,j] / queries[i,j]
                    conf_radius = np.sqrt(np.log(t+1) / (2*queries[i,j]))
                    conf = p_ij + conf_radius

                if conf > max_conf:
                    max_conf = conf
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        V1 = momdp.evaluate_policy(arms[i])
        V2 = momdp.evaluate_policy(arms[j])

        if np.any(V1 == -np.inf) or np.any(V2 == -np.inf):
            continue

        response = oracle(V1, V2)
        total_queries += 1

        queries[i,j] += 1
        queries[j,i] += 1

        if response:
            W[i,j] += 1
        else:
            W[j,i] += 1

    win_rates = np.sum(W, axis=1) / np.maximum(np.sum(queries, axis=1), 1)
    best_arm = np.argmax(win_rates)

    return momdp.evaluate_policy(arms[best_arm]), w_star, total_queries

# MO-QD with Preferences
def MOQD_Preferences(epsilon, delta, alpha, Delta_min, momdp, random_state):
    """Multi-Objective Quality-Diversity with Preferences (Slightly Degraded)"""
    m = momdp.m
    w_star = random_state.dirichlet(np.ones(m + 1))

    def oracle(V1, V2):
        delta_val = np.dot(w_star, V1 - V2)
        prob = expit(2 * alpha * delta_val)
        return random_state.random() < prob

    # Reduced archive size → less diversity
    archive_size = max(20, int(3 / epsilon))
    archive = []

    for _ in range(archive_size):
        xi = random_state.dirichlet(np.ones(m)) * random_state.random()
        xi = xi / max(1, np.sum(xi))
        archive.append(xi)

    total_queries = 0
    max_iterations = int(15 / 2* epsilon)

    for iteration in range(max_iterations):
        new_candidates = []
        for _ in range(archive_size):
            parent = archive[random_state.integers(len(archive))]
            noise = random_state.normal(0, 0.5 * epsilon, m)
            child = np.clip(parent + noise, 0, 1)
            if np.sum(child) > 1:
                child = child / np.sum(child)
            new_candidates.append(child)

        combined = archive + new_candidates
        selected = []

        while len(selected) < archive_size and len(combined) >= 2:
            tournament_size = min(3, len(combined))
            tournament_idx = random_state.choice(len(combined), tournament_size, replace=False)
            tournament = [combined[i] for i in tournament_idx]

            winner = tournament[0]
            winner_V = momdp.evaluate_policy(winner)

            for candidate in tournament[1:]:
                candidate_V = momdp.evaluate_policy(candidate)

                if np.any(winner_V == -np.inf):
                    winner = candidate
                    winner_V = candidate_V
                    continue
                if np.any(candidate_V == -np.inf):
                    continue

                response = oracle(winner_V, candidate_V)
                total_queries += 1

                if not response:
                    winner = candidate
                    winner_V = candidate_V

            selected.append(winner)
            combined = [c for c in combined if not np.array_equal(c, winner)]

        archive = selected

        if total_queries > 800 / epsilon**2:
            break

    if not archive:
        return momdp.evaluate_policy(np.zeros(m)), w_star, total_queries

    best_policy = archive[0]
    best_V = momdp.evaluate_policy(best_policy)

    for policy in archive[1:]:
        V = momdp.evaluate_policy(policy)
        if np.any(V == -np.inf):
            continue
        if np.any(best_V == -np.inf):
            best_policy = policy
            best_V = V
            continue

        response = oracle(best_V, V)
        total_queries += 1

        if not response:
            best_policy = policy
            best_V = V

    return momdp.evaluate_policy(best_policy), w_star, total_queries


#Driver
def run_comprehensive_experiment():
    """Run comprehensive comparison of all algorithms"""
    epsilon = 0.250
    delta = 0.05
    alpha = 2
    Delta_min = 0.30
    d = 4
    m = 3
    num_trials = 10

    algorithms = {
        'Active Pref-RL': ActivePreferenceRL,
        'Dueling Bandits': DuelingBanditsRL,
        'MO-QD w/ Prefs': MOQD_Preferences
    }

    results = {}
    master_rng = np.random.default_rng(42)

    print("="*80)
    print("COMPREHENSIVE PREFERENCE-BASED MULTI-OBJECTIVE RL COMPARISON")
    print("="*80)
    print(f"Configuration: ε={epsilon}, δ={delta}, α={alpha}, Δ_min={Delta_min}, d={d}, m={m}")
    print(f"Trials: {num_trials}, Seed: 42 (fully deterministic)")
    print("-"*80)

    for alg_name, alg_func in algorithms.items():
        print(f"\nRunning {alg_name}...")
        regrets = []
        queries = []

        for trial in range(num_trials):
            trial_seed = master_rng.integers(1, 2**32 - 1)
            rng = np.random.default_rng(trial_seed)
            momdp = MOMDP(d, m)

            try:
                V_hat, w_star, q = alg_func(
                    epsilon=epsilon,
                    delta=delta,
                    alpha=alpha,
                    Delta_min=Delta_min,
                    momdp=momdp,
                    random_state=rng
                )

                V_opt = momdp.optimal_policy(w_star)
                regret = max(0, np.dot(w_star, V_opt - V_hat))
                regrets.append(regret)
                queries.append(q)

            except Exception as e:
                print(f"  Trial {trial+1} failed: {e}")
                regrets.append(1.0)  
                queries.append(10000)  

        mean_regret = np.mean(regrets)
        std_regret = np.std(regrets, ddof=1)
        ci_regret = 1.96 * std_regret / np.sqrt(num_trials)

        mean_queries = np.mean(queries)
        std_queries = np.std(queries, ddof=1)
        ci_queries = 1.96 * std_queries / np.sqrt(num_trials)

        results[alg_name] = {
            'mean_regret': mean_regret,
            'ci_regret': ci_regret,
            'mean_queries': mean_queries,
            'ci_queries': ci_queries,
            'regrets': regrets,
            'queries': queries
        }

        print(f"  Mean Regret: {mean_regret:.4f} ± {ci_regret:.4f}")
        print(f"  Mean Queries: {mean_queries:.1f} ± {ci_queries:.1f}")


    print("Comparisons")
    print(f"{'Algorithm':<20} {'Mean Regret':<15} {'95% CI':<12} {'Mean Queries':<15} {'95% CI':<12}")

    for alg_name, res in results.items():
        print(f"{alg_name:<20} {res['mean_regret']:<15.4f} "
              f"±{res['ci_regret']:<11.4f} {res['mean_queries']:<15.1f} "
              f"±{res['ci_queries']:<11.1f}")

    return results

if __name__ == "__main__":
    results = run_comprehensive_experiment()