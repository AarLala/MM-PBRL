# The Minimax Complexity of Preference Based Decision Making in Multi Objective Reinforcement Learning


## Repository Structure
This repository contains the implementation of the Adaptive Grid Preference Learning algorithm and scripts to reproduce the experiments presented in the paper.

* Ablation.py: Performs ablation studies to evaluate the impact of different algorithmic components (e.g., No Refinement, No Majority Vote).
* BaselineComparisons.py: Compares the proposed method against baseline algorithms, calculating mean regret and query complexity.
* DependencyFigures.py: Generates figures visualizing the relationship between regret/queries and problem parameters (dimension and Pareto front size).
* ParameterSweep.py: Runs a grid search over hyperparameters (epsilon, delta, alpha, Delta_min) to identify optimal configurations.
* 122_The_Minimax_Complexity_of_.pdf: The full research paper.

## Dependencies
The code requires Python 3 and the following libraries:

* numpy
* scipy
* matplotlib

All are in requirements.txt

## Setup

Run the following commands in a terminal:
1. **Create a virtual environment**:

    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**:

    ```bash
    venv\Scripts\activate
    ```
3. **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    ```


# Experimental Validation

To reproduce the results presented in the paper, run the following scripts located in the `Experiments` directory.

## Parameter Sweep

Identifies the optimal hyperparameters (`epsilon`, `delta`, `alpha`, `Delta_min`) for the algorithm.

```bash
python Experiments\ParameterSweep.py
```

## Baseline Comparisons

```bash
python Experiments\BaselineComparisons.py
```

## Ablation Studies

Evaluates the contribution of specific algorithmic components (e.g., refinement, noise handling) by running ablation trials.

```bash
python Experiments\Ablation.py
```

## Dependency Figures

Generates the plots visualizing the relationship between regret, query complexity, and problem parameters.

```bash
python Experiments\DependencyFigures.py
```

## License and Citations

This code is part of the The Minimax Complexity of Preference-Based Decision Making in Multi-Objective Reinforcement Learning research project. If you use this code or the generated figures, please cite the original paper.

