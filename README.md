# Surrogate-Sensitivity-Optimization

A Python-based benchmarking framework designed to integrate surrogate models, sampling techniques, sensitivity analysis, and optimization methods. Developed for testing optimization strategies or exploring mathematical functions with rapid prototyping features for diverse optimization tasks.

---

## Features

- **Surrogate Modeling**:
  - Supported models: Kriging (KRG), Radial Basis Function (RBF).
  - Configurable training and prediction workflows.

- **Sampling Methods**:
  - Methods include Uniform, Latin Hypercube Sampling (LHS), Sobol, and Gaussian distributions.

- **Sensitivity Analysis**:
  - Includes Morris and Sobol sensitivity analysis techniques.
  - Plots and reports generated for visualizing variable importance.

- **Optimization**:
  - Supports gradient-based (SLSQP) and gradient-free (NSGA-II) optimization methods.

- **Test Functions**:
  - Includes classical optimization problems such as Rosenbrock, Rastrigin, Branin, Himmelblau, and Ackley.

- **Visualization**:
  - Automated reporting of results through comparison plots and sensitivity charts.

---

## Installation and Requirements

### Required Packages

This library relies on Python 3.10+ and the following dependencies:

```bash
pip install matplotlib numpy pandas scikit-learn scipy pymoo smt SALib
```

### Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/surrogate-sensitivity-optimization.git
cd surrogate-sensitivity-optimization
```

---

## Usage

To run a benchmarking experiment, provide a configuration file (`.cfg`). Here's an example workflow:

### Configuration File

Create a `config.cfg` file with the following structure:

```ini
[Experiment]
model_type = kriging
test_function = rosenbrock
sensitivity_method = morris
optimizer = gradient_based
output_dir = outputs

[Sampling]
bounds = [[-2, 2], [-1, 3]]
n_samples = 100
sampler_train = lhs
sampler_test = lhs
test_to_train_ratio = 0.10
```

### Running the Experiment

Run the script with:

```bash
python script.py config.cfg
```

---

## Components

### 1. **Test Functions**

Contains classical optimization functions like:
- Rosenbrock
- Branin
- Ackley
- Himmelblau
- Rastrigin

### 2. **Sampling**

Supported sampling techniques:
- Uniform
- Latin Hypercube Sampling (LHS)
- Sobol (requires power-of-2 sample sizes)
- Gaussian (scaled to problem bounds)

### 3. **Surrogate Models**

Train and use surrogate models:
- **Kriging (KRG)**
- **Radial Basis Function (RBF)**

### 4. **Sensitivity Analysis**

Perform sensitivity analysis using:
- **Morris Method** (Elementary Effects, visualized as μ* and σ)
- **Sobol Analysis** (First-order, Total-order indices)

### 5. **Optimization**

Supports:
- **Gradient-Based Optimization**: SLSQP for constrained minimization.
- **Gradient-Free Optimization**: NSGA-II for robust search.

### 6. **Visualization**

Generate accurate visual reports:
- Training/Testing sample distribution.
- Predicted vs. actual values.
- Residual plots.
- Sensitivity analysis results.

---

## Example Outputs

1. **Sampled Points** (Training vs Testing Data)
2. **Predicted vs Actual** with residual plots.
3. **Sensitivity Analysis Outputs** (Morris or Sobol indices).

### Sample Plot: "Predicted vs Actual"
![Predicted vs Actual Example](assets/predicted_vs_actual.png)

---

## Contribution

Contributions, issues, and suggestions are welcome! Feel free to open a [pull request](https://github.com/<your-username>/surrogate-sensitivity-optimization/pulls) or [issue](https://github.com/<your-username>/surrogate-sensitivity-optimization/issues) to improve this repository.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

This repository is built using powerful libraries such as:
- [SMT (Surrogate Modeling Toolbox)](https://smt.readthedocs.io/en/latest/)
- [SALib](https://salib.readthedocs.io/en/latest/)
- [Pymoo](https://pymoo.org/)
