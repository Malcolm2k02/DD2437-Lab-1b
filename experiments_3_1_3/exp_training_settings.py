import numpy as np
import matplotlib.pyplot as plt

from src.data_setup import (
    generate_gaussian_data,
    subsample_gaussian
)

from src.mlp import train_mlp_function, forward_pass


# ============================================================
# Settings
# ============================================================

Nhidden = 100

training_fractions = [
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8
]

n_runs = 20

epochs = 1000
eta = 0.01
alpha = 0.9


# ============================================================
# Data
# ============================================================

patterns, targets, x, y, xx, yy, z = generate_gaussian_data()


mean_errors = []
std_errors = []


# ============================================================
# Experiment
# ============================================================

for fraction in training_fractions:

    errors = []

    print(
        f"\nTraining fraction: {fraction * 100:.0f}%"
    )

    for run in range(n_runs):

        train_patterns, train_targets, _ = subsample_gaussian(
            patterns,
            targets,
            fraction
        )

        W, V, _ = train_mlp_function(
            train_patterns,
            train_targets,
            Nhidden=Nhidden,
            epochs=epochs,
            eta=eta,
            alpha=alpha,
            animate=False
        )

        # Evaluate on ALL points
        _, _, _, _, _, out = forward_pass(
            patterns,
            W,
            V
        )

        full_mse = np.mean(
            (out - targets) ** 2
        )

        errors.append(full_mse)


    mean_error = np.mean(errors)
    std_error = np.std(errors)

    mean_errors.append(mean_error)
    std_errors.append(std_error)

    print(
        f"Full-data MSE: "
        f"{mean_error:.6f} +/- {std_error:.6f}"
    )


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.errorbar(
    np.array(training_fractions) * 100,
    mean_errors,
    yerr=std_errors,
    marker="o",
    capsize=4
)

plt.xlabel("Training Data (%)")
plt.ylabel("MSE on All 441 Samples")

plt.title(
    f"Generalisation vs Training Data "
    f"({Nhidden} Hidden Nodes)"
)

plt.grid()
plt.tight_layout()

plt.show()

# with 10 hidden layers
"""| Training data | Approx. training samples |         Full-data MSE |
| ------------: | -----------------------: | --------------------: |
|           20% |                       88 |     0.03239 ± 0.01452 |
|           30% |                      132 |     0.01417 ± 0.01255 |
|           40% |                      176 |     0.00362 ± 0.00254 |
|           50% |                      220 |     0.00296 ± 0.00239 |
|           60% |                      264 |     0.00180 ± 0.00080 |
|           70% |                      308 |     0.00161 ± 0.00091 |
|           80% |                      352 | **0.00146 ± 0.00075** |
"""
# With 25 hidden layers
"""Training fraction: 20%
Full-data MSE: 0.011667 +/- 0.010201

Training fraction: 30%
Full-data MSE: 0.004226 +/- 0.003425

Training fraction: 40%
Full-data MSE: 0.002187 +/- 0.002069

Training fraction: 50%
Full-data MSE: 0.001070 +/- 0.000790

Training fraction: 60%
Full-data MSE: 0.000872 +/- 0.000349

Training fraction: 70%
Full-data MSE: 0.000704 +/- 0.000259

Training fraction: 80%
Full-data MSE: 0.000923 +/- 0.000491"""

# 10 hidden units with varying eta

# eta = 0.05:
