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

hidden_sizes = [1, 2, 3, 4, 5, 8, 10, 15, 20, 25]

training_fraction = 0.5

n_runs = 20

epochs = 1000
eta = 0.01
alpha = 0.9


# ============================================================
# Generate complete dataset
# ============================================================

patterns, targets, x, y, xx, yy, z = generate_gaussian_data()


# ============================================================
# Store results
# ============================================================

mean_errors = []
std_errors = []


# ============================================================
# Test hidden-layer sizes
# ============================================================

for Nhidden in hidden_sizes:

    errors = []

    print(f"\nHidden nodes: {Nhidden}")

    for run in range(n_runs):

        # ----------------------------------------------------
        # Random training subset
        # ----------------------------------------------------

        train_patterns, train_targets, _ = subsample_gaussian(
            patterns,
            targets,
            training_fraction
        )


        # ----------------------------------------------------
        # Train ONLY on subset
        # ----------------------------------------------------

        W, V, train_errors = train_mlp_function(
            train_patterns,
            train_targets,
            Nhidden=Nhidden,
            epochs=epochs,
            eta=eta,
            alpha=alpha,
            animate=False
        )


        # ----------------------------------------------------
        # Evaluate on ALL 441 samples
        # ----------------------------------------------------

        _, _, _, _, _, out = forward_pass(
            patterns,
            W,
            V
        )

        full_mse = np.mean(
            (out - targets) ** 2
        )

        errors.append(full_mse)


    # ========================================================
    # Mean + standard deviation over runs
    # ========================================================

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
    hidden_sizes,
    mean_errors,
    yerr=std_errors,
    marker="o",
    capsize=4
)

plt.xlabel("Number of Hidden Nodes")
plt.ylabel("MSE on All 441 Samples")

plt.title(
    "Generalisation vs Hidden-Layer Size "
    "(50% Training Data)"
)

plt.grid()
plt.tight_layout()

plt.show()

"""Hidden nodes: 1
Full-data MSE: 0.066473 +/- 0.001248

Hidden nodes: 2
Full-data MSE: 0.049306 +/- 0.012904

Hidden nodes: 3
Full-data MSE: 0.021038 +/- 0.020063

Hidden nodes: 4
Full-data MSE: 0.016675 +/- 0.017934

Hidden nodes: 5
Full-data MSE: 0.010289 +/- 0.017045

Hidden nodes: 8
Full-data MSE: 0.003217 +/- 0.003129

Hidden nodes: 10
Full-data MSE: 0.002148 +/- 0.001822

Hidden nodes: 15
Full-data MSE: 0.001723 +/- 0.000767

Hidden nodes: 20
Full-data MSE: 0.001330 +/- 0.001449

Hidden nodes: 25
Full-data MSE: 0.001111 +/- 0.000635"""