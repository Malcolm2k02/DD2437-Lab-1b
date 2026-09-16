import numpy as np
import matplotlib.pyplot as plt

from src.data_setup import generate_data, subsample_data
from src.mlp import train_mlp


# ============================================================
# Experiment settings
# ============================================================

hidden_sizes = range(1, 21)
n_runs = 20

epochs = 1000
eta = 0.01
alpha = 0.9

scenarios = [
    "25_each",
    "50_A",
    "A_20_80"
]


# ============================================================
# Generate ONE dataset
# ============================================================

patterns, targets, classA, classB = generate_data(ndata=100)


# ============================================================
# Run experiment for each subsampling scenario
# ============================================================

for scenario in scenarios:

    print("\n" + "=" * 60)
    print(f"SCENARIO: {scenario}")
    print("=" * 60)

    # --------------------------------------------------------
    # Create ONE fixed train/validation split
    # --------------------------------------------------------

    (
        train_patterns,
        train_targets,
        val_patterns,
        val_targets
    ) = subsample_data(
        classA,
        classB,
        scenario
    )

    print("Training samples:", train_patterns.shape[1])
    print("Validation samples:", val_patterns.shape[1])

    # Store results for each hidden-layer size
    mean_train_mse = []
    mean_val_mse = []

    mean_train_misclass = []
    mean_val_misclass = []

    # --------------------------------------------------------
    # Test different hidden-layer sizes
    # --------------------------------------------------------

    for hidden_node in hidden_sizes:

        train_mse_runs = []
        val_mse_runs = []

        train_misclass_runs = []
        val_misclass_runs = []

        # ----------------------------------------------------
        # Multiple runs because initial weights are random
        # ----------------------------------------------------

        for run in range(n_runs):

            (
                W,
                V,
                train_errors,
                val_errors,
                train_misclass,
                val_misclass
            ) = train_mlp(
                train_patterns,
                train_targets,
                Nhidden=hidden_node,
                epochs=epochs,
                eta=eta,
                alpha=alpha,
                val_patterns=val_patterns,
                val_targets=val_targets
            )

            # Final error after training
            train_mse_runs.append(train_errors[-1])
            val_mse_runs.append(val_errors[-1])

            # Final misclassification ratio
            train_misclass_runs.append(train_misclass[-1])
            val_misclass_runs.append(val_misclass[-1])

        # ----------------------------------------------------
        # Average over all random initializations
        # ----------------------------------------------------

        train_mse = np.mean(train_mse_runs)
        val_mse = np.mean(val_mse_runs)

        train_wrong = np.mean(train_misclass_runs)
        val_wrong = np.mean(val_misclass_runs)

        mean_train_mse.append(train_mse)
        mean_val_mse.append(val_mse)

        mean_train_misclass.append(train_wrong)
        mean_val_misclass.append(val_wrong)

        print(
            f"Hidden nodes: {hidden_node:2d} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Train wrong: {train_wrong * 100:5.2f}% | "
            f"Val wrong: {val_wrong * 100:5.2f}%"
        )

    # ========================================================
    # Find hidden-layer size with lowest validation MSE
    # ========================================================

    best_index = np.argmin(mean_val_mse)
    best_hidden = list(hidden_sizes)[best_index]

    print("\nBest validation performance:")
    print(f"Hidden nodes: {best_hidden}")
    print(f"Validation MSE: {mean_val_mse[best_index]:.5f}")
    print(
        f"Validation misclassification: "
        f"{mean_val_misclass[best_index] * 100:.2f}%"
    )

    # ========================================================
    # Plot 1: Training vs Validation MSE
    # ========================================================

    plt.figure()

    plt.plot(
        hidden_sizes,
        mean_train_mse,
        marker="o",
        label="Training MSE"
    )

    plt.plot(
        hidden_sizes,
        mean_val_mse,
        marker="o",
        label="Validation MSE"
    )

    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Mean Final MSE")

    plt.title(
        f"Training vs Validation MSE - {scenario}"
    )

    plt.xticks(list(hidden_sizes))
    plt.legend()
    plt.grid()

    plt.show()

    # ========================================================
    # Plot 2: Training vs Validation Misclassification
    # ========================================================

    plt.figure()

    plt.plot(
        hidden_sizes,
        np.array(mean_train_misclass) * 100,
        marker="o",
        label="Training"
    )

    plt.plot(
        hidden_sizes,
        np.array(mean_val_misclass) * 100,
        marker="o",
        label="Validation"
    )

    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Misclassification Ratio (%)")

    plt.title(
        f"Training vs Validation Classification - {scenario}"
    )

    plt.xticks(list(hidden_sizes))
    plt.legend()
    plt.grid()

    plt.show()