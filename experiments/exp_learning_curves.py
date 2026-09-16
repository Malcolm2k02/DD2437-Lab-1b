import numpy as np
import matplotlib.pyplot as plt

from src.data_setup import generate_data, subsample_data
from src.mlp import train_mlp


# ============================================================
# Experiment settings
# ============================================================

Nhidden = 10
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
# Run each sampling scenario
# ============================================================

for scenario in scenarios:

    print("\n" + "=" * 60)
    print(f"SCENARIO: {scenario}")
    print("=" * 60)

    # --------------------------------------------------------
    # Split data into training and validation sets
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


    # --------------------------------------------------------
    # Train MLP
    # --------------------------------------------------------

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
        Nhidden=Nhidden,
        epochs=epochs,
        eta=eta,
        alpha=alpha,
        val_patterns=val_patterns,
        val_targets=val_targets
    )


    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print(
        f"Final training MSE:   {train_errors[-1]:.5f}"
    )

    print(
        f"Final validation MSE: {val_errors[-1]:.5f}"
    )

    print(
        f"Final training misclassification: "
        f"{train_misclass[-1] * 100:.2f}%"
    )

    print(
        f"Final validation misclassification: "
        f"{val_misclass[-1] * 100:.2f}%"
    )


    # ========================================================
    # Plot 1: MSE learning curves
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_errors,
        label="Training MSE"
    )

    plt.plot(
        val_errors,
        label="Validation MSE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.title(
        f"Learning Curves - {scenario} "
        f"({Nhidden} hidden nodes)"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


    # ========================================================
    # Plot 2: Misclassification learning curves
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        np.array(train_misclass) * 100,
        label="Training"
    )

    plt.plot(
        np.array(val_misclass) * 100,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Misclassification Ratio (%)")

    plt.title(
        f"Classification Error - {scenario} "
        f"({Nhidden} hidden nodes)"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()