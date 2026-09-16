import numpy as np
import matplotlib.pyplot as plt

from src.data_setup import generate_data, subsample_data
from src.mlp import train_mlp, forward_pass


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
# Function for plotting one decision boundary
# ============================================================

def plot_boundary(
    train_patterns,
    train_targets,
    val_patterns,
    val_targets,
    W,
    V,
    scenario
):

    # --------------------------------------------------------
    # Create grid covering the complete dataset
    # --------------------------------------------------------

    all_patterns = np.concatenate(
        (train_patterns, val_patterns),
        axis=1
    )

    x_min = all_patterns[0, :].min() - 0.5
    x_max = all_patterns[0, :].max() + 0.5

    y_min = all_patterns[1, :].min() - 0.5
    y_max = all_patterns[1, :].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    # Turn grid into same format as our input patterns
    grid_patterns = np.vstack([
        xx.ravel(),
        yy.ravel()
    ])

    # --------------------------------------------------------
    # Run every grid point through the trained network
    # --------------------------------------------------------

    _, _, _, _, _, grid_output = forward_pass(
        grid_patterns,
        W,
        V
    )

    grid_output = grid_output.reshape(xx.shape)


    # ========================================================
    # Separate Class A and B samples
    # ========================================================

    train_A = train_targets.flatten() == 1
    train_B = train_targets.flatten() == -1

    val_A = val_targets.flatten() == 1
    val_B = val_targets.flatten() == -1


    # ========================================================
    # Plot
    # ========================================================

    plt.figure(figsize=(8, 6))


    # --------------------------------------------------------
    # Training samples
    # --------------------------------------------------------

    plt.scatter(
        train_patterns[0, train_A],
        train_patterns[1, train_A],
        marker="o",
        label="Class A - Training"
    )

    plt.scatter(
        train_patterns[0, train_B],
        train_patterns[1, train_B],
        marker="o",
        label="Class B - Training"
    )


    # --------------------------------------------------------
    # Validation samples
    #
    # x markers make them visually different from training
    # --------------------------------------------------------

    plt.scatter(
        val_patterns[0, val_A],
        val_patterns[1, val_A],
        marker="x",
        s=80,
        linewidths=2,
        label="Class A - Validation"
    )

    plt.scatter(
        val_patterns[0, val_B],
        val_patterns[1, val_B],
        marker="x",
        s=80,
        linewidths=2,
        label="Class B - Validation"
    )


    # --------------------------------------------------------
    # Decision boundary
    #
    # output = 0 is exactly between targets -1 and +1
    # --------------------------------------------------------

    plt.contour(
        xx,
        yy,
        grid_output,
        levels=[0],
        linewidths=2
    )


    # ========================================================
    # Labels
    # ========================================================

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title(
        f"Decision Boundary - {scenario}\n"
        f"{Nhidden} Hidden Nodes"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# ============================================================
# Run experiment
# ============================================================

for scenario in scenarios:

    print("\n" + "=" * 60)
    print(f"SCENARIO: {scenario}")
    print("=" * 60)


    # --------------------------------------------------------
    # Create training / validation split
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


    # --------------------------------------------------------
    # Train network ONLY on training data
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
    # Print performance
    # --------------------------------------------------------

    print(
        f"Training MSE:   {train_errors[-1]:.5f}"
    )

    print(
        f"Validation MSE: {val_errors[-1]:.5f}"
    )

    print(
        f"Training misclassification: "
        f"{train_misclass[-1] * 100:.2f}%"
    )

    print(
        f"Validation misclassification: "
        f"{val_misclass[-1] * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Plot decision boundary
    # --------------------------------------------------------

    plot_boundary(
        train_patterns,
        train_targets,
        val_patterns,
        val_targets,
        W,
        V,
        scenario
    )