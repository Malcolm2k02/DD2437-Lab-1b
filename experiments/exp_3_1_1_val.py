import numpy as np

from src.data_setup import (
    generate_data,
    subsample_data
)

from src.mlp import (
    train_mlp,
    plot_train_validation_errors,
    plot_train_validation_misclassification,
    plot_decision_boundary
)


# ============================================================
# Generate ONE dataset
# ============================================================

patterns, targets, classA, classB = generate_data(
    ndata=100
)


# ============================================================
# Choose scenario
# ============================================================

scenarios = ["25_each", "50_A", "A_20_80"]
for scenario in scenarios:

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


    # ============================================================
    # Train network
    # ============================================================

    W, V, train_errors, val_errors, train_misclass, val_misclass = train_mlp(
        train_patterns,
        train_targets,
        Nhidden=5,
        epochs=1000,
        eta=0.01,
        alpha=0.9,
        val_patterns=val_patterns,
        val_targets=val_targets
    )


    # ============================================================
    # Plot MSE
    # ============================================================

    plot_train_validation_errors(
        train_errors,
        val_errors,
        title=f"MSE - {scenario}"
    )


    # ============================================================
    # Plot misclassification
    # ============================================================

    plot_train_validation_misclassification(
        train_misclass,
        val_misclass,
        title=f"Misclassification - {scenario}"
    )