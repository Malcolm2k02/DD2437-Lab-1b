import numpy as np
import matplotlib.pyplot as plt

from src.data_setup import generate_data, subsample_data
from src.mlp import train_mlp, train_mlp_sequential


# ============================================================
# Settings
# ============================================================

Nhidden = 10
epochs = 1000
eta = 0.01
alpha = 0.9

n_runs = 20

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
# Compare batch and sequential learning
# ============================================================

for scenario in scenarios:

    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario}")
    print("=" * 70)

    # --------------------------------------------------------
    # ONE fixed train / validation split
    # Used for ALL runs and BOTH methods
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

    # ========================================================
    # Store final results from every run
    # ========================================================

    batch_train_mse_runs = []
    batch_val_mse_runs = []
    batch_train_misclass_runs = []
    batch_val_misclass_runs = []

    seq_train_mse_runs = []
    seq_val_mse_runs = []
    seq_train_misclass_runs = []
    seq_val_misclass_runs = []


    # ========================================================
    # Store learning curves
    #
    # We average these later so we can compare how batch and
    # sequential learning behave throughout training.
    # ========================================================

    batch_val_curves = []
    seq_val_curves = []

    batch_val_misclass_curves = []
    seq_val_misclass_curves = []


    # ========================================================
    # Run both methods multiple times
    # ========================================================

    for run in range(n_runs):

        print(f"Run {run + 1}/{n_runs}", end="\r")

        # ====================================================
        # BATCH
        # ====================================================

        (
            W_batch,
            V_batch,
            batch_train_errors,
            batch_val_errors,
            batch_train_misclass,
            batch_val_misclass
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

        # Final performance
        batch_train_mse_runs.append(
            batch_train_errors[-1]
        )

        batch_val_mse_runs.append(
            batch_val_errors[-1]
        )

        batch_train_misclass_runs.append(
            batch_train_misclass[-1]
        )

        batch_val_misclass_runs.append(
            batch_val_misclass[-1]
        )

        # Complete validation learning curves
        batch_val_curves.append(
            batch_val_errors
        )

        batch_val_misclass_curves.append(
            batch_val_misclass
        )


        # ====================================================
        # SEQUENTIAL
        # ====================================================

        (
            W_seq,
            V_seq,
            seq_train_errors,
            seq_val_errors,
            seq_train_misclass,
            seq_val_misclass
        ) = train_mlp_sequential(
            train_patterns,
            train_targets,
            Nhidden=Nhidden,
            epochs=epochs,
            eta=eta,
            alpha=alpha,
            val_patterns=val_patterns,
            val_targets=val_targets
        )

        # Final performance
        seq_train_mse_runs.append(
            seq_train_errors[-1]
        )

        seq_val_mse_runs.append(
            seq_val_errors[-1]
        )

        seq_train_misclass_runs.append(
            seq_train_misclass[-1]
        )

        seq_val_misclass_runs.append(
            seq_val_misclass[-1]
        )

        # Complete validation learning curves
        seq_val_curves.append(
            seq_val_errors
        )

        seq_val_misclass_curves.append(
            seq_val_misclass
        )


    print()


    # ========================================================
    # Convert to NumPy arrays
    # ========================================================

    batch_val_curves = np.array(batch_val_curves)
    seq_val_curves = np.array(seq_val_curves)

    batch_val_misclass_curves = np.array(
        batch_val_misclass_curves
    )

    seq_val_misclass_curves = np.array(
        seq_val_misclass_curves
    )


    # ========================================================
    # Mean learning curves
    # ========================================================

    mean_batch_val_curve = np.mean(
        batch_val_curves,
        axis=0
    )

    mean_seq_val_curve = np.mean(
        seq_val_curves,
        axis=0
    )

    mean_batch_val_misclass_curve = np.mean(
        batch_val_misclass_curves,
        axis=0
    )

    mean_seq_val_misclass_curve = np.mean(
        seq_val_misclass_curves,
        axis=0
    )


    # ========================================================
    # Final mean + standard deviation
    # ========================================================

    batch_train_mse_mean = np.mean(
        batch_train_mse_runs
    )

    batch_val_mse_mean = np.mean(
        batch_val_mse_runs
    )

    batch_val_mse_std = np.std(
        batch_val_mse_runs
    )

    batch_train_wrong_mean = np.mean(
        batch_train_misclass_runs
    )

    batch_val_wrong_mean = np.mean(
        batch_val_misclass_runs
    )

    batch_val_wrong_std = np.std(
        batch_val_misclass_runs
    )


    seq_train_mse_mean = np.mean(
        seq_train_mse_runs
    )

    seq_val_mse_mean = np.mean(
        seq_val_mse_runs
    )

    seq_val_mse_std = np.std(
        seq_val_mse_runs
    )

    seq_train_wrong_mean = np.mean(
        seq_train_misclass_runs
    )

    seq_val_wrong_mean = np.mean(
        seq_val_misclass_runs
    )

    seq_val_wrong_std = np.std(
        seq_val_misclass_runs
    )


    # ========================================================
    # Print results
    # ========================================================

    print("\nBATCH (mean over 20 runs)")
    print(
        f"Training MSE:   "
        f"{batch_train_mse_mean:.5f}"
    )

    print(
        f"Validation MSE: "
        f"{batch_val_mse_mean:.5f} "
        f"+/- {batch_val_mse_std:.5f}"
    )

    print(
        f"Training misclassification: "
        f"{batch_train_wrong_mean * 100:.2f}%"
    )

    print(
        f"Validation misclassification: "
        f"{batch_val_wrong_mean * 100:.2f}% "
        f"+/- {batch_val_wrong_std * 100:.2f}%"
    )


    print("\nSEQUENTIAL (mean over 20 runs)")
    print(
        f"Training MSE:   "
        f"{seq_train_mse_mean:.5f}"
    )

    print(
        f"Validation MSE: "
        f"{seq_val_mse_mean:.5f} "
        f"+/- {seq_val_mse_std:.5f}"
    )

    print(
        f"Training misclassification: "
        f"{seq_train_wrong_mean * 100:.2f}%"
    )

    print(
        f"Validation misclassification: "
        f"{seq_val_wrong_mean * 100:.2f}% "
        f"+/- {seq_val_wrong_std * 100:.2f}%"
    )


    # ========================================================
    # Plot 1:
    # Mean validation MSE over epochs
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        mean_batch_val_curve,
        label="Batch"
    )

    plt.plot(
        mean_seq_val_curve,
        label="Sequential"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Validation MSE")

    plt.title(
        f"Batch vs Sequential Validation MSE - {scenario}"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.show()


    # ========================================================
    # Plot 2:
    # Mean validation misclassification over epochs
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        mean_batch_val_misclass_curve * 100,
        label="Batch"
    )

    plt.plot(
        mean_seq_val_misclass_curve * 100,
        label="Sequential"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Validation Misclassification (%)")

    plt.title(
        f"Batch vs Sequential Classification - {scenario}"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.show()

    # Results:
"""
    ======================================================================
SCENARIO: 25_each
======================================================================
Training samples: 150
Validation samples: 50
Run 20/20

BATCH (mean over 20 runs)
Training MSE:   0.18597
Validation MSE: 0.24018 +/- 0.29170
Training misclassification: 6.27%
Validation misclassification: 7.60% +/- 11.20%

SEQUENTIAL (mean over 20 runs)
Training MSE:   0.26550
Validation MSE: 0.33632 +/- 0.31132
Training misclassification: 8.83%
Validation misclassification: 11.20% +/- 12.22%

======================================================================
SCENARIO: 50_A
======================================================================
Training samples: 150
Validation samples: 50
Run 20/20

BATCH (mean over 20 runs)
Training MSE:   0.09518
Validation MSE: 0.23232 +/- 0.20976
Training misclassification: 2.40%
Validation misclassification: 8.90% +/- 10.38%

SEQUENTIAL (mean over 20 runs)
Training MSE:   0.07488
Validation MSE: 0.20208 +/- 0.07434
Training misclassification: 1.47%
Validation misclassification: 7.10% +/- 1.84%

======================================================================
SCENARIO: A_20_80
======================================================================
Training samples: 150
Validation samples: 50
Run 20/20

BATCH (mean over 20 runs)
Training MSE:   0.29648
Validation MSE: 3.21379 +/- 0.00007
Training misclassification: 8.00%
Validation misclassification: 80.00% +/- 0.00%

SEQUENTIAL (mean over 20 runs)
Training MSE:   0.29673
Validation MSE: 3.21407 +/- 0.00091
Training misclassification: 8.00%
Validation misclassification: 80.00% +/- 0.00%
"""