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

Nhidden = 10
training_fraction = 0.5

learning_rates = [
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.5,
    1.0
]

alpha = 0.9
epochs = 1000
n_runs = 20

# Training MSE used to measure convergence speed
convergence_threshold = 0.005


# ============================================================
# Generate complete Gaussian dataset
# ============================================================

patterns, targets, x, y, xx, yy, z = generate_gaussian_data()

ndata = patterns.shape[1]
nsamp = int(training_fraction * ndata)

print("==========================================")
print("LEARNING RATE EXPERIMENT")
print("==========================================")
print(f"Total samples: {ndata}")
print(f"Training samples: {nsamp}")
print(f"Training fraction: {training_fraction * 100:.0f}%")
print(f"Hidden nodes: {Nhidden}")
print(f"Momentum alpha: {alpha}")
print(f"Maximum epochs: {epochs}")
print(f"Runs per learning rate: {n_runs}")
print(f"Convergence threshold: {convergence_threshold}")
print("==========================================")


# ============================================================
# Storage
# ============================================================

mean_full_mse = []
std_full_mse = []

mean_train_mse = []
std_train_mse = []

mean_convergence_epochs = []
std_convergence_epochs = []

all_mean_curves = []

# Save one trained model for each learning rate
representative_models = {}


# ============================================================
# Experiment
# ============================================================

for eta in learning_rates:

    print(f"\nLearning rate: {eta}")

    full_mse_runs = []
    train_mse_runs = []
    convergence_runs = []
    curves = []

    for run in range(n_runs):

        # ----------------------------------------------------
        # Generate training subset
        # ----------------------------------------------------

        train_patterns, train_targets, _ = subsample_gaussian(
            patterns,
            targets,
            training_fraction
        )


        # ----------------------------------------------------
        # Train network
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

        curves.append(train_errors)


        # ----------------------------------------------------
        # Final training MSE
        # ----------------------------------------------------

        _, _, _, _, _, train_out = forward_pass(
            train_patterns,
            W,
            V
        )

        final_train_mse = np.mean(
            (train_out - train_targets) ** 2
        )

        train_mse_runs.append(final_train_mse)


        # ----------------------------------------------------
        # Generalisation MSE
        #
        # Assignment requests error over ALL available points
        # (training + unseen points)
        # ----------------------------------------------------

        _, _, _, _, _, full_out = forward_pass(
            patterns,
            W,
            V
        )

        final_full_mse = np.mean(
            (full_out - targets) ** 2
        )

        full_mse_runs.append(final_full_mse)


        # ----------------------------------------------------
        # Find first epoch where training MSE falls below
        # convergence threshold
        # ----------------------------------------------------

        convergence_epoch = np.nan

        for epoch_index, mse in enumerate(train_errors):

            if mse < convergence_threshold:
                convergence_epoch = epoch_index + 1
                break

        convergence_runs.append(convergence_epoch)


        # Save first model for visualization
        if run == 0:
            representative_models[eta] = (W, V)


    # ========================================================
    # Convert to NumPy
    # ========================================================

    full_mse_runs = np.array(full_mse_runs)
    train_mse_runs = np.array(train_mse_runs)
    curves = np.array(curves)

    convergence_array = np.array(
        convergence_runs,
        dtype=float
    )


    # ========================================================
    # Statistics
    # ========================================================

    full_mean = np.mean(full_mse_runs)
    full_std = np.std(full_mse_runs)

    train_mean = np.mean(train_mse_runs)
    train_std = np.std(train_mse_runs)

    mean_curve = np.mean(curves, axis=0)

    all_mean_curves.append(mean_curve)

    mean_full_mse.append(full_mean)
    std_full_mse.append(full_std)

    mean_train_mse.append(train_mean)
    std_train_mse.append(train_std)


    # ========================================================
    # Convergence statistics
    # ========================================================

    successful = convergence_array[
        ~np.isnan(convergence_array)
    ]

    if len(successful) > 0:

        conv_mean = np.mean(successful)
        conv_std = np.std(successful)

    else:

        conv_mean = np.nan
        conv_std = np.nan

    mean_convergence_epochs.append(conv_mean)
    std_convergence_epochs.append(conv_std)


    # ========================================================
    # Print results
    # ========================================================

    print(
        f"Training MSE: "
        f"{train_mean:.6f} +/- {train_std:.6f}"
    )

    print(
        f"Full-data MSE: "
        f"{full_mean:.6f} +/- {full_std:.6f}"
    )

    if len(successful) > 0:

        print(
            f"Epochs to MSE < {convergence_threshold}: "
            f"{conv_mean:.1f} +/- {conv_std:.1f}"
        )

        print(
            f"Reached threshold: "
            f"{len(successful)}/{n_runs} runs"
        )

    else:

        print(
            f"Threshold not reached in any run."
        )


# ============================================================
# Convert result lists
# ============================================================

mean_full_mse = np.array(mean_full_mse)
std_full_mse = np.array(std_full_mse)

mean_train_mse = np.array(mean_train_mse)
std_train_mse = np.array(std_train_mse)

mean_convergence_epochs = np.array(
    mean_convergence_epochs
)

std_convergence_epochs = np.array(
    std_convergence_epochs
)


# ============================================================
# Summary table
# ============================================================

print("\n")
print("==============================================================")
print("SUMMARY")
print("==============================================================")

print(
    f"{'eta':<10}"
    f"{'Train MSE':<20}"
    f"{'Full MSE':<20}"
    f"{'Conv. epochs':<20}"
)

print("-" * 70)

for i, eta in enumerate(learning_rates):

    train_text = (
        f"{mean_train_mse[i]:.6f} "
        f"+/- {std_train_mse[i]:.6f}"
    )

    full_text = (
        f"{mean_full_mse[i]:.6f} "
        f"+/- {std_full_mse[i]:.6f}"
    )

    if np.isnan(mean_convergence_epochs[i]):
        conv_text = "Not reached"
    else:
        conv_text = (
            f"{mean_convergence_epochs[i]:.1f} "
            f"+/- {std_convergence_epochs[i]:.1f}"
        )

    print(
        f"{eta:<10}"
        f"{train_text:<20}"
        f"{full_text:<20}"
        f"{conv_text:<20}"
    )


# ============================================================
# Plot 1:
# Mean training MSE vs epoch
# ============================================================

plt.figure(figsize=(9, 6))

for eta, curve in zip(
    learning_rates,
    all_mean_curves
):

    plt.plot(
        range(1, epochs + 1),
        curve,
        label=f"eta = {eta}"
    )


plt.axhline(
    y=convergence_threshold,
    linestyle="--",
    label=f"Threshold = {convergence_threshold}"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Training MSE")

plt.title(
    "Convergence for Different Learning Rates"
)

plt.yscale("log")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# Plot 2:
# Final full-data MSE
# ============================================================

plt.figure(figsize=(8, 5))

plt.errorbar(
    learning_rates,
    mean_full_mse,
    yerr=std_full_mse,
    marker="o",
    capsize=4
)

plt.xscale("log")

plt.xlabel("Learning Rate")
plt.ylabel("Full-data MSE")

plt.title(
    "Generalisation Performance vs Learning Rate"
)

plt.grid()
plt.tight_layout()

plt.show()


# ============================================================
# Plot 3:
# Convergence epochs
# ============================================================

valid = ~np.isnan(mean_convergence_epochs)

plt.figure(figsize=(8, 5))

plt.errorbar(
    np.array(learning_rates)[valid],
    mean_convergence_epochs[valid],
    yerr=std_convergence_epochs[valid],
    marker="o",
    capsize=4
)

plt.xscale("log")

plt.xlabel("Learning Rate")
plt.ylabel(
    f"Epochs to Training MSE < {convergence_threshold}"
)

plt.title(
    "Convergence Speed vs Learning Rate"
)

plt.grid()
plt.tight_layout()

plt.show()


# ============================================================
# Find learning rate with lowest mean full-data MSE
# ============================================================

best_index = np.argmin(mean_full_mse)

best_eta = learning_rates[best_index]

print("\n==========================================")
print("LOWEST MEAN FULL-DATA MSE")
print("==========================================")

print(f"Learning rate: {best_eta}")

print(
    f"Full-data MSE: "
    f"{mean_full_mse[best_index]:.6f} "
    f"+/- {std_full_mse[best_index]:.6f}"
)


# ============================================================
# Plot representative 3D approximation
# ============================================================

W_best, V_best = representative_models[best_eta]

_, _, _, _, _, best_out = forward_pass(
    patterns,
    W_best,
    V_best
)

zz = best_out.reshape(xx.shape)


fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot_surface(
    xx,
    yy,
    zz
)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-0.7, 0.7)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Output")

ax.set_title(
    f"Gaussian Approximation "
    f"(eta = {best_eta})"
)

plt.tight_layout()
plt.show()

# 10 hidden nodes
"""
| Learning rate η |            Training MSE |           Full-data MSE | Epochs to MSE < 0.005 | Reached threshold |
| --------------: | ----------------------: | ----------------------: | --------------------: | ----------------: |
|           0.001 |     0.065614 ± 0.004364 |     0.066830 ± 0.000642 |           Not reached |              0/20 |
|           0.005 |     0.022017 ± 0.016250 |     0.023468 ± 0.016866 |          874.7 ± 71.5 |              3/20 |
|           0.010 |     0.002327 ± 0.001638 |     0.002492 ± 0.001798 |         643.1 ± 106.0 |             18/20 |
|           0.020 |     0.001256 ± 0.000700 |     0.001418 ± 0.000802 |         390.6 ± 101.1 |             20/20 |
|           0.050 |     0.000408 ± 0.000290 |     0.000471 ± 0.000317 |          226.2 ± 53.8 |             20/20 |
|           0.100 |     0.000244 ± 0.000302 |     0.000296 ± 0.000393 |          150.6 ± 36.0 |             20/20 |
|           0.150 |     0.000146 ± 0.000035 |     0.000193 ± 0.000043 |          119.5 ± 30.9 |             20/20 |
|       **0.200** | **0.000141 ± 0.000032** | **0.000172 ± 0.000041** |      **111.3 ± 24.0** |         **20/20** |
|           0.500 |     0.669521 ± 0.019229 |     0.666663 ± 0.000000 |           Not reached |              0/20 |
|           1.000 |     0.672923 ± 0.133780 |     0.681095 ± 0.137483 |           Not reached |              0/20 |
"""

# with 25 hidden nodes

"""| Learning rate η |            Training MSE |           Full-data MSE | Epochs to MSE < 0.005 | Reached threshold |
| --------------: | ----------------------: | ----------------------: | --------------------: | ----------------: |
|           0.005 |     0.006169 ± 0.004432 |     0.006721 ± 0.004791 |          814.2 ± 87.6 |              6/20 |
|           0.010 |     0.001188 ± 0.000663 |     0.001360 ± 0.000784 |         580.9 ± 158.0 |             20/20 |
|           0.020 |     0.000507 ± 0.000246 |     0.000588 ± 0.000344 |          306.6 ± 98.3 |             20/20 |
|           0.050 |     0.000238 ± 0.000098 |     0.000284 ± 0.000114 |          194.5 ± 43.0 |             20/20 |
|           0.100 |     0.000170 ± 0.000076 |     0.000209 ± 0.000076 |      **134.7 ± 38.7** |             20/20 |
|       **0.150** | **0.000141 ± 0.000049** | **0.000172 ± 0.000058** |          144.1 ± 92.4 |             20/20 |
|           0.200 |     0.033169 ± 0.143959 |     0.033496 ± 0.145259 |         108.9 ± 32.2* |             19/20 |
|           0.500 |     0.636494 ± 0.074209 |     0.637162 ± 0.070623 |           Not reached |              0/20 |
|           1.000 |     0.662977 ± 0.115181 |     0.669809 ± 0.118281 |           Not reached |              0/20 |
""" 