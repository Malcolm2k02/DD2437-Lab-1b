import numpy as np
import matplotlib.pyplot as plt

from src.data_setup import generate_gaussian_data
from src.mlp import train_mlp_function, forward_pass


# ============================================================
# SETTINGS
# ============================================================

hidden_sizes = [5, 10, 25, 50]

noise_stds = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2]

training_fraction = 0.5

epochs = 1000
eta = 0.01
alpha = 0.9

n_runs = 20


# ============================================================
# LOAD CLEAN DATA
# ============================================================

patterns, targets, x, y, xx, yy, z = generate_gaussian_data()

ndata = patterns.shape[1]
ntrain = int(training_fraction * ndata)


# Store results
results = {}


# ============================================================
# RUN EXPERIMENT
# ============================================================

for noise_std in noise_stds:

    print("\n" + "=" * 60)
    print(f"NOISE STD = {noise_std}")
    print("=" * 60)

    results[noise_std] = {}

    for Nhidden in hidden_sizes:

        noisy_train_errors = []
        clean_train_errors = []
        clean_val_errors = []

        for run in range(n_runs):

            # ------------------------------------------------
            # Random 50/50 train-validation split
            # ------------------------------------------------

            indices = np.random.permutation(ndata)

            train_indices = indices[:ntrain]
            val_indices = indices[ntrain:]

            train_patterns = patterns[:, train_indices]
            val_patterns = patterns[:, val_indices]

            clean_train_targets = targets[:, train_indices]
            clean_val_targets = targets[:, val_indices]

            # ------------------------------------------------
            # Add Gaussian noise ONLY to training targets
            # ------------------------------------------------

            noise = np.random.normal(
                loc=0.0,
                scale=noise_std,
                size=clean_train_targets.shape
            )

            noisy_train_targets = clean_train_targets + noise

            # ------------------------------------------------
            # Train on noisy targets
            # ------------------------------------------------

            W, V, train_errors = train_mlp_function(
                train_patterns,
                noisy_train_targets,
                Nhidden=Nhidden,
                epochs=epochs,
                eta=eta,
                alpha=alpha
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            _, _, _, _, _, train_output = forward_pass(
                train_patterns,
                W,
                V
            )

            _, _, _, _, _, val_output = forward_pass(
                val_patterns,
                W,
                V
            )

            # ------------------------------------------------
            # 1. Error against NOISY training targets
            # ------------------------------------------------

            noisy_train_mse = np.mean(
                (train_output - noisy_train_targets) ** 2
            )

            # ------------------------------------------------
            # 2. Error against CLEAN training targets
            # ------------------------------------------------

            clean_train_mse = np.mean(
                (train_output - clean_train_targets) ** 2
            )

            # ------------------------------------------------
            # 3. Error on CLEAN validation data
            # ------------------------------------------------

            clean_val_mse = np.mean(
                (val_output - clean_val_targets) ** 2
            )

            noisy_train_errors.append(noisy_train_mse)
            clean_train_errors.append(clean_train_mse)
            clean_val_errors.append(clean_val_mse)

        # ----------------------------------------------------
        # Statistics over runs
        # ----------------------------------------------------

        results[noise_std][Nhidden] = {
            "noisy_train_mean": np.mean(noisy_train_errors),
            "noisy_train_std": np.std(noisy_train_errors),
            "clean_train_mean": np.mean(clean_train_errors),
            "clean_train_std": np.std(clean_train_errors),
            "clean_val_mean": np.mean(clean_val_errors),
            "clean_val_std": np.std(clean_val_errors),
        }

        r = results[noise_std][Nhidden]

        print(
            f"Hidden nodes: {Nhidden:3d} | "
            f"Noisy train MSE: "
            f"{r['noisy_train_mean']:.6f} +/- "
            f"{r['noisy_train_std']:.6f} | "
            f"Clean train MSE: "
            f"{r['clean_train_mean']:.6f} +/- "
            f"{r['clean_train_std']:.6f} | "
            f"Clean val MSE: "
            f"{r['clean_val_mean']:.6f} +/- "
            f"{r['clean_val_std']:.6f}"
        )


# ============================================================
# PLOT: CLEAN VALIDATION ERROR
# ============================================================

plt.figure(figsize=(9, 6))

for noise_std in noise_stds:

    means = [
        results[noise_std][h]["clean_val_mean"]
        for h in hidden_sizes
    ]

    stds = [
        results[noise_std][h]["clean_val_std"]
        for h in hidden_sizes
    ]

    plt.errorbar(
        hidden_sizes,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        label=f"Noise std = {noise_std}"
    )

plt.xlabel("Number of hidden nodes")
plt.ylabel("Clean validation MSE")
plt.title("Effect of hidden-layer size with Gaussian noise")
plt.yscale("log")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# PLOT: NOISY TRAINING ERROR
# ============================================================

plt.figure(figsize=(9, 6))

for noise_std in noise_stds:

    means = [
        results[noise_std][h]["noisy_train_mean"]
        for h in hidden_sizes
    ]

    plt.plot(
        hidden_sizes,
        means,
        marker="o",
        label=f"Noise std = {noise_std}"
    )

plt.xlabel("Number of hidden nodes")
plt.ylabel("Noisy training MSE")
plt.title("Training error with Gaussian noise")
plt.yscale("log")
plt.legend()
plt.grid(True)

plt.show()


"""| Noise std. \(\sigma\) |               5 nodes |          10 nodes |              25 nodes |              50 nodes |
| --------------------: | --------------------: | ----------------: | --------------------: | --------------------: |
|              **0.00** |     0.01247 ± 0.02107 | 0.00577 ± 0.01487 |     0.00111 ± 0.00040 | **0.00086 ± 0.00033** |
|              **0.05** |     0.01738 ± 0.02008 | 0.00279 ± 0.00271 |     0.00177 ± 0.00115 | **0.00108 ± 0.00048** |
|              **0.10** |     0.01086 ± 0.01938 | 0.00511 ± 0.00629 |     0.00224 ± 0.00091 | **0.00179 ± 0.00051** |
|              **0.20** |     0.01394 ± 0.01786 | 0.00546 ± 0.00226 |     0.00428 ± 0.00143 | **0.00382 ± 0.00123** |
|              **0.30** |     0.01350 ± 0.01162 | 0.00948 ± 0.00461 | **0.00873 ± 0.00337** |     0.00891 ± 0.00343 |
|              **0.50** | **0.01884 ± 0.00978** | 0.02378 ± 0.00760 |     0.02090 ± 0.00734 |     0.02108 ± 0.00801 |
"""
"""The overall pattern is:

$$ \boxed{\text{low noise: larger networks perform better}} $$

At \(\sigma=0\), 50 nodes achieves 0.00086 MSE, compared with 0.01247 for 5 nodes. The same basic pattern remains through \(\sigma=0.20\).

At \(\sigma=0.30\), the benefit of additional capacity essentially disappears:

$$ 25 = 0.00873,\qquad 50 = 0.00891. $$

At the highest noise level, \(\sigma=0.50\), the ordering changes completely. The 5-node network has the lowest mean clean-validation error at 0.01884, although the fairly large standard deviations mean we shouldn't claim it is conclusively superior.

A particularly useful comparison at \(\sigma=0.50\) is the noisy-training versus clean-validation error:"""