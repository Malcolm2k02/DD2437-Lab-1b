from src.mlp import *
from src.data_setup import *
import numpy as np
import matplotlib.pyplot as plt
# ============================================================
# Plot decision boundaries for 5 and 10 hidden nodes
# ============================================================
patterns, targets, classA, classB = generate_data(ndata=100)
for hidden_node in [3, 4]:

    best_wrong = np.inf
    best_W = None
    best_V = None
    best_error = None

    # Train several times because initialization is random
    for run in range(50):

        W, V, errors = train_mlp(
            patterns,
            targets,
            Nhidden=hidden_node,
            epochs=1000,
            eta=0.01,
            alpha=0.9
        )

        n_wrong, ratio = classification_error(
            patterns,
            targets,
            W,
            V
        )

        # Save the best network
        if n_wrong < best_wrong:
            best_wrong = n_wrong
            best_W = W.copy()
            best_V = V.copy()
            best_error = errors[-1]

    print(
        f"{hidden_node} hidden nodes | "
        f"Best misclassifications: {best_wrong} | "
        f"MSE: {best_error:.5f}"
    )

    plot_decision_boundary(
        patterns,
        classA,
        classB,
        best_W,
        best_V
    )