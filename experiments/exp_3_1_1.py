from src.data_setup import generate_data
from src.mlp import *
import matplotlib.pyplot as plt
import numpy as np

# Generate ONE fixed dataset
patterns, targets, classA, classB = generate_data(ndata=100)

hidden_sizes = range(1, 21)
n_runs = 50

mean_mse = []
mean_misclassification = []
best_misclassification = []


for hidden_node in hidden_sizes:

    mse_runs = []
    misclassification_runs = []

    for run in range(n_runs):

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

        mse_runs.append(errors[-1])
        misclassification_runs.append(n_wrong)

    mean_mse.append(np.mean(mse_runs))
    mean_misclassification.append(
        np.mean(misclassification_runs)
    )
    best_misclassification.append(
        np.min(misclassification_runs)
    )

    print(
        f"Hidden nodes: {hidden_node:2d} | "
        f"Mean MSE: {np.mean(mse_runs):.5f} | "
        f"Mean wrong: {np.mean(misclassification_runs):.2f} | "
        f"Best wrong: {np.min(misclassification_runs)}"
    )



"""Training completed. Final weights:
W: [[ 0.11164089 -4.34242643 -1.96011465]
 [ 4.98678804 -0.26792421  2.84942423]                
 [ 0.21324857 -1.97478846 -1.07328206]                
 [-1.06965819  0.16137906 -0.22540185]
 [-5.87743964  0.37802453  3.29144491]]
V: [[-4.13699596 -5.2198342  -2.18732586  0.88849931 -5.80022429  0.17527967]]
Final training error: 0.02153593175046333"""

"""Hidden nodes: 1, Mean final error: 0.55312std: 0.05489
Hidden nodes: 2, Mean final error: 0.35598std: 0.23200
Hidden nodes: 3, Mean final error: 0.21939std: 0.22169
Hidden nodes: 4, Mean final error: 0.20319std: 0.21434
Hidden nodes: 5, Mean final error: 0.12573std: 0.14680
Hidden nodes: 6, Mean final error: 0.11834std: 0.15195
Hidden nodes: 7, Mean final error: 0.10396std: 0.12197
Hidden nodes: 8, Mean final error: 0.10933std: 0.12724
Hidden nodes: 9, Mean final error: 0.08233std: 0.07641
Hidden nodes: 10, Mean final error: 0.11969std: 0.14958
Hidden nodes: 11, Mean final error: 0.09944std: 0.11760
Hidden nodes: 12, Mean final error: 0.09441std: 0.10476
Hidden nodes: 13, Mean final error: 0.10132std: 0.12693
Hidden nodes: 14, Mean final error: 0.07852std: 0.06078
Hidden nodes: 15, Mean final error: 0.06993std: 0.03111
Hidden nodes: 16, Mean final error: 0.06981std: 0.03521
Hidden nodes: 17, Mean final error: 0.07379std: 0.05701
Hidden nodes: 18, Mean final error: 0.06745std: 0.02658
Hidden nodes: 19, Mean final error: 0.07585std: 0.07278
Best number of hidden nodes: 18
Best mean error: 0.06744562279893972"""

"""Hidden nodes:  1 | Mean MSE: 0.47664 | Mean wrong: 30.02 | Best wrong: 30
Hidden nodes:  2 | Mean MSE: 0.31571 | Mean wrong: 20.16 | Best wrong: 2
Hidden nodes:  3 | Mean MSE: 0.17626 | Mean wrong: 11.00 | Best wrong: 2
Hidden nodes:  4 | Mean MSE: 0.11817 | Mean wrong: 7.00 | Best wrong: 2
Hidden nodes:  5 | Mean MSE: 0.07344 | Mean wrong: 4.28 | Best wrong: 2
Hidden nodes:  6 | Mean MSE: 0.05732 | Mean wrong: 3.26 | Best wrong: 2
Hidden nodes:  7 | Mean MSE: 0.09528 | Mean wrong: 5.84 | Best wrong: 2
Hidden nodes:  8 | Mean MSE: 0.10976 | Mean wrong: 6.44 | Best wrong: 2
Hidden nodes:  9 | Mean MSE: 0.06173 | Mean wrong: 3.62 | Best wrong: 2
Hidden nodes: 10 | Mean MSE: 0.07920 | Mean wrong: 4.64 | Best wrong: 2
Hidden nodes: 11 | Mean MSE: 0.07601 | Mean wrong: 4.26 | Best wrong: 2
Hidden nodes: 12 | Mean MSE: 0.05218 | Mean wrong: 3.06 | Best wrong: 2
Hidden nodes: 13 | Mean MSE: 0.06468 | Mean wrong: 3.84 | Best wrong: 2
Hidden nodes: 14 | Mean MSE: 0.05323 | Mean wrong: 2.86 | Best wrong: 2
Hidden nodes: 15 | Mean MSE: 0.04628 | Mean wrong: 2.64 | Best wrong: 2
Hidden nodes: 16 | Mean MSE: 0.06542 | Mean wrong: 3.70 | Best wrong: 2
Hidden nodes: 17 | Mean MSE: 0.07390 | Mean wrong: 4.12 | Best wrong: 2
Hidden nodes: 18 | Mean MSE: 0.06063 | Mean wrong: 3.32 | Best wrong: 2
Hidden nodes: 19 | Mean MSE: 0.05536 | Mean wrong: 2.98 | Best wrong: 2
Hidden nodes: 20 | Mean MSE: 0.05743 | Mean wrong: 3.20 | Best wrong: 2"""

