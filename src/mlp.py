import numpy as np
import matplotlib.pyplot as plt

def phi(x):
    """"Activation function for MLP"""
    return 2 / (1+ np.exp(-x)) - 1


def phi_derivative(phi_x):
    """"Derivative of activation function for MLP"""
    return 0.5 * (1 + phi_x) * (1- phi_x)


def forward_pass(patterns, W, V):
    ndata = patterns.shape[1]

    # Adding bias to input layer
    X = np.vstack((patterns, np.ones((1, ndata))))

    # Input to hidden layer
    hin = W @ X
    hidden  = phi(hin)

    # Adding bias to hidden layer
    hout = np.vstack((hidden, np.ones((1, ndata))))

    # Hidden --> output layer
    oin = V @ hout
    out = phi(oin)

    return X, hin, hidden, hout, oin, out


def backward_pass(targets, V, hidden, hout, out):
    # Output layer delta
    delta_o = (out - targets) * phi_derivative(out)

    # Propagating error backwards
    delta_h_full = (V.T @ delta_o) * phi_derivative(hout)

    # Removing bias row
    delta_h = delta_h_full[:-1, :]

    return delta_o, delta_h

def update_weights(W, V, dw, dv, X, hout, delta_h, delta_o, eta, alpha):
    dw = (dw * alpha) - (delta_h @ X.T) * (1 - alpha)
    dv = (dv * alpha) - (delta_o @ hout.T) * (1 - alpha)

    W = W + eta * dw
    V = V + eta * dv

    return W, V, dw, dv


def train_mlp(
    patterns,
    targets,
    Nhidden=5,
    epochs=1000,
    eta=0.01,
    alpha=0.9,
    val_patterns=None,
    val_targets=None
):

    n_inputs = patterns.shape[0]
    n_outputs = targets.shape[0]

    W = np.random.randn(
        Nhidden,
        n_inputs + 1
    ) * 0.1

    V = np.random.randn(
        n_outputs,
        Nhidden + 1
    ) * 0.1

    dw = np.zeros_like(W)
    dv = np.zeros_like(V)

    train_errors = []
    val_errors = []

    train_misclass = []
    val_misclass = []

    for epoch in range(epochs):

        # ========================================================
        # TRAINING FORWARD PASS
        # ========================================================

        X, hin, hidden, hout, oin, out = forward_pass(
            patterns,
            W,
            V
        )

        # ========================================================
        # BACKPROPAGATION
        # ONLY TRAINING DATA IS USED HERE
        # ========================================================

        delta_o, delta_h = backward_pass(
            targets,
            V,
            hidden,
            hout,
            out
        )

        W, V, dw, dv = update_weights(
            W,
            V,
            dw,
            dv,
            X,
            hout,
            delta_h,
            delta_o,
            eta,
            alpha
        )

        # ========================================================
        # TRAINING ERROR
        # ========================================================

        _, _, _, _, _, train_out = forward_pass(
            patterns,
            W,
            V
        )

        train_mse = np.mean(
            (train_out - targets) ** 2
        )

        train_errors.append(train_mse)

        train_predictions = np.where(
            train_out >= 0,
            1,
            -1
        )

        train_ratio = np.mean(
            train_predictions != targets
        )

        train_misclass.append(train_ratio)

        # ========================================================
        # VALIDATION ERROR
        # ========================================================

        if val_patterns is not None:

            _, _, _, _, _, val_out = forward_pass(
                val_patterns,
                W,
                V
            )

            val_mse = np.mean(
                (val_out - val_targets) ** 2
            )

            val_errors.append(val_mse)

            val_predictions = np.where(
                val_out >= 0,
                1,
                -1
            )

            val_ratio = np.mean(
                val_predictions != val_targets
            )

            val_misclass.append(val_ratio)

    return (
        W,
        V,
        train_errors,
        val_errors,
        train_misclass,
        val_misclass
    )


def plot_decision_boundary(patterns, classA, classB, W, V):

    # Define the area we want to visualize
    x_min = patterns[0, :].min() - 0.5
    x_max = patterns[0, :].max() + 0.5
    y_min = patterns[1, :].min() - 0.5
    y_max = patterns[1, :].max() + 0.5

    # Create a dense grid of points
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Turn grid into the same format as our training data
    grid_patterns = np.vstack([
        xx.ravel(),
        yy.ravel()
    ])

    # Feed every grid point through the trained MLP
    _, _, _, _, _, grid_output = forward_pass(
        grid_patterns, W, V
    )

    # Convert output back to grid shape
    grid_output = grid_output.reshape(xx.shape)

    # Plot original data
    plt.scatter(
        classA[0, :],
        classA[1, :],
        label="Class A"
    )

    plt.scatter(
        classB[0, :],
        classB[1, :],
        label="Class B"
    )

    # Output = 0 is the decision boundary
    plt.contour(
        xx,
        yy,
        grid_output,
        levels=[0]
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("MLP Decision Boundary")
    plt.legend()
    plt.show()


def plot_training_error(errors):
    plt.plot(errors)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("MLP Training Error")
    plt.grid()
    plt.show()

def classification_error(patterns, targets, W, V):
    _, _, _, _, _, out = forward_pass(patterns, W, V)

    predictions = np.where(out >= 0, 1, -1)

    n_misclassified = np.sum(predictions != targets)
    misclassification_ratio = n_misclassified / targets.size

    return n_misclassified, misclassification_ratio

def plot_train_validation_errors(
    train_errors,
    val_errors,
    title=""
):
    plt.plot(
        train_errors,
        label="Training"
    )

    plt.plot(
        val_errors,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.show()


def plot_train_validation_misclassification(
    train_misclass,
    val_misclass,
    title=""
):
    plt.plot(
        train_misclass,
        label="Training"
    )

    plt.plot(
        val_misclass,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Misclassification Ratio")
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.show()


def plot_validation_vs_hidden(hidden_sizes, validation_errors, scenario):
    import matplotlib.pyplot as plt

    plt.plot(
        hidden_sizes,
        validation_errors,
        marker="o"
    )

    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Final Validation MSE")
    plt.title(f"Validation Error vs Hidden Layer Size - {scenario}")
    plt.xticks(hidden_sizes)
    plt.grid()

    plt.show()