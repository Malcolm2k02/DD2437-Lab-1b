import numpy as np


def generate_data(ndata=100):
    mA = np.array([1.0, 0.3])
    sigmaA = 0.2

    mB = np.array([0.0, -0.1])
    sigmaB = 0.3

    classA = np.zeros((2, ndata))
    classB = np.zeros((2, ndata))

    # Class A consists of two clusters
    classA[0, :ndata // 2] = (
        np.random.randn(ndata // 2) * sigmaA - mA[0]
    )
    classA[0, ndata // 2:] = (
        np.random.randn(ndata // 2) * sigmaA + mA[0]
    )

    classA[1, :] = np.random.randn(ndata) * sigmaA + mA[1]

    # Class B consists of one cluster
    classB[0, :] = np.random.randn(ndata) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(ndata) * sigmaB + mB[1]

    # Put all samples into one matrix
    patterns = np.concatenate((classA, classB), axis=1)

    # Symmetric targets: A = +1, B = -1
    targets = np.concatenate((
        np.ones(ndata),
        -np.ones(ndata)
    )).reshape(1, -1)

    return patterns, targets, classA, classB


def subsample_data(classA, classB, scenario):

    nA = classA.shape[1]
    nB = classB.shape[1]

    # Initially all samples are training samples
    train_A = np.ones(nA, dtype=bool)
    train_B = np.ones(nB, dtype=bool)

    # ============================================================
    # Scenario 1: remove random 25% from each class
    # ============================================================

    if scenario == "25_each":

        remove_A = np.random.choice(
            nA,
            size=int(0.25 * nA),
            replace=False
        )

        remove_B = np.random.choice(
            nB,
            size=int(0.25 * nB),
            replace=False
        )

        train_A[remove_A] = False
        train_B[remove_B] = False

    # ============================================================
    # Scenario 2: remove random 50% from class A
    # ============================================================

    elif scenario == "50_A":

        remove_A = np.random.choice(
            nA,
            size=int(0.50 * nA),
            replace=False
        )

        train_A[remove_A] = False

    # ============================================================
    # Scenario 3:
    # remove 20% from left A cluster
    # remove 80% from right A cluster
    # ============================================================

    elif scenario == "A_20_80":

        left = np.where(classA[0, :] < 0)[0]
        right = np.where(classA[0, :] > 0)[0]

        remove_left = np.random.choice(
            left,
            size=int(0.20 * len(left)),
            replace=False
        )

        remove_right = np.random.choice(
            right,
            size=int(0.80 * len(right)),
            replace=False
        )

        train_A[remove_left] = False
        train_A[remove_right] = False

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # ============================================================
    # Create training and validation sets
    # ============================================================

    A_train = classA[:, train_A]
    A_val = classA[:, ~train_A]

    B_train = classB[:, train_B]
    B_val = classB[:, ~train_B]

    # Training patterns
    train_patterns = np.concatenate(
        (A_train, B_train),
        axis=1
    )

    train_targets = np.concatenate((
        np.ones(A_train.shape[1]),
        -np.ones(B_train.shape[1])
    )).reshape(1, -1)

    # Validation patterns
    val_patterns = np.concatenate(
        (A_val, B_val),
        axis=1
    )

    val_targets = np.concatenate((
        np.ones(A_val.shape[1]),
        -np.ones(B_val.shape[1])
    )).reshape(1, -1)

    return (
        train_patterns,
        train_targets,
        val_patterns,
        val_targets
    )


def generate_gaussian_data():

    # Equivalent to MATLAB:
    # x = [-5:0.5:5]'
    # y = [-5:0.5:5]'

    x = np.arange(-5, 5.5, 0.5)
    y = np.arange(-5, 5.5, 0.5)

    # Create every possible (x, y) combination
    xx, yy = np.meshgrid(x, y)

    # Gaussian function:
    # f(x,y) = exp(-(x^2 + y^2)/10) - 0.5

    z = np.exp(-(xx**2 + yy**2) / 10) - 0.5

    # Convert grid into pattern matrix
    #
    # patterns:
    # row 0 = x coordinates
    # row 1 = y coordinates
    #
    # shape = (2, 441)

    patterns = np.vstack((
        xx.reshape(-1),
        yy.reshape(-1)
    ))

    # One target value for every (x,y)
    #
    # shape = (1, 441)

    targets = z.reshape(1, -1)

    return patterns, targets, x, y, xx, yy, z



def subsample_gaussian(patterns, targets, fraction):

    ndata = patterns.shape[1]

    # Number of samples used for training
    nsamp = int(fraction * ndata)

    # Random permutation of all sample indices
    indices = np.random.permutation(ndata)

    # Select first nsamp samples
    train_indices = indices[:nsamp]

    # Training subset
    train_patterns = patterns[:, train_indices]
    train_targets = targets[:, train_indices]

    return train_patterns, train_targets, train_indices