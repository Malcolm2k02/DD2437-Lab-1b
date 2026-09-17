import torch
import numpy as np

from src.data_setup import (
    mackey_glass,
    mackey_glass_dataset,
    data_split
)

from src.mackey_mlp import (
    MackeyGlassMLP,
    train_model
)


# ============================================================
# Prepare Mackey-Glass data
# ============================================================

x = mackey_glass()

X, y = mackey_glass_dataset(x)

X_train, y_train, X_val, y_val, X_test, y_test = data_split(X, y)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

nh1_values = [3, 4, 5]
nh2_values = [2, 4, 6]

results = []

for nh1 in nh1_values:
    for nh2 in nh2_values:
        # Same initialisation seed 
        torch.manual_seed(42)

        model = MackeyGlassMLP(hidden_sizes=[nh1, nh2])

        train_losses, val_losses = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate=0.1,
            weight_decay=0.001,
            max_epochs=20000,
            patience=3000
        )

        best_val_mse = min(val_losses)

        results.append({
            "nh1": nh1,
            "nh2": nh2,
            "val_mse": best_val_mse,
            "epochs": len(train_losses) 
        })

        print(
            f"{nh1}x{nh2} | "
            f"Best Val MSE: {best_val_mse:.6f} | "
            f"Epochs: {len(train_losses)}"
        )

best_architecture = min(
    results,
    key=lambda result: result["val_mse"]
)

worst_architecture = max(
    results,
    key=lambda result: result["val_mse"]
)

print("\nMost suitable architecture:")
print(best_architecture)

print("\nLeast suitable architecture:")
print(worst_architecture)

import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("5x2 Architecture")
plt.legend()
plt.show()


best_architecture = [3, 4]
worst_architecture = [3, 2]

def test_initializations(hidden_sizes, n_runs=10):

    val_errors = []

    for seed in range(n_runs):

        torch.manual_seed(seed)

        model = MackeyGlassMLP(
            hidden_sizes=hidden_sizes
        )

        train_losses, val_losses = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate=0.1,
            weight_decay=0.001,
            max_epochs=20000,
            patience=3000
        )

        best_val_mse = min(val_losses)
        val_errors.append(best_val_mse)

        print(
            f"Seed {seed} | "
            f"Best Val MSE: {best_val_mse:.6f}"
        )

    return val_errors

print("\n=== 3x4 architecture ===")
best_errors = test_initializations([3, 4])

print("\n=== 3x2 architecture ===")
worst_errors = test_initializations([3, 2])

print("\nResults:")

print(
    f"3x4: {np.mean(best_errors):.6f} "
    f"+/- {np.std(best_errors):.6f}"
)

print(
    f"3x2: {np.mean(worst_errors):.6f} "
    f"+/- {np.std(worst_errors):.6f}"
)

# ============================================================
# Final test evaluation
# ============================================================

criterion = torch.nn.MSELoss()

architectures = {
    "Most suitable (3x4)": [3, 4],
    "Least suitable (3x2)": [3, 2]
}

test_results = {}

for name, hidden_sizes in architectures.items():

    # Use same seed as original grid search
    torch.manual_seed(42)

    model = MackeyGlassMLP(hidden_sizes=hidden_sizes)

    # Train using ONLY training + validation
    train_losses, val_losses = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=0.1,
        weight_decay=0.001,
        max_epochs=20000,
        patience=3000
    )

    # Evaluate on unseen test data
    model.eval()

    with torch.no_grad():
        predictions = model(X_test)
        test_mse = criterion(predictions, y_test).item()

    test_results[name] = {
        "mse": test_mse,
        "predictions": predictions.numpy()
    }

    print(
        f"{name} | "
        f"Test MSE: {test_mse:.6f}"
    )


import matplotlib.pyplot as plt

target_values = y_test.numpy().flatten()

for name, result in test_results.items():

    predictions = result["predictions"].flatten()

    plt.figure(figsize=(10, 4))

    plt.plot(
        target_values,
        label="Target"
    )

    plt.plot(
        predictions,
        label="Prediction"
    )

    plt.xlabel("Test sample")
    plt.ylabel("x(t+5)")
    plt.title(
        f"{name} - Test Predictions"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


"""3x2 | Best Val MSE: 0.010516 | Epochs: 20000
3x4 | Best Val MSE: 0.009407 | Epochs: 20000
3x6 | Best Val MSE: 0.009889 | Epochs: 20000
4x2 | Best Val MSE: 0.009950 | Epochs: 20000
4x4 | Best Val MSE: 0.010002 | Epochs: 20000
4x6 | Best Val MSE: 0.009859 | Epochs: 20000
5x2 | Best Val MSE: 0.010125 | Epochs: 20000
5x4 | Best Val MSE: 0.010057 | Epochs: 20000
5x6 | Best Val MSE: 0.009766 | Epochs: 20000

Most suitable architecture:
{'nh1': 3, 'nh2': 4, 'val_mse': 0.00940669421106577, 'epochs': 20000}

Least suitable architecture:
{'nh1': 3, 'nh2': 2, 'val_mse': 0.010516034439206123,'epochs': 20000}

=== 3x4 architecture ===
Seed 0 | Best Val MSE: 0.008948
Seed 1 | Best Val MSE: 0.009651
Seed 2 | Best Val MSE: 0.009885
Seed 3 | Best Val MSE: 0.009816
Seed 4 | Best Val MSE: 0.009797
Seed 5 | Best Val MSE: 0.010218
Seed 6 | Best Val MSE: 0.010153
Seed 7 | Best Val MSE: 0.009560
Seed 8 | Best Val MSE: 0.009763
Seed 9 | Best Val MSE: 0.009937

=== 3x2 architecture ===
Seed 0 | Best Val MSE: 0.008355
Seed 1 | Best Val MSE: 0.009377
Seed 2 | Best Val MSE: 0.009932
Seed 3 | Best Val MSE: 0.009879
Seed 4 | Best Val MSE: 0.009929
Seed 5 | Best Val MSE: 0.010059
Seed 6 | Best Val MSE: 0.009749
Seed 7 | Best Val MSE: 0.009965
Seed 8 | Best Val MSE: 0.009416
Seed 9 | Best Val MSE: 0.010139

Results:
3x4: 0.009773 +/- 0.000335
3x2: 0.009680 +/- 0.000502
Most suitable (3x4) | Test MSE: 0.012781
Least suitable (3x2) | Test MSE: 0.014178"""