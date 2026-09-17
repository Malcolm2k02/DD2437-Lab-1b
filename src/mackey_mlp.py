import torch
import torch.nn as nn

class MackeyGlassMLP(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()

        layers = []
        input_size = 5

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Sigmoid())
            input_size = hidden_size

        # One linear output
        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model(model, X_train, y_train, X_val, y_val, learning_rate = 0.01, weight_decay=0.0, max_epochs=20000, patience=100):

    # Mse
    criterion = nn.MSELoss()

    # Gradient descent optimizer with optional weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Training
        model.train()

        optimizer.zero_grad()

        predictions = model(X_train)

        train_loss = criterion(predictions, y_train)

        train_loss.backward()

        optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch + 1}")
            break

    # Restore model with best val error
    model.load_state_dict(best_model_state)

    return train_losses, val_losses

