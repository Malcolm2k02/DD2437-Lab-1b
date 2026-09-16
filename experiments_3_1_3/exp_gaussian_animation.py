import matplotlib.pyplot as plt

from src.data_setup import generate_gaussian_data
from src.mlp import train_mlp_function, forward_pass


# ============================================================
# Generate Gaussian data
# ============================================================

patterns, targets, x, y, xx, yy, z = generate_gaussian_data()

print("Patterns:", patterns.shape)
print("Targets:", targets.shape)


# ============================================================
# Network settings
# ============================================================

Nhidden = 32
epochs = 1000
eta = 0.01
alpha = 0.9


# ============================================================
# Train network
# ============================================================

W, V, train_errors = train_mlp_function(
    patterns,
    targets,
    Nhidden=Nhidden,
    epochs=epochs,
    eta=eta,
    alpha=alpha,
    animate=True,
    xx=xx,
    yy=yy,
    plot_every=10
)


print("Final MSE:", train_errors[-1])


# ============================================================
# Final network output
# ============================================================

_, _, _, _, _, out = forward_pass(
    patterns,
    W,
    V
)

predicted_z = out.reshape(xx.shape)


# ============================================================
# Plot final approximation
# ============================================================

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot_surface(
    xx,
    yy,
    predicted_z
)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-0.7, 0.7)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Output")

ax.set_title(
    f"Final MLP Approximation - {Nhidden} Hidden Nodes"
)

plt.tight_layout()
plt.show()


# ============================================================
# Plot learning curve
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(train_errors)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(
    f"Learning Curve - {Nhidden} Hidden Nodes"
)

plt.grid()
plt.tight_layout()

plt.show()