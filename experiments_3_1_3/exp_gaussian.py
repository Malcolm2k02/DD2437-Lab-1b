import matplotlib.pyplot as plt

from src.data_setup import generate_gaussian_data


# ============================================================
# Generate data
# ============================================================

patterns, targets, x, y, xx, yy, z = generate_gaussian_data()


# ============================================================
# Print dimensions
# ============================================================

print("Patterns shape:", patterns.shape)
print("Targets shape:", targets.shape)


# ============================================================
# Plot original Gaussian function
# ============================================================

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot_surface(
    xx,
    yy,
    z
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")

ax.set_title("Target Gaussian Function")

plt.tight_layout()
plt.show()