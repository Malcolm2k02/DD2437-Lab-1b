from src.data_setup import *
from src.mackey_mlp import *
import matplotlib.pyplot as plt

x = mackey_glass()
X, y = mackey_glass_dataset(x)
X_train, y_train, X_val, y_val, X_test, y_test = data_split(X, y)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

plt.figure(figsize=(10, 4))
plt.plot(x)
plt.xlabel("Time t")
plt.ylabel("x(t)")
plt.title("Noise-free Mackey-Glass Time Series")
plt.show()
            
