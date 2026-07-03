import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Synthetic Data Generation (y = 2x + 3)
X = np.random.rand(100, 1).astype(np.float32)
y = 2 * X + 3 + np.random.randn(100, 1).astype(np.float32) * 0.1

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# 2. Linear Regression Model
model = nn.Linear(1, 1)

# 3. Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training Loop
for epoch in range(100):
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

print(f"Trained Weight: {model.weight.item():.4f}, Bias: {model.bias.item():.4f}")