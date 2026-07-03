import torch
import torch.nn as nn
import torch.optim as optim

# Simple shared model setup comparison framework
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

model_adam = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
model_sgd = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

# Share initial state weights for fairness
model_sgd.load_state_dict(model_adam.state_dict())

opt_adam = optim.Adam(model_adam.parameters(), lr=0.01)
opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Single Step Update Comparison Demo
loss_a = criterion(model_adam(X), y)
opt_adam.zero_grad()
loss_a.backward()
opt_adam.step()

loss_s = criterion(model_sgd(X), y)
opt_sgd.zero_grad()
loss_s.backward()
opt_sgd.step()

print(f"Initial Adam step loss: {loss_a.item():.4f}")
print(f"Initial SGD step loss: {loss_s.item():.4f}")