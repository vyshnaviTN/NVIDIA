import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 1. Model & Component Pipeline
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
# Decay LR by a factor of 0.1 every 5 epochs
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 2. Simulate step tracking over epochs
print("Simulating step adjustments:")
for epoch in range(12):
    # Dummy step update process
    optimizer.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1:02d} -> Active Learning Rate: {current_lr:.5f}")
    scheduler.step()