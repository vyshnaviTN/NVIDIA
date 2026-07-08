import torch
import torch.nn as nn
import torch.optim as optim

# 1. Binary Classification Network (Sigmoid Output)
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# 2. Data (XOR-like simplified or linear separation)
X = torch.randn(200, 2)
y = ((X[:, 0] + X[:, 1]) > 0).float().unsqueeze(1)

# 3. Training Config
model = BinaryClassifier()
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training
for epoch in range(50):
    predictions = model(X)
    loss = criterion(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")