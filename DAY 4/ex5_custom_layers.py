import torch
import torch.nn as nn

# 1. Custom Element-wise Scale Layer
class ElementwiseScaleLayer(nn.Module):
    def __init__(self, num_features):
        super(ElementwiseScaleLayer, self).__init__()
        # Learnable parameter
        self.weights = nn.Parameter(torch.ones(num_features))
        
    def forward(self, x):
        return x * self.weights

# 2. Testing Custom Model Layer Components
model = nn.Sequential(
    nn.Linear(5, 5),
    ElementwiseScaleLayer(5),
    nn.ReLU()
)

sample_input = torch.randn(2, 5)
output = model(sample_input)
print("Input shape:", sample_input.shape)
print("Output tensor sample:\n", output)