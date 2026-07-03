import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DeepClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10, use_dropout=True, drop_rate=0.5):
        super(DeepClassifier, self).__init__()
        self.use_dropout = use_dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        return self.fc2(x)

def run_training_pipeline(optimizer_type='Adam', epochs=10, lr=0.001, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing: {optimizer_type} Engine Setup on Device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST(root='./data', train=False, transform=transform), batch_size=batch_size, shuffle=False)
    
    model = DeepClassifier(use_dropout=True, drop_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    history = {'train_loss': [], 'test_accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_acc = (correct / total) * 100
        history['test_accuracy'].append(epoch_acc)
        print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] -> Loss: {epoch_loss:.4f} | Validation Accuracy: {epoch_acc:.2f}%")
        
    return history

if __name__ == "__main__":
    total_epochs = 10
    adam_history = run_training_pipeline('Adam', epochs=total_epochs, lr=0.001)
    sgd_history = run_training_pipeline('SGD', epochs=total_epochs, lr=0.01)
    
    epochs_range = range(1, total_epochs + 1)
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, adam_history['train_loss'], 'o-', label='Adam Optimizer', color='#3182ce')
    plt.plot(epochs_range, sgd_history['train_loss'], 's--', label='SGD with Momentum', color='#dd6b20')
    plt.title('Optimization Training Loss Descent', fontweight='bold')
    plt.xlabel('Epoch Cycles')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, adam_history['test_accuracy'], 'o-', label='Adam Optimizer', color='#3182ce')
    plt.plot(epochs_range, sgd_history['test_accuracy'], 's--', label='SGD with Momentum', color='#dd6b20')
    plt.title('Generalized Validation Accuracy Curves', fontweight='bold')
    plt.xlabel('Epoch Cycles')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('optimization_curves_comparison.png', dpi=300)
    print("\n>>> Visual verification comparison chart saved as 'optimization_curves_comparison.png'")
    plt.show()
