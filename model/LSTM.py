import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Model and training parameters
input_size = 10  # Number of features per time step
seq_length = 5  # Sequence length
hidden_size = 20  # Hidden size of the LSTM
num_layers = 1  # Number of LSTM layers
num_epochs = 5  # Number of training epochs
batch_size = 16
learning_rate = 0.001

# Generate synthetic data
num_samples = 200
X = torch.randn(num_samples, seq_length, input_size)  # Random data
y = torch.randint(0, 2, (num_samples,))  # Random binary labels

# Create train and test datasets
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size
train_dataset = TensorDataset(X[:train_size], y[:train_size])
test_dataset = TensorDataset(X[train_size:], y[train_size:])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Define LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Use last hidden state
        return out


# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing the model
model.eval()
with torch.no_grad():
    correct, total = 0, 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")