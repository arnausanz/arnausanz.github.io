import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xlstm import mLSTMLayer, mLSTMLayerConfig


class SimpleModel(nn.Module):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        self.mlstmlayer = mLSTMLayer(config)
        # Output layer to map the mLSTM output to 10 classes
        self.fc = nn.Linear(512, 10)  # 10 classes for classification

    def forward(self, x):
        # Forward pass through the mLSTMLayer
        x = self.mlstmlayer(x)
        # Ensure the output has the correct shape: [batch_size, num_classes]
        x = x.view(x.size(0), -1)  # Flatten if needed (e.g., if the output is [batch_size, features])
        # Pass the output through the linear layer (classification head)
        out = self.fc(x)
        return out

# Configuration for the mLSTMLayer
config = mLSTMLayerConfig(
    embedding_dim=16,
    context_length=32,
    dropout=0.1,
    conv1d_kernel_size=4,
    qkv_proj_blocksize=4,
    num_heads=4,
)

# Generate fake data: 100 samples, 32 timesteps, 16 features
x_train = torch.randn(100, 32, 16)
y_train = torch.randint(0, 10, (100,))  # Random target labels (for classification)

# Create a DataLoader for batch processing
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Initialize the model
model = SimpleModel(config)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    running_loss = 0.0
    for inputs, labels in train_loader:
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Track loss
        running_loss += loss.item()

    # Print loss for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
