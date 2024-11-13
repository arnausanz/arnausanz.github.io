import torch
import torch.nn as nn
from keras.src.backend import backend
from xlstm import sLSTMLayer, sLSTMLayerConfig

# Define SimpleModel with sLSTMLayer
class SimpleModel(nn.Module):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        self.sLSTM_layer = sLSTMLayer(config)
        self.fc = nn.Linear(config.embedding_dim, 10)  # Example output layer

    def forward(self, x):
        x = self.sLSTM_layer(x)
        out = self.fc(x[:, -1, :])  # Output from the last time step
        return out


# Define configuration
config = sLSTMLayerConfig(
    embedding_dim=128,
    num_heads=8,
    dropout=0.1,
    conv1d_kernel_size=3,
    group_norm_weight=True,
    backend='vanilla'
)

# Instantiate the model
model = SimpleModel(config)

# Random data (batch_size=32, seq_length=50, embedding_dim=128)
x = torch.randn(32, 50, 128)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    optimizer.zero_grad()

    output = model(x)
    y = torch.randint(0, 10, (32,))  # Random target labels

    loss = criterion(output, y)
    loss.backward()

    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')