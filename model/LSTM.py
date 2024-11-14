import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1: Prepare the Data
X_train_seq = np.random.rand(100, 10, 5)  # (batch_size, seq_len, feature_dim)
y_train_seq = np.random.rand(100, 10, 1)  # (batch_size, seq_len, output_dim)

# Convert to PyTorch tensors
X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32)


# Step 2: Define the Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM layer (output shape: [batch_size, seq_len, hidden_dim])
        lstm_out, (hn, cn) = self.lstm(x)
        # Pass the LSTM output through the fully connected layer
        output = self.fc(lstm_out)
        return output


# Define the model
model = LSTMModel(input_dim=5, hidden_dim=100, output_dim=1)

# Step 3: Define Loss and Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()  # Set model to training mode

    # Zero the gradients before the backward pass
    optimizer.zero_grad()

    # Forward pass
    output = model(X_train_seq)  # Get predictions

    # Compute the loss
    loss = loss_fn(output, y_train_seq)

    # Backward pass (compute gradients)
    loss.backward()

    # Update the weights
    optimizer.step()

    # Print the loss
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")