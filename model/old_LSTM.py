import torch
import torch.nn as nn
from data_prep import get_data
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


class ReservoirLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        """
        LSTM model for predicting reservoir volumes.

        Parameters:
        - input_dim (int): Number of input features per timestep.
        - hidden_dim (int): Number of hidden units in the LSTM.
        - output_dim (int): Number of output targets (reservoirs).
        - num_layers (int): Number of LSTM layers. Default is 1.
        """
        super(ReservoirLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        last_timestep = lstm_out[:, -1, :]  # Use output of the last timestep
        output = self.fc(last_timestep)  # Fully connected layer for prediction
        return output

    def save_model(self, path):
        """
        Save the model to a file.

        Parameters:
        - path (str): File path to save the model.
        """
        torch.save(self.state_dict(), path)

    def model_train(self, X_train, y_train, num_epochs=100, batch_size=32, lr=0.001, verbose=True, criterion=nn.MSELoss(), optimizer=torch.optim.Adam):
        # TODO --> Implementar bé aquesta funció i fer els entrenaments a partir d'aquí
        """
        Train the model on the training data.

        Parameters:
        - X_train (torch.Tensor): Input training data.
        - y_train (torch.Tensor): Target training data.
        - num_epochs (int): Number of training epochs. Default is 100.
        - batch_size (int): Batch size for training. Default is 32.
        - lr (float): Learning rate for the optimizer. Default is 0.001.
        - verbose (bool): Whether to print training loss. Default is True.
        - criterion (torch.nn.Module): Loss function. Default is MSELoss.
        - optimizer (torch.optim.Optimizer): Optimizer for training. Default is Adam.

        Returns:
        - list: Training losses for each epoch.
        """
        # Define DataLoader
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss function and optimizer
        optimizer = optimizer(self.parameters(), lr=lr)

        # Training loop
        training_losses = []
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
            training_losses.append(epoch_loss / len(dataloader))
        return training_losses


data = get_data()
# y columns = '0', '1', '2', '3', '4', '5', '6' , '7', '8'
X, y = data.drop(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8']), data[['0', '1', '2', '3', '4', '5', '6', '7', '8']]

# Define parameters
input_dim = X.shape[1]  # Number of features (from your data preparation)
hidden_dim = 64         # Number of hidden units in the LSTM
output_dim = y.shape[1] # Number of reservoirs (targets)
num_layers = 2          # Number of LSTM layers

# Initialize the model
model = ReservoirLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
print(model)



# Sequential train-test split (e.g., 80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 15
batch_size = 32

# Create DataLoader
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()  # Reset gradients
        predictions = model(batch_X)  # Forward pass
        loss = criterion(predictions, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)  # Predictions
    y_pred = y_pred_tensor.numpy()  # Convert to NumPy for evaluation

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Loop for all reservoir indexes and plot predictions vs. actual values
for i in range(y_test.shape[1]):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:, i], label='Actual', color='blue', alpha=0.6)
    plt.plot(y_pred[:, i], label='Predicted', color='red', alpha=0.6)
    plt.title(f"Predictions vs. Actuals for Reservoir {i + 1}")
    plt.xlabel("Time Steps (Test Data)")
    plt.ylabel("Scaled Volume")
    plt.legend()
    plt.show()