from torch.utils.data import Dataset
import torch
import numpy as np

class DynamicSensorDataset(Dataset):
    def __init__(self, data, window_size, target_col='current_volume'):
        self.data = data
        self.window_size = window_size
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X_categorical = self.data.iloc[idx][['sensor_code', 'station_code']].values.astype(int)
        X_numerical = self._get_window_data(idx).astype(np.float32)
        y = self.data.iloc[idx][self.target_col].astype(np.float32)

        return torch.tensor(X_categorical, dtype=torch.long), torch.tensor(X_numerical, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def _get_window_data(self, idx):
        start_idx = max(0, idx - self.window_size + 1)
        window_data = self.data.iloc[start_idx:idx + 1][['1000', '1300', '1600']].values.flatten()
        other_vars = self.data.iloc[idx].drop(['date', 'sensor_code', 'station_code', 'current_volume', '1000', '1300', '1600']).values
        combined = np.concatenate([window_data, other_vars])
        return combined

def create_sliding_windows(data, window_size):
    dataset = DynamicSensorDataset(data, window_size)
    return dataset

from sklearn.model_selection import train_test_split
import prepare_final_data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Load your data
data = prepare_final_data.get_data()
data = data.iloc[-13920:]

print('Data Loaded')

train_size = int(0.9 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Create datasets
window_size = 1

train_dataset = create_sliding_windows(train_data, window_size)
test_dataset = create_sliding_windows(test_data, window_size)

print("Defining the DataLoader")

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the model
class SensorModel(nn.Module):
    def __init__(self, n_sensors, n_stations, embedding_dim, numerical_dim, lstm_hidden_dim, output_dim):
        super(SensorModel, self).__init__()
        self.sensor_embedding = nn.Embedding(n_sensors, embedding_dim)
        self.station_embedding = nn.Embedding(n_stations, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2 + numerical_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, X_categorical, X_numerical):
        sensor_embedded = self.sensor_embedding(X_categorical[:, 0])
        station_embedded = self.station_embedding(X_categorical[:, 1])
        combined = torch.cat((sensor_embedded, station_embedded, X_numerical), dim=1)
        combined = combined.unsqueeze(1)  # Add sequence dimension
        lstm_out, _ = self.lstm(combined)
        output = self.fc(lstm_out[:, -1, :])
        return output

n_sensors = 9
n_stations = 174
embedding_dim = 4
numerical_dim = window_size * 3 + 4
lstm_hidden_dim = 32
output_dim = 1

model = SensorModel(n_sensors, n_stations, embedding_dim, numerical_dim, lstm_hidden_dim, output_dim)

# Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training the model")

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_categorical_batch, X_numerical_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_categorical_batch, X_numerical_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

import matplotlib.pyplot as plt

# Set the model to evaluation mode
model.eval()

# Initialize dictionaries to store real and predicted values for each sensor
real_values = {sensor: [] for sensor in range(9)}
predicted_values = {sensor: [] for sensor in range(9)}

# Disable gradient calculation for testing
with torch.no_grad():
    for X_categorical_batch, X_numerical_batch, y_batch in test_loader:
        outputs = model(X_categorical_batch, X_numerical_batch)
        for i in range(X_categorical_batch.size(0)):
            sensor = X_categorical_batch[i, 0].item()
            real_values[sensor].append(y_batch[i].item())
            predicted_values[sensor].append(outputs[i].item())

# Plot real vs. predicted values for each sensor
for sensor in range(9):
    plt.figure(figsize=(10, 6))
    plt.plot(real_values[sensor], label='Real Values')
    plt.plot(predicted_values[sensor], label='Predicted Values', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Current Volume')
    plt.title(f'Real vs. Predicted Values for Sensor {sensor}')
    plt.legend()
    plt.show()