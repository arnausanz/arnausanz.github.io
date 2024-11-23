import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# Step 1: Create the dataset
data = {
    'Date': ['2000-01-01', '2000-01-01'],
    'SensorID': [1, 1],
    'StationID': [1, 2],
    'Rain': [1.5, 0.12],
    'Snow': [0.0, 4.5],
    'Distance': [26.2, 56.2],
    '%Type1': [43.0, 43.0],
    '%Type2': [2.3, 2.3],
    '%Type3': [15.0, 15.0],
    '%Type4': [3.35, 3.35],
    'Target': [400.2, 25.3]  # Target is Sensor1-Lvl or Sensor2-Lvl
}
df = pd.DataFrame(data)

print(df)

# Step 2: Encode categorical columns
encoder_sensor = LabelEncoder()
encoder_station = LabelEncoder()

df['SensorID_encoded'] = encoder_sensor.fit_transform(df['SensorID'])
df['StationID_encoded'] = encoder_station.fit_transform(df['StationID'])

# Separate categorical and numerical features
X_categorical = df[['SensorID_encoded', 'StationID_encoded']].values
X_numerical = df[['Rain', 'Snow', 'Distance', '%Type1', '%Type2', '%Type3', '%Type4']].values
y = df['Target'].values

# Step 3: Define the dataset class
class SensorDataset(Dataset):
    def __init__(self, X_categorical, X_numerical, y):
        self.X_categorical = torch.tensor(X_categorical, dtype=torch.long)
        self.X_numerical = torch.tensor(X_numerical, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_categorical[idx], self.X_numerical[idx], self.y[idx]

dataset = SensorDataset(X_categorical, X_numerical, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 4: Define the model
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

n_sensors = len(encoder_sensor.classes_)
n_stations = len(encoder_station.classes_)
embedding_dim = 4
numerical_dim = X_numerical.shape[1]
lstm_hidden_dim = 32
output_dim = 1

model = SensorModel(n_sensors, n_stations, embedding_dim, numerical_dim, lstm_hidden_dim, output_dim)

# Step 5: Train the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_categorical_batch, X_numerical_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_categorical_batch, X_numerical_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")