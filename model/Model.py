import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import data_prep
import matplotlib.pyplot as plt


class ModelConfig:
    def __init__(self, **kwargs):
        self.model_type = kwargs.get('model_type', 'LSTM')
        self.input_dim = kwargs.get('input_dim', 5)
        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.output_dim = kwargs.get('output_dim', 1)
        self.num_layers = kwargs.get('num_layers', 2)
        self.dropout = kwargs.get('dropout', 0.2)


class Model(nn.Module):
    def __init__(self, model_config):
        super(Model, self).__init__()
        self.model_config = model_config
        self.lstm = nn.LSTM(self.model_config.input_dim, self.model_config.hidden_dim, self.model_config.num_layers,
                            batch_first=True, dropout=self.model_config.dropout)
        self.dropout = nn.Dropout(self.model_config.dropout)
        self.fc = nn.Linear(self.model_config.hidden_dim, self.model_config.output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Take the last output of the LSTM
        output = self.fc(lstm_out)
        return output

    def model_train(self, X_train, y_train, num_epochs=100, batch_size=32, lr=0.001, verbose=True,
                    criterion=nn.MSELoss(), optimizer=torch.optim.Adam):
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = optimizer(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if verbose:
                print(f'Epoch {epoch + 1}, Loss: {running_loss}')
        return self

    def model_test(self, X_test, y_test):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        test_data = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                output = self.forward(x_batch)
                loss = nn.MSELoss()(output, y_batch)
                test_loss += loss.item()
        # Plot the real vs predicted values for each reservoir
        for i in range(y_test.shape[1]):
            plt.figure(figsize=(10, 5))
            plt.plot(y_test[:, i], label=f'Reservoir {i} (real)')
            plt.plot(self.forward(X_test_tensor)[:, i].detach().numpy(), label=f'Reservoir {i} (predicted)')
            plt.legend()
            plt.show()
        return test_loss


X, y, scalers = data_prep.get_data(30)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model_config = ModelConfig(model_type='LSTM', input_dim=X.shape[2], output_dim=y.shape[1], num_layers=3, hidden_dim=128,
                           dropout=0.2)

model = Model(model_config)
model.model_train(X_train, y_train, num_epochs=15, batch_size=32, lr=0.001)
test_loss = model.model_test(X_test, y_test)