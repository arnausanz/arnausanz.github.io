import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import data_prep
import matplotlib.pyplot as plt
from xlstm import xLSTMBlockStack
from tqdm import tqdm


class ModelConfig:
    def __init__(self, **kwargs):
        self.model_type = kwargs.get('model_type', None)
        self.input_dim = kwargs.get('input_dim', None)
        self.hidden_dim = kwargs.get('hidden_dim', None)
        self.output_dim = kwargs.get('output_dim', None)
        self.num_layers = kwargs.get('num_layers', None)
        self.dropout = kwargs.get('dropout', None)
        self.xLSTM_config = kwargs.get('xLSTM_config', None)

class Model(nn.Module):
    def __init__(self, model_config):
        super(Model, self).__init__()
        self.model_config = model_config
        self._build_model()

    def _build_model(self):
        if self.model_config.model_type == 'LSTM':
            self.lstm = nn.LSTM(self.model_config.input_dim, self.model_config.hidden_dim, self.model_config.num_layers, batch_first=True, dropout=self.model_config.dropout)
            self.dropout = nn.Dropout(self.model_config.dropout)
            self.fc = nn.Linear(self.model_config.hidden_dim, self.model_config.output_dim)
        elif self.model_config.model_type == 'xLSTM':
            self.xlstm_stack = xLSTMBlockStack(self.model_config.xLSTM_config)
            self.fc = nn.Linear(self.model_config.xLSTM_config.embedding_dim, self.model_config.output_dim)  # Map to 9 output dimensions

    def forward(self, x):
        if self.model_config.model_type == 'LSTM':
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out[:, -1, :])  # Take the last output of the LSTM
            output = self.fc(lstm_out)
            return output
        elif self.model_config.model_type == 'xLSTM':
            xlstm_out = self.xlstm_stack(x)
            xlstm_out = xlstm_out[:, -1, :]
            output = self.fc(xlstm_out)
            return output

    def model_train(self, X_train, y_train, num_epochs=100, batch_size=32, lr=0.001, verbose=True, criterion=nn.MSELoss(), optimizer=torch.optim.Adam):
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        print('X_train_tensor shape:', X_train_tensor.shape)
        print('y_train_tensor shape:', y_train_tensor.shape)

        print('Training the model...')

        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        optimizer = optimizer(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch_idx, (x_batch, y_batch) in progress_bar:
                optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss/(batch_idx+1))
            progress_bar.close()
            if verbose:
                print(f'Epoch {epoch + 1}, Loss: {running_loss}')
        return self

    def model_test(self, X_test, y_test):
        print('Testing the model...')
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