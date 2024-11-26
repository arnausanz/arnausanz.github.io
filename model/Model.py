import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
from xlstm import xLSTMBlockStack
from tqdm import tqdm
import DataExtraction.utils as utils


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
    def __init__(self, model_config, save=True):
        super(Model, self).__init__()
        self.model_config = model_config
        self._build_model()
        if save:
            # If save, save the model and all its information in the specific directory
            self.save = save
            self.model_name = None
            self.this_model_src = None
            self.get_save_directory()
            self.save_model_config()


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
        if self.save:
            self.save_model(self)
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
            plt.figure(figsize=(28, 14))
            plt.plot(y_test[:, i], label=f'Reservoir {i} (real)')
            plt.plot(self.forward(X_test_tensor).cpu()[:, i].detach().numpy(), label=f'Reservoir {i} (predicted)')
            plt.legend(fontsize='x-large')
            plt.title(f'Reservoir {i} real vs predicted values', fontsize='xx-large')
            if self.save:
                self.save_testing(plt, self.model_name + f'_reservoir_{i}')
            # plt.show()
        return test_loss

    def get_save_directory(self):
        src = utils.get_root_dir() + '/model/final_models'
        os.chdir(src)
        models = os.listdir()
        last_model = 0
        for model_dir in models:
            if model_dir.startswith('model'):
                model_num = int(model_dir.split('_')[1])
                last_model = max(last_model, model_num)
        this_model = last_model + 1
        self.model_name = str(this_model)
        os.mkdir(f'model_{this_model}')
        self.this_model_src = src + f'/model_{this_model}'
        os.chdir(self.this_model_src)
        os.mkdir('training')
        os.mkdir('testing')
        os.mkdir('model')
        os.chdir(utils.get_root_dir())

    def save_model_config(self):
        """
        Save the model configuration in the specific directory
        :return: None
        """
        os.chdir(self.this_model_src)
        with open('model_config.txt', 'w') as f:
            f.write(f'Model type: {self.model_config.model_type}\n')
            if self.model_config.model_type == 'LSTM':
                f.write(f'Input dimension: {self.model_config.input_dim}\n')
                f.write(f'Hidden dimension: {self.model_config.hidden_dim}\n')
                f.write(f'Output dimension: {self.model_config.output_dim}\n')
                f.write(f'Number of layers: {self.model_config.num_layers}\n')
                f.write(f'Dropout: {self.model_config.dropout}\n')
            elif self.model_config.model_type == 'xLSTM':
                f.write(f'xLSTM configuration: {self.model_config.xLSTM_config}\n')
        f.close()
        os.chdir(utils.get_root_dir())


    def save_training(self):
        """
        Save the training process in the specific directory
        Things to save:
        - Loss per epoch: A lot of metrics (MSE, RMSE, MAE, etc.)
        - Training time: Log training time per epoch and total training time
        - Accuracy: Log accuracy per epoch
        :return:
        """
        pass

    def save_testing(self, chart, title):
        """
        # TODO --> Acabar de pensar les mètriques a guardar
        If self.save is True, save the chart in the specific directory with the reservoir name, and the model number
        :param title: Title of the chart file
        :param chart: Chart to save
        :return: None
        """
        os.chdir(self.this_model_src + '/testing')
        chart.savefig(f'{title}.png')
        os.chdir(utils.get_root_dir())

    def save_model(self, model):
        """
        Save the model in the specific directory
        :param model: Model to save
        :return: None
        """
        os.chdir(self.this_model_src + '/model')
        torch.save(model.state_dict(), 'model.pth')
        os.chdir(utils.get_root_dir())

    def load_model(self):
        # TODO --> To load the model I should save all the other data: model name, and so on
        pass