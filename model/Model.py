import json
import os
import pickle
import subprocess
import time
from codecarbon import track_emissions
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
from xlstm import xLSTMBlockStack
from tqdm import tqdm

import DataExtraction.utils as utils
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class ModelConfig:
    def __init__(self, **kwargs):
        self.model_type = kwargs.get('model_type', None)
        self.input_dim = kwargs.get('input_dim', None)
        self.hidden_dim = kwargs.get('hidden_dim', None)
        self.output_dim = kwargs.get('output_dim', None)
        self.num_layers = kwargs.get('num_layers', None)
        self.dropout = kwargs.get('dropout', None)
        self.xLSTM_config = kwargs.get('xLSTM_config', None)
        self.device = None


def evaluate_model_performance(y_true, y_pred):
    """
    Evaluate the model performance by calculating the correlation between the true and predicted values for each reservoir individually
    :param y_true: True values
    :param y_pred: Predicted values
    :return: dict with all metrics
    """
    n_reservoirs = y_true.shape[1]
    metrics = {}
    for i in range(n_reservoirs):
        true_values = y_true[:, i]
        pred_values = y_pred[:, i]
        mse = mean_squared_error(true_values, pred_values)
        r2 = r2_score(true_values, pred_values)
        mae = mean_absolute_error(true_values, pred_values)
        rmse = mse ** 0.5

        metrics[f'reservoir_{i}'] = {'mse': mse, 'r2': r2, 'mae': mae, 'rmse': rmse}

    # Save also the general metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mse ** 0.5
    metrics['general'] = {'mse': mse, 'r2': r2, 'mae': mae, 'rmse': rmse}

    return metrics


class Model(nn.Module):
    def __init__(self, model_config = None, save=True, load=None):
        super(Model, self).__init__()
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        torch.set_default_device(self.device)
        print('Device:', self.device)

        self.save = save
        # Check if the model is being loaded
        if load:
            self.this_model_src = load
            self.load_model()
        else:
            self.model_config = model_config
            self._build_model()
            if save:
                # If save, save the model and all its information in the specific directory
                self.model_name = None
                self.this_model_src = None
                self.get_save_directory()
        if self.model_config.device is None:
            self.model_config.device = self.device.__str__()


    def _build_model(self):
        if self.model_config.model_type == 'LSTM':
            self.lstm = nn.LSTM(self.model_config.input_dim, self.model_config.hidden_dim, self.model_config.num_layers, batch_first=True, dropout=self.model_config.dropout)
            self.dropout = nn.Dropout(self.model_config.dropout)
            self.fc = nn.Linear(self.model_config.hidden_dim, self.model_config.output_dim)
        elif self.model_config.model_type == 'xLSTM':
            self.xlstm_stack = xLSTMBlockStack(self.model_config.xLSTM_config)
            self.fc = nn.Linear(self.model_config.xLSTM_config.embedding_dim, self.model_config.output_dim)

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

    @track_emissions(offline=True, country_iso_code="ESP", log_level="critical", default_cpu_power=30)
    def model_train(self, X_train, y_train, num_epochs=100, batch_size=32, lr=0.001, criterion=nn.MSELoss(), optimizer=torch.optim.Adam):
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        print('X_train_tensor shape:', X_train_tensor.shape)
        print('y_train_tensor shape:', y_train_tensor.shape)

        print('Training the model...')

        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=self.device))
        optimizer = optimizer(self.parameters(), lr=lr)
        n_reservoirs = y_train.shape[1]
        training_stats = {'epochs': [], 'mse_loss': [], 'epoch_time': [], 'loss_by_reservoir': {f'reservoir_{i}': [] for i in range(n_reservoirs)}}

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            start_time = time.time()
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f'Epoch {epoch + 1}/{num_epochs}')
            reservoir_individual_loss = {f'reservoir_{i}': 0 for i in range(n_reservoirs)}
            for batch_idx, (x_batch, y_batch) in progress_bar:
                optimizer.zero_grad()
                output = self.forward(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # Save the results of each reservoir to calculate the loss by reservoir
                for i in range(n_reservoirs):
                    reservoir_individual_loss[f'reservoir_{i}'] += criterion(output[:, i], y_batch[:, i]).item()

            progress_bar.close()
            epoch_time = time.time() - start_time
            training_stats['epochs'].append(epoch+1)
            training_stats['mse_loss'].append(running_loss)
            training_stats['epoch_time'].append(epoch_time)
            # Calculate the loss by reservoir
            for i in range(n_reservoirs):
                training_stats['loss_by_reservoir'][f'reservoir_{i}'].append(reservoir_individual_loss[f'reservoir_{i}'])

        if self.save:
            with open(f'{self.this_model_src}/training/training_stats.json', 'w') as f:
                json.dump(training_stats, f, indent=4)
        return self

    def move_track_emissions_file(self):
        """
        Move the track_emissions file automatically created in the current directory to the model directory
        :return: None
        """
        os.rename('emissions.csv', f'{self.this_model_src}/emissions.csv')

    def model_test(self, X_test, y_test, test_mode = True):
        print('Testing the model...')
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        test_data = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, generator=torch.Generator(device=self.device))
        self.eval()
        all_outputs = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                output = self.forward(x_batch).cpu()
                all_outputs.append(output.numpy())

        y_pred = torch.cat([torch.tensor(out).cpu() for out in all_outputs], dim=0).numpy()

        if test_mode:
            metrics = evaluate_model_performance(y_test, y_pred)
            if self.save:
                with open(f'{self.this_model_src}/testing/metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=4)

            # Plot the results for each reservoir
            for i in range(y_test.shape[1]):
                # Plot with a high resolution and big text
                plt.figure(figsize=(32, 18))
                plt.plot(y_test[:, i], label='True', color='#dbbe04', linewidth=3.5)
                plt.plot(y_pred[:, i], label='Predicted', color='#32b8aa', linewidth=3.5)
                plt.legend(fontsize=20)
                plt.title(f'Reservoir {i}', fontsize=24)
                self.save_testing_chart(plt, f'reservoir_{i}')
        return y_pred


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
        os.chdir(utils.get_root_dir())

    def save_testing_chart(self, chart, title):
        """
        Save this
        :param title: Title of the chart file
        :param chart: Chart to save
        :return: None
        """
        os.chdir(self.this_model_src + '/testing/')
        chart.savefig(f'{title}.png')
        os.chdir(utils.get_root_dir())

    def save_model(self):
        """
        Save the model and its information. Specifically, save:
        - Model architecture
        - Model initial state
        - Model hyperparameters
        - Model configuration
        - Model itself
        :param model: Model to save
        :return: None
        """
        os.chdir(self.this_model_src + '/')
        torch.save(self.state_dict(), 'model.pth')
        # Create a dict from config ignoring xLSTM_config as it's not serializable
        model_config_dict = {k: v for k, v in self.model_config.__dict__.items() if k != 'xLSTM_config'}
        # Save the xLSTM_config separately in a pickle file
        if self.model_config.model_type == 'xLSTM':
            with open('xLSTM_config.pkl', 'wb') as f:
                pickle.dump(self.model_config.xLSTM_config, f)
        with open(f'model_{self.model_name}.json', 'w') as f:
            json.dump(model_config_dict, f, indent=4)
        os.chdir(utils.get_root_dir())

    def load_model(self):
        """
        Load the model from the specified directory
        :return: None
        """
        os.chdir(self.this_model_src + '/')
        model_name = self.this_model_src.split('/')[-1]
        with open(model_name+'.json', 'r') as f:
            model_info = json.load(f)
        self.model_config = ModelConfig(**model_info)
        if self.model_config.model_type == 'xLSTM':
            self.model_config.xLSTM_config = pickle.load(open('xLSTM_config.pkl', 'rb'))
        self.model_name = model_name
        self._build_model()
        self.load_state_dict(torch.load('model.pth', weights_only=True))
        self.eval()
        os.chdir(utils.get_root_dir())