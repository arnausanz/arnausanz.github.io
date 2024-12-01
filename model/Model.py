import json
import os
import pickle
import random
import time

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
from xlstm import xLSTMBlockStack
from tqdm import tqdm
import DataExtraction.utils as utils
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class ModelConfig:
    """
    Class to store the configuration of the model
    It then will be used to store model information
    """
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
    """
    Base class for models LSTM and xLSTM
    """
    def __init__(self, model_config = None, save=True, load=None):
        super(Model, self).__init__()
        # Train in mps if available
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        torch.set_default_device(self.device)
        print('Device:', self.device)
        # Work with the same seed for reproducibility
        random.seed(2)
        np.random.seed(2)
        torch.manual_seed(2)

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
        """
        Build the model based on the type specified (LSTM or xLSTM)
        :return: None
        """
        if self.model_config.model_type == 'LSTM':
            self.lstm = nn.LSTM(self.model_config.input_dim, self.model_config.hidden_dim, self.model_config.num_layers, batch_first=True, dropout=self.model_config.dropout)
            self.dropout = nn.Dropout(self.model_config.dropout)
            self.fc = nn.Linear(self.model_config.hidden_dim, self.model_config.output_dim)
        elif self.model_config.model_type == 'xLSTM':
            self.xlstm_stack = xLSTMBlockStack(self.model_config.xLSTM_config)
            self.fc = nn.Linear(self.model_config.xLSTM_config.embedding_dim, self.model_config.output_dim)

    def forward(self, x):
        """
        Forward steps of the model
        :param x: Input data
        :return: Output of the model
        """
        # Decide which model to use based on the model type
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

    def model_train(self, X_train, y_train, num_epochs=100, batch_size=32, lr=0.001, criterion=nn.MSELoss(), optimizer=torch.optim.Adam):
        """
        Train the model
        :param X_train: X data to train
        :param y_train: y data to train
        :param num_epochs: Number of epochs
        :param batch_size: Batch size
        :param lr: Learning rate
        :param criterion: Loss function
        :param optimizer: Optimizer
        :return: None
        """
        # Convert to PyTorch tensors and send to device (to use mps if available)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        # Generate the DataLoader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=self.device))
        optimizer = optimizer(self.parameters(), lr=lr)
        n_reservoirs = y_train.shape[1] # Get number of reservoirs (y columns) used to save information later
        # Define stats to save
        training_stats = {'epochs': [], 'mse_loss': [], 'epoch_time': [], 'loss_by_reservoir': {f'reservoir_{i}': [] for i in range(n_reservoirs)}}
        # Train the model
        for epoch in range(num_epochs):
            self.train()
            # Init metric variables (also to save the loss by reservoir)
            running_loss = 0.0
            reservoir_individual_loss = {f'reservoir_{i}': 0 for i in range(n_reservoirs)}
            start_time = time.time()
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}') # Create a progress bar to show the training progress
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
            # Append metrics to the stat saver
            epoch_time = time.time() - start_time
            training_stats['epochs'].append(epoch+1)
            training_stats['mse_loss'].append(running_loss)
            training_stats['epoch_time'].append(epoch_time)
            # Calculate the loss by reservoir
            for i in range(n_reservoirs):
                training_stats['loss_by_reservoir'][f'reservoir_{i}'].append(reservoir_individual_loss[f'reservoir_{i}'])
        # If save, save the training stats into a file
        if self.save:
            with open(f'{self.this_model_src}/training/training_stats.json', 'w') as f:
                json.dump(training_stats, f, indent=4)
        return self

    def model_predict(self, X_test, y_test = None, test_mode = True, show = False):
        """
        Make predictions with the model trained
        :param show: Show the testing charts
        :param X_test: X data to test (or predict)
        :param y_test: y data to test (or predict)
        :param test_mode: If test mode, evaluate the model performance and save testing information
        :return: Predictions if not in test mode else None
        """
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        # If not in test mode and y_test is None, return the predictions (real case)
        if y_test is None and not test_mode:
            return self.forward(X_test_tensor).cpu().numpy()
        # If the code reaches this point, it's in test mode
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        test_data = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_data, batch_size=24, shuffle=False, generator=torch.Generator(device=self.device))
        self.eval()
        all_outputs = []
        # Make predictions
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                output = self.forward(x_batch).cpu()
                all_outputs.append(output.numpy())
        # Concatenate all the outputs
        y_pred = torch.cat([torch.tensor(out).cpu() for out in all_outputs], dim=0).numpy()
        # Use the evaluate_model_performance function to get the metrics and save the information in a test eval file
        metrics = evaluate_model_performance(y_test, y_pred)
        if self.save:
            with open(f'{self.this_model_src}/testing/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            # Save predictions
            np.save(f'{self.this_model_src}/testing/predictions.npy', y_pred)
        # Generate the testing charts
        for i in range(y_test.shape[1]):
            # Plot with a high resolution, big text and specific colors
            plt.figure(figsize=(32, 18))
            plt.plot(y_test[:, i], label='True', color='#dbbe04', linewidth=3.5)
            plt.plot(y_pred[:, i], label='Predicted', color='#32b8aa', linewidth=3.5)
            plt.legend(fontsize=20)
            plt.title(f'Reservoir {i}', fontsize=24)
            # Save the chart
            if self.save:
                self.save_testing_chart(plt, f'reservoir_{i}')
            # Show the chart
            if show:
                plt.show()


    def get_save_directory(self):
        """
        Generate the directory where the model will be saved (including the metrics and the charts)
        :return: none
        """
        src = utils.get_root_dir() + '/model/final_models'
        os.chdir(src)
        models = os.listdir()
        # Get last model saved to set the new model name and directory
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
        # Create the training and testing directories
        os.mkdir('training')
        os.mkdir('testing')
        # Return to main directory
        os.chdir(utils.get_root_dir())

    def save_testing_chart(self, chart, title):
        """
        Save charts in the testing directory
        :param title: Title of the chart file
        :param chart: Chart to save
        :return: None
        """
        os.chdir(self.this_model_src + '/testing/')
        chart.savefig(f'{title}.png')
        # Return to main directory
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
        # Save the rest of the model configuration into a JSON file
        with open(f'model_{self.model_name}.json', 'w') as f:
            json.dump(model_config_dict, f, indent=4)
        # Return to main directory
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
        # As the xLSTM_config is not serializable, load it from the separate pickle file
        if self.model_config.model_type == 'xLSTM':
            self.model_config.xLSTM_config = pickle.load(open('xLSTM_config.pkl', 'rb'))
        self.model_name = model_name
        self._build_model()
        # Load the weights of the model
        self.load_state_dict(torch.load('model.pth', weights_only=True))
        # Return to main directory
        os.chdir(utils.get_root_dir())