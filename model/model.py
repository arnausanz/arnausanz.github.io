import json
import os
import pickle
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from xlstm import xLSTMBlockStack
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as rmse
from DataExtraction.utils import get_root_dir
from . import data_prep

def get_split_data(model_type, window_size, subwindow_size = None, steps_fwd=0, train_size = 0.8, device = 'mps'):
    if model_type == 'LSTM':
        X, y, scalers = data_prep.get_data(window_size, steps_fwd)
    elif model_type == 'xLSTM':
        X, y, scalers = data_prep.get_data_x(window_size, subwindow_size, steps_fwd)
    else:
        raise ValueError("Invalid model type")
    train_size = int(train_size * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    # Convert data to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    return X_train, X_test, y_train, y_test, scalers


def _get_model_name():
    os.chdir(get_root_dir() + '/model/final_models')
    if len([model for model in os.listdir() if model.startswith('model_')]) == 0:
        return 'model_1'
    max_model_num = max([int(model.split('_')[1]) for model in os.listdir() if model.startswith('model_')])
    model_name = f'model_{max_model_num + 1}'
    os.chdir(get_root_dir())
    return model_name

def _create_model_directory(model_name):
    os.chdir(get_root_dir() + '/model/final_models')
    os.mkdir(model_name)
    # Also create train and test directories
    os.mkdir(model_name + '/train')
    os.mkdir(model_name + '/test')
    os.chdir(get_root_dir())
    return get_root_dir() + '/model/final_models/' + model_name

def save_model(model):
    model_src = model.model_config.model_src
    torch.save(model.state_dict(), model_src + '/model.pth')
    # Save pickle file with model config
    with open(model_src + '/model_config.pkl', 'wb') as f:
        pickle.dump(model.model_config, f)
    print(f'Model {model.model_config.model_name} saved')

def load_model(model_name):
    model_src = get_root_dir() + '/model/final_models/' + model_name
    # Read model config
    with open(model_src + '/model_config.pkl', 'rb') as f:
        model_config = pickle.load(f)
    model = Model(model_config)
    model.load_state_dict(torch.load(model.model_config.model_src+'/model.pth', weights_only=True))
    model.loaded = True
    return model

class ModelConfig:
    def __init__(self, steps_forward, **kwargs):
        self.model_name = _get_model_name()
        self.model_type = kwargs.get('model_type', None)
        self.train_size = kwargs.get('train_size', None)
        self.window_size = kwargs.get('window_size', None)
        self.subwindow_size = kwargs.get('subwindow_size', None)
        self.num_epochs = kwargs.get('num_epochs', None)
        self.batch_size = kwargs.get('batch_size', None)
        self.lr = kwargs.get('lr', None)
        self.num_layers = kwargs.get('num_layers', None)
        self.input_dim = kwargs.get('input_dim', None)
        self.hidden_dim = kwargs.get('hidden_dim', None)
        self.dropout = kwargs.get('dropout', None)
        self.output_dim = kwargs.get('output_dim', None)
        self.xLSTM_config = kwargs.get('xLSTM_config', None)
        self.criterion = kwargs.get('criterion', nn.MSELoss)
        self.optimizer = kwargs.get('optimizer', torch.optim.Adam)
        self.model_src = _create_model_directory(self.model_name)
        self.device = kwargs.get('device', 'mps')
        self.steps_forward = steps_forward

class Model(nn.Module):
    def __init__(self, model_config):
        super(Model, self).__init__()
        self.model_config = model_config
        # Set default device
        torch.set_default_device(self.model_config.device)
        self._build_model()
        self.loaded = False

    def _build_model(self):
        # Save window size and subwindow size
        self.window_size = self.model_config.window_size
        self.subwindow_size = self.model_config.subwindow_size if self.model_config.model_type == 'xLSTM' else None
        if self.model_config.model_type == 'LSTM':
            self.lstm = nn.LSTM(
                input_size=self.model_config.input_dim,
                hidden_size=self.model_config.hidden_dim,
                num_layers=self.model_config.num_layers,
                batch_first=True
            )
            self.dropout = nn.Dropout(self.model_config.dropout)
            self.fc = nn.Linear(self.model_config.hidden_dim, self.model_config.output_dim)
        elif self.model_config.model_type == 'xLSTM':
            self.xlstm_stack = xLSTMBlockStack(self.model_config.xLSTM_config)
            self.fc = nn.Linear(self.model_config.xLSTM_config.embedding_dim, self.model_config.output_dim)
        else:
            raise ValueError("Invalid model type")
        print(f"Model {self.model_config.model_name} created")

    def forward(self, x):
        if self.model_config.model_type == 'LSTM':
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out[:, -1, :])
            output = self.fc(lstm_out)
        elif self.model_config.model_type == 'xLSTM':
            x_lstm_out = self.xlstm_stack(x)
            x_lstm_out = self.dropout(x_lstm_out[:, -1, :])
            output = self.fc(x_lstm_out)
        else:
            raise ValueError("Invalid model type")
        return output

    def model_train(self, X_train, y_train):
        if self.loaded:
            raise ValueError("Model already trained")
        criterion = self.model_config.criterion()
        optimizer = self.model_config.optimizer(self.parameters(), lr=self.model_config.lr)
        # Vars to store training information
        training_stats = {'epoch_num': [], 'epoch_times': [], 'epoch_losses': [], 'loss_by_reservoir': {f'reservoir_{i}': [] for i in range(self.model_config.output_dim)}}
        for epoch in range(self.model_config.num_epochs):
            self.train()
            epoch_loss = 0
            start_time = time.time()
            reservoir_individual_losses = {f'reservoir_{i}': 0 for i in range(self.model_config.output_dim)}
            with tqdm(total=len(X_train), desc=f'Epoch {epoch + 1}/{self.model_config.num_epochs}', unit='records') as pbar:
                for i in range(0, len(X_train), self.model_config.batch_size):
                    X_batch = X_train[i:i+self.model_config.batch_size]
                    y_batch = y_train[i:i+self.model_config.batch_size]
                    output = self(X_batch)
                    loss = criterion(output, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * X_batch.size(0)
                    for r in range(self.model_config.output_dim):
                        reservoir_individual_losses[f'reservoir_{r}'] += (criterion(output[:, r], y_batch[:, r]).item() * X_batch.size(0))
                    pbar.update(X_batch.size(0))
                epoch_loss /= len(X_train)
                pbar.set_postfix({'loss': epoch_loss})
                # Calc the mean of the losses
                for r in range(self.model_config.output_dim):
                    reservoir_individual_losses[f'reservoir_{r}'] /= len(X_train)
                time_total = time.time() - start_time
                training_stats['epoch_num'].append(epoch + 1)
                training_stats['epoch_times'].append(time_total)
                training_stats['epoch_losses'].append(epoch_loss)
                for r in range(self.model_config.output_dim):
                    training_stats['loss_by_reservoir'][f'reservoir_{r}'].append(reservoir_individual_losses[f'reservoir_{r}'])
        # Save training stats
        with open(self.model_config.model_src + '/train/training_stats.json', 'w') as f:
            json.dump(training_stats, f)


    def model_predict(self, X_test, y_test, save=True, plot=True):
        print('Predicting...')
        self.eval()
        with torch.no_grad():
            y_pred = self(X_test)
            loss = self.model_config.criterion()(y_pred, y_test)
            print(f'Loss: {loss.item():.4f}')
        # Save predictions, losses, and charts
        save = False if self.loaded else save
        if save:
            # Save predictions in a numpy file
            os.chdir(self.model_config.model_src + '/test')
            y_pred = y_pred.cpu().numpy()
            y_test = y_test.cpu().numpy()
            np.save('y_pred.npy', y_pred)
            np.save('y_test.npy', y_test)
            # Calc the losses for each reservoir and the total losses
            losses = {'total': {'mse': float(mse(y_test, y_pred)),
                                'mae': float(mae(y_test, y_pred)),
                                'rmse': float(rmse(y_test, y_pred)),
                                'r2': float(r2_score(y_test, y_pred))}}
            for i in range(self.model_config.output_dim):
                losses[f'reservoir_{i}'] = {'mse': float(mse(y_test[:, i], y_pred[:, i])),
                                            'mae': float(mae(y_test[:, i], y_pred[:, i])),
                                            'rmse': float(rmse(y_test[:, i], y_pred[:, i])),
                                            'r2': float(r2_score(y_test[:, i], y_pred[:, i]))}
            with open('losses.json', 'w') as f:
                json.dump(losses, f)
        if save or plot:
            # Save or the charts of the predictions for each reservoir
            for i in range(self.model_config.output_dim):
                # Ensure tensor is converted to numpy
                y_test = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test
                y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
                plt.figure(figsize=(32, 18))
                plt.plot(y_test[:, i], label='True', color='#dbbe04', linewidth=3.5)
                plt.plot(y_pred[:, i], label='Predicted', color='#32b8aa', linewidth=3.5)
                plt.legend(fontsize=20)
                plt.title(f'Reservoir {i + 1} predicted vs true water level')
                plt.legend()
                if save:
                    plt.savefig(f'reservoir_{i+1}.png')
                if plot:
                    plt.show()
                plt.close()
        os.chdir(get_root_dir())
        return y_pred

"""
X_train, X_test, y_train, y_test, scalers = get_split_data('LSTM', 180)
input_size = X_train.shape[2]
output_size = y_train.shape[1]
a = ModelConfig(model_type='LSTM', input_dim=input_size, output_dim=output_size, num_layers=5, hidden_dim=128, dropout=0.2, num_epochs=3, batch_size=24, lr=0.00001)
model = Model(a)

# model = load_model('model_8')


model.model_train(X_train, y_train)
y_pred = model.model_predict(X_test, y_test)
save_model(model)

"""