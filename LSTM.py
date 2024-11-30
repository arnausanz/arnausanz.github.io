import torch

from model.Model import Model, ModelConfig
from model.data_prep import get_data
from DataExtraction.utils import get_root_dir

"""
MODEL 1
X, y, scalers = get_data(180)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def new_model():
    model_config = ModelConfig(model_type='LSTM', input_dim=X.shape[2], output_dim=y.shape[1], num_layers=5, hidden_dim=128,
                           dropout=0.2)
    model = Model(model_config)
    model.model_train(X_train, y_train, num_epochs=300, batch_size=24, lr=0.00001)
    model.model_test(X_test, y_test)
    model.save_model()
    model.move_track_emissions_file()
"""

"""
MODEL 3
X, y, scalers = get_data(360)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def new_model():
    model_config = ModelConfig(model_type='LSTM', input_dim=X.shape[2], output_dim=y.shape[1], num_layers=3, hidden_dim=64,
                           dropout=0.2)
    model = Model(model_config)
    model.model_train(X_train, y_train, num_epochs=200, batch_size=64, lr=0.0001)
    model.model_test(X_test, y_test)
    model.save_model()
    model.move_track_emissions_file()
    
"""
X, y, scalers = get_data(360)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def new_model():
    model_config = ModelConfig(model_type='LSTM', input_dim=X.shape[2], output_dim=y.shape[1], num_layers=7, hidden_dim=64,
                           dropout=0.2)
    model = Model(model_config)
    model.model_train(X_train, y_train, num_epochs=100, batch_size=64, lr=0.0001)
    model.model_test(X_test, y_test)
    model.save_model()
    model.move_track_emissions_file()

def load_existing(name):
    model = Model(load=get_root_dir() + '/model/final_models/' + name, save=False)
    preds = model.model_test(X_test, y_test, test_mode=False)
    print('Predictions:', preds)

new_model()