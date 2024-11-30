from model.Model import Model, ModelConfig
from model.data_prep import get_data
from DataExtraction.utils import get_root_dir

"""
----------------- MODEL 1 -----------------

X, y, scalers = get_data(180)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def new_model():
    model_config = ModelConfig(model_type='LSTM', input_dim=X.shape[2], output_dim=y.shape[1], num_layers=5, hidden_dim=128,
                           dropout=0.2)
    model = Model(model_config)
    model.model_train(X_train, y_train, num_epochs=300, batch_size=24, lr=0.00001)
    model.model_predict(X_test, y_test)
    model.save_model()
    # model.move_track_emissions_file() # Deleted function to save emissions data
"""

"""
----------------- MODEL 3 -----------------

X, y, scalers = get_data(360)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def new_model():
    model_config = ModelConfig(model_type='LSTM', input_dim=X.shape[2], output_dim=y.shape[1], num_layers=3, hidden_dim=64,
                           dropout=0.2)
    model = Model(model_config)
    model.model_train(X_train, y_train, num_epochs=200, batch_size=64, lr=0.0001)
    model.model_predict(X_test, y_test)
    model.save_model()
    # model.move_track_emissions_file() # Deleted function to save emissions data
    
"""


X, y, scalers = get_data(360)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def new_model():
    """
    Function to create a new model with a specific configuration
    It's used once and then the model is saved.
    Once this function is used, it's copied-pasted into a comment above to save the configuration
    :return: None
    """
    model_config = ModelConfig(model_type='LSTM', input_dim=X.shape[2], output_dim=y.shape[1], num_layers=7, hidden_dim=64, dropout=0.2)
    model = Model(model_config)
    model.model_train(X_train, y_train, num_epochs=100, batch_size=64, lr=0.0001)
    model.model_predict(X_test, y_test)
    model.save_model()

def load_existing(name):
    """
    Function to load an existing model and make predictions
    :param name: name of the model directory (e.g. 'model_1')
    :return: pretrained model
    """
    model = Model(load=get_root_dir() + '/model/final_models/' + name, save=False)
    return model

new_model()