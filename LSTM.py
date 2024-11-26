import torch

from model.Model import Model, ModelConfig
from model.data_prep import get_data

torch.set_default_device('mps')

X, y, scalers = get_data(10)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model_config = ModelConfig(model_type='LSTM', input_dim=X.shape[2], output_dim=y.shape[1], num_layers=3, hidden_dim=128,
                           dropout=0.2)

model = Model(model_config)
model.model_train(X_train, y_train, num_epochs=3, batch_size=32, lr=0.0005)
test_loss = model.model_test(X_test, y_test)