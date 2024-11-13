import xlstm
from keras import Sequential
from keras import layers as l
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim

# Generate fake stationary data for the LSTM model
X_train_seq = np.random.rand(100, 10, 5)
y_train_seq = np.random.rand(100, 10, 1)


model = Sequential()
model.add(l.LSTM(100, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=True))
model.add(l.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_seq, y_train_seq, epochs=10, verbose=1)