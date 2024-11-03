import keras
from tkan import TKAN
import numpy as np

# Create fake X and y data
X_train_seq = np.random.rand(100, 10, 5)
y_train_seq = np.random.rand(100, 5)

class Model:
      model = keras.Sequential([
            keras.layers.InputLayer(shape=X_train_seq.shape[1:]),
            TKAN(100, sub_kan_configs=[{'spline_order': 3, 'grid_size': 10}, {'spline_order': 1, 'grid_size': 5}, {'spline_order': 4, 'grid_size': 6}, ], return_sequences=True, use_bias=True), #Define the params of the KANLinear as dict as here
            TKAN(100, sub_kan_configs=[1, 2, 3, 3, 4], return_sequences=True, use_bias=True), #Use float or int to specify only the exponent of the spline
            TKAN(100, sub_kan_configs=['relu', 'relu', 'relu', 'relu', 'relu'], return_sequences=True, use_bias=True), #Or use string to specify the standard tensorflow activation using Dense in sublayers instead of KANLinear
            TKAN(100, sub_kan_configs=[None for _ in range(3)], return_sequences=False, use_bias=True), # Or put None for default activation
            keras.layers.Dense(y_train_seq.shape[1]),
      ])

      model.compile(optimizer='adam', loss='mse')
      model.fit(X_train_seq, y_train_seq, epochs=10, verbose=1)
