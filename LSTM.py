from model.model import Model, ModelConfig, save_model, load_model, get_split_data

"""
----------------- MODEL 1 -----------------
X_train_1, X_test_1, y_train_1, y_test_1, scalers_1 = get_split_data('LSTM',180)
m_cfg_1 = ModelConfig(
    model_type = 'LSTM',
    num_layers = 5,
    hidden_dim = 128,
    dropout = 0.2,
    num_epochs = 300,
    batch_size = 24,
    lr = 0.00001,
    input_dim = X_train_1.shape[2],
    output_dim = y_train_1.shape[1]
)
m1 = Model(m_cfg_1)
print('Model 1 # Parameters:', sum(p.numel() for p in m1.parameters()))
m1.model_train(X_train_1, y_train_1)
m1.model_predict(X_test_1, y_test_1)
save_model(m1)

# m_1 = load_model('model_1')
# m_1.model_predict(X_test_1, y_test_1)
"""

"""
----------------- MODEL 2 -----------------"""
X_train_2, X_test_2, y_train_2, y_test_2, scalers_2 = get_split_data('LSTM',90)
m_cfg_2 = ModelConfig(
    model_type = 'LSTM',
    num_layers = 5,
    hidden_dim = 128,
    dropout = 0.2,
    num_epochs = 300,
    batch_size = 24,
    lr = 0.00001,
    input_dim = X_train_2.shape[2],
    output_dim = y_train_2.shape[1]
)
m1 = Model(m_cfg_2)
print('Model 1 # Parameters:', sum(p.numel() for p in m1.parameters()))
m1.model_train(X_train_2, y_train_2)
m1.model_predict(X_test_2, y_test_2)
save_model(m1)

# m_2 = load_model('model_2')
# m_2.model_predict(X_test_2, y_test_2)