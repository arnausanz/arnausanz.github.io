from model.model import Model, ModelConfig, save_model, load_model, get_split_data

"""
----------------- MODEL 1 -----------------"""
X_train_1, X_test_1, y_train_1, y_test_1, scalers_1 = get_split_data('LSTM',7)
m_cfg_1 = ModelConfig(
    model_type = 'LSTM',
    num_layers = 5,
    hidden_dim = 128,
    dropout = 0.2,
    num_epochs = 30,
    batch_size = 24,
    lr = 0.00001,
    input_dim = X_train_1.shape[2],
    output_dim = y_train_1.shape[1],
    steps_forward = 30
)
m_1 = Model(m_cfg_1)
print('Model 1 # Parameters:', sum(p.numel() for p in m_1.parameters()))
m_1.model_train(X_train_1, y_train_1)
m_1.model_predict(X_test_1, y_test_1)
save_model(m_1)

# m_1 = load_model('model_1')
# m_1.model_predict(X_test_1, y_test_1)
