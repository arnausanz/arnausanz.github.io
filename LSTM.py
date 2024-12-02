import pickle

from model.model import Model, ModelConfig, save_model, load_model, get_split_data


# ----------------- MODEL 1/2/3/4 -----------------
steps_forward = (30, 90, 180, 365)
for step_forward in steps_forward:
    X_train, X_test, y_train, y_test, scalers = get_split_data('LSTM', 180, steps_fwd=step_forward)
    print(X_train.shape)
    m_cfg = ModelConfig(
        model_type = 'LSTM',
        num_layers = 5,
        hidden_dim = 128,
        dropout = 0.2,
        num_epochs = 300,
        batch_size = 24,
        lr = 0.00001,
        input_dim = X_train.shape[2],
        output_dim = y_train.shape[1],
        steps_forward = step_forward
    )
    m = Model(m_cfg)
    print('Model 1 # Parameters:', sum(p.numel() for p in m.parameters()))
    m.model_train(X_train, y_train)
    m.model_predict(X_test, y_test, plot=False)
    save_model(m)
    # Save scalers in a pickle file
    with open(m.model_config.model_src + '/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

# ----------------- MODEL 5/6/7/8 -----------------
steps_forward = (30, 90, 180, 365)
for step_forward in steps_forward:
    X_train, X_test, y_train, y_test, scalers = get_split_data('LSTM', 365, steps_fwd=step_forward)
    print(X_train.shape)
    m_cfg = ModelConfig(
        model_type = 'LSTM',
        num_layers = 5,
        hidden_dim = 128,
        dropout = 0.2,
        num_epochs = 300,
        batch_size = 24,
        lr = 0.00001,
        input_dim = X_train.shape[2],
        output_dim = y_train.shape[1],
        steps_forward = step_forward
    )
    m = Model(m_cfg)
    print('Model 1 # Parameters:', sum(p.numel() for p in m.parameters()))
    m.model_train(X_train, y_train)
    m.model_predict(X_test, y_test, plot=False)
    save_model(m)
    # Save scalers in a pickle file
    with open(m.model_config.model_src + '/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

# m = load_model('model_1')
# m.model_predict(X_test_1, y_test_1)
