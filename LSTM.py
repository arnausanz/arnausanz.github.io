import pickle

import torch

from model.model import Model, ModelConfig, save_model, load_model, get_split_data

"""
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
    print('Model # Parameters:', sum(p.numel() for p in m.parameters()))
    m.model_train(X_train, y_train)
    m.model_predict(X_test, y_test, plot=False)
    save_model(m)
    # Save scalers in a pickle file
    with open(m.model_config.model_src + '/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
"""

"""
# ----------------- MODEL 5/6/7/8 -----------------
steps_forward = (30, 90, 180, 365)
for step_forward in steps_forward:
    X_train, X_test, y_train, y_test, scalers = get_split_data('LSTM', 270, steps_fwd=step_forward)
    print(X_train.shape)
    m_cfg = ModelConfig(
        model_type = 'LSTM',
        num_layers = 3,
        hidden_dim = 256,
        dropout = 0.3,
        num_epochs = 350,
        batch_size = 64,
        lr = 0.00001,
        input_dim = X_train.shape[2],
        output_dim = y_train.shape[1],
        steps_forward = step_forward
    )
    m = Model(m_cfg)
    model_name = m.model_config.model_name
    print('Model # Parameters:', sum(p.numel() for p in m.parameters()))
    m.model_train(X_train, y_train)
    save_model(m)
    # Save scalers in a pickle file
    with open(m.model_config.model_src + '/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    # Delete the model to save memory and delete also the cache from pytorch
    del m
    torch.mps.empty_cache()
    m = load_model(model_name)
    m.model_predict(X_test, y_test, plot=False, force_save=True)
    del m
    torch.mps.empty_cache()
"""

# ----------------- MODEL 17/18/19/20 -----------------
steps_forward = (30, 90, 180, 365)
for step_forward in steps_forward:
    X_train, X_test, y_train, y_test, scalers = get_split_data('LSTM', 90, steps_fwd=step_forward)
    print(X_train.shape)
    m_cfg = ModelConfig(
        model_type = 'LSTM',
        num_layers = 10,
        hidden_dim = 64,
        dropout = 0.25,
        num_epochs = 300,
        batch_size = 32,
        lr = 0.0001,
        input_dim = X_train.shape[2],
        output_dim = y_train.shape[1],
        steps_forward = step_forward
    )
    m = Model(m_cfg)
    model_name = m.model_config.model_name
    print('Model # Parameters:', sum(p.numel() for p in m.parameters()))
    m.model_train(X_train, y_train)
    save_model(m)
    # Save scalers in a pickle file
    with open(m.model_config.model_src + '/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    # Delete the model to save memory and delete also the cache from pytorch
    del m
    torch.mps.empty_cache()
    m = load_model(model_name)
    m.model_predict(X_test, y_test, plot=False, force_save=True)
    del m
    torch.mps.empty_cache()

# m = load_model('model_1')
# m.model_predict(X_test_1, y_test_1)
