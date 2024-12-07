import pickle

import torch
from xlstm import xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig
from model.model import Model, ModelConfig, save_model, load_model, get_split_data


# ----------------- MODEL 9/10/11/12 -----------------
steps_forward = (30, 90, 180, 365)
for step_forward in steps_forward:
    X_train, X_test, y_train, y_test, scalers = get_split_data('xLSTM', 180, num_subwindows=6, steps_fwd=step_forward)
    print(X_train.shape)
    config = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
        slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4, conv1d_kernel_size=2,
                                                            bias_init="powerlaw_blockdependent"),
                                     feedforward=FeedForwardConfig(proj_factor=2, act_fn="gelu")),
        context_length=X_train.shape[1],
        num_blocks=5,
        embedding_dim=X_train.shape[2],
        slstm_at=[1, 3]
    )
    m_cfg = ModelConfig(
        model_type = 'xLSTM',
        xLSTM_config=config,
        dropout = 0.2,
        num_epochs = 200,
        batch_size = 32,
        lr = 0.00001,
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
    m.model_predict(X_test, y_test, plot=False, force_save = True)
    del m
    torch.mps.empty_cache()

# ----------------- MODEL 13/14/15/16 -----------------
steps_forward = (30, 90, 180, 365)
for step_forward in steps_forward:
    X_train, X_test, y_train, y_test, scalers = get_split_data('xLSTM', 360, num_subwindows=12, steps_fwd=step_forward)
    print(X_train.shape)
    config = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
        slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4, conv1d_kernel_size=2,
                                                            bias_init="powerlaw_blockdependent"),
                                     feedforward=FeedForwardConfig(proj_factor=2, act_fn="gelu")),
        context_length=X_train.shape[1],
        num_blocks=5,
        embedding_dim=X_train.shape[2],
        slstm_at=[1, 3]
    )
    m_cfg = ModelConfig(
        model_type = 'xLSTM',
        xLSTM_config=config,
        dropout = 0.2,
        num_epochs = 200,
        batch_size = 32,
        lr = 0.00001,
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
    m.model_predict(X_test, y_test, plot=False, force_save = True)
    del m
    torch.mps.empty_cache()