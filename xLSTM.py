from model.Model import Model, ModelConfig
from model.data_prep import get_data_x
from xlstm import (
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from DataExtraction.utils import get_root_dir

"""
----------------- MODEL 1 -----------------

X, y, scalers = get_data_x(180, 6)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def new_model():
    config = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)), 
            slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4, conv1d_kernel_size=2, bias_init="powerlaw_blockdependent"),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")),
        context_length=X.shape[1],
        num_blocks=7,
        embedding_dim=X.shape[2],
        slstm_at=[1]
    )
    model_config = ModelConfig(model_type='xLSTM', xLSTM_config=config, output_dim=y.shape[1])
    model = Model(model_config)
    model.model_train(X_train, y_train, num_epochs=100, batch_size=25, lr=0.00001)
    model.model_predict(X_test, y_test)
    model.save_model()
"""

X, y, scalers = get_data_x(360, 12)
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
    config = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
        slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4, conv1d_kernel_size=2, bias_init="powerlaw_blockdependent"),
                                     feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")),
        context_length=X.shape[1],
        num_blocks=7,
        embedding_dim=X.shape[2],
        slstm_at=[2, 4]
    )
    model_config = ModelConfig(model_type='xLSTM', xLSTM_config=config, output_dim=y.shape[1])
    model = Model(model_config)
    model.model_train(X_train, y_train, num_epochs=150, batch_size=36, lr=0.00001)
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