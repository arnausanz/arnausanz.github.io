from model.Model import Model, ModelConfig
from model.data_prep import get_data_x
import torch
from xlstm import (
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from DataExtraction.utils import get_root_dir

torch.set_default_device('mps')


X, y, scalers = get_data_x(180, 6)

print('X data shape:', X.shape)
print('y data shape:', y.shape)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


def new_model():
    config = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",
                num_heads=4,
                conv1d_kernel_size=2,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
        context_length=X.shape[1],
        num_blocks=7,
        embedding_dim=X.shape[2],
        slstm_at=[1],

    )
    model_config = ModelConfig(model_type='xLSTM', xLSTM_config=config, output_dim=y.shape[1])
    model = Model(model_config)
    model.model_train(X_train, y_train, num_epochs=100, batch_size=25, lr=0.00001)
    model.model_test(X_test, y_test)
    model.save_model()

def load_existing(name):
    model = Model(load=get_root_dir() + '/model/final_models/' + name, save=False)
    preds = model.model_test(X_test, y_test, test_mode=False)
    print('Predictions:', preds)

new_model()