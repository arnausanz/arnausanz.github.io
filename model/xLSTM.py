from Model import Model, ModelConfig
from data_prep import get_data
import torch
from xlstm import (
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

ctxt = 2

# Use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, y, scalers = get_data(ctxt)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

config = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=2, qkv_proj_blocksize=2, num_heads=2
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla" if device == "cpu" else "cuda",
            num_heads=2,
            conv1d_kernel_size=2,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.1, act_fn="gelu"),
    ),
    context_length=ctxt,
    num_blocks=2,
    embedding_dim=258,
    slstm_at=[1],

)

model_config = ModelConfig(model_type='xLSTM', xLSTM_config=config, output_dim=y.shape[1], device=device)

model = Model(model_config)
model.model_train(X_train, y_train, num_epochs=1, batch_size=ctxt, lr=0.0005)
test_loss = model.model_test(X_test, y_test)