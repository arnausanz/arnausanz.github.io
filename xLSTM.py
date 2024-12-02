from xlstm import xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig
from model.model import Model, ModelConfig, save_model, load_model, get_split_data


X_train_x, X_test_x, y_train_x, y_test_x, scalers_x = get_split_data('LSTM',90)

config = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
    slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4, conv1d_kernel_size=2,
                                                        bias_init="powerlaw_blockdependent"),
                                 feedforward=FeedForwardConfig(proj_factor=2, act_fn="gelu")),
    context_length=X_train_x.shape[1],
    num_blocks=3,
    embedding_dim=X_train_x.shape[2],
    slstm_at=[1]
)