from xlstm import xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig

config = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
    slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla", num_heads=4, conv1d_kernel_size=2,
                                                        bias_init="powerlaw_blockdependent"),
                                 feedforward=FeedForwardConfig(proj_factor=2, act_fn="gelu")),
    context_length=X.shape[1],
    num_blocks=3,
    embedding_dim=X.shape[2],
    slstm_at=[1]
)