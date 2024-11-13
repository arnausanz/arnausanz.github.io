import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=256,
    num_blocks=7,
    embedding_dim=128,
    slstm_at=[1],

)

xlstm_stack = xLSTMBlockStack(cfg)

x = torch.randn(4, 256, 128).to("cpu")  # Example input of batch size 4, sequence length 256, and feature dimension 128
xlstm_stack = xlstm_stack.to("cpu")  # Move the model to the same device as the input
y = xlstm_stack(x)  # Forward pass
print(y.shape)  # The output tensor's shape

loss_fn = nn.MSELoss()  # Mean Squared Error Loss (for regression tasks)
optimizer = Adam(xlstm_stack.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001


class FakeDataset(Dataset):
    def __init__(self, num_samples, seq_len, feature_dim):
        """
        Initializes the dataset with fake random data.

        :param num_samples: Number of samples in the dataset.
        :param seq_len: The sequence length (e.g., 256).
        :param feature_dim: The feature dimension (e.g., 128).
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Generates a random sample and a target."""
        # Random input tensor (e.g., batch_size, sequence_length, feature_dim)
        x = torch.randn(self.seq_len, self.feature_dim)  # Shape: (seq_len, feature_dim)

        # Random target tensor (could be same shape or something else depending on the task)
        y = torch.randn(self.seq_len, self.feature_dim)  # Same shape here (seq_len, feature_dim)

        return x, y


# Create a DataLoader to load fake data
fake_dataset = FakeDataset(num_samples=100, seq_len=256, feature_dim=128)
fake_loader = DataLoader(fake_dataset, batch_size=4, shuffle=True)

# Example data (train_data is a DataLoader object that yields input-output pairs)
for epoch in range(10):  # For 10 epochs
    xlstm_stack.train()  # Set model to training mode
    running_loss = 0.0

    for batch_idx, (x_batch, y_batch) in enumerate(fake_loader):  # Assuming train_loader is your DataLoader
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = xlstm_stack(x_batch)  # Get the model's prediction

        # Calculate the loss
        loss = loss_fn(output, y_batch)

        # Backward pass (compute gradients)
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track the loss
        running_loss += loss.item()

        # Print the loss every 100 steps
        if (batch_idx + 1) % 25 == 0:
            print(f"Epoch [{epoch+1}/10], Step [{batch_idx+1}/{len(fake_loader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0