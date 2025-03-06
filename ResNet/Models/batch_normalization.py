import torch
import torch.nn as nn

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        """
        Custom implementation of 2D Batch Normalization.

        Args:
            num_features (int): Number of feature maps (channels).
            eps (float): Small value to avoid division by zero. Default: 1e-5.
            momentum (float): Momentum for running mean and variance. Default: 0.1.
            affine (bool): If True, includes learnable scale (gamma) and shift (beta) parameters. Default: True.
            track_running_stats (bool): If True, maintains running mean and variance. Default: True.
        """
        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Learnable affine parameters: γ (scale) and β (shift)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))  # Scale factor (γ)
            self.beta = nn.Parameter(torch.zeros(num_features))  # Shift factor (β)
        else:
            self.gamma = None
            self.beta = None

        # Running mean and variance for inference
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.running_mean = torch.zeros(num_features)
            self.running_var = torch.ones(num_features)
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        """
        Forward pass of Batch Normalization.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_features, height, width].

        Returns:
            Tensor: Batch normalized output.
        """
        if self.training:
            # Compute batch mean and variance along (N, H, W) axes
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # Shape: [1, C, 1, 1]
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)  # Shape: [1, C, 1, 1]

            # Update running statistics (if enabled)
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # Use running statistics during inference
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        # Normalize input
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation (γx + β) if enabled
        if self.affine:
            x_norm = self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)

        return x_norm