import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Custom implementation of a linear layer (fully connected layer).

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (num_classes).
        """
        super(MLP, self).__init__()
        
        # Initialize weights and biases manually
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.01)  # Xavier initialization
        self.bias = torch.nn.Parameter(torch.zeros(out_features))  # Initialize bias to 0

    def forward(self, x):
        """
        Forward pass for the custom MLP layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return x @ self.weight.T + self.bias  # Matrix multiplication and bias addition
