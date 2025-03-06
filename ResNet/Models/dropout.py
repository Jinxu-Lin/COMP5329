import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Custom implementation of Dropout without using nn.Dropout.

        Args:
            p (float): Probability of dropping a unit (default: 0.5)
        """
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        """
        Forward pass of Dropout.

        Args:
            x (Tensor): Input tensor of any shape.

        Returns:
            Tensor: Output tensor with dropout applied.
        """
        if not self.training or self.p == 0.0:
            return x  # No dropout during evaluation

        # Create a dropout mask (same shape as x, with elements being 0 with probability p)
        mask = (torch.rand_like(x) > self.p).float()  # Generates 0s and 1s
        
        # Scale the output to keep the expected value of activations the same
        return x * mask / (1.0 - self.p)