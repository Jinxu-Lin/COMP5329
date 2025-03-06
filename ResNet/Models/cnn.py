import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        Custom implementation of 2D Convolution without using torch.nn.Conv2d.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int or tuple, optional): Zero-padding added to both sides. Default: 0.
            bias (bool, optional): If True, includes learnable bias. Default: True.
        """
        super(Conv2d, self).__init__()

        # Handle kernel size and stride as tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias

        # Initialize weights (out_channels, in_channels, kernel_height, kernel_width)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size) * 0.01)
        
        # Initialize bias if needed
        if self.bias_flag:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        """
        Forward pass of the custom Conv2d layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            Tensor: Output tensor after applying convolution.
        """
        batch_size, _, height, width = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # Apply padding
        x_padded = F.pad(x, (pw, pw, ph, ph), mode="constant", value=0)

        # Extract sliding window patches using unfold
        unfolded_x = F.unfold(x_padded, kernel_size=self.kernel_size, stride=self.stride)

        # Reshape weight tensor for matrix multiplication
        weight_flat = self.weight.view(self.out_channels, -1)  # Shape: [out_channels, in_channels * kh * kw]

        # Compute convolution via matrix multiplication
        conv_out = torch.matmul(weight_flat, unfolded_x)  # Shape: [out_channels, batch_size * output_size]
        
        # Reshape back to convolutional output format
        output_height = (height + 2 * ph - kh) // sh + 1
        output_width = (width + 2 * pw - kw) // sw + 1
        conv_out = conv_out.view(batch_size, self.out_channels, output_height, output_width)

        # Add bias if needed
        if self.bias is not None:
            conv_out += self.bias.view(1, -1, 1, 1)  # Broadcast bias across output feature maps

        return conv_out
