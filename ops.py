import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=True):
        super(ConvLayer, self).__init__()
        pad = kernel_size // 2 if padding == 'same' else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad, bias=bias)

    def forward(self, x):
        return self.conv(x)

def conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding='same'):
    pad = kernel_size // 2 if padding == 'same' else 0
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad)

def deconv2d(in_channels, out_channels, kernel_size=5, stride=2, padding='same'):
    pad = kernel_size // 2 if padding == 'same' else 0
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=pad)

def lrelu(x, leak=0.2):
    return torch.maximum(x, leak * x)

def prelu(channels):
    return nn.PReLU(num_parameters=channels)

def max_pool_2x2():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def conv(inputs, kernel_size, output_num, stride_size=1, padding='same', activation_func=F.relu):
    in_channels = inputs.size(1)
    pad = kernel_size // 2 if padding == 'same' else 0
    conv_layer = nn.Conv2d(in_channels, output_num, kernel_size, stride_size, padding=pad)
    x = conv_layer(inputs)
    return activation_func(x) if activation_func else x

def fc(inputs, output_size, activation_func=F.relu):
    if len(inputs.shape) > 2:
        inputs = torch.flatten(inputs, 1)
    in_features = inputs.size(1)
    linear_layer = nn.Linear(in_features, output_size)
    x = linear_layer(inputs)
    return activation_func(x) if activation_func else x

def lrn(inputs, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return F.local_response_norm(inputs, size=depth_radius, alpha=alpha, beta=beta, k=bias)

# Batch Norm Utility
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=True)

    def forward(self, x):
        return self.bn(x)

# Placeholder for concat operation
def concat(tensors, axis):
    return torch.cat(tensors, dim=axis)