"""Adapted from https://github.com/jhjacobsen/fully-invertible-revnet
"""
import ipdb

import torch
import torch.nn as nn


'''
Convolution Layer with zero initialization
'''
class Conv2dZeroInit(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=0, logscale=3.):
        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=padding)
        self.register_parameter("logs", nn.Parameter(torch.zeros(channels_out, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out * torch.exp(self.logs * self.logscale_factor)

'''
Convolution Interlaced with Actnorm
'''
class Conv2dActNorm(nn.Module):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=None, an_stable_eps=0):
        from invertible_layers2_mem import ActNorm
        super(Conv2dActNorm, self).__init__()
        padding = (filter_size - 1) // 2 or padding
        self.conv = nn.Conv2d(channels_in, channels_out, filter_size, padding=padding, bias=False)
        self.actnorm = ActNorm(channels_out, stable_eps=an_stable_eps)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.conv(x)
        x = self.actnorm.forward_(x, -1)[0]
        return x

'''
Linear layer zero initialization
'''
class LinearZeroInit(nn.Linear):
    def reset_parameters(self):
        self.weight.data.fill_(0.)
        self.bias.data.fill_(0.)

'''
Shallow NN used for skip connection. Labelled `f` in the original repo.
'''
def NN(in_channels, hidden_channels=128, channels_out=None, actnorm=False, an_stable_eps=0, relu_inplace=False, zero_init=True):
    channels_out = channels_out or in_channels

    if zero_init:
        last_conv_layer = Conv2dZeroInit
    else:
        last_conv_layer = nn.Conv2d

    if actnorm:
        return nn.Sequential(
            Conv2dActNorm(in_channels, hidden_channels, 3, stride=1, padding=1, an_stable_eps=an_stable_eps),
            nn.ReLU(inplace=relu_inplace),
            Conv2dActNorm(hidden_channels, hidden_channels, 1, stride=1, padding=0, an_stable_eps=an_stable_eps),
            nn.ReLU(inplace=relu_inplace),
            last_conv_layer(hidden_channels, channels_out, 3, stride=1, padding=1))
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0),
            nn.ReLU(inplace=relu_inplace),
            last_conv_layer(hidden_channels, channels_out, 3, stride=1, padding=1))
