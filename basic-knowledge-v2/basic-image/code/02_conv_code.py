# -*- coding:utf-8 -*-
# Author: lqxu

#%%

import torch
 
from torch import nn, Tensor 
from torch.nn import functional as F

#%%

conv_layer = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=(3, 6), stride=1, padding=0)

# weight: [out_channels, in_channels, kernel_height, kernel_width]
# 每一个 kernel 的 shape 是 [kernel_height, kernel_width, in_channels], 一共有 out_channels 个 kernel
# bias: [out_channels, ]
print(
    conv_layer.weight.shape, conv_layer.bias.shape
)

# %%

input_tensor = torch.randn(3, 224, 224)

weight = torch.randn(10, 3, 5, 5)

torch.conv2d(input_tensor, weight).shape

# %%


def conv2d_v1(input_img: Tensor, weight: Tensor):
    
    in_channels, in_height, in_width = input_img.shape 
    
    out_channels, in_channels, k_height, k_width = weight.shape

    out_height = in_height - k_height + 1 
    out_width = in_width - k_width + 1
    
    kernel = weight.reshape(out_channels, -1)
    
    output_img = torch.zeros(out_channels, out_height, out_width)
    
    for idx_height in range(out_height):
        for idx_width in range(out_width):
            receptive_field = input_img[:, idx_height:idx_height+k_height, idx_width:idx_width+k_width]
            
            output_img[:, idx_height, idx_width] = kernel @ receptive_field.flatten()
            
            print(kernel.shape, receptive_field.flatten().shape)
            return None
    
    return output_img


conv2d_v1(torch.randn(3, 224, 224), torch.randn(10, 3, 7, 7))
# %%


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros(
        (in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


conv_trans = nn.ConvTranspose2d(
    3, 3, kernel_size=4, padding=1, stride=2, bias=False
)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));


#%%


def trans_conv(X: Tensor, K: Tensor):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i + h, j:j + w] += X[i, j] * K
    return Y


x = torch.randn(10, 10)
k = torch.randn(3, 3)

trans_conv(x, k)
# %%

with torch.no_grad():
    m = nn.ConvTranspose2d(1, 1, 3, bias=False)

    m.weight.copy_(k.data)

# %%


conv2d_m = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

input = torch.randn(1, 10, 10)

conv2d_m(input).backward(torch.randn(1, 8, 8))

conv2d_m.weight.grad

# %%

nn.UpsamplingNearest2d
