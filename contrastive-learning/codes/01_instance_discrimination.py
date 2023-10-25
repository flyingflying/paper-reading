# -*- coding:utf-8 -*-
# Author: lqxu
# Reference: https://github.com/zhirongw/lemniscate.pytorch 
# Paper: Unsupervised Embedding Learning via Invariant and Spreading Instance Feature
# Paper Link: https://arxiv.org/abs/1904.03436

"""
在 instance discrimination 的论文中, 作者是用 01_word2vec.md 笔记中的公式 (5.7) 实现的, 而不是公式 (5.8) 和 (5.9) 实现的。
除此之外, 这里的 噪声分布 直接选用的 均匀分布, 但是 噪声样本 非常多, 默认是 4096 个。
在 word2vec 的负采样中, 噪声分布 是根据词频来的, 较为复杂, 但是 噪声样本 很少, 一般 5 - 20 个。 
"""

import math

import torch
from torch import nn, Tensor, LongTensor


class NCEAverage(nn.Module):

    # reference: https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCEAverage.py 
    # 原始的代码中使用的是 torch.autograd.Function 实现的, 符合 caffe 的实现风格, 不符合 PyTorch 的实现风格, 这里按照 PyTorch 的风格重构了

    def __init__(self, num_images: int, hidden_size: int = 128, num_noise: int = 4096, temperature: float = 0.07, momentum: float = 0.5):
        """
        用于计算 logits 值, 等价于 softmax regression 中的最后一个线性层 !
        num_images: 训练集中的图片数量, 也就是 实体判别 的类别数量
        hidden_size: 图片向量的维度
        num_noise: NCE 中噪声样本的数量
        temperature: 计算 logit 值时的 tau
        momentum: 更新 memory bank 时的 momentum 值
        """
        super(NCEAverage, self).__init__()

        self.num_images = num_images
        self.hidden_size = hidden_size
        self.num_noise = num_noise
        self.temperature = temperature
        self.momentum = momentum

        # 初始化 memory_bank, 这样初始化得到的 图片向量 模长的均值约为 1
        stdv = 1. / math.sqrt(hidden_size / 3)
        # torch.rand: 均匀分布 U(0, 1); torch.randn: 正态分布 N(0, 1)
        memory_bank = torch.rand(num_images, hidden_size).mul_(2 * stdv).add_(-stdv)
        self.register_buffer("memory_bank", memory_bank)
        # torch.norm(self.memory_bank, dim=1).mean()
        
        normalization_constant = torch.tensor(-1.)
        self.register_buffer("normalization_constant", normalization_constant)
 
    def forward(self, image_vectors: Tensor, image_index: LongTensor) -> Tensor:
        """
        image_vectors: 图片向量 (经过了标准化, 模长为 1), shape: [batch_size, hidden_size]
        image_idx: 图片的索引值, shape: [batch_size, ]
        :ret probs 值 (不是 logits 值!!!) [batch_size, num_noise + 1]
        """
        batch_size = image_vectors.size(0)
        
        # 噪声分布是均匀分布, 因此用 torch.randint 即可, 否则需要用 torch.multinomial
        # 对应原始实现的 AliasMethod 类
        sample_indices = torch.randint(
            low=0, high=self.num_images, 
            size=(batch_size, self.num_noise + 1), 
            device=image_vectors.device
        )
        sample_indices[:, 0] = image_index
        # 等价于: sample_indices.select(dim=1, index=0).copy_(image_index)

        weights = self.memory_bank[sample_indices, ...]  # [batch_size, num_noise + 1, hidden_size]

        logits = torch.bmm(weights, image_vectors.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_noise + 1]
        probs = logits.div_(self.temperature).exp_()

        if (self.normalization_constant < 0).item():
            self.normalization_constant = probs.mean() * self.num_images
        
        probs.div_(self.normalization_constant)
        
        if self.training:
            # 根据 https://github.com/zhirongw/lemniscate.pytorch/issues/11 中的说法, 动量更新 等价于 Proximal Regularization
            self.memory_bank[image_index] = (1 - self.momentum) * self.memory_bank[image_index] + self.momentum * image_vectors

        return logits


class NCECriterion(nn.Module):

    # reference: https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCECriterion.py 进行了适当重构
    # 按照 01_word2vec.md 笔记中的公式 (5.7) 实现的, 而不是公式 (5.8) 和 (5.9) 实现的

    def __init__(self, num_images: int, num_noise: int = 4096, eps: float = 1e-7):
        super(NCECriterion, self).__init__()
        self.noise_prob = num_noise * (1. / num_images)
        self.eps = eps

    def forward(self, probs: Tensor) -> Tensor:
        # probs: [batch_size, num_noise + 1]
        batch_size = probs.size(0)
        
        pos_probs = probs[:, 0]  # [batch_size, ]
        pos_loss = torch.log(pos_probs / (pos_probs + self.noise_prob + self.eps))

        neg_probs = probs[:, 1:]  # [batch_size, num_noise]
        neg_loss = torch.log(self.noise_prob / (neg_probs + self.noise_prob + self.eps))
        
        return -(pos_loss.sum() + neg_loss.sum()) / batch_size


if __name__ == "__main__":

    # 重要参数: batch_size=256, num_epoches=200
    lemniscate_ = NCEAverage(num_images=2000, hidden_size=128, num_noise=200, temperature=0.07, momentum=0.5)
    
    probs_ = lemniscate_(
        torch.randn(100, 128), 
        torch.randint(low=0, high=2000, size=(100, ))
    )
    
    criterion_ = NCECriterion(num_images=2000, num_noise=200)
    
    loss_ = criterion_(probs_)
    
    print(loss_)
    