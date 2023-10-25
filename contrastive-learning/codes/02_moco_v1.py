# -*- coding:utf-8 -*-
# Author: lqxu
# Reference: https://github.com/facebookresearch/moco 
# Paper: Momentum Contrast for Unsupervised Visual Representation Learning 
# Paper Link: https://arxiv.org/abs/1911.05722

from typing import *

import torch
from torch import nn, Tensor 


class MoCo(nn.Module):

    # reference: https://github.com/facebookresearch/moco/blob/main/moco/builder.py 
    # 和原始的代码相比, 我删除了 batch shuffle 那一部分, 提高代码的可读性

    def __init__(self, base_encoder: Callable, hidden_size: int = 128, num_noise: int = 65536, momentum: float = 0.999, temperature: float = 0.07):
        """
        base_encoder: torchvision.models 库中的函数, 比方说 torchvision.models.resnet50
        hidden_size: 图片向量的维度
        num_noise: 负样本的个数 (K), 队列的长度
        momentum: 动量编码器的 momentum
        temperature: 计算 logit 值时的 tau
        """
        super(MoCo, self).__init__()

        self.num_noise = num_noise
        self.momentum = momentum
        self.temperature = temperature

        # ## 初始化编码器
        # 在 torchvision.models 中, num_classes 表示最终输出的 类别 数量
        # 这里是简化的写法, 直接用其作为 hidden_size, 个人认为, 这是一种 hack (inelegant solution) 的写法
        self.encoder_q = base_encoder(num_classes=hidden_size)  # encoder_q 是实际上用于 下游任务 的编码器
        self.encoder_k = base_encoder(num_classes=hidden_size)  # 动量编码器, 不会用于 下游任务

        # 动量编码器不需要计算梯度, 完全使用 动量 来更新
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # ## 初始化队列
        self.register_buffer("queue", torch.randn(hidden_size, num_noise))
        self.queue = nn.functional.normalize(self.queue, dim=0)  # 直接 L2 标准化 成 方向向量

        self.register_buffer("queue_ptr", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """ 更新动量编码器 """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor) -> None:
        # keys: [batch_size, hidden_size]
        batch_size = keys.shape[0]
        
        if self.num_noise % batch_size != 0:
            # 需要注意, 这里为了简化, batch_size 必须是 num_noise 的倍数
            # 同时, DataLoader 中的 drop_last 参数必须要设置为 True (不然最后一个 batch_size 的大小不可控)
            raise ValueError("batch_size 必须是 num_noise 的倍数")

        # 更新队列
        ptr = self.queue_ptr.item()
        self.queue[:, ptr : ptr + batch_size] = keys.T  # 使用 index 时一定要严格区分 类型, 属于 int, LongTensor 还是 BoolTensor

        # 更新指针
        self.queue_ptr = (self.queue_ptr + batch_size) % self.num_noise

    def forward(self, image_q: Tensor, image_k: Tensor) -> Tensor:
        """
        image_q 和 image_k 是同一个图片经过两次不同的数据增强后得到的图片, shape 是 [batch_size, n_channels, img_heights, img_weights]
        :ret loss
        """

        # ## step1: 对 image_q 进行编码
        q = self.encoder_q(image_q)  # [batch_size, hidden_size]
        q = nn.functional.normalize(q, dim=1)

        # ## step2: 对 image_k 进行编码
        with torch.no_grad():
            self._momentum_update_key_encoder()  # 更新动量编码器

            k = self.encoder_k(image_k)  # [batch_size, hidden_size]
            k = nn.functional.normalize(k, dim=1)

        # ## step3: 计算 logits
        l_pos = torch.einsum("bh,bh->b", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("bh,hk->bk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)  # [batch_size, num_noise + 1]
        logits /= self.temperature

        # ## step4: 计算 loss (InfoNCE)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # [batch_size, ]
        # https://github.com/facebookresearch/moco/blob/main/main_moco.py 中的第 289 行和 417-418 行
        loss = nn.functional.cross_entropy(logits, labels)

        # ## step5: 将 k 存入 queue 中
        self._dequeue_and_enqueue(k)

        return loss


if __name__ == "__main__":
    from torchvision.models import resnet18

    # 重要参数: batch_size=256, num_epoches=200
    image_q_ = torch.randint(low=0, high=256, size=(16, 3, 224, 224), dtype=torch.float) / 255.
    image_k_ = torch.randint(low=0, high=256, size=(16, 3, 224, 224), dtype=torch.float) / 255.
    
    model_ = MoCo(resnet18)
    
    print(model_(image_q_, image_k_))
    