# -*- coding:utf-8 -*-
# Author: lqxu

import torch 
from torch import Tensor, nn

 
class GateFusionLayer(nn.Module):
    """ 完全按照论文 4.2 节的 Gate Fusion Mechanism 实现的 """
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.transform = nn.Linear(
            in_features=hidden_size * 2, out_features=hidden_size, bias=True
        )

    def forward(self, input_p: Tensor, input_q: Tensor) -> Tensor:
        # input1 和 input2 的 shape 应该是一致的, 是 [*, hidden_size]
        gate_input = torch.cat([input_p, input_q], dim=-1)  # [*, hidden_size * 2]

        gate = torch.sigmoid(self.transform(gate_input))  # [*, hidden_size]

        result = gate * input_p + (1 - gate) * input_q

        return result


if __name__ == "__main__":

    def test_gate_fusion_layer():
        batch_size_, num_tokens_, hidden_size_ = 10, 20, 768

        layer_ = GateFusionLayer(hidden_size_)
        
        print(layer_(
            input_p=torch.randn(batch_size_, num_tokens_, hidden_size_),
            input_q=torch.randn(batch_size_, num_tokens_, hidden_size_)
        ).shape)
        
        print(layer_(
            input_p=torch.randn(batch_size_, num_tokens_, hidden_size_),
            # 由于 concat 没有 broadcast 机制, 只能人为 broadcast 了
            input_q=torch.randn(batch_size_, hidden_size_).unsqueeze(dim=1).broadcast_to(batch_size_, num_tokens_, hidden_size_)
        ).shape)
    
    test_gate_fusion_layer()
