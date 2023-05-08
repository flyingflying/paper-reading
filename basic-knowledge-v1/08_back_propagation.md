
# 反向传播

[TOC]

## 简介

我们知道, 在深度学习中, 参数的梯度计算采用的是 **反向传播** 机制, 用公式表示如下:

$$
\frac{\partial{loss}}{\partial{{weight}_{k}}} =
    \frac{\partial{loss}}{\partial{{output}_{n}}} \cdot
    \frac{\partial{{output}_{n}}}{\partial{{output}_{n-1}}} \cdots
    \frac{\partial{{output}_{k}}}{\partial{{weight}_{n}}}
\tag{1}
$$

在 PyTorch 中, 可以像下面这样进行验证

```python
import torch 
from torch import nn
from torch.nn import functional as F

batch_size, hidden_size, n_classes = 2, 768, 10

# 初始化模型
linear_model = nn.Linear(in_features=hidden_size, out_features=n_classes)

# 初始化输入
x = torch.randn(batch_size, hidden_size)
y = torch.randint(low=0, high=2, size=(batch_size, n_classes), dtype=torch.float32)  # y_hat 是预测结果

# 前向传播
output = linear_model(x)
# output 属于 non-leaf tensor, 因此需要 retain_grad 来保存梯度, 否则无法通过 grad 属性获得梯度
# 而所有的参数张量属于 leaf tensor, 因此可以直接通过 grad 属性获得梯度
output.retain_grad()
# 计算 loss
loss = F.binary_cross_entropy_with_logits(output, y)
# 反向传播
loss.backward()

print(torch.all(output.grad.T @ x == linear_model.weight.grad).item())
print(torch.all(output.grad.sum(axis=0) == linear_model.bias.grad).item())
```
