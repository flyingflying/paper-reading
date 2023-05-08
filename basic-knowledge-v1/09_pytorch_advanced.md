# PyTorch 相关的内容总结

[TOC]

## 爱因斯坦求和约定

`einsum` 的意思是 "爱因斯坦求和约定", 其中 `ein` 就是 `Einstein` 的缩写。向量点乘的本质是: 两个向量对应坐标的数相乘后得到一个新的向量, 再将这个新的向量中所有的元素相加。这样, 向量的点乘就可以拆成两部分: 元素级别的乘法 (element-wise product) 和 维度求和。在深度学习中, 矩阵和张量的主要作用是让样本向量的计算可以并行化, 其本质还是向量点乘。很多运算如果要转化成标准的向量运算或者矩阵运算, 需要进行大量的转置 (`transpose` 和 `permute`) 和 广播 (`broadcast`) 操作, 此时就有了 `einsum` 运算, 可以自动帮助我们在张量运算时省去大量的转置和广播操作。

关于如何使用, 可以看 [einsum函数介绍-张量常用操作](https://www.cnblogs.com/qftie/archive/2022/05/08/16245124.html) 这篇博客。简单来说, 在 `einsum` 中, 主要分为以下两种索引: **自由索引**和**求和索引**, 在运算过程中, `einsum` 会自动将两个张量的自由索引对齐, 缺失的索引会进行广播操作, 然后进行乘法操作, 最后在求和索引上面进行求和。

下面是用 `einsum` 实现矩阵点乘操作。注意如果方便使用点乘 (`.dot`, `@`, `torch.bmm` 等等) 的情况下应当尽可能地使用点乘操作, 因为 PyTorch 中对其进行了优化, 运算效率高。

```python
import torch 

A_Matrix = torch.randn(3, 4)
B_Matrix = torch.randn(4, 5)

# 3.52 us
result1 = A_Matrix @ B_Matrix
# 13.56 us
result2 = torch.sum(A_Matrix.unsqueeze(dim=-1) * B_Matrix.unsqueeze(dim=0), dim=1)
# 24.51 us
result3 = torch.einsum("xy,yz->xz", A_Matrix, B_Matrix)

# result1 和 result2 运算方式不一致, 但是运算的结果是同一个东西
print(torch.all(
    abs(result1 - result2) < 1e-6
))

# result2 和 result3 运算方式完全是一致的
print(torch.all(result2 == result3))
```

## Gradient Checkpoint

这是一种节省显存的方式, 在 PyTorch 中的文档是 [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html), 详细博客可以参考 [PyTorch之Checkpoint机制解析](http://www.manongjc.com/detail/27-npfegllifhhciob.html) 。这里我总结一下我的理解:

在 PyTorch 中, 显存的占用主要分成四个部分: 模型参数 (parameters), 模型参数的梯度 (gradients), 优化器状态 (optimizer states) 和 中间结果 (intermediate results)。前三个很好理解, 最后一个 "中间结果" 指的是在前向传播时为了计算梯度而保留的数据。gradient checkpoint 技术就是在前向传播时让模型不要保存 "中间结果"。

如果原来的代码写法是: `self.bert(input_ids, attention_mask, token_type_ids)`, 用 gradient checkpint 的写法如下: `torch.utils.checkpoint.checkpoint(self.bert, input_ids, attention_mask, token_type_ids)`, 其计算方式如下：

+ 前向传播时是在 `torch.no_grad()` 模式下运行的, 也就是 `torch.no_grad(): self.bert(input_ids, attention_mask, token_type_ids)`, 此时不会保存任何的 "中间结果";
+ 不保存 "中间结果", 取而代之的是保存 `self.bert` 函数的输入;
+ 反向传播时, 由于没有 "中间结果", 重新进行一次前向传播, 此时不是在 `torch.no_grad()` 模式下运行的, 会有 "中间结果", 可以计算梯度了;
+ 计算完成梯度后, 将 "中间结果" 所占用的显存立刻释放

这样的话, "中间结果" 只在反向传播时存在了, 其它时候就不存在, 这样就可以达到节省显存的目的。设置 gradient checkpoint 可以是任意的代码片段, 用函数封装即可。需要注意以下几点:

+ checkpoint 不宜设置地过多, 因为程序还是要保存函数的输入值, 它们也是要占用显存的
+ checkpoint 设置的模型跨度不宜太大, 因为 "中间结果" 在反向传播时还是会计算出来的
+ 传入函数的返回值建议是 `torch.Tensor` 或者 `Tuple[torch.Tensor]` 类型, 如果计算出来的张量放在其它数据结构中, 可能会导致反向传播失效
+ 传入函数参数的类型没有限制, 但是至少有一个是 `torch.Tensor` 且 `requires_grad=True`, 否则这个 checkpoint 的设置是没有意义的
+ 传入函数的参数必须全部以 positional arguments 的形式传入, 不能以 keyword arguments 的形式传入

本质上来说, 是一种用时间换空间的做法, 仅在训练时节省显存, 所有设置了 checkpoint 的函数都会计算两次。
