
# 正弦波位置编码

[TOC]

## 基础知识

这一部分涉及到一些三角函数的知识, 这里复习一下:

三角函数属于 **周期函数**。对于函数 $\sin(\omega \cdot x)$ 或者 $\cos(\omega \cdot x)$ 来说, 其 周期 是 $\frac{2\pi}{|\omega|}$ 。

和角公式:

$$
\sin(\alpha + \beta) = \sin\alpha \cdot \cos\beta + \cos\alpha \cdot \sin\beta
\tag{1}
$$

$$
\cos(\alpha + \beta) = \cos\alpha \cdot \cos\beta - \sin\alpha \cdot \sin\beta
\tag{2}
$$

## 公式

**位置编码** 指的是将句子中的 token **位置索引** 编码成 **位置向量**, 方便输入到模型中, 进行计算。我们设 **位置向量** 的维度是 $D$。也就是说, 对于每一个 **位置索引**, 我们需要为其寻找 $D$ 个数字, 构成 **位置向量**。

正弦波 (sinusoid) 位置编码是 [Transformers](https://arxiv.org/abs/1706.03762) 中使用的编码方式。我们用 $pe^{(i)}_d$ 表示 **位置索引** 为 $i$, 第 $d$ 个维度编码后的值, 其用公式表示如下:

$$
pe^{(i)}_d =
\begin{cases}
    \sin(\theta_{d} \cdot i) & d = 0, 2, 4, \cdots \\
    \cos(\theta_{d-1} \cdot i) & d = 1, 3, 5, \cdots
\end{cases}
\tag{3}
$$

其中, $\theta_d = 10000^{-d / D}$, 单调递减函数, 从 $1$ 递减到 $10^{-4}$ 。

每一个 **位置向量** 可以表示为: $\overrightarrow{p}^{(i)} = [pe^{(i)}_0, pe^{(i)}_1, \cdots, pe^{(i)}_D]$ 。

如果觉得公式难以理解, 可以看下面的表格 (每一行表示一个 **位置向量**):

| |$d=0$| $d=1$ | $d=2$ | $d=3$ | $\cdots$ | $d=766$ | $d=767$ |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|$i=0$|$\sin(\theta_0 \cdot 0)$|$\cos(\theta_0 \cdot 0)$|$\sin(\theta_2 \cdot 0)$|$\cos(\theta_2 \cdot 0)$| $\cdots$ |$\sin(\theta_{766} \cdot 0)$|$\cos(\theta_{766} \cdot 0)$|
|$i=1$|$\sin(\theta_0 \cdot 1)$|$\cos(\theta_0 \cdot 1)$|$\sin(\theta_2 \cdot 1)$|$\cos(\theta_2 \cdot 1)$| $\cdots$ |$\sin(\theta_{766} \cdot 1)$|$\cos(\theta_{766} \cdot 1)$|
|$\cdots$|$\cdots$|$\cdots$|$\cdots$|$\cdots$| $\ddots$ |$\cdots$|$\cdots$|
|$i=511$|$\sin(\theta_0 \cdot 511)$|$\cos(\theta_0 \cdot 511)$|$\sin(\theta_2 \cdot 511)$|$\cos(\theta_2 \cdot 511)$| $\cdots$ |$\sin(\theta_{766} \cdot 511)$|$\cos(\theta_{766} \cdot 511)$|

用代码表示如下:

```python
import math 
import torch 


def create_sinusoid_table(num_tokens: int = 512, hidden_size: int = 768):
    d = torch.arange(start=0, end=hidden_size, step=2).float()
    theta_d = 10000 ** (-d / hidden_size)
    # theta_d = torch.exp(-d / hidden_size * math.log(10000.))  # 数值稳定的写法
    
    idx = torch.arange(0, 512).float()
    
    table = torch.zeros(num_tokens, hidden_size)
    table[:, 0::2] = torch.sin(idx.unsqueeze(1) @ theta_d.unsqueeze(0))
    table[:, 1::2] = torch.cos(idx.unsqueeze(1) @ theta_d.unsqueeze(0))
    
    return table
```

## 公式解析

观察上述内容, 我们可以发现: 对于 $D$ 维的 **位置向量**, 我们可以将相邻维度的分成一组 ($d=0$ 和 $d=1$ 是一组, $d=2$ 和 $d=3$ 是一组, $d=766$ 和 $d=767$ 是一组, 以此类推), 这样一共有 $D/2$ 组。那么, 我们可以这样理解公式 $(3)$:

在一组中, 我们分别使用 $\sin$ 和 $\cos$ 函数进行编码; 不同组之间使用不同周期的三角函数进行编码, 周期由 $\theta_d$ 控制, 函数的输入始终是 **位置索引** $i$。或者说, 作者一共寻找了 $D$ 个不同的三角函数, 用来编码生成 **位置向量**。

每一组三角函数的周期是 $\frac{2\pi}{\theta_d}$, 取值范围在 $2\pi$ 到 $10000 \cdot 2\pi$ 之间。越靠后的维度, $\theta_d$ 越小, 三角函数的周期性也就越大。

在一组内, **位置向量** 可以写成 $\overrightarrow{p}^{(i)} = [\sin(\theta_d \cdot i), \cos(\theta_d \cdot i)]$。根据 公式 $(1)$ 和 $(2)$, 我们可以进行下面的推导:

$$
\begin{aligned}
\overrightarrow{p}^{(i+k)}
&= \begin{bmatrix}
    \sin(\theta_d \cdot i + k) \\
    \cos(\theta_d \cdot i + k)
\end{bmatrix} \\
&= \begin{bmatrix}
\sin(\theta_d \cdot i) \cdot \cos(k) + \cos(\theta_d \cdot i) \cdot \sin(k) \\
\cos(\theta_d \cdot i) \cdot \cos(k) - \sin(\theta_d \cdot i) \cdot \sin(k)
\end{bmatrix} \\
&=\begin{bmatrix}
    \cos(k) & \sin(k) \\
    -\sin(k)& \cos(k)
\end{bmatrix}
\cdot \begin{bmatrix}
    \sin(\theta_d \cdot i) \\ \cos(\theta_d \cdot i)
\end{bmatrix}  \\
&= \begin{bmatrix}
    \cos(k) & \sin(k) \\ -\sin(k) & \cos(k)
\end{bmatrix}
\cdot \overrightarrow{p}^{(i)}
\end{aligned}
\tag{4}
$$

因此, $\overrightarrow{p}^{(i+k)}$ 和 $\overrightarrow{p}^{(i)}$ 之间是 **线性关系**。作者认为, 由于这种线性关系的存在, 模型很容易捕捉到 **相对位置** 的信息。

关于 正弦波位置编码 的更多信息, 可以参考苏剑林的博客: [Transformer升级之路：1、Sinusoidal位置编码追根溯源](https://kexue.fm/archives/8231)。个人认为里面的假设太强了, 不具备普适性。额外提醒一下, 忘记 二元函数的泰勒展开 的同学可以先去复习一下。

## 引用

+ [BERT为何使用学习的position embedding而非正弦position encoding?](https://www.zhihu.com/question/307293465/answer/1039311514)
