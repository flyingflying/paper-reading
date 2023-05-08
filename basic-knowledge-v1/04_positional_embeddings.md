
# Positional Embeddings

[TOC]

## 一、引言

CNN 和 RNN 架构本身就能捕捉到位置信息:

+ 对于 `Conv1d` 而言, 在一个 receptive field 中的词语是 BOW (bag-of-words) 的, 不考虑词的顺序, 但是整体计算出来的类似于 n-gram 模型, 还是有词序的;
+ 对于 `LSTM` 而言, 本身就是时序模型, 将句子中的每一个词都当成时序中的一个时刻, 一步步输入

Attention 的过程可以描述为:

> 目标: 将某一个词向量 $\vec{q}$ 重新编码成新的词向量 $\vec{o}$
> 设 $key$ 矩阵和 $value$ 矩阵中词向量的数目相同, 且一一对应, 那么整个过程为:
> 编码后的词向量 $\vec{o}$ 是 $value$ 矩阵中词向量的线性组合, 其系数由词向量 $\vec{q}$ 和 $key$ 矩阵中每一个词向量进行点乘, 然后用 softmax 函数归一化得到。

理解过程后不难发现, Attention 是无法捕捉到位置信息的: 对于某一个 query 向量而言, 交换 $key$ 矩阵和 $value$ 矩阵中词向量的位置 ($key$ 矩阵和 $value$ 矩阵中的词向量是一一对应的), 并不会改变其计算结果。在这种情况下, 就需要对位置进行编码。

本篇博客将会介绍四种类型的位置编码, 涉及到大量的数学知识, 做好准备。

## 二、训练式绝对位置编码

首先定义一下绝对位置: 每一个 token 在句子中的索引值称为绝对位置。假设模型输入的最大 token 数为 $s$, 那么 token 的索引范围是 $[0, s-1]$, 一共有 $s$ 中不同的值。

如何对这 $s$ 个不同的位置进行编码变成向量呢? 一种简单的方式是采用和词向量一样的编码方式, 使用 `nn.Embedding` 让模型自行去学习每一个位置的 "位置向量"。由于是模型自己通过迭代确定的, 这种编码方式也被称为 "训练式"。

BERT 中采用的就是这种编码方式的, 然后通过 "逐位相加" 的方式融入词向量中。

一般两个特征向量的融合有三种方式: "逐位相加", "逐位相乘" 和 concat, 具体选择哪一种一般是看实验效果。在大多数论文和实验中, 位置编码都是采用 "逐位相加" 的形式融入的。

## 三、正弦波位置编码

原版的 Transformer 使用的是 **正弦波** (sinusoid) 的位置编码方式, 用 "逐位相加" 的方式将 "位置向量" 融入词向量中。在看公式前, 我们先复习两个三角函数的相关知识:

> 1. 对于一个三角函数 $f(x) = \sin(\omega x)$ 而言, 其周期为 $\frac{2\pi}{|\omega|}$
> 2. 正余弦公式:
>    + $sin(\alpha + \beta) = sin\alpha \cdot cos\beta + cos\alpha \cdot sin\beta$
>    + $cos(\alpha + \beta) = cos\alpha \cdot cos\beta - sin\alpha \cdot sin\beta$

复习后我们来看公式:

设 $i$ 是词语在句子中的索引值, 位置向量一共有 $D$ 维, $d$ 表示位置向量维度的索引值, 其范围在 $[0, D)$ 之间的整数, 设 $\theta_{d} = 10000^{-d/D}$ ($d$ 是偶数), 则编码公式为:

$$
PE_{i, d} =
\begin{cases}
    \sin(i * \theta_{d}) & \text{d 是偶数} \\
    \cos(i * \theta_{d-1}) & \text{d 是奇数}
\end{cases}
\tag{3.1}
$$

我们假设 $D=2$, 那么位置编码向量可以写成: $p_{i} = [sin(i), cos(i)]$, 我们设 $k$ 是索引值的增量, 根据正余弦公式, 我们有:

$$
\begin{aligned}
p_{i+k} &= [sin(i+k), cos(i+k)] \\
        &=[sin(i)cos(k) + cos(i)sin(k), cos(i)cos(k) - sin(i)sin(k)] \\
        &=\begin{bmatrix}
            cos(k) & sin(k) \\
            -sin(k)& cos(k)
          \end{bmatrix} \cdot
          \begin{bmatrix}
            sin(i) \\
            cos(i)
          \end{bmatrix}  \\
        &=\begin{bmatrix}
            cos(k) & sin(k) \\
            -sin(k)& cos(k)
          \end{bmatrix} \cdot p_i
\end{aligned}
\tag{3.2}
$$

如果 $k$ 是一个定值, 那么我们可以知道 $p_{(i+k)}$ 和 $p_i$ 之间是线性关系, 这证明了论文 [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) 在 3.5 小节中所说的话:

> We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, $PE_{(pos+k)}$ can be represented as a linear function of $PE_{pos}$.

这个证明参考的是 [知乎](https://www.zhihu.com/question/307293465/answer/1039311514) 上网友的回答。也就是说, 作者认为模型可以通过这样的线性关系来捕捉到相对位置关系。

延申到 $D = 768$ 的情况, 根据公式 $(3.1)$, 我们将 "位置向量" 从起始位置开始每两个分成一组, 一组内的 $\theta_d$ 是相同的, 不同组的 $\theta_d$ 不相同, 一共可以分成 $768 / 2 = 384$ 组。$\theta_d$ 影响的是正余弦函数的周期, 每一组的正余弦函数周期是一致的, 为 $\theta_d / 2\pi$。每一组都可以推导出类似公式 $(3.2)$ 的结论。一句话概括就是, 正弦波位置编码实际上是用周期不同的正余弦函数对位置进行编码。

观察 $\theta_{d} = 10000^{-d/D}$, 我们可以得到 $\theta_d$ 的取值范围是 $[1, 1/10000)$, 对应周期范围是 $[2\pi, 2\pi * 10000)$, 为什么要将周期的最大值设置成 $2\pi * 10000$, 我的理解是为了保证位置编码的唯一性。维度索引越高, 正余弦函数的周期就越大, 重复的概率就越低。当周期为最大值时, 句子的长度要达到 3.14 万以上, 才可能出现重复的编码值。一般情况下句子是不可能到这么长的, 如果有, 一般的计算机也计算不了。

正弦波位置编码是绝对位置编码, 同时内部包含了相对位置信息, 可以说是非常巧妙。苏神在其博客 [Transformer升级之路：1、Sinusoidal位置编码追根溯源](https://kexue.fm/archives/8231) 中论述了很多其特性, 但由于其推理是基于很强的假设进行的 ~~(更主要的原因是我太菜了, 里面的公式写的太数学化了, 看不懂)~~, 这里就不说了。

## 四、训练式相对位置编码

什么是相对位置呢? 我们定义两个词向量之间的坐标差就是相对位置, 如果 $\vec{q}$ 在句子中的索引值为 $q$, $\vec{k}$ 在句子中的索引值是 $k$, 那么我们认为 $(q - k)$ 的值就是相对位置, 英文名是 relative position。注意这里的索引是 0-based index, 也就是说如果输入句子最多有 $m$ 个 token, 那么相对位置的最小值为 $ 0 - (m - 1) = -m + 1$, 最大值为 $(m - 1) - 0 = m - 1$, 一共有 $(2m - 1)$ 种不同的值。

由于深度学习是黑箱模型, 我们只能确定模型使用了显式表示的信息, 对于隐式表示的信息, 我们很难去确定模型是否捕捉到了。无论是上面提及的正弦波位置编码还是训练式绝对位置编码, 都仅仅是显式表示绝对位置信息, 没有显式表示相对位置信息。理论上可以通过绝对位置信息计算出相对位置信息, 但实际上我们无法判断模型是否进行了相关操作。因此, 谷歌团队在 2018 年提出了相对位置编码。

这里的相对位置的编码采用了和 BERT 一样的方式, 采用了 "训练" 的方式, 希望模型自己学习出每一个相对位置的 "相对位置向量", 使用的也是 `nn.Embedding`。那么问题来了, 怎么将 "相对位置向量" 融入 "词向量" 中呢? 答案是在计算 attention 时直接融入 query 和 key 向量中, 这样做的原因有两个:

+ 在整个 Transformer 架构中, 涉及到的计算主要有两种类型: `nn.Linear` 和 Attention。`nn.Linear` 实际上是对每一个词向量进行线性变换, 不需要考虑词序, 需要词序信息的只有 Attention 架构。
+ "绝对位置" 是 token 级别的, 而 "相对位置" 是 token-pairs 级别的, 也就是每两个 token 之间就有一个 "相对位置"。而 Attention 中 `attention_scores` 的计算也恰恰是 token-pairs 级别的 (每一个 query 向量都要和每一个 key 向量进行点乘), 正好可以融合进去。

具体的过程如下:

我们设 $x_{i}$ 和 $x_j$ 分别为索引是 $i$ 和 $j$ 的词向量, $W^Q$ 为计算 QUERY 矩阵的参数矩阵, $W^K$ 为计算 KEY 矩阵的参数矩阵, $e_{ij}$ 为 $x_{i}$ 和 $x_j$ 之间的注意力分数, 则:

$$
e_{ij} = (x_i W^Q) (x_j W^K) ^ T
$$

我们设索引 $i$ 和索引 $j$ 之间的 "相对位置向量" 为 $a_{(i - j)}$, 融入相对位置信息的 $e_{ij}$ 算法如下:

$$
e_{ij} = (x_i W^Q) (x_j W^K + a_{(i-j)}) ^ T
       = (x_i W^Q)(x_j W^K) ^ T + (x_i W^Q)(a_{(i-j)}) ^ T
\tag{4.1}
$$

式 $(4.1)$ 就是 论文 [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf) 中提出的计算方式, 对应论文中的公式 $(4)$ 和 $(5)$。一句话概括就是将 "相对位置向量" 通过 "逐位相加" 的方式融入 key 向量中。

论文中还尝试了将 "相对位置向量" 通过 "逐位相加" 的方式融入 value 向量中 (`context_layer` 的计算也是 token-pairs 级别的)，但是效果不如融入 key 向量中好, 这里也就不再说明了, 具体的可以见论文中的公式 $(3)$。

在 BERT 源码中, 这一部分对应 `relative_key`:

```python
# ## 申明编码的嵌入层
self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
...

# ## 计算相对位置
position_ids = torch.arange(self.max_position_embeddings)
distance = position_ids.view(-1, 1) - position_ids.view(1, -1)  # [seq_len_query, seq_len_key]
# ## 获取 "相对位置向量"
positional_embedding = self.distance_embedding(distance + (self.max_position_embeddings - 1))  # [seq_len_query, seq_len_key, head_size]
# ## 融入 attention_scores 中
relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)  # [batch_size, num_heads, seq_len_query, seq_len_key]
attention_scores = attention_scores + relative_position_scores
```

这里给出 `relative_position_scores` 的循环写法:

```python
import torch
batch_size, num_heads, seq_len, head_size = 2, 3, 4, 5
query_layer = torch.randn(batch_size, num_heads, seq_len, head_size)
positional_embedding = torch.randn(seq_len, seq_len, head_size)
result_test = torch.ones(batch_size, num_heads, seq_len, seq_len)
for batch_idx in range(batch_size):
    for head_idx in range(num_heads):
        for query_idx in range(seq_len):
            query_vec = query_layer[batch_idx, head_idx, query_idx]
            for key_idx in range(seq_len):
                relative_qk_vec = positional_embedding[query_idx, key_idx]
                result_test[batch_idx, head_idx, query_idx, key_idx] = query_vec @ relative_qk_vec  # 向量点乘
  
result_gold = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
print(torch.all(abs(result_test - result_gold) < 1e-6))
```

在论文 [Improve Transformer Models with Better Relative Position Embeddings](https://arxiv.org/pdf/2009.13658.pdf) 的 3.5.3 和 3.5.4 小节中, 亚马逊团队提出了一种改进方式: 将 "相对位置信息" 同时融入 query 向量和 key 向量中。其中 3.5.3 采用的是 "逐位相乘" 的融合方式 (或者说三个向量在一起计算相关性), 3.5.4 采用的是 "逐位相加" 的融合方式。由于论文的 4.4 小节中的实验, 在 BERT 上只使用了 3.5.4 中所说的编码方式, 同时 BERT 源码中也只复现了 3.5.4 的方法, 这里只说明 3.5.4 中提出的方法。3.5.4 中的公式如下:

$$
e_{ij} = (x_i W^Q + a_{(i-j)}) (x_j W^K + a_{(i-j)}) ^ T - \langle a_{(i-j)}, a_{(i-j)}\rangle
       = (x_i W^Q)(x_j W^K) ^ T + (x_i W^Q) (a_{(i-j)}) ^ T + (x_i W^K) (a_{(i-j)}) ^ T
\tag{4.2}
$$

需要注意的是融入 query 向量和 key 向量的 "相对位置向量" 是同一个向量。对应 BERT 源码中的 `relative_query_key` 这一部分:

```python
# ## 申明编码的嵌入层
self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
...

# ## 计算相对位置
position_ids = torch.arange(self.max_position_embeddings)
distance = position_ids.view(-1, 1) - position_ids.view(1, -1)  # [seq_len_query, seq_len_key]
# ## 获取 "相对位置向量"
positional_embedding = self.distance_embedding(distance + (self.max_position_embeddings - 1))  # [seq_len_query, seq_len_key, head_size]
# ## 融入 attention_scores 中
# relative_position_scores_query 和 上面的 relative_position_scores 是一致的, 往 key 向量中融入信息
relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
# relative_position_scores_key 是往 query 向量中融入信息, shape 为: [batch_size, num_heads, seq_len_query, seq_len_key]
relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
```

代码中的 `relative_position_scores_query` 和上面的 `relative_position_scores` 是一致的, 这里给出 `relative_position_scores_key` 的循环写法:

```python
import torch
batch_size, num_heads, seq_len, head_size = 2, 3, 4, 5
key_layer = torch.randn(batch_size, num_heads, seq_len, head_size)
positional_embedding = torch.randn(seq_len, seq_len, head_size)
result_test = torch.ones(batch_size, num_heads, seq_len, seq_len)
for batch_idx in range(batch_size):
    for head_idx in range(num_heads):
        for key_idx in range(seq_len):
            key_vec = key_layer[batch_idx, head_idx, key_idx]
            for query_idx in range(seq_len):
                relative_qk_vec = positional_embedding[query_idx, key_idx]
                result_test[batch_idx, head_idx, query_idx, key_idx] = key_vec @ relative_qk_vec  # 向量点乘

result_gold = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
print(torch.all(abs(result_test - result_gold) < 1e-6))
```

思考: 每一个 attention 层都需要一个相对位置编码, 由于 `nn.Embedding` 比较难训练, 这实际上将模型的训练难度提升了很多。能否让所有的 attention 共用同一个相对位置编码呢?

## 五、旋转式位置编码

旋转式位置编码是苏神在其博客 [Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265) 中提出的一种相对位置编码。全称是 Rotary Position Embedding, 简称 RoPE。里面涉及到很多关于复数的运算知识, 首先总结一下:

### 5.1 Prerequisite: 复数的相关知识

**复数** (complex number) 的一般表示: 设 $x$ 和 $y$ 为实数, $i = \sqrt{-1}$, 那么 $z = x + i \cdot y$ 就是复数, 其中 $x$ 被称为实部, $y$ 被称为是虚部。

这样一个复数就可以用两个实数来表示了。接下来, 我们可以用这两个实数组成一个向量, 从而构建一个空间平面, 这个空间平面就是**复平面** (complex plane)。在复平面内, 我们约定 $x$ 轴 (横坐标) 为实数轴, $y$ 轴 (纵坐标) 为虚数轴。这样, $x$ 轴上的点都是实数, 因此也被称为 **实轴** (real axis), $y$ 轴上除了原点外都是纯虚数, 因此也被称为 **虚轴** (imaginary axis)。

有了复平面后, 我们就可以求复数的模长了, 也就是复数在复平面上的点到原点之间的距离, 即 $r = \lvert z \rvert = \sqrt{x^2 + y^2}$ 。

不仅如此, 复数还可以用三角函数来表示: 设 $\theta$ 是向量 $(x, y)$ 与 $x$ 轴正方向的夹角, 并且 $\theta \in (-\pi, \pi]$, 那么 $z = r(cos \theta + i \cdot sin \theta)$, 此时也可以认为将复平面从直角坐标系转化到极坐标系中。

关于复数的运算法则, 其实和多项式的运算法则是一样的, 只是要注意多了一个条件: $i^2 = -1$ 。

+ 加法公式: 就是单纯的多项式展开, 减法类似

$$
(a + i \cdot b) + (c + i \cdot d) = (a + c) + i \cdot (b + d)
$$

+ 乘法公式: 也就是单纯的多项式展开, 注意 $i^2 = -1$

$$
(a + i \cdot b) * (c + i \cdot d) = (ac - bd) + i \cdot (bc + ad)
$$

+ 除法公式: 通用公式如下, 实际上运算时可以使用各种多项式运算技巧

$$
\frac{(a + i \cdot b)}{(c + i \cdot d)} = \frac{(a + i \cdot b) * (c - i \cdot d)}{(c + i \cdot d) * (c - i \cdot d)} = \frac{ac + bd}{c^2 + d^2} + i \cdot \frac{bc -ad}{c^2 + d^2}
$$

那么如何将复数和指数关联起来呢? 这就要用到 **欧拉公式** (Euler's Formula) 了:

$$
cos \theta + i \cdot sin \theta = e ^ {i \cdot \theta} \tag{5.1}
$$

这个公式的证明方式有多种, 具体参考可以维基百科, 这里不在论述。这个公式非常重要, 在下一部分的推导有很大作用, 因此我给它打上编号 $(5.1)$ 。

根据欧拉公式, 复数的指数表示方法为: $z = r * e ^ {i \cdot \theta}$ , 这样和指数的关系就建立起来了。

我们一般将复数 $\overline{z} = x - i \cdot y$ 称为是复数 $z = x + i \cdot y$ 的共轭复数, 在上面的除法公式中, 我们已经使用了相关概念, 为了使分母实数化, 我们给分子和分母同乘以分子的共轭复数。

有了共轭复数后, 我们给出三个结论, 这三个结论在下一部分的推导中起到了重要作用:

> 结论一: 两个二维向量的内积, 等于把它们当作复数时, 一个复数与另一个复数的共轭复数的乘积的实部。
>
> 结论二: 复数 $e^{i \cdot \theta}$ 的共轭复数为 $e^{-i \cdot \theta}$ 。
>
> 结论三: 如果 $a$ 和 $b$ 是两个复数, 那么: $\overline{a*b} = \overline{a}*\overline{b}$

三个证明都很简单, 结论二的证明需要用到欧拉公式, 结论一和结论三都是用多项式展开就可以证明了。这里给出结论一的证明:

设 $\vec{a} = (a_1, a_2)$, $\vec{b} = (b_1, b_2)$, 那么 $\vec{a}$ 所对应的复数为 $a_1 + i \cdot a_2$ , $\vec{b}$ 所对应的复数的共轭复数为 $b_1 - i \cdot b_2$ , $\vec{a} \cdot \vec{b} = a_1*b_1 + a_2*b_2$ 。

$$
(a_1 + i \cdot a_2) * (b_1 - i \cdot b_2) = (a_1 * b_1 + a_2 * b_2) + i \cdot (a_2 * b_1 - a_1 * b_2) = \vec{a} \cdot \vec{b} + i \cdot (a_2 * b_1 - a_1 * b_2)
$$

证明完毕。

最后说一下我的猜想: 实数的英文是 real number, 其字面含义应该是在这个世界上真实存在的数, 纯虚数的英文是 imaginary number, 其字面含义应该是虚构出来的数, 而复数 (complex number) 应该包含三部分: 实数, 纯虚数 以及 由实数和纯虚数组合而成的数。在中文语境中, 虚数指的是后两部分, 也就是 纯虚数 和 由实数和纯虚数组合而成的数。有时候字面含义可以更好地帮助我们理解这些概念, 毕竟数学语言有时候过于抽象。

### 5.2 旋转式位置编码 RoPE

首先定义一系列的数学符号: 设 $\vec{q}$ 为 $query$ 矩阵中的某一词向量, 其在句子中的位置索引为 $m$, $\vec{k}$ 为 $key$ 矩阵中的词向量, 其在句子中的位置索引为 $n$ 。由于本小节涉及到虚数的运算, 因此在本小节中 $i$ 表示虚根。

我们在计算注意力分数时是将 $\vec{q}$ 和 $\vec{k}$ 两个向量做点乘, 并且我们希望点乘的结果中包含两个向量的位置信息, 而不是像初版 transformer 或者 BERT 一样, 直接通过相加的方式给 word embedding 融入位置信息。基于这样的构想, 我们就需要对 $\vec{q}$ 和 $\vec{k}$ 进行一定的运算, 使其包含位置信息, 也就是我们需要寻找一个函数 $f$, 使其满足以下的关系:

$$
f(\vec{q}, m) \cdot f(\vec{k}, n) = g(\vec{q}, \vec{k}, m - n)
\tag{5.2}
$$

其中, 函数 $g$ 的输出值就是 $\vec{q}$ 和 $\vec{k}$ 的注意力分数, 也就是说分数由三部分共同决定: $\vec{q}$, $\vec{k}$, 以及两者之间相对位置关系 $(m - n)$ 。那么这个函数 $f$ 是什么呢？

我们设 $\vec{q}$ 和 $\vec{k}$ 都是二维向量, 也就是说词嵌入的维度为 2, 我们设 $\vec{q}$ 所对应的复数为 $q$, 共轭复数为 $\overline{q}$, $\vec{k}$ 所对应的复数为 $k$, 共轭复数为 $\overline{k}$, 另外, 设函数 $real: \mathbb{C} \to \mathbb{R}$ 的作用是获取复数的实部, 根据结论一, 我们可以得到下面的公式:

$$
\vec{q} \cdot \vec{k} = real(q * \overline{k})
$$

如果正向推导的话, 涉及到的数学语言太多, 不便于理解 ~~(也可能是我太菜了)~~。这里我们反向推导我们。我们假设函数 $f$ 的形式为:

$$
f(\vec{t}, j) = t * e^{i \cdot j\theta}
\tag{5.3}
$$

其中 $\vec{t} = (t_0, t_1)$, $j$ 表示位置索引, $\theta$ 是一个常数。我们将公式 $(5.3)$ 代入公式 $(5.2)$ 中, 并根据结论一, 结论二和结论三化简, 可以得到:

$$
g(\vec{q}, \vec{k}, m - n) =
f(\vec{q}, m) \cdot f(\vec{k}, n) =
real(q * e^{i \cdot m\theta} * \overline{k * e^{i \cdot n\theta}}) =
real(q * e^{i \cdot m\theta} * \overline{k} * e^{-i \cdot n\theta}) =
real(q * \overline{k} * e^{i \cdot (m - n) \theta})
\tag{5.4}
$$

公式 $(5.4)$ 正好符合我们一开始的想法, 非常的好!!! 也就是说我们需要的函数 $f$ 的形式是公式 $(5.3)$。将 $\vec{q}$ 和 $\vec{k}$ 经过公式 $(5.3)$ 的变换后, 两者的乘积就包含相对位置信息了。可是公式 $(5.3)$ 过于抽象, 我们可以用公式 $(5.1)$ 对其进行化简:

$$
f(\vec{t}, j) =
t * e^{i \cdot j\theta} =
(t_0 + i \cdot t_1) * (\cos (j * \theta) + i \cdot \sin (j * \theta)) =
\begin{bmatrix}
cos \enspace j\theta & -sin \enspace j\theta \\
sin \enspace j\theta &  cos \enspace j\theta
\end{bmatrix} \cdot
\begin{bmatrix}
t_0 \\ t_1
\end{bmatrix}
\tag{5.5}
$$

如果熟悉线性变换中的旋转矩阵, 就会发现公式 $(5.5)$ 中的矩阵和旋转矩阵是一致的, 因此被称为旋转式位置编码。

我们在代码中应该使用下面的公式:

$$
f(\vec{t}, j) =
\begin{bmatrix}
cos \enspace j\theta & -sin \enspace j\theta \\
sin \enspace j\theta &  cos \enspace j\theta
\end{bmatrix} \cdot
\begin{bmatrix}
t_0 \\ t_1
\end{bmatrix} =
\begin{bmatrix}
t_0 * \cos j\theta - t_1 * \sin j\theta \\
t_0 * \sin j\theta + t_1 * \cos j\theta
\end{bmatrix} =
\begin{bmatrix}
t_0 \\ t_1
\end{bmatrix} \otimes
\begin{bmatrix}
\cos j\theta \\ \cos j\theta
\end{bmatrix} +
\begin{bmatrix}
t_1 \\ t_0
\end{bmatrix} \otimes
\begin{bmatrix}
-\sin j\theta \\ \sin j\theta
\end{bmatrix}
\tag{5.6}
$$

其中 $\otimes$ 表示的是 element-wise production, 至此我们就解决了词向量是二维向量下的情况。

实际上词向量的维度肯定不止二维, 那怎么办呢？苏神采用的方法是**分组**, 两个为一组, 即第 1 维和第 2 维是一组, 第 3 维和第 4 维是一组, 以此类推。这和正弦波位置编码的做法是相似的。我们设词向量一共有 $D$ 维 ($D$ 是偶数), 那么一共要分成 $D / 2$ 组, 然后每一组旋转的角度是不同的。那么公式 $(5.6)$ 可以扩展成:

$$
f(\vec{t}, j) =
\begin{bmatrix}
t_0 \\ t_1 \\ t_2 \\ t_3 \\ \vdots \\ t_{D-2} \\ t_{D-1}
\end{bmatrix} \otimes
\begin{bmatrix}
\cos j\theta_0 \\ \cos j\theta_0 \\ \cos j\theta_2 \\ \cos j\theta_2 \\ \vdots \\ \cos j\theta_{D-2} \\ \cos j\theta_{D-2}
\end{bmatrix} +
\begin{bmatrix}
t_1 \\ t_0 \\ t_2 \\ t_1 \\ \vdots \\ t_{D-1} \\ t_{D-2}
\end{bmatrix} \otimes
\begin{bmatrix}
-\sin j\theta_0 \\ \sin j\theta_0 \\ -\sin j\theta_2 \\ \sin j\theta_2 \\ \vdots \\ -\sin j\theta_{D-2} \\ \sin j\theta_{D-2}
\end{bmatrix}
\tag{5.7}
$$

苏神采用了和正弦波位置编码一样的 $\theta_d$ 设置方式。我们设 $d$ 是维度索引, 取值范围是 `[0, D-1)`, 那么 $\theta_d = 10000^{-d/D}$ ($d$ 是偶数)。注意虽然设置的是一样的, 但是含义完全不一样, 一个表示的是周期, 一个表示的是角度。

相比之前的位置编码, 旋转式位置编码最大的特点在于不再构建 "位置向量" 了, 而是直接对特征向量进行旋转变换, 使其乘积中包含位置信息, 可以说改变了之前的思路。

如果我们只看第一个分组, 此时 $\theta_0 = 1$ , 对于正弦波位置编码来说, 索引为 $j$ 的 "位置向量" 就是 $[sin(j), cos(j)]$; 对于旋转式位置编码来说, 就是将特征向量绕原点逆时针旋转 $j$ 角度。推广到一般情况, 如果某一分组有 $\theta_d=\theta$，对于正弦波位置编码来说, 索引为 $j$ 的 "位置向量" 就是 $[sin(\theta j), cos(\theta j)]$; 对于旋转式位置编码来说, 就是将特征向量绕原点逆时针旋转 $\theta j$ 角度。随着 $d$ 增加, $\theta_d$ 减小, 周期增加, 旋转的角度变小。

对于正弦波位置编码来说, $D = 768$, 但是对于旋转式位置编码来说, $D=768/12=64$。

### 5.3 References

+ [复数的三角表示](https://zhuanlan.zhihu.com/p/350055459)
+ [复数的三角表达与指数表达](https://www.jianshu.com/p/8685c2a669bf)
+ [Euler's formula - Wikipedia](https://en.wikipedia.org/wiki/Euler%27s_formula)
+ [Complex plane - Wikipedia](https://en.wikipedia.org/wiki/Complex_plane)
+ [Transformer升级之路：1、Sinusoidal位置编码追根溯源](https://kexue.fm/archives/8231)
+ [Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265)
+ [让研究人员绞尽脑汁的Transformer位置编码](https://kexue.fm/archives/8130)

## 六、总结

位置编码可以说是 Transformer 架构的重要问题之一。本篇博客涉及到的思想有很多, 这里总结一下:

+ 两个特征向量的融合方式: concat, 逐位相加, 逐位相乘
+ 在深度学习中, 没有显式表示的信息我们很难去确定模型是否捕捉到了
+ 高维特征可以看成是由多个二维特征拼接而成 (分组的思想)
+ 正余弦公式和正弦波位置编码
+ 复数平面和旋转式位置编码
+ 位置编码不一定要构建 "位置向量"
+ 绝对位置是 token 级别的, 相对位置是 token-pairs 级别的, attention 也是 token-pairs 级别的
