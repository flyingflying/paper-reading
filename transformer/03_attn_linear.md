
# Linear Attention

[TOC]

## 预备知识

首先, 明确一些概念。对于 [加权平均数](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean) 来说, **权重值** 只要是 **正数** 即可, 在计算时会自动 **标准化**。而 **标准化权重值** 则额外要求所有的 **权重值** 和为 $1$。

本文所说的 **标准化** 指的是 [加权平均数](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean) 中权重值的标准化。标准化后的数据取值范围在 $0$ 到 $1$ 之间, 并且和为 $1$。

如果 $\bold{A} \in \mathbb{R}^{m \times n}$, $\bold{B} \in \mathbb{R}^{n \times k}$, 那么 $\bold{A} \cdot \bold{B}$ 的复杂度 (complexity) 应该是 $O(m \times n \times k)$。对此有疑问的, 建议参考: [Computational complexity of matrix multiplication](https://en.wikipedia.org/wiki/Computational_complexity_of_matrix_multiplication)。

标准的 Attention 计算公式是:

$$
\bold{O} = \mathrm{softmax_{row}} \left(
    \frac{\bold{Q} \cdot \bold{K}^{\mathsf{T}}}{\sqrt{d}}
\right) \cdot \bold{V}
\tag{1.1}
$$

其中, $\bold{Q} \in \mathbb{R}^{n \times d}$, $\bold{K} \in \mathbb{R}^{n \times d}$, $\bold{V} \in \mathbb{R}^{n \times d}$, $\bold{O} \in \mathbb{R}^{n \times d}$。

本文统一使用, $\mathrm{softmax_{row}}$ 表示对矩阵的每一行进行 softmax 操作, $\mathrm{softmax_{column}}$ 表示对矩阵的每一列进行 softmax 操作。

为了方便理解和推理, 也可以写成:

$$
\bold{o}^{\mathsf{T}} = \mathrm{softmax} \left(
    \frac{\bold{q}^{\mathsf{T}} \cdot \bold{K}^{\mathsf{T}}}{\sqrt{d}}
\right) \cdot \bold{V}
\tag{1.2}
$$

其中, 向量 $\bold{q}$, $\bold{k}$, $\bold{v}$ 和 $\bold{o}$ 分别表示矩阵 $\bold{Q}$, $\bold{K}$, $\bold{V}$ 和 $\bold{O}$ 中的 词向量。

## 线性 Attention 原理

从公式 $(1.1)$ 可以看出, Attention 的复杂度应该是 $O(n^2d)$。由于 $d$ 是常数 (BERT 模型中是 $64$), 大多数情况下是小于 $n$ 的, 因此可以忽略不计, 复杂度就是 $O(n^2)$。

我们知道, 矩阵的乘法虽然不满足交换律, 但是满足结合律, 即 $(\bold{Q} \cdot \bold{K}^{\mathsf{T}}) \cdot \bold{V} = \bold{Q} \cdot (\bold{K}^{\mathsf{T}} \cdot \bold{V})$。

如果我们先将 $\bold{K}^{\mathsf{T}}$ 和 $\bold{V}$ 相乘, 再乘以 $\bold{Q}$, 此时的复杂度就是 $O(nd^2)$, 忽略掉 $d$ 后就变成了 $O(n)$。此时的复杂度就是 **线性** 的！通过这种方式实现的 Attention 被称为 **线性 Attention**。

对于 Attention 中的 softmax 函数来说, 我们可以这样理解:

向量 $\bold{q}$ 和 $\bold{k}$ 点乘的结果有正有负, 不能直接作为 **权重值**。那么, 我们对其进行 **指数运算**, 这样一定是正数了。我们将其作为 **权重值**。除此之外, softmax 还对所有的 **权重值** 进行了 **标准化**。那么, **权重值** 公式为:

$$
sim(\bold{q}, \bold{k}) = \exp(\frac{
    \bold{q}^{\mathsf{T}} \cdot \bold{k}
}{
    \sqrt{d}
})
\tag{2.1}
$$

那么, 公式 $(1.2)$ 可以写成如下的形式:

$$
\begin{align*}
\bold{o} &= \sum_{i=1}^n
\frac{
    sim(\bold{q}, \bold{k}_i)
}{
    \sum_{j=1}^n sim(\bold{q}, \bold{k}_j)
} \cdot \bold{v}_i \\
&= \frac{
    \sum_{i=1}^n sim(\bold{q}, \bold{k}_i) \cdot \bold{v}_i
}{
    \sum_{j=1}^n sim(\bold{q}, \bold{k}_j)
}
\end{align*}
\tag{2.2}
$$

道理很简单, 向量 $\bold{o}$ 是矩阵 $\bold{V}$ 中词向量的线性组合, 线性组合的系数就是 $sim(\bold{q}, \bold{k})$ 经过标准化后的结果。公式 $(2.2)$ 描述的就是这样的过程。

我们可以将公式 $(2.1)$ 中的指数函数 $\exp$ 理解为 **转换函数**, 其将 **实数域** 映射到 **正数域**。现在的问题是, 我们对 $\bold{q}^\mathsf{T} \cdot \bold{k}$ 的结果进行转换, 无法将 $\bold{K}^\mathsf{T}$ 和 $\bold{V}$ 先进行运算。

如果我们先对 $\bold{Q}$ 和 $\bold{K}$ 进行转换, 保证其在 **正数域** 内, 那么他们点乘的结果也在 **正数域** 内。此时就可以改变运算顺序了。推理过程如下:

设 $\phi$ 和 $\varphi$ 分别是矩阵 $\bold{Q}$ 和 $\bold{K}$ 的 **转换函数**。根据上面所说, 此时的 **权重** 计算公式为:

$$
sim (\bold{q}, \bold{k}) = \phi (\bold{q})^{\mathsf{T}} \cdot \varphi (\bold{k})
\tag{2.3}
$$

将公式 $(2.3)$ 代入 $(2.2)$ 中, 可以得到:

$$
\begin{align*}
\bold{o} &= \frac{
    \sum_{i=1}^n \left [ \phi (\bold{q})^{\mathsf{T}} \cdot \varphi (\bold{k}_i) \right ] \cdot \bold{v}_i
}{
    \sum_{j=1}^n \phi (\bold{q})^{\mathsf{T}} \cdot \varphi (\bold{k}_j)
}
\\ &= \frac{
    \phi (\bold{q})^{\mathsf{T}} \cdot \varphi (\bold{K})^{\mathsf{T}} \cdot \bold{V}
}{
    \phi (\bold{q})^{\mathsf{T}} \cdot \sum_{j=1}^n \varphi (\bold{k}_j)
}
\end{align*}
\tag{2.4}
$$

在公式 $(2.4)$ 中, 第一步到第二步 分子 的化简过程实际上是公式 $(1.2)$ 到 $(2.2)$ 的逆过程。分母实际上是一个标量。额外提醒一点, 分子和分母的 $\phi (\bold{q})^{\mathsf{T}}$ 不能约掉, 因为这里是 点乘 运算。

将公式 $(2.4)$ 整理成公式 $(1.1)$ 的形式, 可以得到:

$$
\bold{O} = \frac{
    \phi (\bold{Q}) \cdot \left[ \varphi (\bold{K})^{\mathsf{T}} \cdot \bold{V} \right]
}{
    \phi (\bold{Q}) \cdot \mathrm{sum_{column} (\varphi (\bold{K}))}
}
\tag{2.5}
$$

其中, $\mathrm{sum_{column}}$ 表示对矩阵的列求和, 也就是所有的行向量 (词向量) 相加。利用公式 $(2.5)$, 我们就可以实现 **线性 Attention** 了。

**转换函数** 应该选择什么呢? 也就是 $\phi$ 和 $\varphi$ 的形式是什么?

首先, 我认为 指数函数 就是一种选择, 即 $\phi (x) = \varphi (x) = \exp (x)$, 不过会有 指数爆炸 的问题。

[Transformers are RNNs](https://arxiv.org/abs/2006.16236) 的作者提出, $\phi (x) = \varphi (x) = \mathrm{elu} (x) + 1$ 。

除此之外, 还有一些工作不能完全套用公式 $(2.5)$, 不过核心思想是不变的, 即 改变矩阵的运算顺序。

## Efficient Attention

Efficient Attention 出自 2020 年商汤的论文: [Efficient Attention: Attention with Linear Complexities](https://arxiv.org/abs/1812.01243)。其计算方式如下:

$$
\bold{O} =
\mathrm{softmax_{row}} \left(\frac{\bold{Q}}{\sqrt{d}} \right)
\cdot
\mathrm{softmax_{column}} \left( \frac{\bold{K}}{\sqrt{d_k}} \right)^{\mathsf{T}} \cdot \bold{V}
\tag{3.1}
$$

为了方便描述, 我们记 $\bold{Q}^{\prime} = \mathrm{softmax_{row}} \left( \frac{\bold{Q}}{\sqrt{d}} \right)$, $\bold{K}^{\prime} = \mathrm{softmax_{column}} \left( \frac{\bold{K}}{\sqrt{d_k}} \right)$。

公式 $(3.1)$ 利用了如下的性质: 如果 $\bold{Q}^{\prime}$ 的行向量是 **标准化** 的, 且 $\bold{K}^{\prime}$ 的列向量是 **标准化** 的, 那么 $\bold{Q}^{\prime} \cdot \bold{K}^{\prime \mathsf{T}}$ 的行向量也是 **标准化** 的。证明如下:

我们用 $\bold{q}^{\prime}$ 和 $\bold{k}^{\prime}$ 表示矩阵 $\bold{Q}^{\prime}$ 和 $\bold{K}^{\prime}$ 中的 词向量。用 $\mathrm{ele \_ sum}$ 表示对向量中所有的元素求和。

由于矩阵 $\bold{Q}^{\prime}$ 中每一行是 **标准化** 的, 那么:

$$
\mathrm{ele\_sum} (\bold{q}) = 1 \tag{3.2}
$$

由于矩阵 $\bold{K}^{\prime}$ 中每一列是 **标准化** 的, 那么:

$$
\sum_{i=1}^n \bold{k}_i^{\prime} = \overrightarrow{1} \tag{3.3}
$$

其中, $\overrightarrow{1}$ 表示元素全是 $1$ 的向量。这里可能很多人没有想到。

现在, 我们可以进行如下推导:

$$
\begin{align*}
    &\enspace \mathrm{ele\_sum} ( \bold{q}^{\prime \mathsf{T}} \cdot \bold{K}^{\prime \mathsf{T}}) \\
    &= \bold{q}^{\prime \mathsf{T}} \cdot \bold{k}^{\prime}_1 + \bold{q}^{\prime \mathsf{T}} \cdot \bold{k}^{\prime}_2 + \cdots + \bold{q}^{\prime \mathsf{T}} \cdot \bold{k}^{\prime}_n \\
    &= \bold{q}^{\prime \mathsf{T}} \cdot \sum_{i=1}^n \bold{k}_i^{\prime} \\
    &= \bold{q}^{\prime \mathsf{T}} \cdot \overrightarrow{1} \\
    &= \mathrm{ele\_sum} ( \bold{q}^{\prime}) \\
    &= 1
\end{align*}
\tag{3.4}
$$

证明完毕。回到公式 $(3.1)$ 中, 我们可以改变矩阵乘法的运算顺序, 以减少复杂度到线性。其和公式 $(2.5)$ 还是有一定差别的。

## 线性 Attention 缺点

原始 attention 的复杂度是 $O(n^2d)$, 线性 attention 的复杂度是 $O(nd^2)$。只有当 $n > d$ 时, 线性 attention 才能发挥优势, 否则不行。

对于 BERT 来说, $d = 64$, 也就是说, 如果你的文本非常短, 只有 20 - 40 个字左右, 用线性 attention 都是不划算的。

那么, 当 $n > d$ 时, 线性 attention 有什么问题呢?

我们知道, 对于矩阵 $\bold{A} \in \mathbb{R}^{n \times d}$ 和矩阵 $\bold{B} \in \mathbb{R}^{d \times n}$ 来说, 如果 $rank(\bold{A}) = rank(\bold{B}) = d$, 那么 $rank(\bold{A} \cdot \bold{B}) = d$。

也就是说, 对于公式 $(2.5)$ 和 $(3.1)$ 来说, attention 矩阵的 秩 最大只能是 $d$。

但是对于原始的 attention 矩阵来说, 我们对 $\bold{Q} \cdot \bold{K}^{\mathsf{T}}$ 的结果进行了 $\exp$ 转换, 这会改变矩阵的 秩。如果你测试训练好 BERT 模型 attention 矩阵的秩, 会发现几乎都是 满秩 的 (等于 $n$)。

attention 矩阵是 **满秩矩阵** 意味着什么呢? 不同 token 的关注点是不一致的! 这极大的增加了特征的多样性。

除此之外, 我们知道 softmax 函数是 one-hot argmax 函数的光滑近似函数。也就是说, 原始的 attention 矩阵倾向于 **稀疏**, 这符合我们的认知: 一个 token 只需要关注部分 token 就可以了。如果所有 token 都关注, 那和平均池化就没有区别了。

综上所说, 原始 attention 的优点在于: attention 矩阵是一个 **满秩** 的, 倾向于 **稀疏** 的矩阵。显然, 线性 attention 达不到这样的效果。

额外说明一点, 你或许会产生疑问: 一般 稀疏矩阵 不应该是 低秩 的吗? 确实, 但是 attention 矩阵没有那么稀疏, 一般 稀疏度 (接近 0 元素的占比) 最大在 60% 左右。

## FLatten Transformer

FLatten Transformer 出自 2023 年清华大学的论文: [FLatten Transformer: Vision Transformer using Focused Linear Attention](https://arxiv.org/abs/2308.00442)。其尝试解决上面提到的 缺点。

这里的 **转换函数** 是针对 词向量 而言的, 形式如下:

$$
\phi (\bold{x}) = \varphi (\bold{x}) = f_p (relu (\bold{x})) \tag{4.1}
$$

$relu$ 就是常规的激活函数, 其保证向量 $\bold{x}$ 中每一个元素都是非负的。 $f_p$ 的公式如下:

$$
f_p (\bold{x}) = ||\bold{x}||_2 \cdot \frac{\bold{x}^{**p}}{||\bold{x}^{**p}||_2}
\tag{4.2}
$$

其中, $||\bold{x}||_2$ 表示向量 $\bold{x}$ 的模长, $\bold{x}^{**p}$ 表示对向量 $\bold{x}$ 中的元素逐位做 $p$ 次方的运算。公式上还是很简单的, 就是表达上有点复杂。

按照作者的说法, 这样可以使得 attention 矩阵的 **稀疏性** 增加。原理如下:

当 $p > 1$ 时, 如果向量 $\bold{q}$ 和 $\bold{k}$ 最大的元素值在同一个维度上, 那么一定 **存在** $p$, 使得 $\phi(\bold{q})^{\mathsf{T}} \cdot \phi(\bold{k}) > \bold{q}^{\mathsf{T}} \cdot \bold{k}$; 如果向量 $\bold{q}$ 和 $\bold{k}$ 最大的元素值 **不在** 同一个维度上, 那么一定 **存在** $p$, 使得 $\phi(\bold{q})^{\mathsf{T}} \cdot \phi(\bold{k}) < \bold{q}^{\mathsf{T}} \cdot \bold{k}$。

看到上面的结论是不是很懵? 没错, 当我看懂时, 也是很懵的:

+ 作者认为, 如果向量 $\bold{q}$ 和 $\bold{k}$ 是 **相似** 的, 那么他们元素的最大值应该在同一个维度上
+ 是 **存在** $\exists$ $p$, 不是 **任意** $\forall$ $p$。对, 你没有看错！！！

怎么说呢? 实验效果好就行, 这些东西就不要去较真了。上述结论的证明见 [论文](https://arxiv.org/abs/2308.00442) 附件 A。根据消融实验, 最佳的 $p$ 值是 $3$。

那么, 如何解决 attention 矩阵的低秩性呢? 作者给 $\bold{V}$ 矩阵额外进行了 depthwise convolution (词向量的每一个维度 (特征图/通道) 进行卷积), 加在矩阵 $\bold{O}$ 上。此时, 公式 $(2.5)$ 变成:

$$
\bold{O} = \frac{
    \phi (\bold{Q}) \cdot \left[ \phi (\bold{K})^{\mathsf{T}} \cdot \bold{V} \right]
}{
    \phi (\bold{Q}) \cdot \mathrm{sum_{column} (\phi (\bold{K}))}
} + DWC(\bold{V})
\tag{4.3}
$$

作者说, 其有理由相信, $DWC(\bold{V})$ 是一个 满秩矩阵, 这样就可以增加 $\bold{O}$ 矩阵的 秩 了。

怎么说呢? 给人一种感觉, 绕了半天又绕回 卷积 了。那么, 究竟是 注意力 对性能影响的大呢还是 卷积 对性能影响的大呢? 这值得深思。总之, 算是一种尝试吧。

## Linformer

Linformer 出自 2020 年 FaceBook 的论文: [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768), 代码建议参考 FaceBook 发布的工具集 [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/linformer)。

首先, 作者提出, 在 self attention 中, attention 矩阵是 低秩矩阵。这似乎和上面所说的内容是矛盾的。这里说一说我的理解:

所谓 **低秩矩阵**, 就是说我们可以用矩阵中的 某些向量 来表示 另外一些向量。上面说过, attention 矩阵是 **满秩矩阵**, 也就意味着我们不能用 某些向量 来 **明确表示** 另外一些向量。

而 Linformer 中说 attention 矩阵是 **低秩矩阵**, 是因为我们可以用矩阵中的 某些向量 来 **近似表示** 另外一些向量。由于矩阵中存在近似的 线性关系, 我们可以去掉矩阵中的一些向量, 这样我们就可以用更少的维度信息来表示 attention 矩阵了。

什么是 近似的 线性关系呢? 我们可以简单的理解为: $c_1 \cdot \vec{r}_1 + c_2 \cdot \vec{r}_2 \approx \vec{r_3}$ 。以上只是我目前的理解, 可能是错误的, 也可能是不完善的。

总结一下, attention 矩阵是 **满秩矩阵**, 也就意味着矩阵中的向量可以作为空间中的一组 基向量 (两两之间 **线性无关**), 同时这个矩阵也是倾向于 **稀疏** 的。不仅如此, 这些 向量 之间有 近似的 线性关系, 因此我们可以用 低维矩阵 来近似的表示 attention 矩阵。

在论文中, 作者从两个角度来说明 attention 矩阵的这个特性。一个是从 [Johnson–Lindenstrauss lemma](https://kexue.fm/archives/8679), 这个之后有机会再分析, 另一个是从 SVD 分解的角度来说明。作者使用 Roberta 模型对一个 $\mathbb{R}^{512 \times 512}$ 维度的 attention 矩阵进行 SVD 分解, 发现前 128 个 奇异值 之和占全部 奇异值 之和的 90% 以上。不仅如此, 随着 Transformers 层数的增加, 前 128 个 奇异值 之和的占比还在逐步增加。具体效果可以见论文的 Figure 1。

顺带复习一下 SVD 分解的相关知识: 奇异值 一定是 非负实数, 默认 奇异值 是从大到小排列的, 非零奇异值的个数等于矩阵的秩。

既然如此, 那么我们可以像 [LoRA](https://arxiv.org/abs/2106.09685) 一样将 attention 矩阵和 $\bold{V}$ 矩阵降维后再相乘, 作者提出的方式如下:

$$
\bold{O} = \mathrm{softmax_{row}} \left(
    \frac{\bold{Q} \cdot (\bold{E} \cdot \bold{K})^{\mathsf{T}}}{\sqrt{d}}
\right) \cdot (\bold{F} \cdot \bold{V})
\tag{5.1}
$$

其中, 矩阵 $\bold{E} \in \mathbb{R}^{s \times n}$, $\bold{F} \in \mathbb{R}^{s \times n}$。作者将对 attention 的低维投影转换到 $\bold{K}$ 上。这样, 复杂度就降到了 $O(nsd)$ 了。如果忽略 $s$ 和 $d$, 那么复杂度就是 $O(n)$, 属于线性 Attention。

其中, 矩阵 $\bold{E}$ 和 $\bold{F}$ 是可训练参数, 其中 $n$ 取序列长度的最大值即可。如果实际的序列长度小于最大长度, 将矩阵 $\bold{E}$ 和 $\bold{F}$ 截断即可。

除此之外, 作者还实验了, 将矩阵 $\bold{E}$ 和 $\bold{F}$ 参数共享, 不同 head 和不同层之间的矩阵 $\bold{E}$ 和 $\bold{F}$ 参数共享, 发现效果相差的不多。

论文中的实验并不是特别多, 按照实验结果, 当最大的序列长度是 512 或者 1024 时, $s$ 取值是 256 效果最好。同时, 参数共享比不共享效果要好。

这个方法的好坏暂不做评价。说一个小问题: 如果实际序列长度比 $s$ 还小, 那就不是 低维投影 矩阵了, 而是 高维投影 矩阵了。此时的计算复杂度会增加很多。

## 总结

本文只是简单介绍了一些简化 Attention 计算的方式。线性 Attention 主要就是运用 线性代数 中的一些技巧来简化 Attention。相较于 CNN 和 RNN 而言, Attention 可以说是出现最晚的网络结构了, 现在还很难说什么样子的计算方式是最好的。大家也在 费尽心思 的设计更好的计算方法。

除了本文介绍的以外, 还有一些方法之后有机会再介绍, 比方说:

+ [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)
+ [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902)

主要是我的数学还要加强 QAQ

## References

+ [线性Attention的探索：Attention必须有个Softmax吗？](线性Attention的探索：Attention必须有个Softmax吗？)
+ [Performer：用随机投影将Attention的复杂度线性化](https://spaces.ac.cn/archives/7921)
+ [Nyströmformer：基于矩阵分解的线性化Attention方案](https://spaces.ac.cn/archives/8180)
+ [Transformer升级之路：3、从Performer到线性Attention](https://spaces.ac.cn/archives/8338)
+ [让人惊叹的Johnson-Lindenstrauss引理：理论篇](https://kexue.fm/archives/8679)
+ [让人惊叹的Johnson-Lindenstrauss引理：应用篇](https://kexue.fm/archives/8706)
