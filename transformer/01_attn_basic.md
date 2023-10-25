
# Attention

[TOC]

## 简介

随着 [BERT](https://arxiv.org/abs/1810.04805) 和 [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) 等语言模型的广泛应用, [Transformers](https://arxiv.org/abs/1706.03762) 和 [Attention 机制](https://paperswithcode.com/methods/category/attention-mechanisms) 也越来越受到重视。本文将系统的说一说 Attention 机制, 以及相关衍生的内容。

目前主流的 Attention 计算方式是 [Transformers](https://arxiv.org/abs/1706.03762) 中使用的 **Scaled Dot-Product Attention**, 计算方式如下:

$$
\overrightarrow{o}^{\mathsf{T}} = \mathop{\mathrm{softmax}} (\frac{\overrightarrow{q}^{\mathsf{T}} \cdot K^{\mathsf{T}}}{\sqrt{d_k}}) \cdot V \tag{1.1}
$$

其中, $\overrightarrow{q}$, $K$, $V$ 和 $\overrightarrow{o}$ 的维度分别是 $\mathbb{R}^{d_k}$, $\mathbb{R}^{n_{kv} \times d_k}$, $\mathbb{R}^{n_{kv} \times d_v}$ 和 $\mathbb{R}^{d_v}$。

$\overrightarrow{o}$ 是矩阵 $V$ 中词向量的 **加权平均** (行向量的 **线性组合**)。权重值 (线性组合的系数) 由 $\overrightarrow{q}$ 和矩阵 $K$ 中的词向量点乘, 然后经过 softmax 函数标准化后得到。

矩阵 $K$ 和矩阵 $V$ 中的词向量是 **一一对应** 的关系。两者的词向量数量必须一致, 但是维度可以不一致。如果调换矩阵 $K$ 中词向量的位置, 矩阵 $V$ 中词向量的位置也要相应的改变。这样就会产生一个问题: 矩阵 $K$ 和矩阵 $V$ 中词向量的位置对于 $\overrightarrow{o}$ 没有任何的影响。如果需要位置关系, 那么就需要 **positional encoding**。

虽然 query, key 和 value 是检索领域的用语, 但是 attention 和检索领域的关系并不大。其是 [Transformers](https://arxiv.org/abs/1706.03762) 作者用来形象化, 统一化的描述。如果你阅读早期的论文, 会发现 Attention 的表述方式各不相同, 但是本质上都是一致的。本文统一采用 query, key 和 value 的方式进行表述, 方便理解。

将公式 $(1.1)$ 中的 $\overrightarrow{q}$ 和 $\overrightarrow{o}$ 变成矩阵后, 公式如下:

$$
O = \mathop{\mathrm{softmax}} (\frac{Q \cdot K^{\mathsf{T}}}{\sqrt{d_k}}) \cdot V \tag{1.2}
$$

其中, 矩阵 $Q$, $K$, $V$ 和 $O$ 的维度分别是 $\mathbb{R}^{n_q \times d_k}$, $\mathbb{R}^{n_{kv} \times d_k}$, $\mathbb{R}^{n_{kv} \times d_v}$ 和 $\mathbb{R}^{n_q \times d_v}$。一般情况下, 矩阵 $K$ 和 $V$ 源于同一个矩阵, 但是会进行不同的转换 (激活函数 或者 线性变换), 因此 $d_k$ 一般等于 $d_v$。而矩阵 $Q$ 和矩阵 $K$ 可以源于同一个矩阵, 也可以不源于同一个矩阵, 因此 $n_q$ 和 $n_{kv}$ 不一定相等。

Attention 在 NLP 领域的用途主要有三个, 分别是: (1) 构建句向量; (2) self attention; (3) cross attention。无论进行多么复杂的计算, 一般也不会逃离这三种形式。

## 用途: 构建句向量

通过 **词向量** 来构建 **句向量** 是 NLP 中常见的问题。常见的方式有:

+ 池化: 用 **最大池化** (max pooling) 或者 **平均池化** (average pooling) 作为句向量
+ 位置: 用整个序列的 **首词向量** (eg. BERT) 或者 **尾词向量** (eg. GPT) 作为 句向量
+ 双向 RNN: 将第一个 token 的 **反向词向量** 和最后一个 token 的 **正向词向量** 拼接作为句向量

使用 Attention 来构建句向量属于 **平均池化** 的延申, 我们希望使用 **加权平均** 的方式来构建句向量。具体的, $\overrightarrow{q}$ 是 **可训练参数**, $K$ 和 $V$ 矩阵都是词向量矩阵。句向量 $\overrightarrow{o}$ 是 所有词向量 (矩阵 $V$) 的加权平均, 权重值由 参数向量 $\overrightarrow{q}$ 和 所有词向量 (矩阵 $K$) 点乘, 然后通过 softmax 函数标准化后得到。

这样的好处是, 我们可以构建多个句向量, 这对于 多标签分类 来说是一种好方案。实现时, 可能还会添加一些线性变换层和激活层, 具体参考下面两篇论文:

+ [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://aclanthology.org/P16-2034/) 的 $3.3$ 节
+ [Document-level Event Extraction via Heterogeneous Graph-based Interaction Model with a Tracker](https://arxiv.org/abs/2105.14924) 的 $3.3$ 节

## 用途: self attention

self attention 指的是矩阵 $Q$, $K$ 和 $V$ 都源于同一个词向量矩阵, 有时也被称为 intra attention。

在 [Transformers](https://arxiv.org/abs/1706.03762) 架构中, 就是使用 self attention 模块来替换 RNN 模块的。观察公式 $(1.2)$, 此时的矩阵 $Q$, $K$, $V$ 和 $O$ 维度都是 $\mathbb{R}^{n \times d}$。其存在一个很大的问题: 那就是没有可训练参数。因此, 作者对这四个矩阵都额外增加了一个线性层, 我们可以用 $W^Q$, $W^K$, $W^V$ 和 $W^O$ 来表示, 他们的维度都是 $\mathbb{R}^{d \times d}$。

对于标准的语言模型来说, 我们是根据左边的词语来推测右边的词语。因此, 在计算每一个词向量时, 只能用其左边词语的信息, 不能使用右边词语的信息。对应到 Attention 模块中, 我们需要将 $Q \cdot K^{\mathsf{T}}$ 矩阵中 **右上三角** 部分的给 mask 掉。这样的 Attention 模块也被称为 masked self attention, 常用于 GPT 架构和 Transformers 的 decoder 架构。

而对于不用 mask 的, 计算词向量可以使用整个句子中所有词语信息的 Attention, 我们仿照 RNN 中的用语, 将其称为 bidirectional attention。BERT 模型中的 B 含义就是 bidirectional。

上面也说过了, attention 机制是缺少位置信息的。如果使用 self attention 模块来代替 CNN 和 RNN 模块, 就需要人为添加位置信息, 也就是 positional encoding。

## 用途: cross attention

cross attention 指的是矩阵 $K$ 和 $V$ 源于同一个词向量矩阵, 矩阵 $Q$ 源于另一个词向量矩阵。大部分论文的 attention 都可以归为这一类。

cross attention 在 [Transformers](https://arxiv.org/abs/1706.03762) 中被称为 encoder-decoder attention, 其 $Q$ 矩阵来自于 decoder 每一次编码后的词向量, $K$ 和 $V$ 矩阵来自于 encoder 最终生成的词向量。几乎所有的 encoder-decoder 架构都是这样的。

上面也说过了, attention 机制缺少位置信息是由于 $\overrightarrow{q}$ 和矩阵 $K$ 相乘时缺少位置信息。而 encoder 生成的词向量一般不会缺少位置信息 (否则其编码的向量就有问题), 因此 cross attention 原则上可以不用考虑 positional encoding (加入位置信息也可以)。

除了 encoder-decoder 架构, 其在 NLU 中也有大量的应用。在很多关于 QA, Multiple-Choice, zero-shot learning 等论文中, 很多人会构建出多个 词向量矩阵 (比方说 文本词向量矩阵 和 问题词向量矩阵), 然后就会使用 cross attention 的方式让两者之间进行交互, 比方说:

+ [DUMA: Reading Comprehension with Transposition Thinking](https://arxiv.org/abs/2001.09415)
+ [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)
+ [Query and Extract: Refining Event Extraction as Type Oriented Binary Decoding](https://arxiv.org/abs/2110.07476)

## 注意力分数

目前公认的, 最早使用 Attention 机制的论文是: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)。其做法和上面所说的 cross attention 差不多, 但是表述方式相差巨大。如果想要深入了解, 可以结合 [代码](https://github.com/bentrevett/pytorch-seq2seq) 一起理解。

除此之外, 早期 Attention 还有一篇非常著名的论文: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)。

早期的 Attention 和 [Transformers](https://arxiv.org/abs/1706.03762) 相比, 最大的不同在于 query 和 key 向量之间 **注意力分数** 的计算方式。[Transformers](https://arxiv.org/abs/1706.03762) 中的计算方式被称为 scaled dot product, 用公式表示如下:

$$
score(\overrightarrow{q}, \overrightarrow{k}) = \frac{\overrightarrow{q}^{\mathsf{T}} \cdot \overrightarrow{k}}{\sqrt{d_k}} \tag{5.1}
$$

除此之外, 还有 [additive](https://paperswithcode.com/method/additive-attention) 和 [multiplicative](https://paperswithcode.com/method/multiplicative-attention) 两种比较经典的方式。

**Additive Attention** 的注意力分数计算方式如下:

$$
score(\overrightarrow{q}, \overrightarrow{k}) = \overrightarrow{v_a}^{\mathsf{T}} \cdot \mathop{\mathrm{tanh}} (W_q \cdot \overrightarrow{q} + W_k \cdot \overrightarrow{k}) \tag{5.2}
$$

其中, $W_q$ 和 $W_k$ 的维度都是 $\mathbb{R}^{d \times d}$ 。 $\overrightarrow{q}$, $\overrightarrow{k}$ 和 $\overrightarrow{v_a}$ 的维度都是 $\mathbb{R}^{d}$ 。上述公式 $(5.2)$ 还有另外一种写法:

$$
score(\overrightarrow{q}, \overrightarrow{k}) = \overrightarrow{v_a}^{\mathsf{T}} \cdot \mathop{\mathrm{tanh}} (W_a \cdot \left[ \overrightarrow{q}; \overrightarrow{k} \right]) \tag{5.3}
$$

其中, $W_a = \left[ W_q; W_k \right] \in \mathbb{R}^{d \times 2d}$。两个公式计算结果是一致的, 如果不理解, 可以想一想矩阵点乘的过程。后续的论文使用公式 $(5.3)$ 的偏多。

**Multiplicative Attention** 的注意力分数计算方式如下:

$$
score(\overrightarrow{q}, \overrightarrow{k}) = \overrightarrow{q}^{\mathsf{T}} W_a \overrightarrow{k} \tag{5.4}
$$

其中, $W_a$ 的维度是 $\mathbb{R}^{d \times d}$。

Additive Attention 和 Multiplicative Attention 都是有额外参数的, 这会增加计算量。现在一般都采用 scaled dot product 的方式, 后续所说的改进也都是基于这种方式的改进, 一般不用考虑前两种。

无论哪一种方式, 注意力分数的计算都是 token-pair 级别的。在这之后, 出现了大量基于 token-pair 的信息抽取方案, 用于解决实体嵌套的问题。他们的计算方式和这里的计算方式都是相似的, 不知道有没有借鉴关系。具体可以参考下面的内容:

+ scaled dot product 方式: Global Pointer
  + [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)
  + [Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877)
  + [GPLinker：基于GlobalPointer的实体关系联合抽取](https://spaces.ac.cn/archives/8888)
  + [GPLinker：基于GlobalPointer的事件联合抽取](https://spaces.ac.cn/archives/8926)
+ additive 方式: Multi-Head NER
  + [实体识别之Multi-Head多头选择方法](https://zhuanlan.zhihu.com/p/369784302)
  + [Joint entity recognition and relation extraction as a multi-head selection problem](https://arxiv.org/abs/1804.07847)
+ multiplicative 方式: Biaffine NER
  + [Named Entity Recognition as Dependency Parsing](https://arxiv.org/abs/2005.07150)
  + [Unified Named Entity Recognition as Word-Word Relation Classification](https://arxiv.org/abs/2112.10070)

## 引用

+ [Cross-Attention is All You Need: Adapting Pretrained Transformers for Machine Translation](https://arxiv.org/abs/2104.08771)
