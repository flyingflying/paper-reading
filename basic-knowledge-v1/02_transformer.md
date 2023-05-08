
# Transformer 架构

[TOC]

## 简介

就目前的发展趋势来说, `Transformer` 可以说是继 `MLP` (多层感知机), `CNN` 和 `RNN` 后的第四个基础框架, 其在图像处理, 自然语言处理, 视频处理 和 多模态等等领域都有广泛的应用。这篇博客主要是借助李沐的视频, 从 `Transformer` 入手, 了解以下 `Transformer` 以及之后的和 NLP 相关的模型。

## 注意力运算

<!-- 2023年5月8号重新编辑了一下 -->

对 **query 向量** 进行重新编码, 编码后的 **output 向量** 是 **value 矩阵** 中 **向量** 的线性组合, 线性组合的系数由 **query 向量** 和 **key 矩阵** 中的 **向量** 点乘再用 softmax 归一化后得到。

从上面可以看出, **query 向量** 和 **key 向量** 的维度必须一致, 同时 **key 矩阵** 的向量数和 **value 矩阵** 的向量数必须一致, 否则没有办法计算。

如果 **value 矩阵** 中只有一个向量, 那么 **output 向量** 和这个向量是一致的。

### 要点一: 线性组合

由于 **output 向量** 是 **value 矩阵** 中向量的线性组合, 那么 **output 向量** 和 **value** 向量就具有某一种 **相似度**。我们可以用同一种方式来度量这两个向量。(具体的例子见 CasEE 3.2 节的 sigma 函数, 两个 sigma 函数是共享参数的)

除此之外, 利用这个特性, 我们可以构建 **句向量**。很多论文都使用这一特性, 比方说 [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://aclanthology.org/P16-2034/) 。

常用的通过 **词向量** 构建 **句向量** 的方式有:

+ 池化: 包括 最大池化 (max pooling) 和 平均池化 (average pooling)
+ 位置: 用整个序列的 首词向量 (eg. BERT) 或者 尾词向量 (eg. GPT) 作为 句向量
+ 双向 LSTM: 将第一个 token 的 **反向词向量** 和最后一个 token 的 **正向词向量** 拼接

那么怎么通过 **attention 运算** 构建 **句向量** 呢? 我们设置一个 $w$ 向量, 作为 query 向量, 所有的词向量构成 key 矩阵和 value 矩阵, 进行上述运算, 得到的 output 向量就是 **句向量**。

其中, $w$ 向量是可训练参数。也就是说, **句向量** 是所有 **词向量** 的一种线性组合。

这种构建 **句向量** 的方式常常用于 **文本分类**。如果是 多任务分类, 或者说 多标签分类 (多个二分类任务), 我们可以为每一个任务设置一个 **句向量**。在篇章级事件抽取中, 事件类型检测几乎都是采用这种方式。

### 要点二: softmax 函数

我们知道, 两个向量的点乘可以计算两者间的相关性, 但是这个相关性受词向量的维度影响很大, 维度越多, 点乘后的值也倾向于越大。然而 softmax 函数受极端值的影响非常大, 因此将 **query 向量** 和 **key 矩阵** 点乘的结果直接输入到 softmax 是一个非常危险的操作。

最长见的方式是缩小点乘后的结果。一般会缩小 $\sqrt{d}$ 倍 ($d$ 是词向量的维度)。这种方式被称为 scaled dot product。

另一种不常见的方式是对 key 矩阵用 tanh 激活函数激活, 将其值转化到 -1 至 1 之间。上面提到的论文就是采用这种方式的。

关于 softmax 的另一个注意事项是 mask。NLP 任务向量化运算的一个问题就是每个句子的 token 数是不一致的。此时我们需要通过 padding 的方式将句子的长度补成一致的。

在一般的线性层中, 如果这些 padding 向量为全零向量, 那么对于运算结果是没有影响的。

但是在 attention 的 softmax 运算中, 就会出问题, 因为, $e^0 = 1$。在用 softmax 归一化时, 对于序列中 padding 的部分, 需要将其转化为 `-inf`, 使其计算出来后的权重值为 0, 因此需要 `attention_mask` 参数。

### 要点三: 重新编码

**attention 运算** 可以理解成是对 **query 向量** 的重新编码, 其是从 **用途** 的角度出发来说明的, 也就是说 **query 向量** 和 **output 向量** 的作用是一致的。

需要注意的是, 从本质上来说, **output 向量** 和 **value 向量** 的性质是 **相似** 的, 我们可以 **重新编码** 其理解为用 **value 矩阵** 来表示 **query 向量**。

## Attention 层

transformer 架构中的 attention 层主要分成三种:

+ multi-head self attention
+ multi-head masked self attention
+ multi-head cross attention

**attention 层** 和 **attention 运算** 的主要区别是: query, key, value 和 output 四个矩阵都有一个线性投影层。

在 self attention 中, query, key 和 value 矩阵都是 **词向量矩阵**。

在 cross attention 中, query 矩阵是 decoder 部分的 **词向量矩阵**, key 矩阵和 value 矩阵是 encoder 部分输出的 **词向量矩阵**。

在 transformer 架构中, attention 运算本身是没有可学习的参数的, 所有的参数都集中在 线性投影层中。

我们可以这样理解 transformer 架构: 模型本身对语言没有任何假设 (`CNN` 是对位置建模, `RNN` 是对时序建模)。

正是因为这样, attention 层没有涉及到任何的位置信息, 这样会产生一个问题: 如果改变一句话中词语的顺序, 也就是改变 **key 矩阵** 和 **value 矩阵** 中词向量的位置, 编码后的 query 词向量是一致的。这显然不是我们想要的, 此时我们需要在词向量中添加位置信息, 也就需要位置编码 (positional embedding)。

什么是 **多头注意力** (multi-head attention) 呢? 就是进行多次 **attention 运算**。

为什么需要 多头注意力 呢? 为的是模拟 CNN 中多通道的效果:

+ 对于 CNN 来说, 一个 kernel 表示一个 pattern, 将图片中每一个 receptive field 都和 pattern 通过点乘的方式计算相关性分数 (互相关), 有几个 pattern 就有几个相关性分数
+ 对于 transformer 来说, pattern 存在于 **attention 运算** 前的线性投影中。一个 head 对应一个 pattern 对应三个 `[hidden_size, head_size]` 的参数矩阵 (三个参数矩阵分别将词向量投影成 query, key 和 value 向量), 有几个 head 表示捕捉到几组 token 之间的关系

## LayerNorm

`BatchNorm` 是将一个 mini-batch 中词向量中的每一个维度都标准化成均值是 $\beta$, 标准差是 $\gamma$ 的值 (其中 $\beta$ 和 $\gamma$ 是可学习参数), 公式见: [BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) 。按照官方文档来解释, `BatchNorm` 是对 mini-batch 中的每一个通道 (channel) 进行标准化, 在 NLP 领域, 通道指的就是词向量的维度, 一个通道对应一个词向量维度。

`LayerNorm` 是将一个词向量中所有的特征维度进行标准化, 公式和可学习参数和上面一样。由于 NLP 领域中, 句子长度不一致时需要补零成一样长的序列, 此时用 `BatchNorm` 不是很好, 因为 `0` 是会影响到 `BatchNorm` 计算的, 而 `LayerNorm` 仅仅对一个词向量进行标准化, 这样就不会受 `0` 值的影响了。

由于 `Transformer` 中使用的 `LayerNorm` 是对每一个词向量进行标准化, 因此一开始的 `nn.Embedding` 学出来的值倾向于模长较小的向量, 也就是说词向量中的每一个维度值都很小。而 `Transformer` 中的正弦波位置编码值的范围在 `[-1, 1]` 之间, 因此作者将训练出来的词向量都扩大了 $\sqrt{d}$ 倍 ($d$ 是词向量的维度, 这里是 512)。

## Others

`Transformer` 原本是一个 序列转录模型 (sequence transduction model), 主要是用于机器翻译任务的。其解码器是一个 auto-regressive (自回归) 模型, 关于这一部分具体研究时再探索。

作者使用的 `regularization` (防止过拟合) 的策略是: dropout 和 label smoothing。

## Conclusion

在 `Transformer` 之前, NLP 领域主要的神经网络就是 `CNN` 和 `RNN`:

+ `RNN` 的缺点:
  + 不能并行计算, 必须要一步步算, 在这种情况下, 你的 GPU 有再多的线程也算不快
  + 由于是一步步地算, 时序早期的信息容易丢失, 增大隐藏层的维度可以缓解这个问题, 但相对应的显存开销也变大
+ `CNN` 的缺点:
  + 由于 `CNN` 一次只看一个较小的窗口, 因此对较长的句子难以建模

`Transformer` 可以说是解决了上面的问题:

+ 绝大部分运算都是矩阵乘法的运算, 这些运算都是可以并行的
+ `Attention` 运算是和句子中的每一个词向量进行运算, 因此是一次看完整个句子, 这样就不会将句子中的信息丢失了, 对长句子也很容易建模

但是其也存在一些缺陷:

+ `Attention` 运算是 token-pair 级别的, 如果句子特别长, 显存开销会很恐怖
+ `Transformer` 架构几乎取消了建模过程中的所有假设, 同时 `Attention` 运算中是不包含参数的, 绝大部分参数都在线性层, 这样导致的结果是模型的参数必须要很多才行, 否则效果会很不好, 因此基于 `Transformer` 往后的模型参数量都非常大, 倾向于 100 亿参数以上

最后说一下论文的表一是怎么算出来的:

由于 restricted attention 是作者在论文中未来探索的目标, 没有具体的说明, 这里就将其忽略了。这里主要是为了综合复习 `CNN`, `RNN` 和 `Attention` 的相关知识, 如果对其不了解, 是看不懂的。

`complexity` 是复杂度, 表示的是运算中包含的乘法数:

+ 两个维度是 $c$ 的向量, 点乘的复杂度是 $c$
+ 一个维度是 $(a, c)$ 的矩阵和维度是 $c$ 的向量进行点乘, 复杂度是 $a \times c$
+ 一个维度是 $(a, c)$ 的矩阵和维度是 $(c, b)$ 的矩阵进行点乘, 复杂度是 $a \times b \times c$

现在我们设序列长度为 $n$, 词向量维度为 $d$, 则:

+ 对于 `RNN` 来说, 每一步包含两个矩阵和向量的乘法, 一共有 $n$ 步, 输入的词向量和输出的词向量维度都是 $d$, 则乘法数为 $2*n*d^2$, 则复杂度是 $O(n d^2)$
+ 对于 `Attention` 来说, 一共有两个矩阵和矩阵的乘法, 如果 QUERY, KEY 和 VALUE 矩阵都是 $(n, d)$ 维度, 则乘法数为 $2*n^2*d$, 则复杂度是 $O(n^2 * d)$
+ 对于 `CNN` 来说, 设 kernel 的大小为 $k$, 则每次运算向量的维度是 $kd$, 一共进行 $n-k+1$ 次这样的运算 (一般 $k$ 值不大, 最多是 $5$, 这里可以简化成 $n$), 如果输出的词向量维度是 $d$, 也就是 kernel 的数量是 $d$, 则乘法数为 $(k \times d) \times n \times d$, 则复杂度是 $O(k \times n \times d^2)$

观察, 我们可以得到以下结论:

+ 由于 $k$ 值一般不会取很大, 也就是 2, 3 或者 5, 因此 `RNN` 和 `CNN` 的复杂度是相近的。
+ 当输入的句子很短时, `Attention` 明显会比其它的复杂度要低
+ 这样的比较我认为是不合理的, 一般情况下, `CNN` 和 `RNN` 的网络中不会有那么多的线性层, `Attention` 是要配合大量的线性层的, 但是这里的计算不包含线性层

`Sequential Operations` 很简单, 表示序列 "运算" 数。除了 `RNN` 是以 `RNNCell` 为一个 "运算" 单位的, 其它的 "运算" 单位是其本身。因此, `RNN` 对应的值是 `n`, 其它都是 `1`。

`Maximum Path Length` 表示一句话中第一个词向量到最后一个词向量要进行多少步 "运算" 才会产生相关性的计算:

+ 对于 `Attention`, 其是 token-pair 级别的, 一步 `Attention` "运算" 即可。
+ 对于 `RNN`, 其 "运算" 单位是 `RNNCell`, 因此需要 `n` 步 `RNNCell` "运算" 才行。
+ 对于 `CNN`, 需要多层卷积才行, 如果每层卷积的 kernel 大小为 $k$, 那么第 $m$ 层卷积出来的词向量可以融合 $k^m$ 个初始的词向量信息, 因此需要 $\log_k(n)$ 步 `CNN` "运算" 才行。

## GPT & BERT

我们知道, 标准的语言模型和序列模型很像: 通过 $w$ 词之前的 $n$ 个词来预测 $w$ 词发生的概率, 也就是在知道 $w$ 词之前的 $n$ 个词情况下计算 $w$ 词出现的概率。在序列模型中, 这种方式被称为自回归模型 (auto regressive model, AR model)。

`GPT` 就是采用这种方式的, 即在 `Attention` 操作中, 在对每一个 query 向量进行编码时, 只能用到在其之前 (包括其本身) 位置的 value 向量, 在其位置之后的 value 向量不能使用 (对应着权重值为 0)。在预训练和微调中都是采用这种方式的。其正好和 `Transformer` 中 decoder 架构中的 masked self attention 是一致的。

在预训练阶段, 每一个词输出的正是其下一个词发生的概率。正是这样, 这种预训练模型被称为 因果语言模型 (causal language model, CLM)。

总结一下, GPT 架构就是去掉 cross attention 的 Transformer decoder 架构, 或者说是将 self attention 变成 masked self attention 的 Transformer encoder 架构。

这种模型在 NLG 问题上效果较好, 因为这类模型要一个一个词的输出, 和 CLM 预训练任务很类似; 但是在 NLU 上面效果不好, 因为对句子的理解往往要看完一整个句子才行, 不能每个词只看其前面部分。

`BERT` 正好和其相反, 预训练采用的是 掩码语言模型 (masked langugage model, MLM), 即挖掉句子中的一部分词, 让模型去预测被挖掉的词是什么。其和 auto-encoder 架构很像, 但不是标准的 auto-encoder, 因此往往被称为自编码模型 (auto encoder model, AE model)。

`BERT` 就是 Transformer encoder 架构, 很显然, 其在 NLG 问题上表现的不好, 但是在 NLU 问题上表现的很好。

在 `GPT` 的 `Attention` 运算中, 由于对 query 向量的重新编码只用到其位置左边的 value 向量, 没有用到其右边的 value 向量, 因此也被称为单向编码; 而在 `BERT` 的 `Attention` 运算中, 对 query 向量的重新编码用到了所有的 value 向量, 包括其左边和右边的 value 向量, 因此也被称为双向编码。(这里的单向和双向的概念应该是从 `RNN` 架构中借鉴过来的)。

GPT-2 和 GPT-3 只是对 GPT 模型进行了微小的改动, 最主要的变化是加深和加宽了神经网络, 用了更多的数据集。GPT-2 有 15 亿参数, 其在 NLU 上的效果和 `BERT` 相近, GPT-3 有 1760 亿参数。

除此之外, GPT-2 和 GPT-3 的另一个贡献是 few-shot learning, one-shot learning 和 zero-shot learning。

GPT-3 的参数非常多, 很难微调, 作者提出了一种 in-context learning, 将 few-shot 提供的少量数据和 prompt (任务提示词) 一起当作输入, 让模型直接输出我们想要的文本。这种方式将 NLU 的任务转变成 NLG 的任务, 而 NLG 任务也正好是 GPT 模型所擅长的。

当然, 这种 in-context learning 虽然能解决 NLU 的任务, 但是仅仅是比传统的 few-shot learning 的实验效果要好。对于有一定标注量的数据集而言, 效果肯定是比不上其它微调模型的。现在很多人都在研究 prompt, 原因是其属于统一的信息抽取范式, 不需要再针对不同的信息抽取任务设计相关的任务和模型了, 向强人工智能阶段迈进, 但是还有很长的路要走。

## References

+ [Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE) (Accessed on 2022-09-16)
+ [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
+ [BERT 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1PL411M7eQ) (Accessed on 2022-09-16)
+ [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
+ [GPT，GPT-2，GPT-3 论文精读【论文精读】](https://www.bilibili.com/video/BV1AF411b7xQ) (Accessed on 2022-09-16)
+ [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) (Accessed on 2022-09-18)
+ [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Accessed on 2022-09-18)
+ [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) (Accessed on 2022-09-18)
+ [NLP预训练模型 | 按时间线整理10种常见的预训练模型](https://zhuanlan.zhihu.com/p/210077100) (Accessed on 2022-09-18)
+ [Glossary (huggingface.co)](https://huggingface.co/docs/transformers/v4.22.1/en/glossary#general-terms) (Accessed on 2022-09-18)
