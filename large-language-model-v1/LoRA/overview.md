
# LoRA: Low-Rank Adaptation of Large Language Models

[TOC]

## 简介

关键词:

+ parameter-efficient fine-tuning
+ adapter tuning
+ low-rank matrix

相关资料:

+ 论文地址: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
](https://arxiv.org/abs/2106.09685)
+ 原始代码: [LoRA](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
+ HuggingFace 复现: [PEFT](https://github.com/huggingface/peft)
+ 中文讲解博客1: [LoRA:大型语言模型的低秩适配器](https://zhuanlan.zhihu.com/p/610943445)
+ 中文讲解博客2: [LoRA论文回顾](https://zhuanlan.zhihu.com/p/619468521)

## PEFT 少参数微调

一般情况下, 对于 **微调** 来说, 是要调整所有的参数。但是对于大模型来说, 这样的成本非常之高。此时的解决办法就是: 冻结预训练模型的参数, 只训练下游任务相关的新网络层。

我们将这种方式称为 parameter-efficient fine-tuning, 简称 PEFT。我们知道 fine-tuning 指的是 **微调**。在英语中, efficient 是一个用途广泛的词。由于我们只训练下游任务相关的网络层, 训练参数变少, 训练效率变高, 因此称为 parameter-efficient。考虑到其本身的含义, 我建议将 PEFT 翻译成 **少参数微调**。

对于传统的 ResNet, BERT 等模型来说, 一般是直接将预训练模型层作为 特征提取器, 然后在后面添加 **任务相关 (task-specified)** 的全连接层和分类层。在微调时, 我们只训练新增加的网络层即可。

但是对于 T5, GPT 等生成大模型来说, 下游任务是不需要新的全连接层和分类层的, 结果是直接 "生成" 的。比方说, 如果下游任务是文本分类, 那么应该是直接生成 **标签文本**, 而不是得到每一个标签的概率。

此时, 如果你的资源足够充分, 是不需要添加任务相关的网络层, 直接所有参数训练就好了。那如果资源不充分, 只支持训练部分参数, 那么应该怎么办呢? 答案是在神经网络的 **开始** 或者 **中间** 添加任务相关的网络层。

基于这样的想法, 产生了两个分支: prompt tuning 和 adapter tuning。

有关 prompt tuning 可以参考这篇 [文档](https://huggingface.co/docs/peft/conceptual_guides/prompting), 之后有时间再总结。

adapter tuning 中的 adapter 就是神经网络层, 像插件一样插在大模型中。由于这些插入的层是 **任务相关** 的, 我们可以将其理解为是 下游任务 的 适配器, 因此叫 adapter。

大模型的问题除了训练资源要求高外, 还有就是推理慢。无论是 **prompt tuning**, 还是 **adapter tuning**, 都会添加新的网络层, 这样会导致推理更加缓慢!

这篇论文尝试用 **低秩矩阵** 的思想, 使用 adapter tuning 的方式, 添加网络层。并且, 在推理时, adapter 层是可以融入到原始的大模型中, 不会使得推理变慢, 可以说是一个非常好的方案。

那么什么是 **低秩矩阵** 呢?

## Low-Rank Matrix 低秩矩阵

什么是矩阵的 **秩** (rank) 呢? 参考 [博客](https://blog.csdn.net/qq_33542428/article/details/106481508) 进行简单的复习。

在初学线性代数时, 我们是从 **方程组** 的概念延申到 **矩阵** 的概念。如果方程组中的某一个方程可以用其它的方程来表示, 那么这个方程就是多余的, 不包含有效信息, 对方程组的求解没有任何帮助作用, 可以剔除掉。

为了从方程组中去掉多余的方程，我们引出了矩阵的 **秩** 概念。矩阵的 **秩** 度量的是矩阵行列之间的相关性。

为了求矩阵 $\mathbf{A}$ 的 **秩**，我们是通过矩阵的 **初等变换** 把 $\mathbf{A}$ 转化为 **阶梯型矩阵**。 若 **阶梯型矩阵** 有 $r$ 个非零行，那 $\mathbf{A}$ 的 **秩** 就等于 $r$。我们用 $rank(\mathbf{A})$ 表示 **秩**。

在实际应用中, 矩阵不单单表示方程组, 还可以表示 数据集, 图片, 模型参数等等。在机器学习中, 只要是将 **向量化** 的东西堆积在一起, 那么就可以构成 矩阵 (张量)。此时, 我们用 **线性相关** 来表述矩阵的 **秩**。

如果矩阵的各行或列是 **线性无关** 的，矩阵就是 **满秩** (full rank) 的，也就是 **秩** 等于行数或者列数中的较小值。如果矩阵中存在 **线性相关** 的行或列, 矩阵就是 **秩亏** (rank-deficiency) 的, 也就是 **秩** 比行数和列数都小。

**满秩矩阵** 在数学上有很多优秀的特性, 比方说可逆, 可解等等。**秩亏矩阵** 有多种处理方式, 比方说 PCA, SVD 等等。这些方法可以用来降维, 去噪, 填补缺失信息, 提取共性信息等等。

**低秩矩阵** 就是 **秩亏矩阵**, 不仅如此, 其 **秩** 往往是远远小于矩阵的行数和列数。也就是说, 矩阵内部包含的信息量是很少的, 可以被较少的 特征向量 或 奇异值 来表示。

那么如何将 **低秩矩阵** 应用于 adapter tuning 呢?

## LoRA 低秩矩阵适配器

在深度学习的过程中, 我们采用如下的方式更新参数:

$$
\mathbf{W} := \mathbf{W} + \Delta \mathbf{W} = \mathbf{W} - lr \cdot gradient
$$

**下游任务微调** 可以看作是每一步更新的 $\Delta \mathbf{W}$ 矩阵累加起来。此时我们可以这样做:

冻结预训练模型中的所有参数, 为每一个 权重矩阵 $\mathbf{W}$ 额外设置一个 可训练的 $\Delta \mathbf{W}$ (初始化为全零矩阵)。在前向传播时, 使用 $\mathbf{W} + \Delta \mathbf{W}$ 作为 权重矩阵, 在反向传播时, 只更新 $\Delta \mathbf{W}$, 不更新 $\mathbf{W}$ 。

如果你熟悉链式求导法则, 你会发现这种方式和原始方式没有什么区别, 好处是我们保留了原始的预训练模型参数。如果你的模型非常大, 达到千亿级别, 采用 API 接口的方式提供服务, 同时希望不同用户拥有属于自己的模型, 那么这是一种很好的方式。

作者认为, 我们可以假设 $\Delta \mathbf{W}$ 是一个 **低秩矩阵**, 那么我们可以将其拆分成两个小的参数矩阵。具体的:

我们设 $\mathbf{W}$ 和 $\Delta \mathbf{W}$ 矩阵的维度是 $\mathbb{R}^{d \times k}$, 同时 $\Delta \mathbf{W}$ 的秩为 $r$, 那么我们可以使用 $\mathbf{A}$ 和 $\mathbf{B}$ 相乘的方式来表示 $\Delta \mathbf{W}$。$\mathbf{A}$ 和 $\mathbf{B}$ 的维度分别是 $\mathbb{R}^{d \times r}$ 和 $\mathbb{R}^{r \times k}$, 用公式表示如下:

$$
\Delta \mathbf{W} = \mathbf{A} \mathbf{B}
$$

我们将 $\mathbf{A}$ 初始化为全零矩阵, $\mathbf{B}$ 高斯随机数初始化。这样可以保证 $\Delta \mathbf{W}$ 一开始是全零矩阵。同时, 在前向传播时, 我们还会让 $\mathbf{A} \mathbf{B}$ 乘以 $\frac{\alpha}{r}$ 。其中 $\alpha$ 是超参数, 一般情况下和 $r$ 值保持一致即可。

整个 `nn.Linear` 前向传播过程可以表示为:

$$
\boldsymbol{h} = (\mathbf{W} + \Delta \mathbf{W}) \cdot \boldsymbol{x} + \boldsymbol{b} = (\mathbf{W} + \frac{\alpha}{r} \cdot \mathbf{A} \mathbf{B}) \cdot \boldsymbol{x} + \boldsymbol{b}
$$

其中只有矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 是可训练参数, bias 一般也设置成不可训练参数。

在推理时, 我们可以将 $\Delta \mathbf{W}$ 事先融入 $\mathbf{W}$ 中, 这样就没有推理速度上的损失了。

## 总结

整个方案可以说是非常的巧妙。如果我们将 $\alpha$ 和 $r$ 设置成一样, 那么实际的超参数只有一个, 那就是 $r$。根据论文中的实验, 一般设置成 4 或者 8 即可。

另一个问题是哪些层需要 LoRA 。论文中主要是针对 attention 层的 query, key, value 和 output 的投影层进行实验。从效果来看, 四个都设置, 同时 $r=2$ 或者 $r=4$ 时效果最佳。

在模型性能上, 几乎没有下降, 甚至还有 0.1% 的提升。
