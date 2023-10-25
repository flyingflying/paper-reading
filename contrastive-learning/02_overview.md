
# 对比学习 简介

[TOC]

## 简介

[对比学习 (contrastive learning)](https://paperswithcode.com/task/contrastive-learning) 是一种无监督的深度学习方法, 其作用是为 图片, 文字, 音频 等事物寻找向量化的表示方式 (或者说编码器), 使得:

+ 相似的 事物 在 向量空间 中相近的位置, 不相似的 事物 在 向量空间 中尽可能远的位置
+ 迁移到其它任务中, 如 分类, 结构化预测 (如 目标检测, NER) 等等

在 [word2vec](01_word2vec.md) 中已经说过了, 我们通过训练 **代理任务** (pretext task) 来达成这一目的, 其作用和 预训练任务 (如 MLM, CLM, ImageNet) 是相似的。不同的是:

+ 预训练任务 一般是有实际应用价值的, 但是 代理任务 则没有, 训练完成后就可以丢弃了, 我们不会去评估 代理任务 的效果
+ 预训练任务 一般只在 迁移学习 (微调) 上效果好, 在 相似性判断 上效果不好, 但是 代理任务 在两者上效果都不错

**实体判别** (instance discrimination) 是 对比学习 中常见的 代理任务, 属于 分类任务。其将 训练集 中的每一个 样本 都作为一个 类别。如果训练集有 128 万张图片, 那么就有 128 万个类别; 如果有 10 亿张图片, 那么就有 10 亿个类别。

现在, 在每一个类别中, 都只有一个样本。为了扩充样本数量, 我们对样本 **数据增强** (data augmentation), 将新得到的样本也归为同一类。这样, 每一个类别中的样本数量就变多了。常见的 **数据增强** 方式有:

+ CV: 裁剪, 翻转, 旋转, 灰度化, jitter, 高斯模糊 等等 (参考 [SimCLR](https://arxiv.org/abs/2002.05709) 中的 Figure 4)
+ NLP: 随机插入, 删除, 重复 词语 等等 (参考 [ESimCSE](https://arxiv.org/abs/2109.04380) 中的 Table 2)

如果按照正常的 softmax 回归来, 需要为每一个样本设置一个 线性函数, 用于计算 logit 值, 即:

$$
logit(\vec{v}, i) = \vec{v} \cdot \vec{w}_i + b_i \tag{1}
$$

其中, $\vec{v}$ 表示 样本向量, $i$ 表示类别索引, $\vec{w}_i$ 和 $b_i$ 表示第 $i$ 个类别线性函数的参数。这样, 需要设置大量的 线性函数, 并不是我们所期待的。因此, 我们将 logit 的计算方式修改为:

$$
logit(\vec{v}, i) = \frac{1}{\tau} \cdot \frac{\vec{v}}{||\vec{v}||_2} \cdot \frac{\vec{v}_i}{||\vec{v}_i||_2} = \frac{\cos(\vec{v}, \vec{{v}_i})}{\tau} \tag{2}
$$

其中, $\vec{v}$ 表示 样本向量, $i$ 表示类别索引, $\vec{v}_i$ 表示第 $i$ 个类别中的样本向量。也就是说, 这里的 logit 是在计算两个类别中样本向量的 cosine 相似度。需要注意的是, 如果 $\vec{v}$ 是第 $i$ 个类别中的样本向量, 那么 $\vec{v}_i$ 应该取其它的样本向量, 两者不能一致, 否则计算出来的 logit 值一定为 $1$。

我们知道, cosine 相似度的取值范围在 $[-1, 1]$ 之间, 这样的值直接作为 logit 输入 softmax 函数中, 概率分布会十分平缓。我们希望分布变得陡峭起来, 因此将 cosine 相似度除以 $\tau$ ($\tau < 0$), 改变 logit 的取值范围。论文中将 $\tau$ 称为温度, 取值为 $0.07$, 这样 logit 的取值范围就是 $[-14.28, 14.28]$ 了。

这里也说明了为什么 对比学习 得到的模型在 相似性判断 上的效果好了, 因为其就是用 cosine 相似性作为 logit 值的。

如果训练集中有 $V$ 个样本, 那么就有 $V$ 个类别, 按照上面的计算方式, 需要进行 $V+1$ 次编码, 这显然是不能接受的。在 [word2vec](01_word2vec.md) 中说过了, 只能使用 **估算** 的方式。常见的估算方式包括:

+ [NCE (noise contrastive estimation)](https://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf)
+ [Negative Sampling](https://arxiv.org/abs/1310.4546)
+ [in-batch negatives](https://arxiv.org/abs/1706.07881)
+ [InfoNCE](https://arxiv.org/abs/1807.03748)

在上述方法中, 我们需要为 每一个样本 选择一些 **负样本** 类别进行 **估算**。相对应的, 原始样本的类别 记作 **正样本**, 让两者 **对比** 去学习 (增加模型估算 **正样本** 类别的分数值, 减少模型估算 **负样本** 类别的分数值)。我们将 **负样本** 的个数记作 $K$。

其中, NCE 和 negative sampling 在 [word2vec](01_word2vec.md) 中已经说明了, 是将 $V+1$ 类的 softmax 回归任务变成 $K+1$ 个逻辑回归任务; 而 InfoNCE 和 in-batch negatives 是将 $V+1$ 类的 softmax 回归简化成 $K+1$ 类的 softmax 回归任务。相关的数学推导不再说明, 后面会说具体应该怎么做。

也正是因为此, 很多 **对比学习** 的论文中不会提及 **代理任务** 这一概念, 而是直接指明 **正样本** 标签和 **负样本** 标签从哪里来。

上述过程也可以从 检索 的角度来描述。我们可以将 样本 称为 **query**, 正样本 和 负样本 一起称为 **key**。对比学习要做的事情是: 根据 **query**, 从 $K+1$ 个 **key** 中找出最相似的那一个。由于 **key** 每一次是采样产生的, 因此也被称为 **动态字典** (dynamic dictionary)。

还有一种描述方式: 我们将 样本 称为 **anchor**, 正样本称为 **positive**, 负样本称为 **negative**。我们希望 **anchor** 和 **positive** 尽可能地接近, 和 **negative** 尽可能地远。

本文重点说明三篇论文:

+ [**InstDisc** | Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination](https://arxiv.org/abs/1805.01978)
+ [**MoCo** | Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
+ [**SimCLR** | A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

## InstDisc

实体判别 任务就是在这一篇文章中提出来的, 可以说是 **对比学习** 中相当经典的一篇论文了。

在 InstDisc 中, 作者构建了 memory bank, 用来存放每一个类别的 样本向量。每一次迭代时, **anchor** 向量是用编码器产生的, **positive** 和 **negative** 向量来自 memory bank。迭代完成后, 用 **anchor** 向量来更新 memory bank 中的 **positive** 向量。

训练流程大体上如下:

1. 随机初始化 memory bank
2. 从数据集中采样出来一个样本 $a$, 经过 **数据增强** 和 **编码器** 编码, 记作 $\vec{a}$
3. $a$ 类别在 memory bank 中对应的向量就是 **正样本**, 记作 $\vec{p}$
4. 从 memory bank 中按照均匀分布采样出 $K$ 个 **负样本**
5. 使用 NCE, 计算 loss, 反向传播, 更新 **编码器** 参数
6. 用 $\vec{a}$ 替代 memory bank 中 **正样本** 向量 $\vec{p}$
7. 重复运行 2-6 步, 直至收敛

NCE 相关的内容在 [word2vec](01_word2vec.md) 中已经说明了, 需要补充的有:

+ 其 噪声分布 选取的是 **均匀分布**
+ 负样本的数量 ($K$ 值) 是 $4096$, 非常大
+ 配分函数 依然使用 常数 代替, 但是选取的不是 $1$, 而是 $V \cdot \mathbb{E}(\exp (logit))$ 进行估算

在一次 epoch 结束后, memory bank 才能更新完成。也就是说, 我们从其中获得的 **样本向量** 可能和当前时刻的编码器相差甚远, 训练受 随机性 的影响很大, 这不是我们想要的。

为了使训练稳定, 作者在 论文 中提出 Proximal Regularization: 在 loss 中加入 $\lambda \cdot ||\vec{a} - \vec{p}||^2_2$ 项, 以减缓 **编码器** 更新的速度。

在 [官方代码](https://github.com/zhirongw/lemniscate.pytorch) 中, 作者没有使用 Proximal Regularization, 而是使用 **动量更新** 的方式来减缓 memory bank 的更新速度, 公式如下:

$$
\vec{p} \coloneqq (1-m) \cdot \vec{a} + m \cdot \vec{p} \tag{3}
$$

其中, $m$ 是动量值, 代码中取的 $0.5$。

第一次看到 memory bank, 可能很多人和我一样, 是难以接受的: 论文中究竟在写什么, 这样做真的没问题吗。但是仔细想想, 其实也是合理的。样本向量在空间中的位置是没有固定答案的, 我们只是期待其在空间中的位置分布有区分度而已。

## MoCo

这篇论文是何凯明在 2019 年发表的论文, 其思想可以看作是 InstDisc 的延续。这里, 我们用 **query** 和 **key** 的方式进行表述。

在 InstDisc 中, memory bank 的更新过于缓慢问题是更新过于缓慢, 样本向量可能是上一个 epoch 任意一次迭代的结果。因此, 作者用 队列 来替代 memory bank。

同时, 为了保证 **key** 向量的一致性, 作者还提出了 **动量编码器** 的概念。最理想的状态是, **query** 和 **key** 向量都是由一个编码器产生的, 但是这样所需要的硬件资源很多。怎么办呢? 将 **query** 和 **key** 向量用两个编码器 $EN^Q$ 和 $EN^K$ 进行编码。其中, $EN^Q$ 是正常的编码器, 而 $EN^K$ 是动量编码器, 其参数完全根据 $EN^Q$ 进行更新, 更新方式如下:

$$
\theta^K \coloneqq m \cdot \theta^K + (1 - m) \cdot \theta^Q \tag{4}
$$

其中, $\theta^K$ 表示 $EN^K$ 中的参数, $\theta^Q$ 表示 $EN^Q$ 中的参数, 初始时刻, 两者是相等的。$m$ 表示动量值, 论文中的消融实验表示取 $0.999$ 是最好的。可以看出, $\theta^K$ 的更新是非常缓慢的。

整体的流程如下:

1. 随机初始化 **query** 编码器 $EN^Q$
2. 初始化 **key** 编码器 $EN^K$, 其参数值和 $EN^Q$ 一致, 并且不需要计算梯度
3. 随机初始化一个 队列, 用于存放负样本向量, 大小和 $K$ 值一致, 论文中取 $65536$
4. 从数据集中采样一个样本, 经过两次 **数据增强**, 得到 anchor 样本 和 正样本
5. 将 anchor 样本用 $EN^Q$ 编码器进行编码, 得到 anchor 向量
6. 将 正样本 用 $EN^K$ 编码器进行编码, 得到 positive 向量
7. 将整个队列中的向量作为 negative 向量, 根据 InfoNCE 计算 loss 值
8. 反向传播, 更新 $EN^Q$ 编码器, 再使用公式 $(4)$ 动量更新 $EN^K$ 编码器
9. 将 positive 向量添加入队列中 (队列是 FIFO, 最先加入队列的 positive 向量被抛出)
10. 反复运行 4-9 步, 直至收敛

我们用 $q$ 表示 anchor 向量, $k_0$ 表示 正样本向量, $k_1, k_2, ..., k_K$ 表示 负样本向量, 则 InfoNCE 的公式如下:

$$
\mathrm{loss} = - \log \frac{\exp(\cos(q, k_0) / \tau)}{\sum_{i=0}^{K} \exp(\cos(q, k_i) / \tau)} \tag{5}
$$

可以看出, InfoNCE 不再使用 $K+1$ 个二分类 loss 之和的方式了, 而是 $K+1$ 多分类的 loss。从数学的角度来说, 这是在 最大化互信息的下界, 这里就不多介绍了。其中, 负样本的采样按照数据集的分布来即可。

从上面可以看出, 是将整个队列中的向量作为负样本向量, 同时也没有了 负样本采样 的过程了, 而是和 数据集采样 合并在一起了。也就是说, 负样本是 均匀采样, 符合 InfoNCE 的要求。

需要注意的是, $\tau$ 并不是 InfoNCE 的一部分, 而是 logit 的一部分, 很多初学者会弄错相关概念。

## SimCLR

上面说过了, 最好的方式是 **query** 向量和 **key** 向量用同一个 编码器 计算得到, 而不是使用 memory bank 或者 queue 数据结构进行存储。财大气粗的谷歌用 TPU 实现了这一方案, 于 2020 年发表了 SimCLR。

在 SimCLR 中, 也不进行 负样本采样, 而是将一个 batch 中的其它样本作为 负样本, loss 计算的方式和公式 $(5)$ 一样。这种 loss 计算方式我们不称为 InfoNCE, 而是 in-batch negatives。

整体的流程如下:

1. 随机初始化 编码器
2. 从数据集中采样 $B$ 个样本, 每个样本进行两次数据增强, 得到 $2B$ 个样本, 构成一个 batch
3. 将 batch 中的每一个样本用 编码器 进行编码, 得到样本向量
4. 对于 batch 中的每一个样本来说, 和其同一类的作为正样本向量, 剩下的 $2(B-1)$ 个都是都是负样本向量
5. 使用公式 $(5)$ 计算 loss, 反向传播, 更新 编码器 参数
6. 反复运行 2-5 步, 直至收敛

相比于上面的方案, 这个方案确实 "简单", 但是对显存要求很高, 一般人也玩不起。论文中使用的 batch_size 是 4096, 也就意味着每一个样本有 8190 个负样本。

这么看下来, 反而是 InstDisc 的负样本数量最少, SimCLR 其次, MoCo 最多。

## 网络架构

这三篇论文都是针对图像训练的模型。它们的骨干网络选取的都是 ResNet, 其中包含 `BatchNorm` 层, 这一层会给 对比学习 带来很大的麻烦。因为 `BatchNorm` 层会让一个 batch 中所有的样本之间进行交互, 存在 信息泄漏 的可能性:

+ 对于 InstDisc, 则没有这个问题, 因为 **key** 向量都是从 memory bank 中获取的
+ 对于 MoCo, 正样本向量 是 动量编码器 编码的, 可能也会有信息泄漏 (原因不明, 但是下游任务效果不好)
+ 对于 SimCLR 来说, 所有的样本都是放在一个 batch 中编码的, 肯定会有问题

MoCo 和 SimCLR 提出的解决办法都是基于 [DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 的, 单卡是解决不了的, 只能换 标准化 方式 (比方说 `GroupNorm`, `LayerNorm` 等等)。

MoCo 提出的方式是 shuffle BN, 即将几张卡上面的 queue 队列给打乱, 再去计算 loss 值。由于 queue 队列中的向量是 动量编码器 更新的, 没有梯度, 因此训练不存在问题。

SimCLR 则使用 [SyncBatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html), 让几张卡上面的 BatchNorm 汇总在一起更新参数, 以解决问题。

目前, 我们很难说明 `BatchNorm` 层对训练的影响, 只能明确的是: 经过上面的操作后, 下游任务的效果变好了。

在 MoCo 中, 其 task-specific 层就是一个简单的线性层 (`Linear`), 而在 SimCLR 中, 作者将其改成 非线性层, 即: `Linear` + `BatchNorm` + `relu` + `Linear` + `BatchNorm`。结果, 下游任务性能的提升有 5 到 10 个点, 非常厉害。因此在 [MoCo v2](https://arxiv.org/abs/2003.04297) 中也采用了这种方式。

在 InstDisc 的论文中, 没有提及 数据增强 的内容, 但是代码中有相关的内容。在 MoCo 中, 作者明确说明 数据增强 是按照 InstDisc 中的方式进行的 (有时候只读论文不读代码是会出问题的)。而在 SimCLR 中, 作者则强调了, 用更多的 数据增强 方式可以提升效果, 其非常重要!

## 相似性判断

对比学习 得到模型 的作用之一就是判断 两个样本 的相似性。上面也说过了, cosine 相似度的计算已经包含在训练 loss 中了。因此判断相似性最好的方式就是 cosine 相似度。

但是, 在某些情况下, 比方说 [KMeans 聚类](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), 我们不能使用 cosine 距离进行计算, 必须使用 欧式距离, 那么应该怎么办呢?

答案是可以直接用, 但是要将 样本向量 标准化, 变成 单位向量。理解方式如下:

在 二维空间 中, 两个 单位向量 的 **欧式距离** 就是 单位圆 的 **弦长**。我们设 $\theta$ 是两个 单位向量 的夹角, 也就是 单位圆 中的 圆心角。

根据三角函数的定义, 我们可以知道, 弦长等于 $2 \cdot \sin (\frac{\theta}{2})$。再根据 半角公式, 我们可以知道, 弦长等于 $\sqrt{2 - 2 \cdot \cos \theta}$。

也就是说, 欧式距离 和 cosine 相似度呈反比, 和 cosine 距离呈正比。

在更高维的空间中, 我们可以用代码来验证:

```python
import numpy as np
from sklearn.metrics import pairwise

# 随机生成样本向量
sample_vectors = np.random.randn(10, 128)

# 计算 cosine 相似度
cos_results = pairwise.cosine_similarity(sample_vectors, sample_vectors)

# 计算 欧式距离
sample_vectors = sample_vectors / np.linalg.norm(sample_vectors, axis=1, keepdims=True)  # 单位向量
euc_results = pairwise.euclidean_distances(sample_vectors, sample_vectors)

# 证明两者之间的关系
temp = 2 - 2 * cos_results
temp[np.abs(temp) < 1e-8] = 0.0  # 防止 负零 的发生
print(np.all(
    np.abs(np.sqrt(temp) - euc_results) < 1e-8
))
```

这里的结论很重要: 对于单位向量来说, cosine 相似度 和 欧式距离 之间是可以相互转换的, 并且两者之间呈反比!

## References

相关论文:

+ [**CPC** | Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
+ [**SimCSE** | SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
+ [**ESimCSE** | ESimCSE: Enhanced Sample Building Method for Contrastive Learning
of Unsupervised Sentence Embedding](https://arxiv.org/abs/2109.04380)
+ [**InstDisc** | Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination](https://arxiv.org/abs/1805.01978)
+ [**MoCo** | Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
+ [**SimCLR** | A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

相关视频:

+ [MoCo 论文逐段精读【论文精读】 | BV1C3411s7t9](https://www.bilibili.com/video/BV1C3411s7t9)
+ [对比学习论文综述【论文精读】 | BV19S4y1M7hm](https://www.bilibili.com/video/BV19S4y1M7hm)
