
# 对抗神经网络 (二) GAN 简介

[TOC]

## 简介

上篇博客简单介绍了 **图片生成** 任务, 以及 VAE 算法。这篇博客就开始介绍 GAN 模型了。

在 VAE 中, 训练时, 我们有两个神经网络: **编码器** 和 **解码器**, 在推理时, 我们只用 解码器 来生成图片。整个训练过程是: 用 编码器 将图片编码成符合标准正态分布的 code, 再用解码器将 code 解码成原来图片。

GAN 的全称是 **生成式对抗网络**, 出出自 2014 年的 paper: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) 。其和 VAE 相似, 也有两个神经网络: **生成器** 和 **判别器**。生成器用来生成图片, 判别器用来判断生成图片的质量。

训练时, 让生成器和判别器进行对抗。也就是对于同一个 **目标函数** $V$, 生成器希望最小化函数值, 判别器希望最大化函数值。

下面, 让我们分别来看 **生成器** 和 **判别器**。

## 生成器

GAN 中 **生成器** (generator) 做的事情是: 将符合 **正态分布** 的点映射成为符合 **样本分布** 的点。换一种说法是: 模型输入的是一个从 **正态分布** 中采样出来的 **随机向量**，模型输出的直接是一个 **图片向量**。

我们用 $G$ 表示生成器模型, $z$ 表示模型的输入, $G(z)$ 表示模型的输出, $P_G$ 表示模型输出的图片向量的概率分布, 样本图片向量的概率分布为 $P_{data}$。

我们要做的事情是寻找一个最佳的 $G^{\star}$, 让 $P_G$ 和 $P_{data}$ 之间的距离尽可能地小, 公式表示如下:

$$
G^{\star} = arg \min_G Div(P_G, P_{data}) \tag{1}
$$

现在, 我们不知道 $P_G$ 和 $P_{data}$ 的概率分布公式, 但是我们可以从中进行 **抽样**。

问题来了, 我们应该如何通过 **抽样** 来估算两个概率分布之间的距离呢? GAN 中给出的答案是用 **判别器** (discriminator) 来模拟这一过程。

## 判别器

我们用 $D$ 来表示判别器模型, 其是用来判断 图片 是否是生成器生成的, 也就是说判别器是一个 分类模型。

我们用 $x$ 表示图片, 也就是 $D$ 模型的输入, 其可能是生成器产生的, 也可能是数据集中的图片。

我们用 $D(x)$ 表示 $x$ 图片是真实的概率, $1 - D(x)$ 表示 $x$ 图片是生成器生成的概率。

对于一个逻辑回归任务来说, 其训练过程是 最小化目标类概率的负对数, 也就是 最大化目标类概率的对数, 用公式表示如下:

$$
V(G, D) = \mathbb{E}_{x \sim P_{data}} [\log D(x)] + \mathbb{E}_{x \sim P_G} [\log (1 - D(x))] \tag{2}
$$

在公式 $(2)$ 中, $x \sim P_{data}$ 表示从数据集中采样出来的图片, $x \sim P_G$ 表示生成器生成的图片。$\mathbb{E}$ 表示期望。在深度学习中, 我们会取一个 mini-batch 中的数据计算 loss, 然后取平均, 这和 **采样** 的思想是相似的。

也就是说, 训练判别器的过程是: 我们要寻找一个最优的判别器 $D^{\star}$, 其满足:

$$
D^{\star} = arg \max_D V(G, D) \tag{3}
$$

你可以通过一系列的推导, 得出如下结论: (具体推导过程见论文)

$$
\max_D V(G, D) = -2 \cdot \log 2 + 2 \cdot JS (P_{data} || P_G) \tag{4}
$$

在公式 $(1)$ 中, 如果用 JS 散度来衡量 $P_G$ 和 $P_{data}$ 之间的距离, 然后将公式 $(4)$ 代入公式 $(1)$ 中, 就可以得到:

$$
\begin{align*}
    G^{\star} &= arg \min_G \max_D V(G, D) \\
    &= arg \min_G \max_D E_{x \sim P_{data}} [\log D(x)] + E_{x \sim P_G} [\log (1 - D(x))]
\end{align*}
\tag{5}
$$

整个过程就是:

1. 初始化生成器 $G_0$
2. 一直更新 **判别器** 的参数, 找到一个 $D_0^{\star}$ 可以最大化公式 $(2)$ 中的 $V$ 值
3. 此时的 $V(G_0, D_0^{\star})$ 可以理解为 $P_{data}$ 和 $P_{G_0}$ 之间的 JS 散度
4. 然后用公式 $(2)$ 计算 $V$ 值, 更新 **生成器** 参数, 最小化 $V$ 值 (JS 散度), 得到 $G_1$
5. 此时你可以认为 **生成器** 的作用是为了计算 loss 值, 也可以认为其是给 **判别器** 参数提供梯度的
6. 反复执行 2 至 5 步操作, 直到训练结束

需要注意的是, 在更新生成器时, $\mathbb{E}_{x \sim P_{data}} [\log D(x)]$ 的值和生成器的参数没有任何关系, 因此实际计算的公式如下:

$$
\begin{align*}
    V_G(G, D) &= \mathbb{E}_{x \sim P_G} [\log (1 - D(x))] \\
              &= \mathbb{E}_{z \sim N(0, 1)} [\log (1 - D(G(z)))]
\end{align*}
\tag{6}
$$

## 实际的过程

上面的过程只是理论上的过程, 实际的过程有一定性质的差别。主要体现在: 在估算 JS 散度时, 我们不会真正去找最佳的 $D^{\star}$, 而是只会更新判别器 $D$ 的参数一次。如果真的使用最佳的 $D^{\star}$ 来估算 JS 散度, 这个训练会崩溃的。

比较正规的解释是: $P_G$ 和 $P_{data}$ 分布的重合度是比较低的, 如果真的估算 JS 散度, 其值很可能趋近于常数 $\log 2$ 。因此, 我们只需要往 JS 散度的方向找一个下界即可, 那么只需要更新一次参数即可。

虽然说训练的过程是两者在 **对抗**, 但是实际上它们是 **亦敌亦友** 的关系, 类似于鸣人和佐助的关系。在训练的过程中, 如果一方的能力远远超过另一方, 训练就会崩溃。

如果生成器的效果很好, 判别器的效果很差, 也就意味着生成器生成好的图片会被判别器判定为假的, 此时肯定没有办法去训练。

如果判别器的效果很好, 生成器的效果很差, 会发生什么呢? 感觉上是没有问题的, 实际上生成器大概率会陷入局部最优当中, 或者说 JS 散度趋于常数, 生成器部分的梯度趋于零。

总而言之, 在训练的过程中, 两者必须旗鼓相当, 这也是 GAN 不容易训练的原因。

一般的训练过程如下:

1. 让 生成器 生成图片 $\hat{x}$, 并从数据集中采样出图片 $x$
2. 让 生成器 判断 $\hat{x}$ 是否是真的图片, 在计算 loss 时, 认定它们是真的图片
3. 反向传播, 然后只更新 生成器 部分的参数, 清空所有参数的梯度
4. 让 判别器 判断 $\hat{x}$ 是否是真的图片, 计算 loss 时, 和上面不同, 认定它们是假的图片
5. 让 判别器 判断 $x$ 是否是真的图片, 计算 loss 时, 认定它们是真的图片
6. 两个 loss 取平均, 反向传播, 然后只更新 判别器 部分的参数, 清空所有参数的梯度
7. 重复执行上面的六个步骤, 直到训练完成

整个过程可以参考 [代码](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py), 这里给出核心的内容:

```python
import torch

# 我们假设生成器输入的向量维度是 128, 输出的图片向量维度是 86 * 64 * 3
BATCH_SIZE, GEN_INPUT_DIM, IMAGE_DIM = 2, 128, 64 * 64 * 3

generator = torch.nn.Linear(in_features=GEN_INPUT_DIM, out_features=IMAGE_DIM)
# 判别器是用来判断图片是否是生成器生成的, 因此是一个 二分类 任务
discriminator = torch.nn.Linear(in_features=IMAGE_DIM, out_features=1)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer_G = torch.optim.SGD(generator.parameters(), lr=0.01)  # 只包含 生成器 中的参数
optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.01)  # 只包含 判别器 中的参数

# ## 下面是某一个 epoch 中某一次迭代的代

# 训练生成器
optimizer_G.zero_grad()  # 清空 生成器 中参数的梯度
z = torch.randn(BATCH_SIZE, GEN_INPUT_DIM)  # 输入的随机向量 z
gen_images = generator(z)
loss = loss_fn(discriminator(gen_images).squeeze(), torch.ones(BATCH_SIZE))  # 对抗 !!!
loss.backward()
optimizer_G.step()  # 只更新 生成器 中参数, 不更新 判别器 中的参数

# 训练判别器
optimizer_D.zero_grad()  # 清空 判别器 中参数的梯度 (注意, 由于先训练的生成器, 判别器参数部分是有梯度的, 这里都清除了)
real_images = torch.randn(BATCH_SIZE, IMAGE_DIM)  # 这个是训练的数据集, 这里用 torch.randn 随机生成了
loss1 = loss_fn(discriminator(real_images).squeeze(), torch.ones(BATCH_SIZE))
loss2 = loss_fn(discriminator(gen_images.detach()).squeeze(), torch.zeros(BATCH_SIZE))  # 注意这里使用了 detach 方法, 生成器中的参数没有梯度更新
loss = (loss1 + loss2) / 2
loss.backward()
optimizer_D.step()  # 只更新 判别器 中的参数, 不更新 生成器 中的参数
```

可以看出, 整个过程是生成器和判别器在 **对抗**:

+ 在更新判别器部分的参数时, 生成的图片是 **假的**, 实际的图片是 **真的**, 希望判别器能够分辨出来图片是否是生成器生成的
+ 在更新生成器部分的参数时, 假设生成的图片都是 **真的**, 然后计算 loss, 希望生成器生成的图片尽可能地是真的, 或者说尽量骗过判别器模型

除此之外, 你如果对 [逻辑回归公式](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) 熟悉的话, 观察上面的代码, 你会发现, 在更新生成器参数时, $V$ 值的计算方式并不是公式 $(6)$, 而是:

$$
V_G(G, D) = E_{x \sim P_G} [-\log (D(x))] \tag{7}
$$

公式 $(6)$ 和公式 $(7)$ 的单调性是一致的, 但是梯度不一致。公式 $(6)$ 在训练开始时梯度小, 训练结束时梯度大; 公式 $(7)$ 则相反, 训练开始时梯度大, 结束时梯度小。显然, 公式 $(7)$ 更符合我们的预期。

公式 $(6)$ 被称为 MMGAN, 公式 $(7)$ 被称为 NSGAN。这两种 GAN 我简单的实验过, NSGAN 效果比 MMGAN 的效果要好很多。使用 MMGAN 很容易出现判别器比生成器效果好的情况, 从而使训练崩溃。

## 总结

看完上面的内容, 估计你会有疑问: 使用公式 $(7)$ 还是在模拟 JS 散度吗? 只找 JS 散度的下界真的可以吗? 虽然有证明过程 (可能还看不懂), 但公式 $(4)$ 总感觉怪怪的。

实际上, GAN 的理论基础远远比不上 VAE, 很多时候只是在实作上效果好即可。虽然有很多种类的 GAN, 但是其研究还处于早期阶段, 现在也被 diffusion model 压了一头, 希望未来能有更好地解释。

额外说明一下, 在训练 GAN 的时候, 经常会遇到 mode collapse 和 mode dropping 的问题。前者指的是 生成器 最终只能生成几乎相似的图片; 后者指的是 生成器 只能生成某一类的图片 (比方说训练集中有男性人脸和女性人脸, 但是最终生成器只能生成男性人脸, 不会生成女性人脸)。在实作上一定要注意这些问题。

## References

+ [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
+ [GAN Lecture 4 (2018): Basic Theory](https://www.youtube.com/watch?v=DMA4MrNieWo)
