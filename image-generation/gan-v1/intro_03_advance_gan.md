
# 对抗神经网络 (三) GAN 进阶

[TOC]

## 简介

上篇博客简单介绍了 GAN 的基本架构和思想。本文作为 GAN 的进阶篇, 介绍一些有意思的变种, 帮助读者更好地理解 GAN 和 VAE 的架构。

在这之前, 我们还需要对 EM 距离和 f 散度有一定的认识。

## KL & JS divergence

我们知道, [KL 散度](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) 和 [JS 散度](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) 的计算公式如下:

$$
\mathrm{KL} (P || Q) = - \sum_x p(x) \log \frac{q(x)}{p(x)}
\tag{1}
$$

$$
\mathrm{JS} (P || Q) = \frac{1}{2} \cdot \mathrm{KL} (P \big|\big| \frac{P + Q}{2}) + \frac{1}{2} \cdot \mathrm{KL} (Q \big|\big| \frac{P + Q}{2})
\tag{2}
$$

那么, 这些散度有什么问题呢? 对于 KL 散度, 我们从公式就可以看出, 其不具有 **对称性**, 即 $\mathrm{KL} (P||Q) \ne \mathrm{KL} (Q || P)$; 除此之外, 其还不满足 **三角不等式** 。

对于 JS 散度来说, 其有对称性, 也满足三角不等式, 但是如果两个分布没有交集, 其值恒定为 $\log 2$ 。我们可以从下面的代码来理解:

```python
import torch 
from torch import Tensor 


def kl_div(p_dist: Tensor, q_dist: Tensor, eps: float = 1e-8):
    p_dist = p_dist + eps
    q_dist = q_dist + eps
    return torch.sum(p_dist * (torch.log(p_dist) - torch.log(q_dist)))


def js_div(p_dist: Tensor, q_dist: Tensor):
    m_dist = (p_dist + q_dist) / 2
    return kl_div(p_dist, m_dist) / 2 + kl_div(q_dist, m_dist) / 2


yh1_dist = torch.tensor([0.4, 0.6, 0.0, 0.0, 0.0])
yh2_dist = torch.tensor([0.0, 0.4, 0.6, 0.0, 0.0])

y_dist = torch.tensor([0.0, 0.0, 0.0, 0.4, 0.6])

print(f"yh1_dist 和 y_dist 之间的 JS 散度是: {js_div(yh1_dist, y_dist)}")
print(f"yh2_dist 和 y_dist 之间的 JS 散度是: {js_div(yh2_dist, y_dist)}")
```

如果随机变量是离散且 **无序** 的, 这样的结果是没有问题的。但是如果变量是连续的呢? 比方说有三个正态分布, 它们的方差都是 1, A 分布的均值是 0, B 分布的均值是 6, C 分布的均值是 12 。根据 3-sigma 准则, 三个正态分布之间几乎没有交集, 也就意味着 JS(A || C) 和 JS(B || C) 的值是非常接近的, 大约为 log 2 。但是实际上, 由于随机变量具有 **大小关系**, 我们希望 A 到 C 的距离大于 B 到 C 之间的距离。

额外说明一点, (A + C) / 2 分布不是正态分布, 而应该是一个双峰的分布, 采样到的点应该集中在 0 附近或者 12 附近。这对于初学者来说容易弄混淆。

应该如何解决上述的问题? 那就必须要在计算公式中加入随机变量 $x$ 的距离计算。

## Earth-Mover Distance

假设随机变量 $X$ 有 $N$ 类: $x_1, x_2, \cdots, x_n$。用 $D(x_i, x_j)$ 表示 $x_i$ 类到 $x_j$ 类之间的距离。假设关于随机变量 $x$ 有两个概率分布, $P$ 和 $Q$。

$P$ 分布和 $Q$ 分布的 EM 距离的是: 将 $P$ 分布变成 $Q$ 分布所需要距离的最小值。假设 $P$ 分布是 `[0.5, 0.1, 0.4]`, $Q$ 分布是 `[0.5, 0.0, 0.5]` 。如果想将 $P$ 分布变成 $Q$ 分布, 需要将第二类 0.1 分量的 "土" 移动给第三类。此时的 EM 距离是: $0.1 \times D(x_2, x_3)$ 。

这里用 "土" 形象化的表示 "概率", 因此也被称为 "推土机距离"。当然, "推土方案" 不一定只有一种, 可能会有很多种。比方说, 对于上面的例子, 我们可以推 $x_1$ 类 0.1 的 "土" 给 $x_3$ 类, 再推 $x_2$ 类 0.1 的 "土" 给 $x_1$ 类。这样算出来的距离会大很多。

因此, EM 距离定义的是所有的 "推土方案" 中 **最小** 的距离。我们用 $\gamma$ 表示一个 "推土方案", $\gamma (x_i, x_j)$ 表示从 $x_i$ 类推给 $x_j$ 类 "土" 的分量。那么我们可以得到:

$$
W(P, Q) = \min_{\gamma \in \prod} \sum_{i=1}^N \sum_{j=1}^N \gamma(x_i, x_j) D(x_i, x_j) \tag{3}
$$

其中 $\prod$ 表示所有可能的 "推土方案"。

"土" 只是 "概率" 形象化的表述。实际上, $\gamma$ 可以理解为 $P$ 和 $Q$ 分布的某一种联合概率分布。$P$ 和 $Q$ 有多少种联合概率分布, 就有多少种 "推土方案" (从边缘概率的角度来理解, 不要从定义来理解)。因此, 很多地方也有用下面公式表示的:

$$
W(P, Q) = \inf_{\gamma \sim \prod(P, Q)} \mathbb{E}_{(x_i, x_j) \sim \gamma} || x_i - x_j || \tag{4}
$$

其中, $\inf$ 表示 "下界", 和 "最小值" 的意思差不多。$\prod (P, Q)$ 表示 $P$ 和 $Q$ 所有可能的联合概率分布, $\gamma$ 是其中的一种。$||x_i - x_j||$ 表示的是两者之间的距离, 和 $D(x_i, x_j)$ 意思是一样的。

可以看出, EM 距离和 JS 距离最大的区别在于是否考虑 随机变量 之间的距离。当然, 其计算复杂度更高了, 因为需要求解一个 "最小值问题"。EM 距离更正式的名称是 Wasserstein 距离。

## f-divergence

实际上, 衡量两个分布之间距离的方式是多种多样的。除了像 JS 散度, KL 散度外, 还有 α 散度, 海林格距离等等。这些距离有一个统一的形式: [f 散度](https://en.wikipedia.org/wiki/F-divergence)。

$$
D_f(P || Q) = \sum_x q(x) f\left(\frac{p(x)}{q(x)}\right) \tag{5}
$$

其中, $f$ 函数需要是凸函数 (convex function), 并且 $f(1) = 0$, 这样可以保证 $D_f(P || Q) \ge 0$ 。

如果 $f(t) = t \cdot \log t$, 代入公式 $(5)$, 可以得到 KL 散度; 如果 $f(t) = -(t + 1) \cdot \ln \left( \frac{t+1}{2} \right) + t \cdot \ln t$, 代入公式 $(5)$, 可以得到 JS 散度。

对于每一个凸函数 $f(t)$, 其都有一个 [共轭凸函数](https://en.wikipedia.org/wiki/Convex_conjugate) (conjugate function) $f^*(t)$ 。其属于 凸优化 (convex optimization) 中的内容, 这里不展开描述了。

## GAN 架构

对于 **生成器** 来说, 其输入的是一个从 **正态分布** 中采样得到的向量, 输出的是一个 **图片张量**。一般会用 `tanh` 函数作为 **激活函数**, 取值在 -1 到 1 之间, 需要自行映射回 0 到 255 之间。一般情况下, 网络架构是: ConvTranspose2d + BatchNorm2d + ReLU 。

对于 **判别器** 来说, 其输入的是一个 **图片张量**, 输出的是一个 **数值** (scale)。一般情况下, 网络架构是: Conv2d + BatchNorm2d + LeakyReLU。

在后续的部分, 我们用 $x$ 表示判别器的输入, 可能是实际的图片, 也可能是生成器生成的图片。用 $D$ 表示判别器, $D(x)$ 表示判别器输出的标量数值, 其没有经过任何激活函数, 就是单纯线性层的输出。如果使用逻辑回归, 那么 $D(x)$ 就是 logit 值。如果用 $\sigma$ 表示 sigmoid 函数, 则目标函数是:

$$
V(G, D) = \mathbb{E}_{x \sim P_{data}} [\log \sigma(D(x))] + \mathbb{E}_{x \sim P_G} [\log (1 - \sigma(D(x))] \tag{6}
$$

在更新判别器时, 我们最大化目标函数的值, 在更新生成器时, 我们最小化目标函数的值。其和 JS 散度之间的关系是: (其中, $P_{data}$ 是样本集合的概率分布, $P_G$ 是生成器生成图片的概率分布)

$$
\max_D V(G, D) = -2 \cdot \log 2 + 2 \cdot JS (P_{data} || P_G) \tag{7}
$$

## f-divergence GAN

[f-GAN](https://arxiv.org/abs/1606.00709) 这篇论文做的事情是用 判别器 来模拟各种散度的计算。上面说过, f 散度将很多散度的计算统一了起来, 如果我们可以用判别器模拟 f 散度, 那么就可以模拟其包含的所有散度, 包括 KL 散度, JS 散度等等。作者通过大量的推理得到如下的公式:

$$
V_f(G, D) = \mathbb{E}_{x \sim P_{data}} [g_f(D(x))] - \mathbb{E}_{x \sim P_G} [f^*(g_f(D(x)))]
\tag{8}
$$

$$
D_f (P_{data} || P_G) = \max_D V_f(G, D) \tag{9}
$$

在公式 $(8)$ 和 $(9)$ 中, $f$ 表示 f 散度支持的某一个散度, 其有一个对应的激活函数 $g_f$ 和共轭凸函数 $f^*$。在计算真实图片目标值时, 我们只需要将 $D(x)$ 经过 $g_f$ 转换即可; 在计算生成器生成图片的目标值时, 我们需要将 $D(x)$ 经过 $g_f$ 和 $f^*$ 两个函数进行转换。

如果是原本的 GAN, $g_f$ 和 $f^*$ 公式如下:

$$
g_f (v) = \log \sigma (v) = -\log(1 + e^{-v}) \tag{10}
$$

$$
f^*(t) = - \log (1 - e^t) \tag{11}
$$

你将公式 $(10)$ 和 $(11)$ 代入公式 $(8)$ 中, 就可以得到公式 $(6)$ 。需要注意的是, 公式 $(11)$ 是要代入公式 $(10)$ 中的。

如果你想模拟 KL 散度, 那么 $g_f$ 和 $f^*$ 公式如下:

$$
g_f (v) = v \tag{12}
$$

$$
f^*(t) = e^{t - 1} \tag{13}
$$

更多相关的公式请参考论文中的 表六 (Table 6), 其一共列举了 12 种模拟散度的公式。说到这里实现已经比较简单了, 代码可以参考 [f-GAN-pytorch](https://github.com/minlee077/f-GAN-pytorch)。通过实验可以发现, mode collapse 和 mode dropping 问题并没有得到有效的解决。

## Wasserstein GAN

原始的 GAN 有什么问题呢? 一种解释是 $P_{data}$ 和 $P_G$ 分布重叠部分偏低 (可以简单理解为是由于采样导致的)。在这种情况下, $P_{data}$ 和 $P_G$ 的 JS 散度趋近于一个定值, 从而导致在更新生成器参数时, 生成器参数的梯度值很低。

原始 GAN 的方案中, 已经包含了解决办法, 那就是不要将生成器训练的太好! 即不是要真正求解公式 $(7)$ 中的最佳 $D$, 而是找一个 $D$ 的下界即可, 因此在更新 $D$ 的参数时往往只会更新一次。

怎么解决上述问题呢? 有人提出了 WGAN, 不再使用 f 散度, 而使用的时 EM 距离。其相关的论文有三篇:

+ [TOWARDS PRINCIPLED METHODS FOR TRAINING GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/abs/1701.04862)
+ [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
+ [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

作者经过一系列复杂的推理后, 得出如下结论:

$$
V(G, D) = \mathbb{E}_{x \sim P_{data}} D(x) - \mathbb{E}_{x \sim P_G} D(x)
\tag{14}
$$

$$
W(P_{data}, P_G) = \max_{D \in 1-Lipschitz} V(G, D) \tag{15}
$$

从公式 $(14)$ 来看, 整个过程都变简单了, 直接用判别器输出的标量作为 目标值, 不需要任何函数进行转换。但是其增加了一个限制, 要求判别器必须时 1-Lipschitz 函数。什么是 1-Lipschitz 函数呢?

如果 $f(u)$ 是 [Lipschitz 函数](https://en.wikipedia.org/wiki/Lipschitz_continuity), 其满足 $||f(u_1) - f(u_2)|| \le k \cdot ||u_1 - u_2||$。当 $k=1$ 时, 就是 1-Lipschitz 函数。

也就是说, 对于 $D$ 来说, 其输出值的变化率要小于输入值的变化率。我们可以简单认为 $D$ 必须是一个平滑的函数, 不能剧增剧减。

那么问题来了, 如何保证 $D$ 是一个 Lipschitz 函数呢? 作者给出了两个方案: weight clipping 和 gradient penalty。

一开始作者没有想到好的办法, 就使用 weight clipping 的做法。其想法很简单, 让参数的值域在 `[-c, c]` 之间。每一次更新完成后, 如果参数的值大于 $c$, 就让其等于 $c$, 如果参数的值小于 $-c$, 就让其等于 $-c$。

后来, 作者发现, 如果 $D$ 是一个 1-Lipschitz 函数, 那么对于任意的 $x$, $D$ 关于 $x$ 梯度的 norm 值应该都是小于 1 的。(注意, 是 $D$ 关于输入 $x$ 的梯度, 不是关于参数的梯度 !!!)

如何实现呢? 作者给出的答案是改变目标函数 $V$ 的值:

$$
\begin{align*}
V(G, D) = & \mathbb{E}_{x \sim P_{data}} D(x) - \mathbb{E}_{x \sim P_G} D(x) \\
          & + \lambda \cdot \mathbb{E}_{x \sim P_{penalty}} (||\nabla_x D(x)||_2 - 1)^2
\end{align*}
\tag{16}
$$

需要注意以下几点:

+ 这里采用的和 L2 Regularation 类似的方式, 通过改变 目标值 的方式来添加限制
+ $\nabla_x D(x)$ 表示的是 $D$ 关于输入 $x$ 的梯度, 不是关于参数的梯度
+ 我们不可能穷举所有的 $x$ 来进行计算, 那么只能根据某种规则采样进行计算
+ 原本只是希望关于 $x$ 梯度的 norm 值小于 1 即可, 作者通过实验发现是越接近 1 越好

$P_{penalty}$ 是我们假设的计算惩罚项的概率分布, 作者认为是 $P_{data}$ 和 $P_G$ 分布的中间区域, 即将 $P_{data}$ 和 $P_G$ 中图片随机相连, 取连线上随机的图片。

至于关于 $x$ 梯度的计算, 可以参考 [torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html) 函数。完整版的代码可以参考 [pytorch-wgan](https://github.com/Zeleni9/pytorch-wgan) 。

在这之后, 还有人提出了 spectrum norm 的方式实现 1-Lipschitz, 具体可以参考:

+ [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
+ [torch.nn.utils.spectral_norm](https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html)

## Least Squares GAN

出自论文 [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)。其将原始 GAN 的判别器模型从 分类任务 改成了 回归任务。如果是真的图片, 标签值为 1, 如果是生成器生成的图片, 标签值为 0。然后用 MSE 计算目标值。按照作者的推导, 其过程实际上是在模拟 Pearson $\chi^2$ 散度。

## Energy-based GAN

出自论文 [Energy-based Generative Adversarial Network](https://arxiv.org/abs/1609.03126) 。其主要的思想是将上面所说的 判别器 模型改成 AutoEncoder 架构。

在 AutoEncoder 架构中, encoder 部分负责将图片编码成向量, decoder 部分负责将向量解码成图片。最后, 我们根据 encoder 的输入和 decoder 的输出计算 欧式距离, 作为 reconstruct error, 这个 error 越小越好。

在 Energy-based GAN 中, 作者使用 reconstruct error 作为判别器的输出, 其值越小越好。其基于的假设是: 一张图片被重构的效果越好, 那么是真实图片的可能性就越大。

为什么要用 AutoEncoder 架构呢? 答案是判别器模型可以 **预训练**。在原始的 GAN 中, 判别器和生成器是一起训练的。如果你想要事先训练好一个判别器模型, 会出现一个很大的问题: 很难收集负样本。生成器一开始生成的图片很可能是没有意义且无规律的, 想要全面的收集负样本是一件很困难的事情。

但是, 如果训练的是 自编码器 模型, 就不需要负样本了, 只需要正样本即可。对于一个训练好的自编码器模型来说, 只要图片和训练集差别很大, 其 reconstruct error 就会非常地高。但是对于分类模型来说, 如果是训练阶段没有见过的样本, 其分错的概率会很高。

需要额外注意的是, 在训练 EB-GAN 时, 我们希望真实的图片 error 越小越好, 生成器生成的图片 error 越大越好。但是, error 是有最小值 0 的, 但是没有最大值。这样的训练是很容易崩溃的。我们可以设置一个 margin 值, 作为 error 的最大值。如果 error 的值比 margin 值大, 那么直接设置 error 值为 margin 值, 这样训练才比较稳定。

## infoGAN

出自论文 [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) 。

在原始的 GAN 中, 输入的是从 正态分布 中抽样出来的随机向量, 这个向量的每一个维度有什么含义, 我们并不知道。比方说, 如果用 MNIST 数据集, 生成的是 "手写数字" 的图片, 里面包含了 10 种数字 (0 到 9), 能不能有某一个维度来控制输出的数字是什么呢? infoGAN 做的事情就是尝试给某些维度赋予实际的意义, 具体的做法如下:

在 infoGAN 中, 总共有三个神经网络: 生成器, 判别器 和 辅助器。

判别器和原始的 GAN 保持一致, 是一个逻辑回归分类任务。

生成器的输入不仅仅是符合 **正态分布** 的随机向量 $z$, 还包括一个十维的 one-hot 向量 $c$, 用来表示图片中数字的类别。两个向量拼接在一起, 作为生成器的输入。需要注意的是, $c$ 向量是从 **均匀分布** 中采样得到的。其他架构和原来一致。

辅助器的功能是判断图片是哪一个数字, 其和十维的 one-hot 向量 $c$ 是对应的, 也就是一个 softmax 回归任务。其余的部分和 判别器 是一致的。

实际的训练过程是:

1. 更新判别器参数, 和原始的 GAN 保持一致
2. 更新生成器和辅助器的参数, 在原始目标值的基础上添加上 辅助器 的目标值即可

需要说明的一点是: 辅助器的训练属于 **无监督学习**, 我们只需要额外设置好标签的数量, 不需要提供标签额外训练辅助器, 希望模型自动可以学习到维度的含义。如果将 生成器 和 辅助器 连在一起, 和 "自编码器" 架构很像, 是对十维 one-hot 向量 $c$ 的 "自编码", 但是生成器多了一个 随机向量 的干扰。作者在 MNIST 上实验成功了, 但是如果换成更复杂的图片, 并不能保证一定能学到你期望的标签类型。

除此之外, $c$ 向量还可以是一个 **连续型随机变量**, 比方说数字的倾斜角度。此时的判别器并不是直接预测 $c$ 的值, 而是预测一个 **正态分布** (预测 mu 和 var), 希望 $c$ 是从预测的 **正态分布** 中抽样得到的。更多细节可以参考: [InfoGAN-PyTorch](https://github.com/Natsu6767/InfoGAN-PyTorch) 。

## VAE-GAN

出自论文 [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300)。

我们知道, VAE 是由 编码器 和 解码器 组成的, GAN 是由 生成器 和 判别器 组成的。VAE-GAN 的思想就是将这两个神经网络组合起来。现在一共有三个神经网络:

+ 编码器: 负责将 图片 编码成 code
+ 解码器 (生成器): 负责将 code 解码成图片
+ 判别器: 判断图片是 真的 还是由 解码器 (生成器) 生成的

训练流程如下:

+ 从 数据集 中采样出图片 $x$
+ 将图片 $x$ 用编码器编码成 $\widetilde{z}$, 再用解码器 (生成器) 解码成 $\widetilde{x}$
+ 从标准正态分布中采样出 $\hat{z}$, 再通过 解码器 (生成器) 生成图片 $\hat{x}$
+ 更新 编码器 参数, 希望 $x$ 和 $\widetilde{x}$ 之间的 reconstruction error 越小越好, 同时 $\widetilde{z}$ 和 标准正态分布 之间的 KL 散度越低越好
+ 更新 解码器 参数, 希望 $x$ 和 $\widetilde{x}$ 之间的 reconstruction error 越小越好, 同时 $\hat{x}$ 和 $\widetilde{x}$ 都能骗过 判别器
+ 更新 判别器 参数, 希望 判别器 能判断出 $x$ 是真实的图片, $\widetilde{x}$ 和 $\hat{x}$ 是假的图片

这种方案很好的缓解了两个算法的问题:

对于 VAE 算法来说, 采用什么距离来计算 reconstruction error 是一个问题, 如果是 欧氏距离, 当生成的图片是输入图片向左移动 1 个 pixel 时, 其 error 值可能都会非常大。有了 判别器 后, 我们不单单用 reconstruction error 来更新参数, 这个问题得到了缓解。

对于 GAN 来说, 其训练很不稳定。有了 编码器 后, 这个问题得到了一定程度的缓解。

## bi-GAN

出自论文 [Adversarial Feature Learning](https://arxiv.org/abs/1605.09782) 和 [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704)。其思想是让 AutoEncoder 的 编码器 和 解码器 分离, 然后通过 判别器 建立联系。

编码器和解码器的输入和输出没有发生变化。判别器的输入是由 图片 和 code 组成的二元组, 输出是来组编码器还是解码器。整个训练过程如下:

+ 从数据集中采样出图片 $x$, 通过编码器编码成 $\hat{z}$
+ 从标准正态分布中采样出 $z$, 通过解码器解码成图片 $\hat{x}$
+ 更新判别器的参数, 让 $(x, \hat{z})$ 为正类, $(z, \hat{x})$ 为负类
+ 更新编码器的参数, 让 $(x, \hat{z})$ 为负类, 企图骗过判别器
+ 更新解码器的参数, 让 $(z, \hat{x})$ 为正类, 企图骗过判别器

简单来说, 就是让 编码器 和 解码器 联手 骗过判别器。也就是说, 让 $P(x, \hat{z})$ 和 $Q(\hat{x}, z)$ 的概率分布越接近越好 (注意, 这里是输入和输出的联合概率分布)。训练到最后, 其应该和普通的 AutoEncoder 是一样的, 即 $x$ 和 $\hat{x}$ 是很接近的。

这个方案可以认为是用 AutoEncoder 实现 图片生成 的另一种想法。其不同于 VAE, 不需要再计算 reconstruction error 了, 泛化性也得到了提高。

## References

+ [GAN Lecture 4 (2018): Basic Theory](https://www.youtube.com/watch?v=DMA4MrNieWo)
+ [GAN Lecture 5 (2018): General Framework](https://www.youtube.com/watch?v=av1bqilLsyQ)
+ [GAN Lecture 6 (2018): WGAN, EBGAN](https://www.youtube.com/watch?v=3JP-xuBJsyc)
+ [GAN Lecture 7 (2018): Info GAN, VAE-GAN, BiGAN](https://www.youtube.com/watch?v=sU5CG8Z0zgw)
