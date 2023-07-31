
# 图文生成概述 (四) 条件 GAN

## 简介

本博客仅仅是简单介绍一下 条件图片生成, 包括 text-to-image generation 和 image-to-image translation。这里面涉及到的领域非常多, 在后续介绍完 diffusion model 后可能会专门写博客介绍其中的一项技术。

本文的重点是 CycleGAN, 同时也计划以此为基础训练一个模型。

## Text-to-Image

在实际使用 GAN 时, 我们希望能够加上一些文字上的限制。比方说, 我现在需要红眼睛的动漫头像, 那么我只需要向模型输入 "红眼睛", "动漫头像" 这些关键词, 模型就能自动按照要求输出图片。这样的任务被称为 Text-to-Image。

这种任务被归为 conditional GAN, "文字" 就是 "条件", 后面用 $c$ 来表示。

对于这类问题, 传统的做法是: 收集一些图片数据 $x$, 并为这些数据标记上期望的文字 $c$, 然后训练一个神经网络, 输入是文字 $c$, 输出是图片 $\hat{x}$, 目标函数是 $x$ 和 $\hat{x}$ 之间的欧式距离。

这种做法对于 生成式 的模型有一个很大的问题: 那就是输入和输出并不是一一对应的关系。

如果输入的是 "火车", 而训练集中关于 火车 的图片有很多种, 此时模型学习的不是生成其中的一种, 而是这些图片的 "平均值"。转化为分类问题描述是, 对于同一个输入 $c$ 而言, 一会告诉模型是 正类, 一会告诉模型是 负类, 那么最终训练的结果很可能是输出正类和负类的概率都接近 0.5 。

怎么解决上述问题呢? 我们需要给 模型 的输入加上一个 随机向量 $z$, 将这些 不确定性 往这个随机向量上映射。这个就是 GAN 的生成器。同时也不要使用 距离 来衡量生成器生成的好坏, 而使用 分类模型 来判断生成的好坏。这个就是 GAN 的判别器。

具体来说, 此时生成器的输入是条件 $c$ 和随机向量 $z$, 输出是图片 $\hat{x}$; 判别器的输入是图片 $x$ 和条件 $c$, 输出是标量分数, 表示此次图片生成的质量。训练过程中一个 step 的流程如下:

+ 从数据集中采样出条件 $c$, 与之相匹配的图片 $x$, 以及与之不相匹配的图片 $\widetilde{x}$
+ 从正态标准分布中采样出 $z$ 向量, 使用生成器, 根据 $c$ 和 $z$ 生成图片 $\hat{x}$
+ 对于判别器, $(c, x)$ 是正类, $(c, \widetilde{x})$ 和 $(c, \hat{x})$ 都是负类, 计算目标值, 更新参数
+ 对于生成器, 假定 $(c, \hat{x})$ 是正类, 计算目标值, 更新参数, 希望其能骗过 判别器

更多相关内容, 可以参考: [paperswithcode](https://paperswithcode.com/task/text-to-image-generation), 里面包含数据集, 论文和代码。

## Image-to-Image

Image-to-Image 的意思是根据图片生成图片, 可以理解为将图片转换成我们想要的样式。其和 Text-to-Image 是相似的, 只是条件从 文本 转换成了 图片。

我们可以用这种方式实现 超分辨率 (super resolution), 图片补全 (Image Completion) 等功能。图片补全可用于去水印, 马赛克之类的功能, 但是需要注意的是这样仅仅是让图片变得合理, 其原本包含的信息是不能够还原的。

GAN 的相关技术还可以用于 PhotoShop 中, 相关内容可以参考论文: [Generative Visual Manipulation on the Natural Image Manifold](https://arxiv.org/abs/1609.03552) 和 [NEURAL PHOTO EDITING WITH INTROSPECTIVE ADVERSARIAL NETWORKS](https://arxiv.org/abs/1609.07093)。

更多相关内容, 可以参考: [paperswithcode](https://paperswithcode.com/task/image-to-image-translation)。

## 无监督的条件 GAN (一) CycleGAN

对于 Image-to-Image 的问题来说, 很多时候我们并没有配对的数据。比方说, 对于 风格迁移 任务来说, 我们希望将 照片 变成 动漫风格的图片。按照上面的做法, 我们需要雇佣一大批的画家, 根据照片来画动漫图, 然后来训练。有没有可能通过 无监督 的方式来训练呢?

假设我们有一组照片和一组动漫图, 但是它们之间没有配对关系, 我们希望将照片转化为动漫图。

更宽泛地说, 我们有一组 $A$ domain 和 $B$ domain 的图片, 它们之间没有配对关系, 如何将 $A$ domain 的图变成 $B$ domain 的图片呢?

一种比较直接的方式是套用 GAN 的框架。对于 生成器 $G$ 来说, 其输入的是 $A$ domain 的图片, 输出的是 $B$ domain 的图片; 对于 判别器 $D$ 来说, 其作用是判断图片是 $A$ domain 的图片还是 $B$ domain 的图片。我们用 $x$ 表示图片, 那么目标函数如下:

$$
V_{GAN} = \mathbb{E}_{x \sim B} \log D(x) + \mathbb{E}_{x \sim A} \log [1 - D(G(x))] \tag{1}
$$

接下来和 GAN 一致, $G$ 希望最小化 $V$ 值, $D$ 希望最大化 $V$ 值。但是, 这样可能会出一个问题: 那就是生成的 $G(x)$ 图片虽然是 $B$ domian 的, 但是和输入的 $x$ 图片之间没有任何关系。比方说, 我们有一张风景照, 结果经过了 $G$ 的转换, 变成了梵高的自画像, 这肯定不是我们想要的。

因此, 我们需要添加限制, 让 $x$ 和 $G(x)$ 之间保持一定的相关性。[Domain Transfer Network](https://arxiv.org/abs/1611.02200) 采用的方式如下:

假设 $E$ 是一个已经训练好的图片编码模型, 可以是自编码器的 encoder 部分, 可以是 ResNet 模型去掉 softmax 层, 还可以是用对比学习训练好的模型。我们在目标函数中添加下面这一项:

$$
V_{PT} = ||E(x) - E(G(x))|| \tag{2}
$$

即希望用 $E$ 编码后的 $x$ 和 $G(x)$ 向量距离越近越好。需要注意的是, 公式 $(2)$ 和 $D$ 参数没有关系, 因此在更新 $D$ 参数时, 不用计算这一项。

另一种更著名的方法是 CycleGAN, 出自论文 [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)。

CycleGAN 中做的事情是同时训练两个生成器模型, $G_1$ 将 $A$ domain 的图片转换成 $B$ domain 的图片, $G_2$ 将 $B$ domain 的图片转换成 $A$ domain 的图片。$D_1$ 和 $D_2$ 的功能是判断图片是 $A$ domain 的还是 $B$ domain 的, 分别用于辅助 $G_1$ 和 $G_2$ 模型的训练, 也就是说对于 $G_1$ 而言, $B$ domain 是正类, 对于 $G_2$ 而言, $A$ domain 是正类。

整个目标函数分为两部分:

$$
\begin{align*}
    \mathrm{loss}_{GAN} &= \mathbb{E}_{x \sim B} \log D_1(x) + \mathbb{E}_{x \sim A} \log [1 - D_1(G_1(x))] \\
    &+ \mathbb{E}_{x \sim A} \log D_2(x) + \mathbb{E}_{x \sim B} \log [1 - D_2(G_2(x))]
\end{align*}
\tag{3}
$$

$$
\begin{align*}
    \mathrm{loss}_{cycle} &= \mathbb{E}_{x \sim A} ||G_2(G_1(x)) - x|| \\
    &+ \mathbb{E}_{x \sim B} ||G_1(G_2(x)) - x||
\end{align*}
\tag{4}
$$

公式 $(3)$ 就是上面所说的 GAN 的公式 $(1)$。公式 $(4)$ 希望 $G_1$ 和 $G_2$ 可以互逆, 即 $A$ domain 的图片经过 $G_1$ 和 $G_2$ 的变换后和原图是一致的。我们认为, 如果 $G_1$ 和 $G_2$ 可以互逆, 那么 $x$ 和 $G_1(x)$ 就会具有相关性。从实验结果来看, 确实如此。想要实践, 了解更多细节, 可以参考 [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)。

## 无监督的条件 GAN (二)

除了上面所说的方法外, 还有一种方法, 是将 $A$ domain 和 $B$ domain 投影到同一个 向量空间 中。使用的是 AutoEncoder 技术的扩展。

现在我们有四个神经网络: $EN_A$, $EN_B$, $DE_A$ 和 $DE_B$。$EN_A$ 和 $DE_A$ 构成一个 AutoEncoder 架构; $EN_B$ 和 $DE_B$ 构成一个 AutoEncoder 架构。我们希望 $EN_A$ 和 $EN_B$ 编码后的 code 在同一个向量空间中, 这样 $EN_A$ 和 $DE_B$ 网络在一起就构成了一个 风格迁移 的模型。如何达成这一目标呢?

[Couple GAN](https://arxiv.org/abs/1606.07536) 和 [UNIT](https://arxiv.org/abs/1703.00848) 中提出的方式是让 $EN_A$ 和 $EN_B$ 的后几层共享参数, $DE_A$ 和 $DE_B$ 的前几层共享参数, 然后直接用 reconstruction error 作为目标函数, 来训练模型。

当然, 我们也可以采用 VAE-GAN 的架构来实现。现在四个神经网络不共享参数了, 而是再添加一个 判别器模型, 让判别器判断编码器生成的 code 是 $EN_A$ 的输出还是 $EN_B$ 的输出。如果最终判别器无法区分 code 是从哪一个 domain 中的图片编码得到的, 我们就认为它们属于同一个向量空间。当然, 还要加上 reconstruction error。

[ComboGAN](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Anoosheh_ComboGAN_Unrestrained_Scalability_CVPR_2018_paper.pdf) 的想法和 CycleGAN 很相似, 对于一个 $A$ domain 的图片 $x$, 其经过 $EN_A$, $DE_B$, $EN_B$, $DE_A$ 四个神经网络得到的图片是 $\hat{x}$。我们希望 $x$ 和 $\hat{x}$ 之间的 reconstruction error 越小越好。这种被称为 cycle consistency, 即饶了一圈后依然保持一致。

[XGAN](https://arxiv.org/abs/1711.05139) 的想法是, 对于一个 $A$ domain 的图片 $x$, 其经过 $EN_A$ 后得到 $z_1$, 然后经过 $DE_B$ 和 $EN_B$ 后得到 $z_2$。我们希望 $z_1$ 和 $z_2$ 之间的某一种距离越小越好。这种被称为 semantic consistency, 我们可以将 $z_1$ 和 $z_2$ 理解为语义, 即保持语义上的一致性。

ComboGAN 和 XGAN 除了上述过程外, 和 CycleGAN 一样, 需要两个额外的判别器模型, 判断图片是 $A$ domain 还是 $B$ domain 生成的。整个过程由 6 个神经网络组成的, 还是非常复杂的。

## 总结

从本篇博客和上篇博客说的 bi-GAN, 你会发现 GAN 可以是一个通用的框架。在训练的过程中, 如果有两个同类的东西, 都可以采用 GAN 框架, 在语音和文本领域都有相关的应用。

## References

+ [GAN Lecture 3 (2018): Unsupervised Conditional Generation](https://www.youtube.com/watch?v=-3LgL3NXLtI)
+ [GAN Lecture 8 (2018): Photo Editing](https://www.youtube.com/watch?v=Lhs_Kphd0jg)
