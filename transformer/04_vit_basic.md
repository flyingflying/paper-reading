
# Vision Transformer & Masked Autoencoder

[TOC]

[Transformer](https://arxiv.org/abs/1706.03762) 和 [BERT](https://arxiv.org/abs/1810.04805) 在 NLP 领域大放光彩, 取代了原本 RNN 的地位。那么, 我们能不能将 Transformer 架构应用于 CV 领域呢? 能不能和 掩码 的视觉任务相互配合呢? 本文将介绍一下最近两年大火的 Vision Transformer (ViT) 和 Masked Autoencoder (MAE)。

## 引言

我们知道, 图片是由 **像素** (pixel) 点构成的。对于一般的图片来说, 每一个像素由 RGB 三个颜色通道构成。我们将每一个颜色通道的值称为 **像素值** (pixel value), 取值范围是 0 到 255 之间的整数, 数值的含义是颜色的 **强度** (intensity), 0 表示颜色强度最弱 (没有颜色), 255 表示颜色强度最强。此时, 一共有 $256^3$ (大约是 1,700 万) 种颜色。

虽然 **像素值** (pixel value) 是离散的, 但是按照统计学 [测量尺度](https://en.wikipedia.org/wiki/Level_of_measurement) 的划分标准, 其属于 **等比尺度** (ratio scale)。然而, 在 NLP 中, 我们处理句子的基本单元是 **token**, 其属于 **名目尺度** (nominal scale)。因此, 两个领域相差非常大, 这里说明两点:

首先, 数据预处理差别很大: 对于图像来说, 我们通常是先进行 [min-max normalization](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)), 再进行 [z-score normalization](https://en.wikipedia.org/wiki/Feature_scaling#Standardization_(Z-score_Normalization)) 即可。然而, token 的向量化一直是 NLP 的难点之一, 比较出名的算法有 word2vec 等等。

其次, 生成式任务的差别很大: 对于文本生成任务来说, 由于 token 是 名目尺度, 我们直接用 softmax 回归即可。相较而言, 图片生成任务难度大很多, 传统的做法是直接计算 生成图片 和 目标图片 之间的欧式距离 (比方说, AutoEncoder 和 VAE), 后续有 GAN 和 Diffusion Model 等方式。

尽管两个领域相差很大, 但是一直是 相互借鉴 的关系。在 [Transformer](https://arxiv.org/abs/1706.03762) 和 [BERT](https://arxiv.org/abs/1810.04805) 出名之前, NLP 借鉴 CV 的方法很多。而现在要介绍的 Vision Transformer (ViT), 属于 CV 借鉴 NLP 中的方法。本文假设你对 Transformer 架构, BERT 以及 GPT 模型非常了解。下面, 就让我们看看图片应该如何应用于 Transformer 架构上。

## 相关工作: ImageGPT

在介绍 ViT 之前, 我们先介绍一下比其早发布四个月的 OpenAI 的一个工作: ImageGPT。其思想很简单, 将图片直接 reshape 成一维的序列, 然后将 像素 当作 token, 一种颜色一个类别, 套用 GPT 或者 BERT 模型即可。具体的细节如下:

首先, 是对颜色类别的处理。刚刚说过, 图片一般有 $256^3$ 种颜色, 如果将每一种颜色作为一个类别, 那么类别数太多了, 同时很多类别是很相近的。于是, 作者将数据集中所有图片的像素点收集起来, 然后用 [k-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) 方法进行聚类, 一共聚成 512 个类别, 这样大大缩减了颜色类别的数量。OpenAI 没有开源相关内容的 [代码](https://github.com/openai/image-gpt), 我在网上找到了针对一张图片进行聚类的代码, 仅供参考: [Color Quantization](https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html)。通过上述方式, 我们将像素从 **等比尺度** 转换成 **名目尺度**, 这样就可以直接套用 NLP 中的 BERT 以及 GPT 模型了。模型的输入和预测的内容都是这 512 个颜色类别。

其次, 是序列长度的问题。我们一般处理的图片大小是 224 x 224, 一共有 50,176 个像素, 对应到 ImageGPT 中就是长度为 50,176 的序列, 这对于硬件资源的要求非常之高。因此, 作者将图片 缩放 或者 裁剪 成较小的图片, 一共实验了三种大小的图片: 32 x 32, 48 x 48 和 64 x 64, 对应的序列长度分别是 1024, 2304 和 4096。

你或许会有疑问, 直接将图片 reshape 成一维序列可以吗? 在 Transformer 架构中, Attention 操作完全没有使用任何 **位置信息**。换言之, Attention 操作就没有将输入当作序列来看, 因此, 是否 reshape 对计算没有任何影响。整个 Transformer 架构的位置信息都集中在 **位置编码** (positional encoding) 中。显然, 这里不能直接使用 [Transformer](https://arxiv.org/abs/1706.03762) 中的 正弦波 (sinusoidal) 位置编码, 也不能直接使用 [旋转式位置编码](https://spaces.ac.cn/archives/8265) (RoPE), 都需要进行改造。苏剑林有将 RoPE 改造成二维的, 具体的可以参考他的博客: [Transformer升级之路：4、二维位置的旋转式位置编码](https://spaces.ac.cn/archives/8397)。在 ImageGPT 中, 作者直接和 BERT 以及 GPT 保持一致, 采用 **可训练式位置编码**, 希望模型能自己学习到二维的位置关系。综上所述, 虽然我们将图片 reshape 成一维序列, 但是二维的空间信息还是保留的, 存在于 位置编码 中。

这里体现了 Transformer 架构的灵活性: 由于 Attention 层没有对位置信息进行建模, 我们也不需要关注输入的数据是否存在位置信息。如果使用 RNN 架构, 就需要对计算方式进行调整了, 否则位置信息会丢失。

那么, ImageGPT 的效果好吗? 并不是太好。OpenAI 在其 [博客](https://openai.com/research/image-gpt) 中展示了用其做 图文生成 的结果, 效果还是可以的, 问题是生成图片的分辨率太低了, 只有 64 x 64, 没有办法展示事物细节。

OpenAI 在其 [论文](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf) 中主要说明了将 ImageGPT 模型迁移到图片分类任务上的效果, 并不是很理想。作者的大部分实验都是使用 GPT 模型, BERT 模型只做了少量的实验。只能说不愧是 OpenAI, 对 GPT 模型情有独钟啊。

为什么 ImageGPT 的效果不好呢? 个人认为有以下一些原因:

+ 信息丢失
  + 输入的图片分辨率太低, 这样会丢失大量图片中的细节
  + 将颜色信息变成名目尺度后, 也会有信息丢失, 同时图片纹理信息可能更加难学习
+ NLP 和 CV 领域的区别
  + 图片中一个像素点包含的信息比句子中 token 包含的信息要少, 但是整张图片包含的信息可能比句子要多很多
  + 图片中 冗余像素的占比 比句子中 冗余 token 的占比 要高很多
  + 句子的 重要信息 出现在开头部分的概率较大, 而图片的 重要信息 出现在中间位置的概率较大

下面, 就让我们看看 Vision Transformer 是怎么做的吧。

## ViT

[ViT](https://arxiv.org/abs/2010.11929) 是谷歌于 2020 年 10 月发布的工作。和 ImageGPT 不同, ViT 不是将一个 像素 作为一个 token, 而是将图片划分成一块一块的, 每一个 **区域块** (patch) 作为一个 token。

假设输入图片的大小是 224 x 224, 每一个 **区域块** (patch) 的大小是 16 x 16, 那么一共有 $\frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196$ 个 **区域块** (patch)。在一个区域块内, 一共有 $16 \times 16 \times 3 = 768$ 个 pixel values。我们将这些 pixel values 直接作为 patch 向量的元素值, 然后进过一次线性变换即可。为了和 NLP 中的概念对应, 我们将这个线性变换称为 patch embedding。由于一共有 196 个 patches, 输入的序列长度也是 196。

需要注意的是, 在 ViT 中, 预训练任务不是生成任务, 还是 ImageNet 分类任务。和 BERT 模型类似, 作者在序列的开头处加入了 `[CLS]` 特殊 token, 用于分类任务。因此, 完整的序列长度是 197。

模型的架构和 Transformer 的 Encoder 没有什么区别, 位置编码 也是采用 可训练式位置编码。作者还进行了测试, 发现每一个 位置向量 和同行或者同列的 位置向量 之间的 cosine 相似度较高, 具体可以参考 [论文](https://arxiv.org/abs/2010.11929) 第 18 页的图 10。如果预训练的图片大小是 224 x 224, 而微调时图片的大小是 384 x 384, 那么应该怎么办呢? 作者直接用 [torch.nn.functional.interpolate](https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html) 方法对 位置编码 进行线性差值。

作者一共设置了三种大小的模型: base, large 和 huge。其中 base 和 BERT-base 差不多大, large 和 BERT-large 差不多大, huge 则更大。同时, patch 的大小也是可以改变的, 作者一共实验了三种尺度的 patch: 32 x 32, 16 x 16 和 14 x 14。越小的 patch 意味着 patch 的数量越多, 输入的序列越长, 所需要的计算资源越多。作者主要训练了 5 种模型:

+ ViT-B/32 : base 模型, patch 大小是 32 x 32
+ ViT-B/16 : base 模型, patch 大小是 16 x 16
+ ViT-L/32 : large 模型, patch 大小是 32 x 32
+ ViT-L/16 : large 模型, patch 大小是 16 x 16
+ ViT-H/14 : huge 模型, patch 大小是 14 x 14

Transformer 架构的特点是 数据量越多, 模型越大, 性能就越好。如果只用 ImageNet 数据集进行预训练, 效果是比不过 ResNet 的, 怎么办呢? 作者就用谷歌内部的 [JFT-300M](https://paperswithcode.com/dataset/jft-300m) 数据集, 一共有 3 亿张带标签的图片, 进行预训练, 效果提升很明显。

2021 年, 同一批人又发表了新的 [工作](https://arxiv.org/abs/2106.04560), 提出 ViT-G/14 (G 表示 giant), 一共有 20 亿参数, 使用 30 亿张带标签的图片 (JFT-3B) 进行预训练, 在 ImageNet-1k 分类任务上进行微调, 最终达到 90.45% 的正确率, 成功霸榜, 诠释了什么是 "有钱任性"。当然, 现在看起来, 和 GPT3 相比, 那是小巫见大巫了。

作者在 ViT 的论文中说用 掩码图片模型 (MIM) 效果不好。我以为接下来的工作是继续挖掘 MIM 任务的可能性, 结果他们去收集了一个更大的 JFT-3B 数据集来进行预训练。下面, 我们来看看清华大学的 SimMIM 以及何凯明的 MAE 工作, 是如何将 ViT 模型和 MIM 任务结合起来进行 **自监督学习** 的。

## SimMIM

[SimMIM](https://arxiv.org/abs/2111.09886) 设计的 掩码图片任务 非常简单:

1. 设置一个可学习的 `[MASK]` 特殊 token
2. 对于输入的所有 patch, 随机选择 50% 用 `[MASK]` token 进行替换
3. 添加位置信息, 用 ViT 模型编码成向量
4. 根据掩码的 patch 向量, 通过一个线性层直接输出 patch 的像素值, 和 原始的像素值 之间计算 L1 距离, 作为最终的 损失值

上述过程可以配合 [论文](https://arxiv.org/abs/2111.09886) 中的图 1 进行理解。

作者使用 ViT-Base 模型, 先用 ImageNet-1K 数据集进行预训练, 再在 ImageNet-1K 分类任务上进行微调, 最高达到 83.8% 的准确率! 作者在论文中没有说明使用 ViT-Huge 的效果, 但是说明了 SwinV2-Giant 的效果, 可以达到 87.1% 的准确率。关于 Swin Transformer 的内容之后有时间再进行细读和介绍。

除此之外, 作者在 [论文](https://arxiv.org/abs/2111.09886) 中花了很大的篇幅来说明和设计 消融实验 (Ablation Study), 涉及到很多论文工作, 也是之后有时间再进行细读和介绍。

个人认为, 这种方式和 BERT 最为相似。对于 NLP 来说, token 是 名目尺度, 那我们就按照 名目尺度 的方式来设计模型的输入和输出; 对于 CV 来说, pixel value 是 等比尺度, 那我们就按照 等比尺度 的方式来设计模型的输入和输出。没有必要强行进行转换。

## MAE

[MAE](https://arxiv.org/abs/2111.06377) 的全称是 Masked Autoencoder, 和 BERT 模型差别还是挺大的。特别说明一下, 这部分所说的 encoder 和 decoder 都是 AutoEncoder 中的概念, 和 Transformer 没有关系。

和 AutoEncoder 类似, 预训练的网络架构分成 encoder 和 decoder 两部分, 用的都是 ViT 模型。具体的做法如下:

+ 对于输入的图片, 随机选择 75% 的 patch 进行掩码
+ 将 25% 的未掩码的 patch 输入 encoder 中, 进行编码, 得到 patch 向量
+ 将 75% 掩码的 patch 用 `[MASK]` 特殊向量代替, 和 encoder 输出的 patch 向量拼接在一起, 输入 decoder 中, 预测原始图片的像素值
+ 计算 掩码处 预测的像素值 和 原始像素值 之间的欧式距离, 作为最终的 loss 值

整个流程可以配合 [论文](https://arxiv.org/abs/2111.06377) 中的图一来理解。整个过程最令人意想不到的是, encoder 的输入仅仅是 25% 未掩码的 patch。能这么做的原因还是 Transformer 架构的灵活性, 即 Attention 的计算过程没有对位置信息进行建模。这些 patch 的位置信息可以通过 **位置编码** 告诉模型! 如果使用 CNN 架构, 需要将掩码的部分用一种统一的颜色替换, 具体可以参考 image inpainting 相关的工作。

75% 的掩码力度非常之大, SimMIM 中也只掩码了 50%, BERT 中只掩码了 15%, 图片中包含的冗余信息偏多, 可能这是一个关键因素吧。

和 AutoEncoder 一样, 只有 encoder 部分用于下游任务, decoder 部分则直接忽略。我们在进行微调时, 输入 encoder 中的是整张图片所有的 patch, 而不是部分 patch。最终, 使用 ViT-Huge 模型, 先用 ImageNet-1K 数据集进行预训练, 再在 ImageNet-1K 分类任务上进行微调, 最高达到 87.8% 的准确率!

进一步分析, 预训练阶段 encoder 的输入只有 未掩码 patch 的优缺点:

+ 优点: 预训练 和 微调 阶段模型输入的类型是一致的 (都不包含 `[MASK]` 特殊向量, BERT 论文中有提到这一点)
+ 缺点: 预训练 和 微调 阶段模型输入的长度不一致

[论文](https://arxiv.org/abs/2111.06377) 的图 2 和图 3 展示了预训练任务的效果, 可以看出, 生成的 patch 比较模糊, 但是是合理的。这个任务仅仅是 预训练任务 (或者说 pretext task), 不会用于 图片生成 任务的, 只要能学习到一个好的 特征表示 即可。

## 总结

[Vision Transformer](https://arxiv.org/abs/2010.11929) 将 Transformer 架构应用于 CV 领域, 并取得了很好的效果。之后, 有工作发现, ViT 模型可以学习到 CNN 模型学习不到的信息, 同时使得 **对抗样本攻击** (Adversarial Attack) 任务变得更加困难, 是非常值得探索的一个领域。

## 引用

论文部分:

+ [**ImageGPT** | Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)
+ [**ViT** | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
+ [**SimMIM** | SimMIM: a Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)
+ [**MAE** | Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
+ [Intriguing Properties of Vision Transformers](https://arxiv.org/abs/2105.10497)

视频部分:

+ [**BV15P4y137jb** | ViT论文逐段精读【论文精读】](https://www.bilibili.com/video/BV15P4y137jb)
+ [**BV1sq4y1q77t** | MAE 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1sq4y1q77t)

HuggingFace 文档部分:

+ [Transformers / MODELS / VISION MODELS / Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)
+ [Transformers / MODELS / VISION MODELS / ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae)
+ [Transformers / MODELS / VISION MODELS / ImageGPT](https://huggingface.co/docs/transformers/model_doc/imagegpt)

其它内容:

+ [What is the difference between a pixel value and a pixel color?](https://www.quora.com/What-is-the-difference-between-a-pixel-value-and-a-pixel-color)
