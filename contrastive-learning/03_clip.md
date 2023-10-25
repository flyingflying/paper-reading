
# CLIP

[TOC]

## 简介

CLIP 是 contrastive language-image pre-training 的简写, 属于 图文检索 领域的工作。

除此之外, 作者还进行了大量关于 零样本学习 (zero-shot learning) 的实验, 将 ImageNet 图片分类任务转换成 图文检索 任务, 取得了非常好的成绩:

+ ImageNet 分类数据集的榜单可以参考 [链接](https://paperswithcode.com/sota/image-classification-on-imagenet), 目前的 SOTA 在 90% 左右
+ CLIP 发表于 2021 年, 那一年的 SOTA 是 90.88% 的准确率
+ 何凯明 2015 年发表的 [ResNet](https://arxiv.org/abs/1512.03385) 最高是 78.57% 的准确率
+ CLIP + 零样本学习可以达到 76.2% 的准确率 (论文中 Table 1.), 可以匹敌 ResNet 的效果
+ 在 CLIP 之前, 零样本学习的正确率大约在 10% 左右

上面足以说明 CLIP 为什么那么出名了。下面让我们来看看 CLIP 的架构, 以及如果做 零样本学习 的。相关资料:

+ [**CLIP** | Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
+ [Official Code | CLIP](https://github.com/openai/CLIP)
+ [HuggingFace Transformers Code | CLIP](https://huggingface.co/docs/transformers/model_doc/clip)

## 任务介绍

用一句话来概括 CLIP, 就是让 图片 和 文本 进行 **对比学习**, 其采用的是 in-batch negatives 作为损失函数。假设我们有一个 图文匹配 的数据集, 每一次迭代时:

+ 对于每一张图片, 与之匹配的文本属于 正样本, batch 中其它的文本样本是 负样本
+ 对于每一段文本, 与之匹配的图片属于 正样本, batch 中其它的图片样本是 负样本

计算两次 loss, 然后求平均即可。其和 [SimCLR](https://arxiv.org/abs/2002.05709) 一样, 虽然网络架构 "简单", 但是工程量是 "巨大的"。为了能更好的估算 loss 值, 其 `batch_size` 达到了 32768, 使用了 256  块 V100 GPU, 运算资源不是一般人所能达到的。对于多 GPU 大模型的训练, 可以参考博客: [How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/)。

或许是因为 `batch_size` 比较大, 作者选择的模型并不是特别大 (相对于 GPT3 而言)。其中, 文本编码器使用的是类似 BERT 的 transformer 架构, 只有 6,300 万参数。图片编码器用的是 [ResNet](https://arxiv.org/abs/1512.03385) 或者 [ViT](https://arxiv.org/abs/2010.11929), 最大的是 ViT/L-14, 大约是 3 亿参数 (和 BERT-Large 相当)。

虽然模型大小在可以接受的范围内, 但是数据集的量非常惊人, 有 4 亿个样本。作者一共训练了 32 个 epoch, 如果使用 ResNet 图片编码器, 最大的模型在 592 张 V100 上训练了 18 天, 如果使用 ViT, 最大的模型在 256 块 V100 上训练 12 天 (大的 CNN 比 Transformers 还耗资源)。

下面让我们简单介绍一下数据集的情况。

## 数据集

目前主流的 图文匹配 数据集一共有三个, 都达不到 OpenAI 的要求, 他们分别是:

[MS-COCO](https://cocodataset.org/#overview) 是微软开源的图像数据集, 其中包含 6 个任务: 目标检测 (Detection), 人体姿态估计 (DensePose), 人体关键点检测 (KeyPoint), 语义分割 (Stuff), 场景分割 (Panoptic) 和 图片描述生成 (Captions)。这个数据集从 2015 年一直更新到 2020 年, 每年还举办比赛, 因此非常出名。其中, **图片描述生成 (Captions)** 符合我们的要求。其是高质量数据集, 但是数据量太少了 (大约 10 万个样本)。

[Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) 是李飞飞团队开源的图像数据集, 目的是将图片的结构化信息和自然语言联系起来。包括的内容有: 目标检测的区域, 区域的文字描述, 区域的树形结构, 以及基于图片的问答 (QA) 数据。其中, **目标检测区域的文字描述** 符合我们的要求。和 MS-COCO 的问题一样, 是高质量数据集, 但是数据量太少了。

[YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) 是雅虎开源的图片数据集, 每一张图片中包含 metadata 信息, 可以作为 图文匹配 的数据集。其中有 1 亿张图片, 但是数据集质量很差: 很多配对的信息并不是图片内容的描述, 而是文件名, 照片的曝光信息等等。经过清洗后, 只有 1500 万个图文匹配的样本, 也达不到我们的要求。

于是, OpenAI 自己收集了一个有 4 亿样本的图文匹配数据集, 称为 **WIT** (WebImageText)。这个数据集和 WebText (训练 GPT2 的数据集) 数据集词语数量是差不多的。论文中貌似没有说这个数据集的收集过程, 也没有开源, 相关信息无从得知。

此时, 你获取会好奇, 从互联网中收集的数据集中是否会包含 ImageNet 数据集呢? 如果包含了, 那么就不算真正的 零样本学习 了。对此, 作者进行了 去重 操作, 具体的参考论文第五节: Data Overlap Analysis。

有了这么大的数据集后, 就有了以下的优势:

+ 不用担心 over-fitting 的问题
+ 不需要做过多的数据增强, 作者只用了 随即裁剪 策略
+ task-specific 层是线性层即可, 不需要是非线性层

但是也有问题, 那就是难以调参。作者说其只在 ResNet50 上调参了, 训练一个 epoch, 之后的大模型都用这一组参数。除此之外, 还将 logit 中的 temperature 参数改成了 可学习参数。

## 零样本学习

零样本学习的过程和 训练过程 差不多。用 文本编码器 将所有的 标签名称 编码成向量, 和 图片向量 进行点乘, 属于 cosine 相似度最大的那一个类别。整个过程和 图文检索 差不多。

考虑到 词语的多样性, 训练文本的特性, 作者还使用了 prompt engineering, 构建 prompt 模板: `A photo of a + 标签名称.` 这样可以得到性能的提升。

除此之外, 作者还使用了 prompt ensembling, 构建多个 prompt 模板, 然后用 文本编码器 编码成向量, 再取平均, 作为分类向量。最终在 ImageNet 上取得了 76.2% 的效果。

不仅如此, 作者还额外测试了 27 个数据集, 其中在 16 个数据集上, CLIP + zero-shot 的效果要优于 resnet50 + linear probe, 可以说是 零样本学习 里程碑式的模型了。

作者还尝试用 CLIP 来做 few-shot 图像分类任务, 做法是: 图像编码器 + linear probe 的方式, 但是 linear probe 线性层权重的初始值是通过 文本编码器 得到的 (方式和 zero-shot 一致)。但是这种方式的效果并不是理想。

CLIP 的局限性有:

+ 目前 ImageNet 的 SOTA 能达到 90%, CLIP 想要达到这个效果需要更大的模型更多的数据, 预计是现在模型计算量的 1000 倍
+ CLIP 在 细分类任务, 抽象任务 (比方说图像中物体个数) 上表现不好, 在异常检测方面表现也不好
+ 如果 测试数据集 和 训练数据集 相差非常大 (out-of-distribution), 效果也不是很好 (比方说在 MNIST 只有 88% 的准确率)
+ 做 零样本学习 时, 还是要预先定义好 类别, 作者认为更好的方式是 直接生成类别, 这样更加灵活
+ 在训练过程中, 每一次都用 ImageNet 的验证集来验证模型的好坏, 然后挑选模型, 这样就存在 数据偏见 了, 也不算真正意义上的 零样本学习
+ 训练的数据集从互联网中获取, 没有经过过滤, 可能会包含社会上大量的偏见
+ CLIP 在 few-shot 上的效果有时候还不如 zero-shot

## 总结

CLIP 可以说是开创性的工作, 但是, 我认为的问题有以下:

+ 并不是所有的 图片分类任务 都可以用语言描述清楚, 这些问题似乎并没有办法解决
+ 如果我的类别是: `dog`, `cat` 和 `others`, 那么 `others` 应该怎么描述可以让 CLIP 理解呢
+ 论文中主要做的是 图搜文任务, 文搜图任务 很少 (见 Table 13.)

CLIP 只公布了 推理 的代码, 没有公布 训练 的代码。我猜测是因为 训练代码 牵涉到过多 分布式计算, 多卡并行的代码, OpenAI 不想公开吧。

## References

+ [CLIP 论文逐段精读【论文精读】 | BV1SL4y1s7LQ](https://www.bilibili.com/video/BV1SL4y1s7LQ)
+ [Understanding the Vision Transformer and Counting Its Parameters](https://medium.com/analytics-vidhya/understanding-the-vision-transformer-and-counting-its-parameters-988a4ea2b8f3)
