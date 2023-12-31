
# CNN 系列 (一) 详解 卷积层 和 池化层

[TOC]

## 一、简介

### 1.1 发展引言

神经网络在 1980-1990 年代就已经提出了。1998 年, 杨立坤 (Yann LeCun) 大佬发表了 [LeNet](https://www.researchgate.net/publication/2985446), 就是现在 CNN 网络的雏形。

在 2000-2010 年这段时期, CV 问题的主要解决办法是 SIFT / SURF 特征工程 + 核函数 + SVM, 他们的优势在于可解释性非常强。

在 2010 之后, CV 问题主要的解决方案变回了神经网络, 标志性的工作有两个: 2008 年的 [ImageNet](https://image-net.org/) 和 2012 年的 [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), 确定了 大数据 + 神经网络解决方案的可行性。在这之后, [VGG](https://arxiv.org/abs/1409.1556), [NiN](https://arxiv.org/abs/1312.4400), [InceptionNet](https://arxiv.org/abs/1409.4842), [ResNet](https://arxiv.org/abs/1512.03385), [EfficientNet](https://arxiv.org/abs/1905.11946) 等等网络层出不穷, 建立了 CNN 网络在 CV 领域的主导地位, 直到 [ViT](https://arxiv.org/abs/2010.11929) 的出现。

目前, 在 图片分类 任务中, [ViT](https://zhuanlan.zhihu.com/p/660058466) 的效果仅仅是小幅度优于 CNN, 并没有达到 "大幅度领先" 的程度。而在 图片生成 任务中, [扩散模型](https://arxiv.org/abs/2006.11239) + [U-Net](https://arxiv.org/abs/1505.04597) 的方案占主导地位, 而 U-Net 就是基于 CNN 架构的网络。本文就来总结一下 CNN (convolutional neural network) 相关的知识。

### 1.2 图片数据简介

对于 图片 来说, 我们有两种描述方式。一种描述方式是: 一张图片是一个由 **像素点** (pixel) 构成的矩阵。一个像素点可以由一个数字构成 (灰度图), 也可以由三个数字构成 (RGB 图), 我们将 像素点 中的一个数字称为 **像素值** (pixel value)。那么, 我们可以用 二维数组 (矩阵) 来表示灰度图, 用 三维数组 (张量) 来表示 RGB 图。

如果用机器学习的方式来描述: 一张图片就是一个样本, 一个样本由大量的 **像素点** 构成, 每一个 **像素点** 由若干 **像素值**, 也就是 **特征** 构成。整体的层级关系是: 图片 (样本) → 像素点矩阵 → 像素值 (特征)。[Vision Transformer](https://zhuanlan.zhihu.com/p/660058466) 中采用的就是这种描述方式。

除此之外, 还有另一种描述方式: 一张图片由一个或者多个颜色 **通道** (channel) 构成, 每一个 **通道** 是一个由 **像素值** 构成的矩阵。灰度图只有一个颜色通道, 而 RGB 图有三个颜色通道。整体的层级关系是: 图片 → 通道 → 像素值矩阵。

在研究图像算法时, 我们一般采用第二种描述方式, 并且默认图片是单通道的, 即像素值矩阵。而对于多通道的情况, 一般有三种处理方法: (1) RGB 转灰度图进行分析; (2) 每一个颜色通道单独处理; (3) 多通道融合。在 CNN 中, 池化层 (pooling) 采用 (2) 处理方式, 卷积层采用 (3) 处理方式, 具体的内容本文后续会介绍。

### 1.3 网络架构简介

无论是 CV 还是 NLP, 处理的数据都有一个共通点: 一个样本中包含多个元素。图片中包含若干像素, 文本中包含若干 token。和 NLP 不同的是, 我们一般不会将一个像素作为基本的运算单元, 而是将一组像素作为基本的运算单元。

CNN 网络中最重要的就是 卷积运算 和 池化运算, 而这两个运算的核心思想都是 **滑动窗口** (sliding window), 简称 **滑窗**。简单来说, 就是将一个窗口内的所有元素进行某种运算: 卷积 对应 点乘, 最大池化 对应 取最大值, 平均池化 对应 取平均值。这些运算并不是很难理解, 难的是 滑窗算法, 比其它领域多出了很多花样。下面, 让我们先来看看 滑窗算法。

## 二、滑动窗口算法

在计算机的很多领域中, 都有滑窗算法的应用, 比方说 [数据传输](https://en.wikipedia.org/wiki/Sliding_window_protocol), [数据压缩](https://en.wikipedia.org/wiki/LZ77_and_LZ78), [词性标注](https://en.wikipedia.org/wiki/Sliding_window_based_part-of-speech_tagging), 时间序列 等等, 在 leetcode 上甚至还有 [滑窗专题](https://leetcode.com/tag/sliding-window/)。然而, 不同任务的处理细节是不一样的, 因此很难形成一个完整的体系。CNN 中的滑窗后文统称 **卷积滑窗**。

网上的大部分 CNN 教程都是直接介绍如何在 图片单通道, 也就是 矩阵 (二维数组) 上进行滑窗的, 后文统称 **二维滑窗**。实际上, 卷积可以在任意维度的数组上面进行运算。PyTorch 官方提供了 [Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html), [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) 和 [Conv3d](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html) 三种情况的卷积。

个人认为, 在 矩阵 上进行滑窗过于抽象, 用文字很难描述整体过程, 用图像也不是很直观, 需要读者大量 "脑补" 才行, 最好的方式是用视频或者动图。那么, 不如先介绍如何在 向量 (一维数组, 列表) 上进行滑窗, 方便大家理解和公式推导。后文统称 **一维滑窗**。

### 2.1 基础滑窗

一维滑窗大致的思路是: 指针初始化在向量首元素的位置, 然后不断向右移动, 并框住右边一定范围内的元素, 记作 **子向量**, 直到不能框为止。

假设有一个向量 `[1.1, 2.2, 3.3, 4.4, 5.5]`, 窗口大小为 3, 那么:

+ 指针初始位置在 `1.1`, 取右边三个元素, 那么 子向量 是 `[1.1, 2.2, 3.3]`
+ 接下来, 指针移动到 `2.2` 的位置, 取右边三个元素, 那么 子向量 是 `[2.2, 3.3, 4.4]`
+ 然后, 指针移动到 `3.3` 的位置, 取右边三个元素, 那么 子向量 是 `[3.3, 4.4, 5.5]`
+ 最后, 指针移动到 `4.4` 的位置, 发现没有办法再取右边三个元素了, 至此结束了

最终, 我们得到了 3 个子向量。观察上述过程, 我们很容易发现规律:

如果向量中有 $w$ 个元素 (或者说向量的维度是 $w$), 窗口大小是 $k$ (或者说子向量的维度是 $k$), 子向量个数是 $o$, 那么:

$$
o = w - k + 1 \tag{2.1}
$$

在 [word2vec](https://zhuanlan.zhihu.com/p/653414844) 算法中, skip-gram 和 CBOW 模型也是基于滑窗的, 主要区别在于, 取元素的方式不同。卷积滑窗取指针右边的 $k$ 个元素 (包含指针本身); 而 word2vec 滑窗取 中心词 (指针) 左右各 $k$ 个词语 (不包含 中心词 本身) 作为 背景词。

除此之外, 还有一点不同:

在 word2vec 滑窗中, 我们并不要求每一个 窗口 (子向量) 中 背景词 的个数相同。如果 中心词 的 左边 (或者右边) 没有 $k$ 个词语, 那么我们将 左边 (或者右边) 所有的词语放入一个 窗口 中即可。也就是说, 如果一句话中有 $w$ 个词语, 那么最终有 $w$ 个窗口, 两者是相同的。

但是在卷积滑窗中, 我们要求每一个 子向量 中的元素个数是固定值 $k$, 如果指针的右边取不到 $k$ 个元素, 直接舍弃掉。如果不这样做, 后续运算可能会出问题。也正是因为此, 子向量 的个数是小于向量中元素个数的, 两者的关系见公式 $(2.1)$。

### 2.2 填充 padding

在设计 CNN 网络时, 有时我们会期待 子向量 的个数是 和 向量的维度是一致的。至于为什么有这样的需求, 后面再介绍。那么应该如何解决这一问题呢?

答案是 **填充** (padding), 其作用是在 向量 的左右添加不影响后续运算的元素值。我们将不影响后续运算的元素值称为 **填充值**。

在 卷积滑窗 中, 我们一般默认向量首尾添加的 **填充值** 数量是一致的, 这个数量记作 $p$。那么, 此时 向量 中的元素个数从 $w$ 变成了 $w + 2p$, 那么公式 $(2.1)$ 变成:

$$
o = w + 2p - k + 1 \tag{2.2}
$$

分析公式 $(2.2)$, 填充主要有三种模式:

(1) valid padding: 不进行任何 填充, 此时 $p = 0$。

(2) same padding: 使得 子向量 个数和 向量维度 相同的填充方式。其满足 $w + 2p - k + 1 = w$, 也就是 $p = (k - 1) / 2$ 。

(3) full padding: 使得 子向量 个数最多的填充方式。也就是说: (a) 第一个 子向量 的最后一个元素值是 向量 的第一个元素值, 其它都是填充值; (b) 最后一个 子向量 的第一个元素值是 向量 的最后一个元素值, 其它的都是填充值。显然, 此时 $p = k - 1$。

观察 same padding 和 full padding 中 $p$ 的计算公式, 发现两者正好相差一倍, 因此 same padding 也被称为 half padding。

### 2.3 步数 stride

上面的过程中, 指针每一次只移动 1 个单位。更一般地, 指针一次可以移动多个单位。我们将指针一次移动的单位数称为 **步数** (stride)。还是用上面的例子, 假设有一个向量 `[1.1, 2.2, 3.3, 4.4, 5.5]`, 窗口大小为 3, 步数为 2, 那么:

+ 指针初始位置在 `1.1`, 取右边三个元素, 那么 子向量 是 `[1.1, 2.2, 3.3]`
+ 接下来, 指针移动两步到 `3.3` 的位置, 取右边三个元素, 那么 子向量 是 `[3.3, 4.4, 5.5]`
+ 最后, 指针移动两步到 `5.5` 的位置, 发现没有办法再取右边三个元素了, 至此结束了

最终, 我们得到了 2 个子向量。那么问题来了: 如果向量中有 $w$ 个元素, 窗口大小是 $k$, 步数为 $s$, 那么最终有多少个子向量呢?

回到上面的例子中, 如果向量是 `[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]`, 有 6 个元素, 窗口大小还是 3, 步数还是 2, 那么最终依然只有两个子向量, 元素 `6.6` 不会包含在任何一个子向量中。同理, 如果向量有 7 个元素, 那么最终就会有三个子向量。

观察上述过程, 我们可以发现, 指针的位置索引是: $1$, $s + 1$, $2s + 1$, $3s + 1$, 以此类推。设最终有 $o$ 个子向量, 根据上述规律, 指针最终的位置索引可以表示为 $(o - 1) \cdot s + 1$。理想状况下, 指针的最终位置索引应该是 $w - k + 1$, 那么可以得到:

$$
\begin{align*}
    (o - 1) \cdot s + 1 &= w - k + 1 \\
    o &= \frac{w - k}{s} + 1
\end{align*}
$$

当然, 实际不一定是理想状况。因此, 最终的公式为:

$$
o = \left\lfloor \frac{w - k}{s} \right\rfloor + 1 \tag{2.3}
$$

滑窗算法的一个重点就是 $o$ 值的计算公式。其难点就在于步数 $s$ 对于 $o$ 的影响, 解决办法就是思考 指针的位置, 上面给出了一种推理思路。从感性上理解, 在去除掉窗口大小 $k$ 后, 指针所在的位置就是带 step 的 for 循环, 最后再加上去除掉的那一个子向量即可。

观察公式 $(2.3)$, 可以发现, 向量的维度 $w$ 大约是 子向量的个数 $o$ 的 $s$ 倍。如果考虑 填充 的情况, 公式 $(2.3)$ 变成:

$$
o = \left\lfloor \frac{w + 2p - k}{s} \right\rfloor + 1 \tag{2.4}
$$

需要注意的是, 上面所说的 same padding 和 full padding 都是在不考虑 stride 参数的情况下, 也就是 $s = 1$。在设计神经网络时, stride 参数和 padding 参数一般不会同时设置。

额外说明一点, 当 $s$ 不为 1 时, 可能会出现 向量中元素 不再任何一个 子向量 中的问题。这个问题可以通过 填充 来解决, 但是一般不会用 填充 的方式来解决。图片可以 resize, 可以 crop, 只要精心设置网络架构, 就不会存在这个问题。

### 2.4 补充: 空洞 dilation

上面所说的就是 卷积滑窗 的基本内容了。这里补充 2016 年有人提出的 **空洞卷积** ([Dilated Convolution](https://arxiv.org/abs/1511.07122))。在经典的神经网络中是用不到相关内容的, 本文后续内容不涉及相关知识, 可以跳过。

dilation 的本意是 **膨胀**, 按照字面意思, Dilated Convolution 应该翻译成 膨胀卷积。但是这里我们一般采用 意译 的方式, 将其翻译成 空洞卷积。其含义是: 在 子向量 元素个数不变的情况下, 使得 子向量 覆盖的区域更广。但是, 实际上, 和 2.3 节的 "步数" 含义是相似的。

2.3 节说的是 指针 移动的步数, 实际上取元素的过程也可以有 "步数", 这个 "步数" 我们记作 $d$。其思路是: 我们取元素时可以跳着取, 每隔 $d - 1$ 个元素取一个, 或者说两个元素之间包含 $d - 1$ 个元素。

举例来说, 现在向量维度是 11: `[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.01, 11.11]`, 如果窗口大小 $k = 3$, 指针移动步数 $s = 2$, 取元素 "步数" $d = 3$, 那么:

+ 初始, 指针在 `1.1` 位置, 我们每隔 2 个取一个, 一共取 3 个元素, 那么子向量是 `[1.1, 4.4, 7.7]`
+ 指针移动两步, 在 `3.3` 位置, 我们每隔 2 个取一个, 一共取 3 个元素, 那么子向量是 `[3.3, 6.6, 9.9]`
+ 指针移动两步, 在 `5.5` 位置, 我们每隔 2 个取一个, 一共取 3 个元素, 那么子向量是 `[5.5, 8.8, 11.11]`
+ 指针移动两步, 在 `7.7` 位置, 我们每隔 2 个取一个, 没有办法取 3 个, 滑窗终止

虽然 子向量 中的元素个数没有变, 依旧是 $k$, 但是我们可以认为窗口大小发生了变化, 变成了 $d \cdot (k - 1) + 1$, 代入到公式 $(2.4)$, 可以得到:

$$
o = \left\lfloor \frac{w + 2p - [d \cdot (k - 1) + 1]}{s} \right\rfloor + 1 \tag{2.5}
$$

公式 $(2.5)$ 就是 [Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) 和 [MaxPool1d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html) 的最终公式。

### 2.5 二维滑窗

上面是在 向量 上滑窗的情况。那么, 如何在 矩阵 上进行滑窗呢? 其动图可以参 [conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic) 项目中 "Convolution animations" 部分。简单来说, 指针变成在平面内移动, 窗口变成一个二维的, 子矩阵 也是按照二维的方式排列的。

设矩阵的维度是 $(h_{in}, w_{in})$, 窗口维度是 $(h_{kernel}, w_{kernel})$, 矩阵的左右 填充 个数是 $w_{padding}$, 指针向右移动步数为 $w_{stride}$, 矩阵的上下 填充 个数是 $h_{padding}$, 指针向下移动步数为 $h_{stride}$, 最终子矩阵可以排列成 $(h_{out}, w_{out})$ 形式的二维数组, 则:

$$
\begin{align*}
    w_{out} &= \left\lfloor \frac{w_{in} + 2 \cdot w_{padding} - w_{kernel}}{w_{stride}} \right\rfloor + 1 \\
    h_{out} &= \left\lfloor \frac{h_{in} + 2 \cdot h_{padding} - h_{kernel}}{h_{stride}} \right\rfloor + 1
\end{align*}
\tag{2.6}
$$

## 三、卷积层

### 3.1 图像 与 全连接层

在介绍 卷积层 之前, 我们先看看 图像如何应用于 全连接层:

+ 假设图像是一个单通道像素值矩阵, 维度是 $(h_{in}, w_{in})$
+ 设置 $f_{out}$ 个线性函数, 每一个线性函数有 $h_{in} \cdot w_{in}$ 个自变量
+ 将矩阵 平铺 成维度是 $h_{in} \cdot w_{in}$ 的向量
+ 将平铺后的向量代入 $f_{out}$ 个线性函数中, 得到新的特征向量

在不考虑 bias 的情况下:

+ 全连接层一共有 $h_{in} \cdot w_{in} \cdot f_{out}$ 个参数
+ 需要进行 $h_{in} \cdot w_{in} \cdot f_{out}$ 次乘法运算
+ 需要进行 $(h_{in} \cdot w_{in} - 1) \cdot f_{out}$ 次加法运算
+ 一共需要 $(2 \cdot h_{in} \cdot w_{in} - 1) \cdot f_{out}$ 次浮点运算 (加法 + 乘法)

全连接层的问题是: 对位置非常敏感。假设一张图片向左平移一个单位, 那么得到的特征向量很可能是完全不同的, 而我们希望是相似的, 甚至是相同的。

更具体地说, 如果相同的 像素块 (子矩阵) 出现在图片不同的位置, 我们希望其计算结果是相同的, 相似也可以。这种性质被称为 **平移不变性** (translation equivariance)。

### 3.2 互相关运算

CNN 中的 卷积层 和 信号处理领域的 **互相关运算** ([cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation)) 的含义是相似的, 其还有一个更直观的名字: **滑窗点乘** (sliding dot product)。

在 CV 领域, 我们将 二维滑窗 得到的 子矩阵 称为 **感受野** (Receptive Field)。使用 2.5 节的符号系统, 那么一共有 $h_{out} \cdot w_{out}$ 个感受野, 每一个感受野是 $(h_{kernel}, w_{kernel})$ 维度的矩阵。

互相关运算 的输入是一个 $(h_{in}, w_{in})$ 的单通道图片矩阵, 参数是 $(h_{kernel}, w_{kernel})$ 的 kernel 矩阵。我们对图片矩阵进行 **二维滑窗**, 将得到的 感受野矩阵 和 kernel 矩阵平铺成向量, 进行点乘。由于点乘的结果是一个数字, 那么可以排列成 $(h_{out}, w_{out})$ 的矩阵, 我们称其为 **特征图** (feature map)。最终输出 特征图矩阵。

整个过程可以理解为将每一个 感受野 "打分", "打分" 方式就是 和 kernel 进行点乘, 以寻找特定样式的 感受野。建议配合 [conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic) 项目中的动图再看一遍, 理解整个运算过程。

这里也解释了 2.1 节中所说的, 所有的 感受野 (子向量) 的维度是相同的, 否则没有办法和 kernel 进行点乘运算。我们一般用 $\star$ 表示 互相关运算, 那么整个过程可以表示成:

$$
feature\_map = image \star kernel \tag{3.1}
$$

互相关运算 的输入和输出都是矩阵。在不使用 填充 的情况下, 输出矩阵的维度是小于输入矩阵的维度, 我们一般将其称为 **下采样** (downsampling)。相对应地, 如果输出矩阵的维度大于输入矩阵的维度, 那就是 **上采样** (upsampling)。也就是说, 在使用 填充 的情况下, 互相关运算也可以变成 上采样 (比方说 full padding)。

那么 2.2 节所说的 **填充值** 应该是什么呢? 我们知道, 向量点乘的运算方式是: 对应位置相乘再求和。而被填充的位置仅仅起到对齐的作用, 不影响计算结果, 那么 **填充值** 自然是 $0$。

在介绍 CNN 时, 很多人都喜欢从 3.1 节所说的 全连接层 推导出本节所说的 互相关运算, 说其具有以下优点: (1) 一个神经元不再考虑整张图片, 只考虑一个感受野, 减少了运算量; (2) 参数共享 (parameter sharing), 减少了参数量。

网上有很多相关的内容, 这里就不介绍了。个人认为, 这种介绍方式具有一定的误导性: 这仅仅是从 全连接层 推导出 互相关运算, 也就是说 kernel 只有 **一个**!!! 卷积层是由多个 互相关运算 构成的, 其有多个 kernel!!! (相关内容见下一节) 在我初学时, 这里坑了我很久!!!

### 3.3 多通道卷积

上面所说的内容都是在 单通道图片 上, 那么 多通道图片 应该怎么办呢? 答案是 多通道融合! 其含义是: 将多个通道在一起计算, 最终得到单通道的输出。

沿用 2.5 节的符号系统, 我们设图片有 $c_{in}$ 个通道, 那么此时的图片就是一个 三维数组, 维度是 $(c_{in}, h_{in}, w_{in})$。

我们依旧进行 二维滑窗, 得到的感受野维度是 $(c_{in}, h_{kernel}, w_{kernel})$。你可以理解为, 在图片的后两个维度进行滑窗, 第一个维度不进行滑窗, 采取 "我全都要" 的方式。如果进行 三维滑窗, 那么感受野的维度是 $(c_{kernel}, h_{kernel}, w_{kernel})$, 注意两者之间的区别。

kernel 的维度始终和 感受野 的维度是一致的, 即 $(c_{in}, h_{kernel}, w_{kernel})$。我们将两者平铺成向量进行点乘。那么最终得到的是 单通道 的 特征图矩阵!

我们希望输出是 多通道 的, 应该怎么办呢? 答案是设置多个 kernel, 进行多次 互相关运算。我们用 $c_{out}$ 表示 互相关运算 的次数, 那么可以得到 $c_{out}$ 个 特征图矩阵。我们将一个 特征图矩阵 作为一个输出的通道, 那么最终卷积层的输出也是一个 三维数组, 维度是 $(c_{out}, h_{out}, w_{out})$。

这样, 卷积层的 输入 和 输出 都是 三维数组 (张量) 了, 那么我们就可以堆叠 卷积层 了: 第一个卷积层的输入是 图片张量, 输出是 特征图张量; 之后每一个卷积层的输入是 上一层输出的特征图张量, 然后输出新的 特征图张量。整个神经网络就是这么构建起来的。

### 3.4 卷积层是特殊的全连接层

3.2 节和 3.3 节主要从 信号处理 的视角介绍卷积层。下面我们从 机器学习 的视角来介绍卷积层: 卷积层的本质 是以 感受野 为单位的 全连接层。kernel 对应一个线性函数的参数, 输出的通道数 对应 线性函数的数量! 沿用 2.5 节的符号系统, 我们可以得到:

+ 一共有 $h_{out} \cdot w_{out}$ 个感受野
+ 每一个感受野是 $(c_{in}, h_{kernel}, w_{kernel})$ 维度的张量
+ 也就是说, 每一个线性函数有 $c_{in} \cdot h_{kernel} \cdot w_{kernel}$ 个自变量
+ 一共有 $c_{out}$ 个线性函数

那么, 在不考虑 bias 的情况下:

+ 一共有 $c_{in} \cdot h_{kernel} \cdot w_{kernel} \cdot c_{out}$ 个参数
+ 一个感受野需要进行 $(2 \cdot c_{in} \cdot h_{kernel} \cdot w_{kernel} - 1) \cdot c_{out}$ 次浮点运算
+ 一个卷积层需要进行 $h_{out} \cdot w_{out} \cdot (2 \cdot c_{in} \cdot h_{kernel} \cdot w_{kernel} - 1) \cdot c_{out}$ 次浮点运算

由此, 我们可以得到结论, 卷积层是特殊的全连接层。在 神经网络 中, 我们采用 全连接层 + 激活层 的方式来拟合任意的函数。因此, 在 CNN 网络中, 每一个 卷积层 后面都要有一个 激活层! 我们还可以得到结论: 卷积层的运算量和 kernel 的大小成正比!

3.1 节所说的是对整张图片进行 全连接操作, 而 卷积层 是对一个感受野进行 全连接操作。也就是说, 从对整张图片 位置敏感 变成了对感受野内部 位置敏感。而 感受野之间 位置不再敏感: 如果两个 感受野 完全相同, 无论其在图片的什么位置, 其计算结果是相同的。换言之, 感受野 具有 平移不变性。

我们常说 卷积层 的 归纳偏置 (inductive biases) 是: 平移不变性 (translation equivariance) 和 局部性 (locality)。其中, 局部性 指的就是 感受野, 全局性 指的是 整张图片。那么, 其含义就是 局部有平移不变性。

当然, 卷积层的问题也很明显: 那就是不具备 **旋转不变性** (rotation equivariance) 和 **缩放不变性** (scaling equivariance): 如果 感受野 被旋转了一定的角度, 或者将 感受野 进行缩放, 计算结果相差会非常大。目前的解决办法是 **数据增强** (data augmentation), 这就是另外一个话题了。

个人认为, 用这个视角介绍 卷积层 是最好的, 方便理解 bias 参数。我们在实现 卷积层 时, 也是用这个视角实现的 (im2col)!

### 3.5 特殊的卷积层

如果 $h_{kernel} = w_{kernel} = 1$, 那么就是以 像素点 为单位的全连接层, 因此也被称为 [pointwise convolution](https://paperswithcode.com/method/pointwise-convolution), 或者 [1x1 Convolution](https://paperswithcode.com/method/1x1-convolution)。

如果 $h_{kernel} = h_{in}$ 且 $w_{kernel} = w_{in}$, 那么就是以 图片 为单位的全连接层, 和 3.1 节所说的全连接层是一致的。

如果让 $h_{stride} = h_{kernel}$ 且 $w_{stride} = w_{kernel}$, 那么就相当于将 图片 划分成一个一个 patch, 卷积层就是以 patch 为单位的全连接层, Vision Transformer 的 嵌入层 就是这么做的。

下面, 额外补充 3.3 节的一个内容: 分组卷积 (Grouped Convolution), 其作用是将一个卷积拆成多组卷积, 以节约计算资源。

我们将 图片张量 按照通道数等分成 $g$ 组, 那么每一组 图片张量 的维度是 $(c_{in} / g, h_{in}, w_{in})$, 然后对每一组图片张量进行 $c_{out} / g$ 次 互相关运算, 得到 $(c_{out} / g, h_{out}, w_{out})$ 维度的 特征图张量。最后将每一组输出的 特征图张量 拼接在一起, 得到 $(c_{out}, h_{out}, w_{out})$ 维度的 特征图张量。

参考 3.4 节的计算方式, 可以发现, 参数量减少了 $g$ 倍, 浮点计算量也减少了 $g$ 倍。需要注意, $c_{in}$ 和 $c_{out}$ 都要能被 $g$ 整除。

额外提醒一下, 分组卷积不能理解为 $c_{kernel} = c_{stride} = c_{in} / g$ 的三维卷积! 因为在分组卷积中, 同一位置不同组的感受野进行的是不同的 互相关运算!

当 $g = c_{in}$ 时, 也就是一个输入通道对应一个或者多个输出通道, 我们称其为 [Depthwise Convolution](https://paperswithcode.com/method/depthwise-convolution)。其优势在于省计算资源!

## 四、池化层

### 4.1 池化运算

池化 (pooling) 的含义是用 一个数字 代表 一组数据, 那么首先想到的应该是 统计学中的集中趋势 (平均数, 中位数, 众数)。在 CV 领域, 我们常用的是 平均数 和 最大值, 分别对应 **平均池化** ([AvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)) 和 (**最大池化** [MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html))。

我们回到 单通道图片 矩阵上, 最大池化 就是选择感受野矩阵中的 最大值, 平均池化就是选择感受野矩阵中的 平均值。和 互相关操作 相同, 都是将一个感受野 "压缩" 成一个数字, 只是采用的运算方式不同。

此时, 你或许会有疑问: 最大值 不是描述 离散程度 的吗? 在 CV 领域, 小范围的感受野一般使用 最大池化, 其更能表示局部的纹理信息; 而大范围的感受野一般使用 平均池化, 其更能表示全局信息。

和 卷积层 相同, 池化层 有完整的 二维滑窗 参数, 包括 kernel, stride 和 padding。下面讨论一下 **填充值** 的问题:

对于 最大池化 来说, 不影响 $\max$ 操作的 **填充值** 自然是 $-\infin$, 这没有什么疑问。

对于 平均池化 来说, 问题就比较大了, 感受野 内元素的个数会影响计算结果! PyTorch 中给出的解决方案是: **填充值** 是 `0`, 同时提供 `count_include_pad` 参数:

+ 如果值为 `True`, 在统计 感受野 中元素个数时包含 填充值 (默认)
+ 如果值为 `False`, 在统计 感受野 中元素个数时不包含 填充值

仔细想想, 和 卷积滑窗 不同, 在 池化滑窗 中, 我们没有必要保证 感受野 的维度相同, 无论 感受野 的维度是什么样子的, 我们都可以取 最大值 和 平均数。在 PyTorch 的 [MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) 和 [AvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html) 中, 有 `ceil_mode` 参数, 如果 感受野 的维度不达标, 我们不舍弃, 依旧进行运算。(注意, 不能和 `padding` 参数一起用, 否则会失效)

和 卷积层 不同, 池化层不采用 多通道 融合的方式, 而采用 单通道 分开处理的方式。每一个通道的 特征图矩阵 经过 池化操作后得到新的 特征图矩阵, 然后将所有新的 特征图矩阵 拼接到一起。也就是说, 输入的通道数和输出的通道数是相同!

### 4.2 经典设计

池化层是没有参数的, 存在一个或者多个 卷积层 + 激活层 的后面。我们在计算神经网络层数时, 一般只统计有参数的层, 因此 池化层 一般不在统计范围内。

一般情况下, 池化层 作为 特征图张量 主要的下采样方式; 而卷积层一般采用 same padding 的方式, 不进行下采样。

对于池化层来说, 一般的设置是 $h_{stride} = h_{kernel}$ 以及 $w_{stride} = w_{kernel}$, 相当于划分出不同的 patch, 对每一个 patch 进行池化操作。我们有时会将 图片 的 "高" 和 "宽" 两个维度称为 **空间维度** (spatial dimension)。那么, 此时 池化层 就是 通道 (特征) 维度不变, 成倍的减小 空间维度。

在设计神经网络时, 一般都采用 **最大池化** 的方式。只有在 全连接层 之前, 对 整个特征图使用 **平均池化** (也就是 $h_{kernel} = h_{in}$, $w_{kernel} = w_{in}$)。由于是对整张特征图进行池化, 这种也被称为 **全局平均池化** (global average pooling)。

经过 全局池化 后, 空间维度 就没有了, 只有 通道 (特征) 维度了, 此时再输入全连接层完成分类任务, 整个网络就这么搭建起来了。

池化层 对于 CNN 网络的作用并不是很明确, 在一些 CNN 网络中, 甚至都不用池化层。目前, 对于池化层作用的介绍是: 缓解 卷积层 中 感受野 内部的 位置敏感性。个人认为, 这个解释很难令人信服。

## 五、总结和引用

本文详细介绍了 CNN 网络中的 卷积层 和 池化层, 之后打算介绍 卷积层的可视化, 经典的 CNN 网络, 目标检测的 anchor 等等内容。至于能不能成为系列, 就看缘分吧。

### 5.1 引用

papers:

+ [LeNet | Gradient-Based Learning Applied to Document Recognition](https://www.researchgate.net/publication/2985446)
+ [AlexNet | ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
+ [VGG | Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
+ [InceptionNet | Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
+ [ResNet | Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
+ [EfficientNet | EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
+ [Dilated Convolution | Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
+ [TextCNN | Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

others:

+ [李宏毅 CNN 课件](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/cnn_v4.pdf)
+ [李沐课件](https://courses.d2l.ai/zh-v2/)