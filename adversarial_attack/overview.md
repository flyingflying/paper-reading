
# 对抗样本攻击 (Adversarial Attack)

[TOC]

## 简介

目标: 对于图片分类问题来说, 我们加入人肉眼难以发现的噪声, 期待模型将其识别错误。

原始的图片被称为 benign image, 其能够被正确识别的概率非常高。加入噪声的图片被称为 attacked image, 其能够被正确识别的概率非常低。

攻击分为两种, non-targeted 和 targeted。对于 non-targeted 来说, 我们期待的是模型识别错误即可。但是对于 targeted 来说, 我们期待模型输出指定的错误类。具体怎么实现呢? 一种很直接的方式就是使用 **梯度**: 将模型的参数固定死, 输入的图片变成要更新的参数, 这样迭代几次就可以获得 attacked image 了。

需要注意的是, 这里的做法是针对图片而言的, 不是针对模型而言的。即为一张图片找一个 attacked image, 使得模型预测错误。并不是为一个模型找一个噪声, 只要这个噪声加入图片中, 就会识别错误。

我们设原始的 benign 图片为 $x$, 加入噪声的 attacked 图片为 $\bar{x}$, 图片真实的类别是 $y$, 模型预测的类别是 $\hat{y}$。对于分类问题来说, $CE$ 表示交叉熵损失函数。

在原始的 softmax 回归中, 我们优化的方向是最小化 $y$ 和 $\hat{y}$ 之间的交叉熵 $CE(y, \hat{y})$。对于 non-targeted 攻击来说, 我们期待模型预测错误, 那么最大化两者之间的交叉熵即可, 此时的损失值是 $L = -CE(y, \hat{y})$。

对于 targeted 攻击, 我们期待模型输出 $y^{target}$ 类, 也就是 $\hat{y}$ 和 $y^{target}$ 之间的交叉熵越小越好, 此时的损失值是 $L = -CE(y, \hat{y}) + CE(y^{target}, \hat{y})$。

但是这样还是存在一个问题, 那就是尽量不要让人感知到图片是被处理过的, 我们设两张图片之间的差距是 $\epsilon$, 我们期待 $d(x, \bar{x}) \le \epsilon$。$d$ 函数的选取一般是 [L-infinity](https://en.wikipedia.org/wiki/L-infinity) 函数。具体的做法如下:

在每一次更新后, 对于 $(c, h, w)$ 位置的 pixel 来说, 如果 $\bar{x}^{(c, h, w)}$ 的值大于 $x^{(c, h, w)} + \epsilon$, 那么就将其赋值为 $x^{(c, h, w)} + \epsilon$; 同理, 如果 $\bar{x}^{(c, h, w)}$ 的值小于 $x^{(c, h, w)} - \epsilon$, 那么就将其赋值为 $x^{(c, h, w)} - \epsilon$。

整体的方案就是这样, 剩下的就是小修小补了。

## FGSM

FGSM 全称是 Fast Gradient Sign Method, 出自 谷歌 2014 年 论文 [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)。整体的思路如下:

我们知道, 在 **梯度下降法** 中, 对于一个参数 $w$ 来说, 其梯度 $w_g$ 的含义是: 如果要让 loss 值变小, $w_g$ 的正负号表示参数应该往正方向还是负方向移动, $w_g$ 值的大小表示建议移动的距离 (实际移动的距离还要乘以 学习率)。

从这里可以看出, $w_g$ 的正负号是至关重要的。FGSM 的想法很直接, 那就是只要 $w_g$ 的正负号, 不需要值, 移动的步长直接是 $\epsilon$, 也就是直接移动到边界。用公式表示如下:

$$
\bar{x} = x - \epsilon \cdot \mathrm{sign} (\nabla_x L)
$$

正常梯度下降的公式应该是:

$$
\bar{x} = x - lr \cdot \nabla_x L
$$

两者对比一下, 就应该能理解了。FGSM 中只更新了一次图片, 并没有进行多次更新, 就可以让模型的准确率下降 40% 左右, 可见效果是非常好的。但是问题也是很明显的: 图片的每一个像素点都更新到了边界处, 更新幅度还是挺大的。

在这之后还有 [迭代版的 FGSM](https://arxiv.org/abs/1607.02533) 和 [动量版的 FGSM](https://arxiv.org/abs/1710.06081), 有兴趣的可以自行了解。

## 黑箱攻击

对抗攻击 为什么能够成功呢? 有一部分人认为, 模型不是主因, 主因在于数据。即使你不使用神经网络, 而用传统的机器学习模型, 都会有相似的问题, 更多内容可以参考: [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175)。

上面所说的问题都是在已知模型参数的情况下, 如果不知道模型参数呢? 也就是说, 我想攻击别的公司的模型, 这个模型的参数是不知道的, 只能通过 API 去访问得到输出, 应该怎么去攻击呢?

我们将不知道参数的模型称为 **黑箱模型**。利用之前所说的性质, 如果我们知道训练 **黑箱模型** 的数据, 然后用这些数据训练一个 **代理模型**。然后在 **代理模型** 上进行攻击, 得到 **对抗样本**。将这个 **对抗样本** 输入到 **黑箱模型** 中, 那么就有一定的概率能够成功。

**代理模型** 的网络架构和 **黑箱模型** 是一致的更容易成功。对于 non-targeted 任务, 黑箱攻击还是比较容易成功的, 但是对于 targeted 任务来说, 就很困难了。更多相关的内容, 可以参考: [DELVING INTO TRANSFERABLE ADVERSARIAL EXAMPLES AND BLACK-BOX ATTACKS](https://arxiv.org/abs/1611.02770)。

那如果连别人模型的训练数据是什么都不知道呢? 一种方式是收集大量的图片, 然后用 **黑箱模型** 的预测结果, 构成一个数据集, 用来训练 **代理模型**。如果是无法访问的模型呢? 那你只能碰运气了。

## 其它方向

我们希望 **对抗样本** 和 原始的样本 差距越小越好。最小是什么样子呢? 那就是只修改一个 pixel。这样的任务被称为 one pixel attack, 更多的内容可以参考: [One Pixel Attack for Fooling Deep Neural Networks](https://arxiv.org/abs/1710.08864)

上面所说的内容都是针对一个样本而言的。对于一个正常的样本, 我们希望能找到一个 **对抗样本**, 使得模型预测错误。那么, 我们能不能针对模型呢? 即为 一个模型 找一个全局的 噪声, 任意图片只要加上这个噪声就会预测错误。答案是可以的, 相关内容参考: [Universal adversarial perturbations](https://arxiv.org/abs/1610.08401)。

对于 targeted 攻击来说, 还有一种更高级的。即可以通过加入指定颜色方块的个数来控制 模型 识别的结果。具体可以参考: [ADVERSARIAL REPROGRAMMING OF NEURAL NETWORKS](https://arxiv.org/abs/1806.11146)。

## 物理世界的攻击

上面所说的都是 数位世界 的攻击, 也就是对一张计算机中的图片进行攻击。那么有没有可能在物理的世界进行攻击呢? 比方说, 对于人脸识别系统, 只要戴上一个眼镜, 无论从哪一个角度拍摄, 无论相机是否高清, 都会识别成为另一个人。这样有可能成功吗? 答案是肯定的, 具体参考论文: [Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition](https://users.cs.northwestern.edu/~srutib/papers/face-rec-ccs16.pdf)。

还有针对 自动驾驶系统 进行攻击的, 让识别系统将路边的 **道路交通标志** 识别错误, 具体参考论文: [Robust Physical-World Attacks on Deep Learning Visual Classification](https://arxiv.org/abs/1707.08945)。

## 对 "训练集" 发起攻击

这和 物理世界 的攻击是一样可怕的。即对 "训练集" 中的部分图片加入噪声。加入噪声的图片看起来是没有什么问题的, 其标注信息也是没有什么问题的。但是通过这个数据集训练出来的模型, 对于另一些图片就会产生错误。更多相关的内容可以参考: [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://arxiv.org/abs/1804.00792)。

## 防御

我们知道, 对于 爬虫技术, 现在有较为成熟的 反爬虫技术。那么对于 对抗攻击, 有没有 防御技术呢? 答案是有的。

最简单的方式是: 在识别图片之前, 对图片进行模糊化处理, 或者加入一些噪声。除此之外, 还可以对图片进行压缩, 或者用 AutoEncoder 进行重构。这些方式都可以 "弱化" 攻击信号, 或者破环攻击信号的内容。相关的内容可以参考:

+ [Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks](https://arxiv.org/abs/1704.01155)
+ [Shield: Fast, Practical Defense and Vaccination for Deep Learning using JPEG Compression](https://arxiv.org/abs/1802.06816)
+ [Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models](https://arxiv.org/abs/1805.06605)
+ [Mitigating Adversarial Effects Through Randomization](https://arxiv.org/abs/1711.01991)

上面的内容被称为 **被动防御** (passive defense), 即在推理阶段, 对图片进行一些处理。还有一种是 **主动防御** (proactive defense), 即在训练阶段, 我们就要考虑 防御 这件事情。或者说, 我们要训练一个不容易被 攻击 的模型。这种 训练 被称为 **对抗训练** (adversarial training)。

想法很容易理解。在模型训练完成后, 对于训练集中的每一张图片, 找到相对应的对抗样本, 标上正确的标签, 放入数据集中再一起进行训练。如此进行三四次, 黑箱攻击就不容易成功了。

这个也是 **数据增强** 的方式之一, 可以增加模型的泛化性。更多相关的内容可以参考: [Adversarial Training for Free!](https://arxiv.org/abs/1904.12843)。

当然, 无论是哪一种方式, 都有被攻击破的可能。对于 **被动防御**, 只要知道 防御方式, 就有可能被攻击破。对于 **主动防御**, 无法防御新一代的攻击方式。这和 爬虫 与 反爬虫, 网络攻击 与 防御 是一样的, 在不断的演化。

## 总结

我们知道, 神经网络的可解释性很差, 无法得知他们是根据什么进行判断或者识别的。但是从来没有想过, 反过来, 利用这种 未知性, 来进行 "攻击"。

这个领域的内容是非常 "综合" 的, 本文仅仅叙述了 FGSM, 列举了 李宏毅 教授视频中的相关论文。未来有机会再慢慢补充。

## 额外: NLP 中的对抗样本

我们知道, 在 NLP 中, token 是离散的, 不是连续的。那么寻找其 对抗样本 的过程和 物理世界的攻击是相似的。

整体过程可以分成四步: 确定攻击目标; 寻找 替换/增加/删除 的候选词; 制定 句子合理性 的判定方式; 使用高级搜索方式寻找对抗样本。

相关内容可以参考: [TextAttack](https://github.com/QData/TextAttack)。

## Reference

+ [Adversarial Attack](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/attack_v3.pdf)
