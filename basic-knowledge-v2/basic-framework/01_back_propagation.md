
# 深度学习 与 梯度

[TOC]

## 一、简介

在深度学习中, **梯度** (gradient) 是一个核心的内容, 其与 随机梯度下降 (stochastic gradient descent; SGD), 反向传播 (backpropagation), 自动微分 (automatic differentiation), 计算图 (computational graph), 优化器 (optimization), 参数初始化 (parameter initialization), loss 函数, 对抗攻击 (adversarial attack) 等等内容都有密切的关系。本文尝试从 **梯度** 的概念入手, 梳理其中一部分内容, 加强对深度学习中一些概念的理解。

正常介绍 **梯度** 应该从 二元函数 入手, 然后扩展到 多元函数。既然能扩展到 多元函数, 那么也能应用于 一元函数。本文另辟蹊径, 从 一元函数 入手, 介绍梯度, 然后扩展到 二元函数, 方便大家理解。下面就让我们开始吧。

## 二、一元函数与梯度

我们知道, **导数** ([derivative](https://en.wikipedia.org/wiki/Derivative)) 的定义如下:

$$
f^{\prime} (x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h} \tag{2.1}
$$

其中, $f^{\prime}(x_0)$ 表示函数 $f(x)$ 在 $x = x_0$ 处切线的 **斜率**。更进一步, **斜率** 就是 **变化率**, **变化率** 就是 **单位变化量**。

也就是说, 如果用 $x = x_0$ 处的切线来近似表示 $f(x)$, 当自变量 $x$ **增加** 单位量时, 函数值 $f(x)$ 的变化量是 $f^{\prime}(x_0)$。换言之, 当自变量 $x$ **增大** 时, 如果 $f^{\prime}(x_0) > 0$, 函数值 $f(x)$ 倾向于变大; 如果 $f^{\prime}(x_0) < 0$, 函数值 $f(x)$ 倾向于变小。一阶 导函数 描述的就是, 当自变量 $x$ **增大** 时, 函数值 $f(x)$ 的变化趋势。

对于 **增加** 和 **增大** 这样的词, 我们可以用更加确切的表述: **沿着数轴正方向移动**。那么, 上面的一段话可以改成:

如果用 $x = x_0$ 处的切线来近似表示 $f(x)$, 当自变量 $x$ 沿着 **数轴正方向** 移动单位量时, 函数值 $f(x)$ 的变化量是 $f^{\prime}(x_0)$。换言之, 当自变量 $x$ 沿着 **数轴正方向** 移动时, 如果 $f^{\prime}(x_0) > 0$, 函数值 $f(x)$ 倾向于变大; 如果 $f^{\prime}(x_0) < 0$, 函数值 $f(x)$ 倾向于变小。总结一下, 一阶 导函数 描述的就是, 当自变量 $x$ 沿着 **数轴正方向** 移动时, 函数值 $f(x)$ 的变化趋势。

可以这么说, **导数** 的概念中已经定义好了 $x$ 移动的 **方向**, 即 **数轴正方向**。

对于一个数字来说, 我们可以将其看作是一个 **一维向量**, 它对应数轴上的一个点。那么和高维向量一样, 其可以表示 **方向** (由 数轴原点 指向 向量点)。举例来说, -5 表示的 **方向** 是: 数轴原点指向数轴 -5 点, 也就是 **数轴负方向**。对于 数字 或者 一维向量 来说, 其能表示的方向只有两个: **数轴正方向** 和 **数轴负方向**。

那么, 自变量 $x$ 可以沿着 **数轴负方向** 移动吗? 答案是可以的, 我们定义:

$$
\nabla_{-1} f(x_0) = \lim_{h \to 0} \frac{f(x_0 - h) - f(x_0)}{h} \tag{2.2}
$$

其中, $\nabla_{-1} f(x_0)$ 表示自变量 $x$ 沿着数轴负方向移动时的导数, 我们称为 **方向导数** ([directional derivative](https://en.wikipedia.org/wiki/Directional_derivative))。其含义是: 如果用 $x = x_0$ 处的切线来近似表示 $f(x)$, 当自变量 $x$ 沿着 **数轴负方向** 移动单位量时, 函数值 $f(x)$ 的变化量。

同理, 我们也定义自变量 $x$ 沿着数轴正方向移动时的导数:

$$
\nabla_{1} f(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h} \tag{2.3}
$$

显然, $\nabla_{1} f(x) = f^{\prime}(x)$, $\nabla_{-1} f(x) = -f^{\prime}(x)$。推导过程见附录。

我们定义, **梯度** ([gradient](https://en.wikipedia.org/wiki/Gradient)) 是, 在 $x = x_0$ 处, 使得函数值 $f(x)$ **增加** 最快的 $x$ 移动的方向, 也就是使得 **方向导数** 值最大的方向, 用符号 $\nabla$ 表示。注意, 虽然 方向导数 和 梯度 的符号表示很相似, 但完全是两个概念: 方向导数 是 导数, 而 梯度 是 方向。

当 $f^{\prime}(x) > 0$ 时, 当自变量 $x$ 沿着 **数轴正方向** 移动时, 函数值 $f(x)$ 变大, 当自变量 $x$ 沿着 **数轴负方向** 移动时, 函数值 $f(x)$ 变小, 此时 **梯度** 是 **数轴正方向**。

当 $f^{\prime}(x) < 0$ 时, 当自变量 $x$ 沿着 **数轴负方向** 移动时, 函数值 $f(x)$ 变大, 当自变量 $x$ 沿着 **数轴正方向** 移动时, 函数值 $f(x)$ 变小, 此时 **梯度** 是 **数轴负方向**。

刚才说过, 一个 数字 可以当作 一维向量, 表示数轴的方向。那么, 恰好, **导数** 表示的方向和 **梯度** 是一致的。那么, 我们就可以将 **梯度** 直接设置成 **导数**, 即 $\nabla f(x) = f^{\prime} (x)$。个人认为, 从严谨的角度来说, 梯度 用 $\frac{f^{\prime} (x)}{|f^{\prime} (x)|}$ 表示更加合理。

需要注意的是, **梯度** 表示的自变量 $x$ 移动的 **方向**, 可以通过 **导数** 的方式求解! 两者是两个概念!

## 三、二元函数与梯度

现在, 让我们来看看 **方向导数** 和 **梯度** 的常规介绍方式。对于 一元函数, 我们研究 微分 和 导数, 对于 二元函数, 我们研究 全微分 和 偏导数 ([partial derivative](https://en.wikipedia.org/wiki/Partial_derivative))。我们知道, 偏导数的定义如下:

$$
f^{\prime}_x (x_0, y_0) = \frac{\partial f(x_0, y_0)}{\partial x} = \lim_{h \to 0}\frac{f(x_0 + h, y_0) - f(x_0, y_0)}{h}
\tag{3.1}
$$

偏导数 和 导数一样, 都是针对函数中定义域内的某一点来定义的。对于 二元函数 来说, 定义域是一个平面, 此时自变量可以移动的 **方向** 不再只有两个, 而是平面内的任意方向, 我们可以用 二维单位向量 $\bold{v} = (v_1, v_2)$ 来表示 方向, 此时 **方向导数** 的定义如下:

$$
\nabla_{\bold{v}} f(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0 + v_1 \cdot h, y_0 + v_2 \cdot h) - f(x_0, y_0)}{h}
\tag{3.2}
$$

根据公式 $(3.2)$ 和 $(3.1)$, 我们不难发现, 和上面的情况一样, 偏导数 也是 方向导数 的特殊情况, 即:

$$
f^{\prime}_x (x_0, y_0) = \nabla_{(0, 1)} f(x_0, y_0) \tag{3.3}
$$

$$
f^{\prime}_y (x_0, y_0) = \nabla_{(1, 0)} f(x_0, y_0) \tag{3.4}
$$

和上面一样的证明方法, 我们可以得到 (推导过程见附录):

$$
\begin{align*}
    \nabla_{\bold{v}} f(x_0, y_0) &= f^{\prime}_x (x_0, y_0) \cdot v_1 + f^{\prime}_y (x_0, y_0) \cdot v_2 \\
    &= \frac{\partial f}{\partial x} \cdot v_1 + \frac{\partial f}{\partial y} \cdot v_2
\end{align*}
\tag{3.5}
$$

公式 $(3.5)$ 不就是向量 $\bold{v}$ 和向量 $(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y})$ 的点积吗。显然, 当两者同向时, 方向导数 值最大, 函数值 增加最快; 当两者反向时, 方向导数 值最小, 函数值 增加最慢。

因此, 我们可以将 梯度 设置成向量 $(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y})$。当然, 转换成单位向量更好, 一般不转换成单位向量。

整体方案思路是, 同时研究两个变量难度较高, 那么我们就分别研究, 再组合起来, 得到我们想要的东西。

对于单个变量来说, 为了使函数值增加, 其只有两个选项: 增加或者减小。

对于两个变量来说, 虽然可以朝着平面内任意方向移动, 但也只有一半的区域可以使得函数值增加, 另一半的区域会使得函数值减小。从公式 $(3.5)$ 可以看出, 如果向量 $\bold{v}$ 和 向量 $(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y})$ 垂直, 那么函数的增加值为 0。这个垂线就是分界线。

更多元的情况也是这样。

## 四、梯度下降法

理解什么是 梯度 后, 梯度下降法就很容易理解了。对于 机器学习 来说, 我们就是需要找一个函数 $f(x; \theta)$ 来 **估算** 目标值或者概率分布。训练过程就是寻找到一组 $\theta$, 使得 loss 值越低越好。

为了使得函数 $f$ 可以拟合任意函数, 我们采用 线性变换 + 激活函数 的方式。此时模型的参数 $\theta$ 会非常多, 我们没有办法定量分析 $\theta$ 和 loss 之间的函数关系, 只能使用 梯度下降法 这种迭代的方式寻找一个 局部最小值 (local minima), 作为我们需要的 $\theta$ 值。

梯度下降法主要分成三步: **前向传播**, **反向传播** 和 **参数更新**。先 **前向传播** 计算出 loss 值, 再 **反向传播** 计算出来参数地梯度值, 最后 **参数更新**。

用一个具体的例子来介绍整体的过程: 对于 线性回归 问题来说: 我们用 $x_i$ 表示样本的特征值, $y_i$ 表示样本的目标值, $\hat{y}_i$ 表示预测值, $n$ 表示训练集的样本数, $\theta$ 表示模型参数。

假定函数 $\hat{y}_i = f(x; \theta) = \theta \cdot x_i$, 用 L2 距离作为 loss 值, 同时学习率 $lr = 0.125$。

如果 $x_i = 2$, $\theta = 5$, $y_i = 9$, 那么某一次迭代的过程如下:

第一步, 正向传播, 计算预测值 $\hat{y}_i$:

$$
\hat{y}_i = x_i \cdot \theta = 2 \times 5 = 10
$$

第二步, 正向传播, 计算损失值 loss:

$$
\mathrm{loss} = (y - \hat{y})^2 = (9 - 10)^2 = 1
$$

第三步, 反向传播, 计算 $\mathrm{loss}$ 关于 $\hat{y}$ 的偏导数:

$$
\frac{\partial \mathrm{loss}}{\partial \hat{y}} = -2 \times (y - \hat{y}) = 2
$$

第四步, 反向传播, 计算 $\hat{y}$ 关于 $\theta$ 的偏导数:

$$
\frac{\partial \hat{y}}{\partial \theta} = x_i = 2
$$

第五步, 反向传播, 根据 链式求导法则, $\mathrm{loss}$ 关于 $\theta$ 的偏导数是:

$$
\frac{\partial \mathrm{loss}}{\partial \theta} = \frac{\partial \mathrm{loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \theta} = 2 \times 2 = 4
$$

第六步, 参数更新:

$$
\theta \gets \theta - lr \cdot \frac{\partial \mathrm{loss}}{\partial \theta} = 5 - 0.125 \times 4 = 4.5
$$

从上述过程可以看出, 反向传播的过程中完全没有使用第二步计算出来的 loss 值, 似乎第二步是不需要的。没错, 在梯度下降法中, 前向传播 起到的作用是支持 反向传播 的计算, 比方说第三步和第五步。 此时, 有些计算是多余的, 不是必要的。

因此, 如果想要深入理解 loss, 必须要对于进行求导! 下面, 让我们看看常见的 线性回归 和 逻辑回归 loss 函数的导数。

## 五、常见 loss 函数的导数

### (一) L2 Loss

$$
\mathrm{loss} = (\hat{y} - y)^2 \tag{5.1}
$$

$$
\frac{\partial \mathrm{loss}}{\partial \hat{y}} = 2 \cdot (\hat{y} - y) \tag{5.2}
$$

可以看出, 对于 L2 Loss 来说, 梯度值就是 预测值 和 目标值 之间的差值。如果 预测值 比 目标值大, 此时 预测值 应该变小, 对应梯度大于 0; 如果 预测值 比 目标值小, 此时 预测值 应该变大, 对应梯度小于 0; 完全符合我们的期待。更多内容参考: [MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)。

### (二) L1 Loss

$$
\mathrm{loss} = |\hat{y} - y| \tag{5.3}
$$

$$
\frac{\partial \mathrm{loss}}{\partial \hat{y}} =
\begin{cases}
    1 & \hat{y} > y \\
    0 & \hat{y} = y \\
    -1 & \hat{y} < y
\end{cases}
\tag{5.4}
$$

和 L2 loss 相比, L1 loss 可以理解为只保留方向的版本。这样的好处是不容易受极端值的影响。如果训练集的 目标值 $y$ 中存在较多极端值, 那么用 L2 loss 很可能会出现问题, 此时只保留 方向 是一个很好的选择。更多内容参考: [L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)。

### (三) 逻辑回归 交叉熵 Loss

在此之前, 我们先了解一下激活函数 sigmoid 求导后的形式:

$$
\mathrm{sigmoid} (x) = \frac{1}{1 + e^{-x}} \tag{5.5}
$$

$$
\mathrm{sigmoid}^{\prime}(x) = \mathrm{sigmoid} (x) \cdot (1 - \mathrm{sigmoid} (x)) \tag{5.6}
$$

$\mathrm{sigmoid} (0) = 0.5$, $\lim_{x \to \infin}\mathrm{sigmoid} (x) = 1$, $\lim_{x \to -\infin}\mathrm{sigmoid} (x) = 0$

$\mathrm{sigmoid}^{\prime} (0) = 0.25$, $\lim_{x \to \infin}\mathrm{sigmoid} (x) = 0$, $\lim_{x \to -\infin}\mathrm{sigmoid} (x) = 0$

sigmoid 函数取值在 0 到 1 之间, 单调递增, 其导函数先增后减, 对称函数, 取值范围在 0 到 0.25 之间。

对于 逻辑回归 来说, 其 loss 形式如下:

$$
\mathrm{loss} = \begin{cases}
    - \log \mathrm{sigmoid} (logit) & y = 1 \\
    - \log (1 - \mathrm{sigmoid}(logit)) & y = 0
\end{cases}
\tag{5.7}
$$

当 logit 值大于 0 时, 预测正类, 当 logit 值小于 0 时, 预测负类, 当 logit 值等于 0 时, 意味着正类和负类的概率相等。下面对 $logits$ 进行求导:

$$
\frac{\partial \mathrm{loss}}{\partial logit} = \begin{cases}
    \mathrm{sigmoid} (logit) - 1 & y = 1 \\
    \mathrm{sigmoid} (logit) & y = 0
\end{cases}
\tag{5.8}
$$

如果我们用 $\hat{y}$ 来表示 $\mathrm{sigmoid} (logits)$, 那么公式 $(5.8)$ 可以写成:

$$
\frac{\partial \mathrm{loss}}{\partial logit} = \hat{y} - y \tag{5.9}
$$

对比公式 $(5.2)$ 和公式 $(5.9)$, 可以发现两者几乎是一致的。线性回归 和 逻辑回归 的一致性就体现在这里, 他们的梯度是相同的。这是数据科学家们精心设计的结果!

预测正确, 梯度的绝对值在 0 到 0.5 之间, 预测错误, 梯度的绝对值在 0.5 到 1 之间。更多相关内容, 参考: [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)。

### (四) softmax 回归 交叉熵 Loss

类别过多不容易分析和计算。假设有三个类别, 他们的 logit 值分别是 $a$, $b$ 和 $c$。其中, 目标类是 $a$, 则:

$$
\mathrm{loss} = -\log \frac{\exp(a)}{\exp(a) + \exp(b) + \exp(c)} \tag{5.10}
$$

那么, 求导可得:

$$
\begin{align*}
    \frac{\partial \mathrm{loss}}{\partial a} &= \frac{\exp(a)}{\exp(a) + \exp(b) + \exp(c)} - 1 \\
    \frac{\partial \mathrm{loss}}{\partial b} &= \frac{\exp(b)}{\exp(a) + \exp(b) + \exp(c)} \\
    \frac{\partial \mathrm{loss}}{\partial c} &= \frac{\exp(c)}{\exp(a) + \exp(b) + \exp(c)} \\
\end{align*}
\tag{5.11}
$$

观察公式 $(5.8)$ 和公式 $(5.11)$, 可以发现 逻辑回归 和 softmax 回归的一致性。如果目标类是 $a$, 那么目标的概率分布是 $[1, 0, 0]$。而 loss 关于 logit 的梯度值就是 目标概率分布 和 预测概率分布 之间的差值。梯度的取值范围在 -1 到 1 之间。

和逻辑回归不同的是, 无论什么情况下, 目标类的 logit 值要增大, 其余类的 logit 值要减小, 只是程度不同罢了。更多相关内容, 参考: [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。

结合 $(5.6)$, 我们也可以看出 sigmoid 函数作为激活函数的问题。对于 CV 和 NLP 来说, 大部分任务都是 分类任务, 那么 loss 层梯度绝对值肯定在 0 到 1 之间。在使用 线性变换 + 激活函数 拟合任意函数的方案中, 每一个线性层后面都要加一个激活函数, 如果是两个连续的线性层, 那么就可以用一个线性层替代。而 sigmoid 梯度值在 0 到 0.25 之间。当神经网络的层数过多时, 多个 sigmoid 层的梯度连乘, 会使得参数梯度的绝对值越来越小, 靠前面的层参数几乎都不更新了, 那么他们起到的作用仅仅是 映射, 这不是我们想要的。一般的解决方案是换成 ReLU 等激活函数, 以及使用 残差 结构。

## 六、雅可比矩阵

反向传播的理论依据不是普通的 链式求导法则 ([Chain rule](https://en.wikipedia.org/wiki/Chain_rule)), 而是 [Multivariable Chain Rules](https://math.libretexts.org/Bookshelves/Calculus/Calculus_(OpenStax)/14%3A_Differentiation_of_Functions_of_Several_Variables/14.05%3A_The_Chain_Rule_for_Multivariable_Functions):

已知: $z = f(x, y)$, $x = g(t)$, $y = h(t)$。也就是通过 $t$ 值可以计算出 $x$ 值和 $y$ 值, 而 $x$ 和 $y$ 值可以计算出 $z$ 值, 那么:

$$
\frac{\partial z}{\partial t} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial t} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial t} \tag{6.1}
$$

和公式 $(3.5)$ 一样, 公式 $(6.1)$ 也可以表示成 向量点乘 的形式, 这是 反向传播 向量化编程的关键!!! 下面举一个多元函数求导的例子: softmax 函数求导。

已知 $\bold{o} = \mathrm{softmax} (\bold{x})$, 假设向量 $\bold{x}$ 是三维向量, 则:

$$
o_1 = \frac{\exp(x_1)}{\exp(x_1) + \exp(x_2) + \exp(x_3)} \tag{6.2}
$$

这样, 可以求出 $o_1$ 关于输入 $\bold{x}$ 的导数:

$$
\begin{align*}
    \frac{\partial o_1}{\partial x_1} &= o_1 \cdot (1 - o_1) \\
    \frac{\partial o_1}{\partial x_2} &= - o_1 \cdot o_2 \\
    \frac{\partial o_1}{\partial x_3} &= - o_1 \cdot o_3
\end{align*}
\tag{6.3}
$$

同理, 也可以求得 $o_2$ 和 $o_3$ 关于输入 $\bold{x}$ 的导数。对求导不熟悉的读者可以拿出草稿纸自己笔划笔划。

loss 值是关于向量 $\bold{o}$ 的一个很复杂的函数, 即 $\mathrm{loss} = f(\bold{o})$, 我们不知道其具体是什么, 但是我们知道 $\frac{\partial \mathrm{loss}}{\partial o_1}$, $\frac{\partial \mathrm{loss}}{\partial o_2}$ 和 $\frac{\partial \mathrm{loss}}{\partial o_3}$ 的数值。

根据公式 $(6.1)$, 我们可以得到:

$$
\begin{align*}
\frac{\partial \mathrm{loss}}{\partial x_1} &=
\frac{\partial \mathrm{loss}}{\partial o_1} \cdot \frac{\partial \mathrm{o_1}}{\partial x_1} +
\frac{\partial \mathrm{loss}}{\partial o_2} \cdot \frac{\partial \mathrm{o_2}}{\partial x_1} +
\frac{\partial \mathrm{loss}}{\partial o_3} \cdot \frac{\partial \mathrm{o_3}}{\partial x_1} \\ &=
\frac{\partial \mathrm{loss}}{\partial o_1} \cdot o_1 \cdot (1 - o_1) +
\frac{\partial \mathrm{loss}}{\partial o_2} \cdot o_2 \cdot (- o_1) +
\frac{\partial \mathrm{loss}}{\partial o_3} \cdot o_3 \cdot (- o_1)
\end{align*}
\tag{6.4}
$$

同理, 我们可以求得 $\frac{\partial \mathrm{loss}}{\partial x_2}$ 和 $\frac{\partial \mathrm{loss}}{\partial x_3}$。

我们可以将上述过程向量化, 设矩阵 $\bold{J}$ 是由向量 $\bold{o}$ 关于向量 $\bold{x}$ 的梯度构成, 形式如下:

$$
\begin{align*}
\bold{J} &= \begin{bmatrix}
    \frac{\partial o_1}{\partial x_1} & \frac{\partial o_1}{\partial x_2} & \frac{\partial o_1}{\partial x_3} \\
    \frac{\partial o_2}{\partial x_1} & \frac{\partial o_2}{\partial x_2} & \frac{\partial o_2}{\partial x_3} \\
    \frac{\partial o_3}{\partial x_1} & \frac{\partial o_3}{\partial x_2} & \frac{\partial o_3}{\partial x_3} \\
\end{bmatrix} \\ &=
\begin{bmatrix*}
    o_1 \cdot (1 - o_1) & o_1 \cdot (0 - o_2) & o_1 \cdot (0 - o_3) \\
    o_2 \cdot (0 - o_1) & o_2 \cdot (1 - o_2) & o_2 \cdot (0 - o_3) \\
    o_3 \cdot (0 - o_1) & o_3 \cdot (0 - o_2) & o_3 \cdot (1 - o_3) \\
\end{bmatrix*}
\end{align*}
\tag{6.6}
$$

设向量 $\bold{v}$ 是由标量 $\mathrm{loss}$ 关于向量 $\bold{o}$ 的梯度构成, 形式如下:

$$
\bold{v} = \begin{bmatrix}
    \frac{\partial \mathrm{loss}}{\partial x_1} \\
    \frac{\partial \mathrm{loss}}{\partial x_2} \\
    \frac{\partial \mathrm{loss}}{\partial x_3}
\end{bmatrix} \tag{6.7}
$$

然后用 $\bold{J}^{\mathsf{T}} \cdot \bold{v}$ 就是我们想要的结果, 即标量 $\mathrm{loss}$ 关于向量 $\bold{x}$ 的梯度。其中, $\bold{J}$ 是 **雅可比矩阵** (Jacobian matrix), $\bold{J}^{\mathsf{T}} \cdot \bold{v}$ 被称为 vector-Jacobian Production。

雅可比矩阵 $\bold{J}$ 的 shape 是 `[n_outputs, n_inputs]`, 而向量 $\bold{v}$ 的 shape 是 `[n_outputs, ]`。

我们可以用下面的代码来验证上述过程的正确性:

```python
import torch 


def softmax_grad(x: torch.Tensor, dy: torch.Tensor):
    if x.ndim != 1:
        raise NotImplementedError

    with torch.no_grad():
        outputs = torch.softmax(x, dim=0)
        labels = torch.eye(x.size(0))
        jacobian_matrix = (labels - outputs) * outputs.unsqueeze(-1)

    return jacobian_matrix.T @ dy


def softmax_grad_pytorch(x: torch.Tensor, dy: torch.Tensor):
    
    x = torch.nn.Parameter(x)
    torch.softmax(x, dim=0).backward(dy)

    return x.grad.detach()


def is_same_tensor(t1: torch.Tensor, t2: torch.Tensor, eps: float = 1e-6):
    return torch.all(torch.abs(t1 - t2) < eps).item()


logits = torch.randn(10)
grad = torch.randn(10)

v1 = softmax_grad(logits, grad)
v2 = softmax_grad_pytorch(logits, grad)

print(is_same_tensor(v1, v2))
print(v1)
print(v2)
```

## 七、总结

总结一下, 在梯度下降法中, **前向传播** 的作用是为 **反向传播** 提供计算支持, 这在上面的例子中已经有所展示了。当然, 我们也可以不用 **前向传播**, 直接使用 loss 关于每一个参数 $\theta$ 的梯度公式也是可以的, 这种做法被称为 **符号微分**。其问题很明显: (1) 表达式膨胀, 靠前面层的表达式会非常复杂; (2) 如果优化的不好, 会产生大量重复的计算。

深度学习中的运算大部分都是由数组 element-wise 运算, 矩阵乘法, 向量点乘, 互相关 (卷积), RNN Cell 等 基本运算 构成。因此, 我们可以将整个神经网络拆分成一个个 基本运算, 然后用这些 基本运算 构建有向无环图 (Directed Acyclic Graph; DAG), 方便数据的流通。我们将这个 DAG 图称为 **计算图** (computation graph), 里面的结点, 或者说 基本运算 称为 **算子** (operator)。

每一个 **算子** 都需要实现 前向 和 反向 过程: 在 **前向过程** 中, 函数记录计算梯度所需要的中间变量, 并返回运算结果; 在 **反向过程** 中, 函数接收 loss 关于 **输出** 的梯度值, 记录 loss 关于 **参数** 的梯度值, 返回 loss 关于 **输入** 的梯度值。

当 **算子** 被拆分的足够小, 那就是 **自动微分** (autograd)。举例来说, 我们一般认为 caffe 没有自动微分的功能, 而 PyTorch 有。从本质上来说, 两者是一致的, 只是 颗粒度 不同。caffe 只实现了部分深度学习常用 **算子** 的 **反向过程**, 而 PyTorch 实现了 numpy 中的绝大部分运算的 **反向过程**。如果你要写一个新的网络层, 并且这个网络层由数个 numpy 操作构成, 使用 PyTorch 框架就不需要自己实现 **反向过程** 了, 但如果使用 caffe 框架, 就需要自己实现 **反向过程**。

比方说, 对于 Attention 运算 来说, 我们可以拆成 四个线性层, 两次矩阵乘法运算 和 一个 softmax 运算。这些基本运算在 PyTorch 中都实现了 反向过程, 但是在 caffe 中没有, 必须要自己实现。这就是两者之间的区别。

更极端的情况, 如果将整个模型作为一个 **算子**, 那就是 **符号微分** 了。可以看出 算子 的 颗粒度 非常重要, 其直接决定了框架的易用性。

上面所说的 **自动微分** 是狭义上。从广义上来说, 应该称为 **自动微分反向模式** (reverse mode), 还有一种是 **自动微分前向模式** (forward mode)。在深度学习领域, 主要使用的是 反向模式。两者的主要区别在于改变了计算顺序, 涉及到 Jacobian-vector Production, 这个之后有时间再探讨。

一阶微分的本质是: 在一个很小的范围内, 用 一次线性函数 来代替原本的函数, 这在公式 $(6.1)$ 和雅可比矩阵中有充分的体现。当然, 我们也可以用 二次函数 来代替, 这就涉及到 海森矩阵 (Hessian Matrix), 这个也是之后有时间再探讨。

## 八、附录

公式 $(3.5)$ 的证明:

当 $h \to 0$ 时, 我们可以得到:

$$
f(x_0 + v_1 \cdot h, y_0 + v_2 \cdot h) - f(x_0, y_0) = 0
\tag{a.1}
$$

$$
f(x_0 + h, y_0) - f(x_0, y_0) = 0
\tag{a.2}
$$

将公式 $(a.2)$ 左右同乘以 $v_1$, 可以得到:

$$
[f(x_0 + h, y_0) - f(x_0, y_0)] v_1 = 0
\tag{a.3}
$$

同理, 可以得到:

$$
[f(x_0, y_0 + h) - f(x_0, y_0)] v_2 = 0
\tag{a.4}
$$

根据公式 $(a.1)$, $(a.3)$ 和 $(a.4)$, 我们可以得到:

$$
\begin{align*}
    &\quad f(x_0 + v_1 \cdot h, y_0 + v_2 \cdot h) - f(x_0, y_0) \\
    &= [f(x_0 + h, y_0) - f(x_0, y_0)] v_1 + [f(x_0, y_0 + h) - f(x_0, y_0)] v_2
\end{align*} \tag{a.5}
$$

在公式 $(a.5)$ 的两边同除以 $h$, 就可以得到公式 $(3.5)$ 了。$\nabla_{-1} f(x) = -f^{\prime}(x)$ 的证明方式是一样的。
