
# Norm

## 单词含义

在日常用语中, norm 的含义是 **行为准则** 和 **规范**, 更多内容可以参考: [剑桥词典](https://dictionary.cambridge.org/zhs/%E8%AF%8D%E5%85%B8/%E8%8B%B1%E8%AF%AD-%E6%B1%89%E8%AF%AD-%E7%AE%80%E4%BD%93/norm)。

在数学中, norm 往往指的是 向量 的模长, 更广义的说, 就是 向量 的长度或者大小。[维基百科](https://en.wikipedia.org/wiki/Norm_(mathematics)) 给出的说法是 **norm** 是一个函数, 将 **向量** 映射为一个 **非负实数**, 并且可以用于描述距离 (满足三角不等式, 在零点值为 0)。

normalization 是 normal 的名词形式, 在统计学中往往翻译成 **标准化** 或者 **归一化**, 无论是 z-score 化还是 min-max 归一化, 都属于 normalization, 更多内容可以参考: [维基百科](https://en.wikipedia.org/wiki/Normalization_(statistics))。

深度学习中也有 normalization, 包括 batch normalization, layer normalization 等等, 更多内容可以参考 [PyTorch 文档](https://pytorch.org/docs/stable/nn.html#normalization-layers)。

为什么这两个单词容易弄混淆呢? 首先, normalization 单词太长了, 很多地方会简写成 norm, 比方说 PyTorch 中的 `BatchNorm` 和 `LayerNorm`, 这简直就是天坑 !!! 其次, normalization 中也有用 norm 作为标准化工具的, 将向量转化为单位向量。

总之, 这里就是一个 天坑, 千万不要弄混淆了。

## p-norm

对于 $\vec{x}$ 来说, 其 norm 的数学形式是: $||\vec{x}||$。p-norm 的数学公式是:

$$
||\vec{x}||_p = \left (\sum_{i=1}^n |x_i|^p \right)^{\frac{1}{p}} \tag{1}
$$

整个计算过程可以描述为: 对向量中的每一个元素, 先取 **绝对值**, 再进行 $p$ 次方的运算, 最后求和得到一个数字, norm 值就是将这个数字进行 $\frac{1}{p}$ 次方运算。

这里面的 **绝对值** 很重要, 保证在进行 **次方** 运算时 底数 一定是正数。这样 $p$ 只要是非零实数都可以计算出来一个 norm 值。同时, 也保证了 norm 值一定是 **非负实数**。(负数进行 **次方** 运算时结果有可能是虚数)

如果 $p=0$ 呢? 开零次方是没有办法计算的, 道理很简单: **开方** 运算是 **次方** 运算的逆运算。任意一个数字的 **零次方** 都是 $1$, 也就意味着 **零次方** 运算是 **不可逆的**, 也就意味着 **开零次方** 无法运算。因此, 我们特别规定: $||\vec{x}||_0$ 表示 $\vec{x}$ 中 **非零元素** 的个数。

如果 $p \to \infin$ 时呢? 此时 $||\vec{x}||_\infin = \max\{|x_1|, |x_2|, \cdots, |x_n|\}$。这和 logsumexp 函数是 max 函数的光滑近似函数, softmax 是 one-hot argmax 函数的光滑近似函数是一个道理。

假设 $a$ 和 $b$ 两个正实数满足 $a < b$。则当 $p \to \infin$ 时, 由于指数爆炸的特性, $a^p \ll b^p$, 也就意味着 $a^p + b^p \approx b^p$, 自然 $(a^p + b^p)^{1/p} \approx b$。更具体地:

+ 如果 $a$ 和 $b$ 都是大于 1 的数字, 上面所说并不难以理解
+ 如果 $a$ 和 $b$ 是在 0 - 1 之间的数字, 此时 $a^p$ 和 $b^p$ 都是趋近于 0 的, 但是趋近的程度并不一样, $a^p$ 一定比 $b^p$ 更加趋近于 0, 此时两者相加一定是 $b^p$ 占主导。

由此可以看出, 指数函数真的是 $\max$ 函数的好朋友!!!

更多相关内容可以参考: [LP Space](https://en.wikipedia.org/wiki/Lp_space) 和 [Norm (mathematics)](https://en.wikipedia.org/wiki/Norm_(mathematics))。

## 不同的名字

我们常用的实际上是 L1 norm 和 L2 norm。这里列举一些常用的名称。

L1 norm 额外的名称:

+ Manhattan Distance (曼哈顿距离)
+ Manhattan Norm
+ Mean-Absolute Error (MAE)

L2 norm 额外的名称:

+ Euclidean Distance (欧式距离)
+ Euclidean Norm
+ Mean-Squared Error (MSE)

## Matrix Norm

除了 向量 有范数外, 矩阵也有范数, 包括 Frobenius norm, nuclear norm, spectral norm 等等, 更多相关内容参考: [Matrix Norm](https://en.wikipedia.org/wiki/Matrix_norm)。

## References

+ [l0-Norm, l1-Norm, l2-Norm, … , l-infinity Norm](https://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/)
