
# 逻辑回归

无论是 **线性回归** 问题还是 **逻辑回归** 问题, 我们都是要求解下面的函数:

$$
z = f(\vec{x}) = \vec{k} \cdot \vec{x} + b \tag{1}
$$

但是其表示的含义不同。对于线性回归问题, 我们可以理解成在 17 维 (加上目标特征) 的空间中找一个 **线性几何体**, 让 1000 个样本点在 $z$ 轴上到 **线性几何体** 的 **误差值** 尽可能地小。需要注意的有:

+ 不是样本点到 **线性几何体** 的距离尽可能地小
+ 随着 $\vec{x}$ 维度的升高, **线性几何体** 可以是点, 直线, 平面等等, 但不可以是曲线, 曲面之类的

<!-- 画图工具: https://www.geogebra.org/classic -->

对于逻辑回归问题, 我们可以理解成在 16 维 (不加上目标值特征) 的空间内找一个 **决策边界**, 在这个边界一边的为正类, 另一边的为负类。更具体地, 决策边界就是 $\vec{k} \cdot \vec{x} + b = 0$ 这个 **线性几何体**。

学过 **线性规划** 应该知道, 对于 $a x + b y + c = 0$ 这条直线来说, 如果 $b > 0$, 则:

+ 若 $(x_0, y_0)$ 在直线 $a x + b y + c = 0$ 上方, 则 $a x_0 + b y_0 + c > 0$
+ 若 $(x_1, y_1)$ 在直线 $a x + b y + c = 0$ 下方, 则 $a x_1 + b y_1 + c < 0$

这样, 我们就可以用 **数值** 来表示 **位置关系**。逻辑回归采用的就是这样的思想。

接着上面的说, 在逻辑回归中, 我们可以用 $f(\vec{x})$ 的值来表示 $\vec{x}$ 和决策边界之间的位置关系。那么, 这个函数值有没有具体的几何含义呢?

学过高数的应该知道, 点 $(x_1^{\prime}, x_2^{\prime})$ 到直线 $ax_1 + bx_2 + c = 0$ 之间的欧式距离是:

$$
d = \frac{|ax_1^{\prime} + bx_2^{\prime} + c|}{\sqrt{a^2 + b^2}}
$$

点 $(x_1^{\prime}, x_2^{\prime}, x_3^{\prime})$ 到平面 $ax_1 + bx_2 + cx_3 + d = 0$ 之间的欧式距离是:

$$
d = \frac{|ax_1^{\prime} + bx_2^{\prime} + cx_3^{\prime} + d|}{\sqrt{a^2 + b^2 + c^2}}
$$

观察上面, 我们不难发现: 分子就是函数值 $f(\vec{x})$ 的绝对值, 分母是 $\vec{k}$ 的模长。由此我们可以推广得到, 对于点 $\vec{x}^{\prime}$, 其到决策边界 $\vec{k} \cdot \vec{x} + b = 0$ 的欧式距离是:

$$
d = \frac{|f(\vec{x}^{\prime})|}{||\vec{k}||} \tag{2}
$$

也就是说, 函数值 $f(\vec{x})$ 和 $\vec{x}$ 到决策边界的欧式距离是有关系的, 如果 $\vec{k}$ 是单位向量, 那么函数值的绝对值就是这个欧式距离。

接下来, 我们分析损失值。我们知道, $\mathrm{logsumexp}$ 是 $\max$ 函数的光滑近似函数。

我们用 $z$ 表示函数值 $f(\vec{x})$, 如果 $\vec{x}$ 是正类, 那么其 loss 值是:

$$
\begin{align*}
loss &= - \log (\mathrm{sigmoid}(z)) \\
     &= - \log \frac{1}{1 + e^{-z}} \\
     &= \log (e^0 + e^{-z}) \\
     &= \mathrm{logsumexp} (0, -z) \\
     &\approx \max(0, -z)
\end{align*}
$$

如果 $z > 0$, 那么 loss 值就是 0; 如果 $z < 0$, 那么 loss 值就是 $-z$。我们优化的方向是 loss 值越小越好, 也就是 $z > 0$ 即可, 此时可以取到最小值 $0$。当然, 实际上我们使用的是 $\mathrm{logsumexp}$ 函数, 只有当 $z > 4$ 时, loss 的值才会接近 0。

同理, 如果 $\vec{x}$ 是负类, 其 loss 值是:

$$
\begin{align*}
loss &= - \log (1 - \mathrm{sigmoid}(z)) \\
     &= \log (e^0 + e^{z}) \\
     &= \mathrm{logsumexp} (0, z) \\
     &\approx \max(0, z)
\end{align*}
$$

如果 $z > 0$, 那么 loss 值就是 $z$; 如果 $z < 0$, 那么 loss 值就是 $0$。我们优化的方向是 loss 值越小越好, 也就是 $z < 0$ 即可, 此时可以取到最小值 $0$。当然, 实际上我们使用的是 $\mathrm{logsumexp}$ 函数, 只有当 $z < -4$ 时, loss 的值才会接近 0。如果 $z = 0$, loss 值大约时 0.69。

总结一下, 就是函数 $f(\vec{x})$ 的绝对值和点 $\vec{x}$ 到线性几何体之间的欧式距离是正相关的。在逻辑回归中, 我们可以认为是用函数值的绝对值作为 loss 值, 同时设置了一个上限, 正类的上限是 `4`, 负类的上限是 `-4`。

对于 softmax 回归来说, 如果是 4 分类问题, 那么我们需要寻找 4 个线性几何体, 每两个线性几何体之间的差值就是对应类别之间的决策边界。一共有 $C_4^2 = 6$ 个决策边界。你可以像上面一样进行推导, 具体参考 [笔记](../../basic-knowledge-v1/07_loss.md)。

整个过程没有涉及到和 概率论 相关的知识, 解释性还是非常强的。另外扩展一下从概率论理解的知识体系:

首先, **概率** 的理解有两种方式:

+ logit 函数 + sigmoid 函数
+ sigmoid 函数是 [单位阶跃函数](https://en.wikipedia.org/wiki/Heaviside_step_function) 的光滑近似函数; softmax 函数是 one-hot argmax 函数的光滑近似函数

其次, 对于 loss 的推荐从两个层面来理解:

+ 信息量 → 信息熵 → 交叉熵
+ 极大似然法 → KL 散度 → 交叉熵

这些内容在其它的笔记中是有的, 之后有时间来一个 **逻辑回归** 概念总结。
