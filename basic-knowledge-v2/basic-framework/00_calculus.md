
# 微积分

## 中英对照表

+ 微积分: calculus
+ 微分: [differential calculus](https://en.wikipedia.org/wiki/Differential_calculus)
+ 积分: integral calculus
+ 导数: derivative

## 微分 和 积分 的关系

微分 和 积分 是互逆运算:

+ 微分: 求 s-t 函数在某一点切线的斜率 (瞬时速度)
+ 积分: 求 v-t 函数在某一区间内的面积 (路程)

注意: 微分 和 积分是互逆运算, 但是针对的函数不同 !!! 牛顿-莱布尼兹公式如下:

$$
\int_a^b V(t) dt = S(b) - S(a)
$$

## 微分 和 导数 的关系

$dy = f^{\prime}(x) dx$

$\Delta y = f(x + \Delta x) - f(x)$

导数是求函数在某一点切线的斜率; 微分是用函数在某一点的切线来替代函数。

其中, $dx = \Delta x$, $dy \approx \Delta y$。$\Delta y$ 表示 函数值 $y$ 的变化量, $dy$ 表示 切线函数值 的变化量。

## 常见导数推导

定义:

$$
f^{\prime} (x_0) = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{x_0 + \Delta x - x_0}
$$

$f(x) = x^2$ 二次函数推导:

$$
\begin{align*}
    f^{\prime}(x) &= \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} \\
    &= \lim_{\Delta x \to 0} \frac{(x + \Delta x)^2 - x^2}{\Delta x} \\
    &= \lim_{\Delta x \to 0} 2x + \Delta x \\
    &= 2x
\end{align*}
$$

[自然常数, 自然底数, 欧拉数, 数学常数 e](https://en.wikipedia.org/wiki/E_(mathematical_constant)) 的定义:

$$
\begin{align*}
    e &= \lim_{n \to \infty} (1 + \frac{1}{n})^n \\
    &= \sum_{n=0}^{\infty} \frac{1}{n!}
\end{align*}
$$

## 导数 和 方向导数 的关系

## 二元函数

只有 偏导数, 偏微分, 全微分, 而 全导数 是针对符合函数设计的, 具体参考: [什么是全导数？ - 湖心亭看雪的回答 - 知乎](https://www.zhihu.com/question/26966355/answer/276813234)

## 科幻: 维度

维度 是一个很泛化的概念, 向量有维度, 数组有维度, 空间有维度, 时间也有维度!

我们生活在一个 三维空间, 单向一维时间的世界里。对于更高维的空间, 我们无法理解, 对于更高维的时间, 我们可以类比空间来理解:

一维时间: 我们可以回到过去
二维时间: 类似于平行宇宙, 不同世界里的你

网上很多科幻片就是以此为基点, 将时间和空间的维度混在一起。一般有两种比较合理的解释:

1. 你穿越到过去就会形成一个新的时间线
2. 你穿越到过去也是历史中的一部分, 无论怎么努力, 都不可能改变历史, 反而会促进历史

对于 二维时间 概念的最大问题是测量尺度的问题。对于空间来说, 我们可以统一用长度来衡量每一个维度, 但是对于时间来说, 我们可以用 秒, 小时这样的单位来衡量两个时间线之间的距离吗? 不同 **时间线** 怎么整合成一个 **时间面** 呢?

## 引用

知识体系:

+ Jacobian Matrix 雅可比矩阵
+ Hessian Matrix 海森矩阵 / 黑森矩阵
  + Natural Gradient Descent & Fisher Information Matrix
  + Hessian-Free (HF) optimization
  + Second-Order Adversarial Attack
  + Second-Order Optimization in GANs
  + Hessian Eigenvalue Analysis

博客:

+ [导数与微分到底有什么区别？](https://zhuanlan.zhihu.com/p/145620564)
+ [通俗理解方向导数、梯度|学习笔记](https://zhuanlan.zhihu.com/p/613651124)
+ [泰勒公式（泰勒展开式）通俗+本质详解](https://blog.csdn.net/qq_38646027/article/details/88014692)
+ [Homework 2-2 Hessian Matrix](https://github.com/ga642381/ML2021-Spring/blob/main/HW02/HW02-2.ipynb)
+ [A Gentle Introduction To Hessian Matrices](https://machinelearningmastery.com/a-gentle-introduction-to-hessian-matrices/)
+ [深度学习参数初始化详细推导：Xavier方法和kaiming方法【一】](https://zhuanlan.zhihu.com/p/532018644)

论文:

+ [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767)
+ [Tangent: Automatic Differentiation Using Source Code Transformation in Python](https://arxiv.org/abs/1711.02712)
+ [Differentiable Programming Tensor Networks](https://arxiv.org/abs/1903.09650)
+ [A Differentiable Programming System to Bridge Machine Learning and Scientific Computing](https://arxiv.org/abs/1907.07587)

PyTorch 文档:

+ [THE FUNDAMENTALS OF AUTOGRAD](https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html)
+ [FORWARD-MODE AUTOMATIC DIFFERENTIATION (BETA)](https://pytorch.org/tutorials/intermediate/forward_ad_usage.html)
+ [JACOBIANS, HESSIANS, HVP, VHP, AND MORE: COMPOSING FUNCTION TRANSFORMS](https://pytorch.org/tutorials/intermediate/jacobians_hessians.html)
