
# Latex 规范

## 规范写法

标量: $\bold{a}$, $\bold{q}$, $\bold{k}$

列向量: $\bold{a}$, $\bold{q}$, $\bold{k}$

行向量: $\bold{a}^{\mathsf{T}}$, $\bold{q}^{\mathsf{T}}$, $\bold{k}^{\mathsf{T}}$

向量点乘: $\bold{q}^{\mathsf{T}} \cdot \bold{k}$, 结果应该是一个标量。

向量矩阵乘法: $\bold{u} \cdot \bold{v}^{\mathsf{T}}$, 结果应该是一个矩阵。

矩阵: $\bold{A}$, $\bold{Q}$, $\bold{K}$, $\bold{V}$

比方说, SVD 分解的结果可以写成:

$$
\begin{align*}
    \bold{A} &= \bold{U} \cdot \bold{\Sigma} \cdot \bold{V}^{\mathsf{T}} \\
    &= \sigma_1 \cdot \bold{u}_1 \cdot \bold{v}_1^{\mathsf{T}} + \sigma_2 \cdot \bold{u}_2 \cdot \bold{v}_2^{\mathsf{T}} + \cdots + \sigma_n \cdot \bold{u}_n \cdot \bold{v}_n^{\mathsf{T}}
\end{align*}
$$

## 常用希腊字母

$\Sigma$, $\sigma$, $\sum$

$\Omega$, $\omega$

$\alpha$, $\beta$, $\gamma$, $\Gamma$

## 效果测试

列向量: $\bold{a}$, $\boldsymbol{a}$, $\overrightarrow{a}$, $\vec{a}$

行向量: $\bold{a}^{\mathsf{T}}$, $\boldsymbol{a}^{\mathsf{T}}$, $\overrightarrow{a}^{\mathsf{T}}$, $\vec{a}^{\mathsf{T}}$

列向量: $\bold{q}$, $\boldsymbol{q}$, $\overrightarrow{q}$, $\vec{q}$

行向量: $\bold{q}^{\mathsf{T}}$, $\boldsymbol{q}^{\mathsf{T}}$, $\overrightarrow{q}^{\mathsf{T}}$, $\vec{q}^{\mathsf{T}}$

列向量: $\bold{k}$, $\boldsymbol{k}$, $\overrightarrow{k}$, $\vec{k}$

行向量: $\bold{k}^{\mathsf{T}}$, $\boldsymbol{k}^{\mathsf{T}}$, $\overrightarrow{k}^{\mathsf{T}}$, $\vec{k}^{\mathsf{T}}$
