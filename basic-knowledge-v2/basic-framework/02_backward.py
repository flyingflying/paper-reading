
# %%

from itertools import zip_longest

import torch 
from torch import nn, Tensor


def is_same_tensor(t1: Tensor, t2: Tensor, eps: float = 1e-8) -> bool:
    return t1.shape == t2.shape and (t1 - t2).abs().max().item() < eps

# %% scalar: sin 

''' scalar: sin '''


@torch.no_grad()
def sin_backward(input: Tensor, output_grad: Tensor) -> Tensor:
    if input.size() != output_grad.size():
        raise ValueError("input 和 output_grad 的 size 应该相同!")

    # step1: 求解 output 关于 input 的导数
    derivative = torch.cos(input)

    # step2: 计算 input 的梯度值
    input_grad = derivative * output_grad

    return input_grad


def check_sin_backward():
    input = nn.Parameter(torch.randn(2, 3, 4, 5).double())
    output_grad = torch.randn_like(input).double()

    torch.sin(input).backward(output_grad)
    result = sin_backward(input, output_grad)

    print(is_same_tensor(input.grad, result))


check_sin_backward()

# %% scalar: sigmoid 

''' scalar: sigmoid '''


@torch.no_grad()
def sigmoid_backward(input: Tensor, output_grad: Tensor) -> Tensor:
    output = torch.sigmoid(input)
    derivative = output * (1 - output)
    input_grad = output_grad * derivative
    return input_grad


def check_sigmoid_backward():
    input = nn.Parameter(torch.randn(2, 3, 4, 5).double())
    output_grad = torch.randn_like(input).double()

    torch.sigmoid(input).backward(output_grad)
    result = sigmoid_backward(input, output_grad)

    print(is_same_tensor(input.grad, result))


check_sigmoid_backward()

# %% scalar: relu 

''' scalar: relu '''

@torch.no_grad()
def relu_backward(input: Tensor, output_grad: Tensor) -> Tensor:
    # torch.heaviside: 单位阶跃函数
    derivative = torch.heaviside(input, torch.tensor(0, dtype=input.dtype))
    input_grad = derivative * output_grad
    return input_grad


def check_relu_backward():
    input = torch.randn(2, 3, 4, 5).double()
    input[0, 0, 0, 0] = 0.  # 强制某一个数是 0
    input = nn.Parameter(input)
    output_grad = torch.randn_like(input).double()

    torch.relu(input).backward(output_grad)
    result = relu_backward(input, output_grad)

    print(is_same_tensor(input.grad, result))


check_relu_backward()

# %% scalar: tanh 

''' scalar: tanh '''

@torch.no_grad()
def tanh_backward(input: Tensor, output_grad: Tensor) -> Tensor:
    # torch.heaviside: 单位阶跃函数
    derivative = 1 - torch.square(torch.tanh(input))
    input_grad = derivative * output_grad
    return input_grad


def check_tanh_backward():
    input = nn.Parameter(torch.randn(2, 3, 4, 5).double())
    output_grad = torch.randn_like(input).double()

    # PyTorch 求导方式
    torch.tanh(input).backward(output_grad)
    result = tanh_backward(input, output_grad)

    print(is_same_tensor(input.grad, result))


check_tanh_backward()

# %% non-saturated activation function pictures 

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.axes import Axes

x = np.linspace(-10, 10, 1000)
y_sigmoid = 1. / (1. + np.exp(-x))

_, axes = plt.subplots(2, 2, figsize=(10, 10))
axes: list[Axes] = axes.flatten()

# sigmoid function
y_sigmoid = 1. / (1. + np.exp(-x))
axes[0].plot(x, y_sigmoid)
axes[0].set_title("sigmoid function")

# sigmoid grad function
y_sigmoid_grad = y_sigmoid * (1 - y_sigmoid)
axes[2].plot(x, y_sigmoid_grad)
axes[2].set_title("sigmoid grad function")

# tanh function
y_tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
axes[1].plot(x, y_tanh)
axes[1].set_title("tanh function")

# tanh grad function
y_tanh_grad = 1 - np.power(y_tanh, 2)
axes[3].plot(x, y_tanh_grad)
axes[3].set_title("tanh grad function")

# %% scalar: add 

''' scalar: add '''

@torch.no_grad()
def add_backward(input1: Tensor, input2: Tensor, output_grad: Tensor) -> Tensor:
    if torch.broadcast_shapes(input1.shape, input2.shape) != output_grad.shape:
        raise ValueError

    # step1: 求 output 关于 input 的导数
    input1_derivative = input2_derivative = torch.ones_like(output_grad)

    # step2: 求 input 的梯度值
    input1_grad = output_grad * input1_derivative
    input2_grad = output_grad * input2_derivative

    # step3: 处理 广播机制
    for _ in range(input1_grad.ndim - input1.ndim):
        input1_grad = input1_grad.sum(dim=0)
    for _ in range(input2_grad.ndim - input2.ndim):
        input2_grad = input2_grad.sum(dim=0)

    for i, (r, s) in enumerate(zip(input1_grad.shape, input1.shape)):
        if r != s:
            input1_grad = input1_grad.sum(dim=i, keepdim=True)
    for i, (r, s) in enumerate(zip(input2_grad.shape, input2.shape)):
        if r != s:
            input2_grad = input2_grad.sum(dim=i, keepdim=True)
    
    return input1_grad, input2_grad


def check_add_backward():
    input1 = nn.Parameter(torch.randn(2, 3, 4, 5).double())
    input2 = nn.Parameter(torch.randn(3, 1, 5).double())
    output_grad = torch.randn_like(input1 + input2).double()

    (input1 + input2).backward(output_grad)
    result1, result2 = add_backward(input1, input2, output_grad)

    print(is_same_tensor(input1.grad, result1))
    print(is_same_tensor(input2.grad, result2))


check_add_backward()

# %% scalar: mul 

''' scalar: mul '''


@torch.no_grad()
def mul_backward(input1: Tensor, input2: Tensor, output_grad: Tensor) -> tuple[Tensor, Tensor]:

    input1_grad = output_grad * input2
    input2_grad = output_grad * input1

    for _ in range(input1_grad.ndim - input1.ndim):
        input1_grad = input1_grad.sum(dim=0)
    for _ in range(input2_grad.ndim - input2.ndim):
        input2_grad = input2_grad.sum(dim=0)

    for i, (r, s) in enumerate(zip(input1_grad.shape, input1.shape)):
        if r != s:
            input1_grad = input1_grad.sum(dim=i, keepdim=True)
    for i, (r, s) in enumerate(zip(input2_grad.shape, input2.shape)):
        if r != s:
            input2_grad = input2_grad.sum(dim=i, keepdim=True)
    
    return input1_grad, input2_grad


def check_mul_backward():
    input1 = nn.Parameter(torch.randn(2, 1, 4, 5).double())
    input2 = nn.Parameter(torch.randn(3, 1, 5).double())
    output_grad = torch.randn_like(input1 * input2).double()

    (input1 * input2).backward(output_grad)
    result1, result2 = mul_backward(input1, input2, output_grad)

    print(is_same_tensor(input1.grad, result1))
    print(is_same_tensor(input2.grad, result2))


check_mul_backward()


# %% scalar: expand

''' scalar: expand '''


@torch.no_grad()
def expand_backward(input: Tensor, output_grad: Tensor, size: list[int]) -> Tensor:

    input_grad = output_grad

    for i, (r, s) in enumerate(zip(input.shape, size)):
        if r != s:
            input_grad = input_grad.sum(dim=i, keepdim=True)
    
    return input_grad


def check_expand_backward():
    input = nn.Parameter(torch.randn(2, 1, 4, 1).double())
    size = (2, 3, 4, 5)
    output_grad = torch.randn(size).double()

    input.expand(size).backward(output_grad)
    result = expand_backward(input, output_grad, size)

    print(is_same_tensor(input.grad, result))


check_expand_backward()

# %% scalar: unsqueeze

''' scalar: unsqueeze '''


@torch.no_grad()
def unsqueeze_backward(input: Tensor, output_grad: Tensor, dim: int) -> Tensor:
    return output_grad.squeeze(dim=dim)


def check_unsqueeze_backward():
    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(2, 1, 3, 4).double()

    input.unsqueeze(dim=1).backward(output_grad)
    result = unsqueeze_backward(input, output_grad, 1)

    print(is_same_tensor(input.grad, result))


check_unsqueeze_backward()

# %% scalar: squeeze

''' scalar: squeeze '''


@torch.no_grad()
def squeeze_backward(input: Tensor, output_grad: Tensor, dim: int) -> Tensor:
    return output_grad.unsqueeze(dim=dim)


def check_squeeze_backward():
    input = nn.Parameter(torch.randn(2, 3, 1, 4).double())
    output_grad = torch.randn(2, 3, 4).double()

    input.squeeze(dim=2).backward(output_grad)
    result = squeeze_backward(input, output_grad, 2)

    print(is_same_tensor(input.grad, result))


check_squeeze_backward()

# %% scalar: indexing 

''' scalar: indexing '''

a = nn.Parameter(torch.randn(10, 10))
a[torch.tensor([1, 3, 3])].sum().backward()
print(a.grad)

a = nn.Parameter(torch.randn(3, 4))
b = a * 1
b[b < 0] = 2.
b.sum().backward()
print(a.grad) 

# %% vector: sum 

''' vector: sum operator '''


@torch.no_grad()
def sum_backward(input: Tensor, output_grad: Tensor, dim: int = None) -> Tensor:
    if dim is None:
        # 如果没有指定 dim, 那么 loss 关于 input 的偏导数就是 output_grad 标量值
        return torch.full_like(input, fill_value=output_grad.item())
    
    # 如果指定 dim, 那么直接 扩展 output_grad 即可
    return output_grad.unsqueeze(dim).expand_as(input)


def check_sum_backward():
    # 测试没有 dim 参数的情况
    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(1)[0].double()
    
    input.sum().backward(output_grad)
    result = sum_backward(input, output_grad)
    print(is_same_tensor(result, input.grad))

    # 测试有 dim 参数的情况
    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(2, 4).double()
    
    input.sum(dim=1).backward(output_grad)
    result = sum_backward(input, output_grad, dim=1)
    print(is_same_tensor(result, input.grad))


check_sum_backward()

# %% vector: prod 

''' vector: prod operator '''


@torch.no_grad()
def prod_backward(input: Tensor, output_grad: Tensor, dim: int = None) -> Tensor:
    if dim is None:
        # 如果没有指定 dim, 那么 output 标量对于 input 张量的导数就是 output / input
        output = torch.prod(input)
        derivative = output / input
        return derivative * output_grad

    output = torch.prod(input, dim=dim)
    derivative = output.unsqueeze(dim=dim) / input
    return derivative * output_grad.unsqueeze(dim)


def check_prod_backward():
    # 测试没有 dim 参数的情况
    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(1)[0].double()
    
    input.prod().backward(output_grad)
    result = prod_backward(input, output_grad)
    print(is_same_tensor(result, input.grad))

    # 测试有 dim 参数的情况
    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(2, 4).double()
    
    input.prod(dim=1).backward(output_grad)
    result = prod_backward(input, output_grad, dim=1)
    print(is_same_tensor(result, input.grad))


check_prod_backward()
# %% vector: mean 

''' vector: mean operator '''


@torch.no_grad()
def mean_backward(input: Tensor, output_grad: Tensor, dim: int = None) -> Tensor:
    if dim is None:
        num_elements = input.numel()
        return torch.full_like(input, fill_value=output_grad.item()) / num_elements

    num_elements = input.size(dim=dim)
    return output_grad.unsqueeze(dim).expand_as(input) / num_elements


def check_mean_backward():
    # 测试没有 dim 参数的情况
    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(1)[0].double()
    
    input.mean().backward(output_grad)
    result = mean_backward(input, output_grad)
    print(is_same_tensor(result, input.grad))

    # 测试有 dim 参数的情况
    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(2, 4).double()
    
    input.mean(dim=1).backward(output_grad)
    result = mean_backward(input, output_grad, dim=1)
    print(is_same_tensor(result, input.grad))


check_mean_backward()

# %% vector: max 

''' vector: max operator '''


@torch.no_grad()
def max_backward(input: Tensor, output_grad: Tensor, dim: int = None) -> Tensor:
    if dim is None:
        # 在没有指定 dim 参数的情况下, torch.max 仅仅返回最大值
        # 此时, 如果有多个值都是最大值, 此时会平分梯度值 (不知道为什么)
        max_value = torch.max(input)
        mask = input == max_value
        return (mask * output_grad) / mask.sum()

    # 在指定 dim 参数的情况下, torch.max 返回 最大值 和 索引
    # 此时, 输出关于输入 在 索引 处导数为 1, 其余为 0, 即使有相同最大值也不需要取平均值
    max_value, max_idx = torch.max(input, dim=dim)
    input_grad = torch.zeros_like(input)
    input_grad.scatter_(
        dim=dim, index=max_idx.unsqueeze(dim=dim), 
        src=output_grad.unsqueeze(dim=dim)
    ) # scatter_ 和 gather 互为反运算
    return input_grad


def check_max_backward():
    # 测试没有 dim 参数的情况
    input = torch.randn(2, 3, 4).double()
    input[0, 0, 0] = input[0, 0, 1] = input.max()
    input = nn.Parameter(input)
    output_grad = torch.randn(1)[0].double()
    
    input.max().backward(output_grad)
    result = max_backward(input, output_grad)
    print(is_same_tensor(result, input.grad))

    # 测试有 dim 参数的情况
    input = torch.randn(2, 3, 4).double()
    input[0, 0, 0] = input[0, 0, 1] = input[0, 0, :].max()
    input = nn.Parameter(input)
    output_grad = torch.randn(2, 3).double()
    
    input.max(dim=2).values.backward(output_grad)
    result = max_backward(input, output_grad, dim=2)
    print(is_same_tensor(result, input.grad))

    # 测试有 dim 参数的情况
    input = nn.Parameter(torch.randn(2, 3, 4, 5, 6).double())
    output_grad = torch.randn(2, 4, 5, 6).double()
    
    input.max(dim=1).values.backward(output_grad)
    result = max_backward(input, output_grad, dim=1)
    print(is_same_tensor(result, input.grad))


check_max_backward()


# %% vector: var 

''' vector: var operator '''


@torch.no_grad()
def var_backward(input: Tensor, output_grad: Tensor, dim: int = None) -> Tensor:
    if dim is None:
        deviation = input - input.mean()  # 偏差
        num_elements = input.numel()

        # 计算 derivative
        derivative = - deviation.sum() / num_elements + deviation
        derivative = 2. / num_elements * derivative

        input_grad = derivative * output_grad
        return input_grad 

    # 将 dim 维移动到最后一个维度, 方便后续处理
    input = input.movedim(dim, -1)

    num_elements = input.size(-1)
    deviation = input - input.mean(dim=-1, keepdim=True)  # 偏差 (b, n)

    # 计算 derivative
    derivative = torch.eye(num_elements) - (1. / num_elements)
    derivative = torch.sum(derivative * deviation.unsqueeze(-2), dim=-1)
    derivative = 2. / num_elements * derivative

    input_grad = derivative * output_grad.unsqueeze(-1)
    input_grad = input_grad.movedim(-1, dim)

    return input_grad


def check_var_backward():
    # 测试没有 dim 参数的情况
    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(1)[0].double()
    
    torch.var(input, unbiased=False).backward(output_grad)
    result = var_backward(input, output_grad)
    print(is_same_tensor(result, input.grad))

    # 测试有 dim 参数的情况 - vector 
    input = nn.Parameter(torch.randn(10).double())
    output_grad = torch.randn(1)[0].double()
    
    input.var(dim=0, unbiased=False).backward(output_grad)
    result = var_backward(input, output_grad, dim=0)
    print(is_same_tensor(result, input.grad))

    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(2, 3).double()
    
    input.var(dim=2, unbiased=False).backward(output_grad)
    result = var_backward(input, output_grad, dim=2)
    print(is_same_tensor(result, input.grad))

    input = nn.Parameter(torch.randn(2, 3, 4).double())
    output_grad = torch.randn(3, 4).double()
    
    input.var(dim=0, unbiased=False).backward(output_grad)
    result = var_backward(input, output_grad, dim=0)
    print(is_same_tensor(result, input.grad))


check_var_backward()

# %% vector: softmax 

''' vector: softmax '''

@torch.no_grad()
def softmax_backward(input: Tensor, output_grad: Tensor, dim: int) -> Tensor:    
    # ## step1: 将 dim 维度移动到最后
    # input / output / input_grad / output_grad : [n, ]
    input = input.movedim(dim, -1)
    output_grad = output_grad.movedim(dim, -1)

    # ## step2: 求雅可比矩阵: [n, n]
    output = torch.softmax(input, dim=-1)
    num_elements = input.size(-1)
    jacobian = (torch.eye(num_elements) - output.unsqueeze(-2)) * output.unsqueeze(-1)
    
    # ## step3: 求 input_grad: [n, n] @ [n, 1]
    input_grad = torch.matmul(jacobian.transpose(-1, -2), output_grad.unsqueeze(-1))
    input_grad = input_grad.squeeze(-1).movedim(-1, dim)
    return input_grad


def check_softmax_backward():
    input = nn.Parameter(torch.randn(5).double())
    output_grad = torch.randn_like(input).double()
    
    torch.softmax(input, dim=0).backward(output_grad)
    result = softmax_backward(input, output_grad, dim=0)
    print("向量测试:", is_same_tensor(input.grad, result))

    for dim in range(5):
        input = nn.Parameter(torch.randn(3, 4, 5, 6, 7).double())
        output_grad = torch.randn_like(input).double()
        
        torch.softmax(input, dim=dim).backward(output_grad)
        result = softmax_backward(input, output_grad, dim=dim)
        print(f"dim={dim}测试:", is_same_tensor(input.grad, result))


check_softmax_backward()

# %% vector: cumsum

''' vector: cumsum '''


@torch.no_grad()
def cumsum_backward(input: Tensor, output_grad: Tensor, dim: int) -> Tensor:    
    # ## step1: 将 dim 维度移动到最后
    # input / output / input_grad / output_grad : [n, ]
    input = input.movedim(dim, -1)
    output_grad = output_grad.movedim(dim, -1)

    # ## step2: 求雅可比矩阵: [n, n]
    num_elements = input.size(-1)
    jacobian = torch.ones(num_elements, num_elements).tril().to(input.dtype)
    
    # ## step3: 求 input_grad: [n, n] @ [n, 1]
    input_grad = torch.matmul(jacobian.transpose(-1, -2), output_grad.unsqueeze(-1))
    input_grad = input_grad.squeeze(-1).movedim(-1, dim)
    return input_grad


def check_cumsum_backward():
    input = nn.Parameter(torch.randn(5).double())
    output_grad = torch.randn_like(input).double()
    
    torch.cumsum(input, dim=0).backward(output_grad)
    result = cumsum_backward(input, output_grad, dim=0)
    print("cumsum 向量测试:", is_same_tensor(input.grad, result))

    for dim in range(5):
        input = nn.Parameter(torch.randn(3, 4, 5, 6, 7).double())
        output_grad = torch.randn_like(input).double()
        
        torch.cumsum(input, dim=dim).backward(output_grad)
        result = cumsum_backward(input, output_grad, dim=dim)
        print(f"cumsum dim={dim}测试:", is_same_tensor(input.grad, result))


check_cumsum_backward()

# %% vector: cumprod 

''' vector: cumprod '''


@torch.no_grad()
def cumprod_backward(input: Tensor, output_grad: Tensor, dim: int) -> Tensor:    
    # ## step1: 将 dim 维度移动到最后
    # input / output / input_grad / output_grad : [n, ]
    input = input.movedim(dim, -1)
    output_grad = output_grad.movedim(dim, -1)

    # ## step2: 求雅可比矩阵: [n, n]
    output = torch.cumprod(input, dim=-1)
    jacobian = torch.tril(output.unsqueeze(-1) / input.unsqueeze(-2))
    
    # ## step3: 求 input_grad: [n, n] @ [n, 1]
    input_grad = torch.matmul(jacobian.transpose(-1, -2), output_grad.unsqueeze(-1))
    input_grad = input_grad.squeeze(-1).movedim(-1, dim)
    return input_grad


def check_cumprod_backward():
    input = nn.Parameter(torch.randn(5).double())
    output_grad = torch.randn_like(input).double()
    
    torch.cumprod(input, dim=0).backward(output_grad)
    result = cumprod_backward(input, output_grad, dim=0)
    print("cumprod 向量测试:", is_same_tensor(input.grad, result))

    for dim in range(5):
        input = nn.Parameter(torch.randn(3, 4, 5, 6, 7).double())
        output_grad = torch.randn_like(input).double()
        
        torch.cumprod(input, dim=dim).backward(output_grad)
        result = cumprod_backward(input, output_grad, dim=dim)
        print(f"cumprod dim={dim}测试:", is_same_tensor(input.grad, result))


check_cumprod_backward()

# %% other: bmm 

@torch.no_grad()
def bmm_backward(input1, input2, output_grad):
    input1_grad = torch.bmm(output_grad, input2.transpose(1, 2))
    input2_grad = torch.bmm(input1.transpose(1, 2), output_grad)
    return input1_grad, input2_grad


def check_bmm_backward():
    input1 = nn.Parameter(torch.randn(3, 4, 5).double())
    input2 = nn.Parameter(torch.randn(3, 5, 6).double())
    output_grad = torch.randn(3, 4, 6).double()
    
    torch.bmm(input1, input2).backward(output_grad)
    result1, result2 = bmm_backward(input1, input2, output_grad)
    
    print(is_same_tensor(result1, input1.grad))
    print(is_same_tensor(result2, input2.grad))


check_bmm_backward()


# %% other: matmul

''' other: matmul '''


@torch.no_grad()
def matmul_backward(input1: Tensor, input2: Tensor, output_grad: Tensor) -> tuple[Tensor, Tensor]:
    if input1.ndim == 1 and input2.ndim == 1:  # [n, ] @ [n, ] ==> []
        # ## case1: 向量点乘: 多对一运算
        # output 对 input1 和 input2 的导数是 input2 和 input1
        input1_grad = output_grad * input2
        input2_grad = output_grad * input1
    
    elif input2.ndim == 1:  # [b, n] @ [n, ] ==> [b, ]
        # ## case2: 矩阵-向量乘法 (需要考虑 广播机制)
        # #  step1: 转换成标准样式
        matrix = torch.flatten(input1, start_dim=0, end_dim=-2)
        vector = input2
        output_grad = torch.flatten(output_grad)
        # #  step2: 求 向量 的梯度, 属于 多对多运算, 此时的雅可比矩阵就是 matrix 
        vector_grad = torch.matmul(matrix.transpose(-1, -2), output_grad)  # [n, b] @ [b, ] ==> [n, ]
        # #  step3: 求 矩阵 的梯度, 属于 多对一运算, 那么方式和 向量点乘 是一致的
        matrix_grad = output_grad.unsqueeze(-1) * vector.unsqueeze(-2)
        # #  step4: 转换回原来的样式
        input1_grad = matrix_grad.reshape_as(input1)
        input2_grad = vector_grad
    
    elif input1.ndim == 1:  # [b, ] @ [b, n] ==> [n, ]
        # ## case3: 向量-矩阵乘法 (不需要考虑 广播机制)
        input1_grad = torch.matmul(input2, output_grad)  # [b, n] @ [n, ] ==> [b, ]
        input2_grad = output_grad.unsqueeze(-2) * input1.unsqueeze(-1)
    
    else:  # [b, n] @ [n, s] ==> [b, s]
        # ## case4: 矩阵-矩阵乘法 (需要考虑 广播机制)
        # #  step1: 求 右矩阵 的梯度
        # 对于 右矩阵 来说, 是 列向量 的多对多运算, 每一个 列向量 的雅可比矩阵是 左矩阵
        # VJP 过程就是 左矩阵的转置 乘以 输出列向量, 拼在一起就是 左矩阵的转置 乘以 输出矩阵
        input2_grad = torch.matmul(input1.transpose(-1, -2), output_grad)  # [n, b] @ [b, s] ==> [n, s]
        for _ in range(input2_grad.ndim - input2.ndim):
            input2_grad = input2_grad.sum(dim=0)
        for i, (r, s) in enumerate(zip(input2_grad.shape[:-2], input2.shape[:-2])):
            input2_grad = input2_grad.sum(dim=i, keepdim=True) if r != s else input2_grad
        # #  step2: 求 左矩阵 的梯度
        # 此时 可以把 左矩阵 变成右矩阵: [s, n] @ [n, b] ==> [s, b]; C = A @ B ==> C^T = B^T @ A^T
        input1_grad = torch.matmul(output_grad, input2.transpose(-1, -2))  # [b, s] @ [s, n] ==> [b, n]
        for _ in range(input1_grad.ndim - input1.ndim):
            input1_grad = input1_grad.sum(dim=0)
        for i, (r, s) in enumerate(zip(input1_grad.shape[:-2], input1.shape[:-2])):
            input1_grad = input1_grad.sum(dim=i, keepdim=True) if r != s else input1_grad

    return input1_grad, input2_grad
    


def check_matmul_backward():
    print("测试 向量点乘 的正确性")
    input1 = nn.Parameter(torch.randn(10).double())
    input2 = nn.Parameter(torch.randn(10).double())
    output_grad = torch.randn(1)[0].double()
    
    torch.matmul(input1, input2).backward(output_grad)
    result1, result2 = matmul_backward(input1, input2, output_grad)
    print(is_same_tensor(result1, input1.grad))
    print(is_same_tensor(result2, input2.grad)) 

    print("测试 矩阵-向量 乘法的正确性")
    input1 = nn.Parameter(torch.randn(2, 3, 10).double())
    input2 = nn.Parameter(torch.randn(10).double())
    output_grad = torch.randn(2, 3).double()
    
    torch.matmul(input1, input2).backward(output_grad)
    result1, result2 = matmul_backward(input1, input2, output_grad)
    print(is_same_tensor(result1, input1.grad))
    print(is_same_tensor(result2, input2.grad)) 

    print("测试 向量-矩阵 乘法的正确性")
    input1 = nn.Parameter(torch.randn(100).double())
    input2 = nn.Parameter(torch.randn(100, 20).double())
    output_grad = torch.randn(20).double()
    
    torch.matmul(input1, input2).backward(output_grad)
    result1, result2 = matmul_backward(input1, input2, output_grad)
    print(is_same_tensor(result1, input1.grad))
    print(is_same_tensor(result2, input2.grad)) 

    print("测试 矩阵-矩阵 乘法的正确性")
    input1 = nn.Parameter(torch.randn(2, 1, 2, 5).double())
    input2 = nn.Parameter(torch.randn(6, 5, 4).double())
    output_grad = torch.randn(2, 6, 2, 4).double()
    
    torch.matmul(input1, input2).backward(output_grad)
    result1, result2 = matmul_backward(input1, input2, output_grad)
    print(result1.shape, input1.grad.shape)
    print(result2.shape, input2.grad.shape)
    print(is_same_tensor(result1, input1.grad))
    print(is_same_tensor(result2, input2.grad)) 


check_matmul_backward()

# %%
