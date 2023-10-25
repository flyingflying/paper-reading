# -*- coding:utf-8 -*-
# Author: lqxu

#%%

import math 
import torch 


def is_same_tensor(t1, t2, eps=1e-8) -> bool:
    return torch.all(torch.abs(t1 - t2) < eps).item()

#%% 测试 additive attention 两种形式的一致性

hidden_size = 768

query = torch.randn(hidden_size, dtype=torch.float64)
key = torch.randn(hidden_size, dtype=torch.float64)

weight_query = torch.randn(hidden_size, hidden_size, dtype=torch.float64)
weight_key = torch.randn(hidden_size, hidden_size, dtype=torch.float64)

r1 = weight_query @ query + weight_key @ key
r2 = torch.cat([weight_query, weight_key], dim=1) @ torch.cat([query, key])

torch.all(torch.abs(r1 - r2) < 1e-8)

# %% 测试 正弦波位置编码 的 远程衰减 性质

import seaborn as sns

def create_table(num_tokens: int = 512, hidden_size: int = 768):
    d = torch.arange(start=0, end=hidden_size, step=2).float()
    theta_d = 10000 ** (-d / hidden_size)
    # theta_d = torch.exp(-d / hidden_size * math.log(10000.))  # 数值稳定的写法
    
    idx = torch.arange(0, 512).float()
    
    table = torch.zeros(num_tokens, hidden_size)
    table[:, 0::2] = torch.sin(idx.unsqueeze(1) @ theta_d.unsqueeze(0))
    table[:, 1::2] = torch.cos(idx.unsqueeze(1) @ theta_d.unsqueeze(0))
    
    return table


pos_vectors = create_table()

sns.heatmap(pos_vectors @ pos_vectors.T)

# %% 验证: 向量经过旋转变换后, 点积结果不变

def rotary(vector, theta):
    transform = torch.Tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])
    
    return transform @ vector

query = torch.randn(2)
key = torch.randn(2)

# 验证正确性 (旋转 180 度)
print(query, rotary(query, math.pi))

# 验证点积的不变性
print(query @ key, rotary(query, 1) @ rotary(key, 1))

# 验证化简的正确性
m, n = 100, 98
r1 = query[0] * key[1] * math.sin(m - n) + query[1] * key[0] * math.sin(n - m) \
   + query[0] * key[0] * math.cos(m - n) + query[1] * key[1] * math.cos(n - m)
r2 = rotary(query, m) @ rotary(key, n)
print(r1, r2)

# %% 测试 Efficient Attention: Attention with Linear Complexities

num_tokens_query, num_tokens_key, hidden_size = 16, 20, 768

query_matrix = torch.softmax(
    torch.randn(num_tokens_query, hidden_size),  # / math.sqrt(hidden_size),
    dim=1
)  # 每一个词向量的和为 1
key_matrix = torch.softmax(
    torch.randn(num_tokens_key, hidden_size),  # / math.sqrt(hidden_size),
    dim=0
)

attn_scores = query_matrix @ key_matrix.T  # [num_tokens_query, num_tokens_key]

print(attn_scores.sum(dim=1))

# torch.softmax(torch.matmul(query_matrix, key_matrix.transpose(-1, -2)), dim=-1).sum(dim=1)

# %% 测试线性 attention

num_tokens_query, num_tokens_key, hidden_size = 512, 512, 20
dtype = torch.float32

query_matrix = torch.rand(num_tokens_query, hidden_size, dtype=dtype)
key_matrix = torch.rand(num_tokens_key, hidden_size, dtype=dtype)
value_matrix= torch.randn(num_tokens_key, hidden_size, dtype=dtype)


def normal_attn(query_matrix, key_matrix, value_matrix):
    # 726 µs ± 3.95 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    attn_scores = query_matrix @ key_matrix.T
    attn_probs = attn_scores / attn_scores.sum(axis=1, keepdim=True)
    output_matrix = attn_probs @ value_matrix
    return output_matrix


def linear_attn(query_matrix, key_matrix, value_matrix):
    # 73.4 µs ± 896 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    z = query_matrix @ key_matrix.sum(dim=0, keepdims=True).T
    output_matrix = query_matrix @ (key_matrix.T @ value_matrix) / z
    return output_matrix


print(is_same_tensor(
    normal_attn(query_matrix, key_matrix, value_matrix), 
    linear_attn(query_matrix, key_matrix, value_matrix), 
    eps=1e-6
))

#%%

def focus_attn(vector, p=3):
    return torch.norm(vector) * torch.nn.functional.normalize(torch.pow(vector, p), dim=0)

torch.manual_seed(0)

case_vec1 = torch.rand(10)
case_vec2 = torch.rand(10)

print(torch.argmax(case_vec1), torch.argmax(case_vec2))
print(focus_attn(case_vec1, p=100) @ focus_attn(case_vec2, p=100))
print(case_vec1 @ case_vec2)

# torch.manual_seed(897932153352800)

# case_vec1 = torch.rand(10)
# case_vec2 = torch.rand(10)

# print(torch.argmax(case_vec1), torch.argmax(case_vec2))
# print(focus_attn(case_vec1) @ focus_attn(case_vec2))
# print(case_vec1 @ case_vec2)

# %%

case = torch.randn(128, 128, dtype=torch.float64)

u_matrix, sigma_vector, v_matrix = torch.linalg.svd(case)

print("u_matrix 的 shape 是:", u_matrix.shape)
print("sigma_vector 的 shape 是:", sigma_vector.shape)
print("v_matrix 的 shape 是:", v_matrix.shape)

sigma_matrix = torch.diag(sigma_vector)

print("测试 SVD 分解的正确性", is_same_tensor(
    u_matrix @ sigma_matrix @ v_matrix,
    case
))

new_case = u_matrix[:, :50] @ sigma_matrix[:50, :50] @ v_matrix[:50, :]

print(
    "case 矩阵的 rank 是:", torch.linalg.matrix_rank(case),
    "\nsvd 压缩后矩阵的 rank 是:", torch.linalg.matrix_rank(new_case)
)

new_case = torch.exp(new_case)

print(torch.linalg.matrix_rank(new_case))

print(
    "进过转换函数转换后矩阵的 rank 是:", torch.linalg.matrix_rank(case)
)

sigma_cum_weights = torch.cumsum(sigma_vector / sigma_vector.sum(), dim=0)

print("case 矩阵 SVD 分解后前 50 个维度 奇异值 占比:", sigma_cum_weights[50])

_, new_sigma_vector, _ = torch.linalg.svd(new_case)
new_sigma_cum_weights = torch.cumsum(new_sigma_vector / new_sigma_vector.sum(), dim=0)

print("new case 矩阵 SVD 分解后前 50 个维度 奇异值 之和占比:", new_sigma_cum_weights[50])

# %%

case = torch.tensor([
    [0., 0., 1.],
    [0., 1., 0.],
    [1., 0., 0.]
])

print(torch.linalg.matrix_rank(case))

_, sigma_vector, _ = torch.linalg.svd(case)

print(sigma_vector)

case[2] = (case[0] + case[1]) + 1e-2

print(torch.linalg.matrix_rank(case))

_, sigma_vector, _ = torch.linalg.svd(case)

print(sigma_vector)

# %%

a = torch.randn(100, 30, dtype=torch.float64)
b = torch.randn(30, 100, dtype=torch.float64)

c = a @ b
_, sigma_vector, _ = torch.linalg.svd(c)

print(torch.linalg.matrix_rank(c))
print(torch.sum(sigma_vector.abs() > 1e-8))

# %%

from einops import repeat

a = torch.randn(10, 10)

repeat(a, "batch_size num_tokens -> batch_size num_heads num_tokens", num_heads=5).shape

# %%
