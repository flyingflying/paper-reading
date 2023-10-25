# -*- coding:utf-8 -*-
# Author: lqxu

#%%
# import math 
# import numpy as np 


# def get_rotary_matrix(radian):
#     return np.array([
#         [math.cos(radian), -math.sin(radian)],
#         [math.sin(radian), math.cos(radian)], 
#     ])


# def is_same_array(a1, a2, eps=1e-8):
#     return np.all(np.abs(a1 - a2) < eps)


# print(is_same_array(
#     get_rotary_matrix(5).T @ get_rotary_matrix(7), 
#     get_rotary_matrix(7 - 5)
# ))

# print(is_same_array(
#     get_rotary_matrix(5).T @ get_rotary_matrix(5),
#     np.identity(2)
# ))

# x_vec = np.random.randn(2)
# y_vec = np.random.randn(2)

# print(is_same_array(
#     (get_rotary_matrix(5) @ x_vec).T @ (get_rotary_matrix(7) @ y_vec),
#     x_vec @ (get_rotary_matrix(7 - 5) @ y_vec)
# ))

# result = x_vec[0] * y_vec[1] * math.sin(5 - 7) + x_vec[1] * y_vec[0] * math.sin(7 - 5) \
#        + x_vec[0] * y_vec[0] * math.cos(5 - 7) + x_vec[1] * y_vec[1] * math.cos(7 - 5)

# print(is_same_array(
#     (get_rotary_matrix(5) @ x_vec).T @ (get_rotary_matrix(7) @ y_vec),
#     result
# ))


# #%%

# import matplotlib.pyplot as plt 

# debug_mode = False
# max_num_tokens = 10000000 # 1_000_000
# head_size = 2 if debug_mode else 128

# # q_vec = np.random.randn(head_size)
# # k_vec = np.random.randn(head_size)
# np.random.seed(61)
# q_vec = np.random.rand(head_size)
# k_vec = np.random.rand(head_size)
# # q_vec = np.full(head_size, 1 / head_size)
# # k_vec = np.full(head_size, 1 / head_size)

# theta = 10000 ** (-np.arange(0, head_size, 2) / head_size)
# theta = theta.reshape(-1, 1).repeat(2, axis=1).flatten()

# def rotary(vec, pos):
#     vec_2 = np.array(vec).reshape(-1, 2)[:, ::-1]
#     vec_2[:, 0] = -vec_2[:, 0]
#     vec_2 = vec_2.flatten()

#     return vec * np.cos(pos * theta) + vec_2 * np.sin(pos * theta)


# if debug_mode:
#     print(is_same_array(
#         get_rotary_matrix(4) @ q_vec,
#         rotary(q_vec, 4)
#     ))

# else:
#     q_result = rotary(q_vec, 8192)
#     k_results = [rotary(k_vec, i) for i in range(max_num_tokens)]
#     k_results = np.stack(k_results)
#     results = (k_results @ q_result)
#     plt.plot(np.arange(max_num_tokens), results)
    
#     print(np.argmax(results))

# %%

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.axes import Axes


def create_sin_cos_cache(max_num_tokens, head_size):
    theta = 10000 ** (-np.arange(0, head_size, 2) / head_size)
    theta = theta.reshape(-1, 1).repeat(2, axis=1).flatten()
    
    pos = np.arange(0, max_num_tokens)
    table = pos.reshape(-1, 1) @ theta.reshape(1, -1)  # [max_num_tokens, head_size]
    
    sin_cache = np.sin(table)
    sin_cache[:, ::2] = -sin_cache[:, ::2]
    
    cos_cache = np.cos(table)
    return sin_cache, cos_cache


def rotate_half(vec):
    return vec.reshape(-1, 2)[:, ::-1].flatten()


def rotary(vec, pos, sin_table, cos_table):
    return vec * cos_table[pos] + rotate_half(vec) * sin_table[pos]


def plot(plt_obj: Axes, pic_index, query_index=0, head_size=256, max_num_tokens=8192, step=1):
    q_vec = np.ones(head_size)
    k_vec = np.ones(head_size)
    sin_table, cos_table = create_sin_cos_cache(max_num_tokens, head_size)

    rotated_q_vec = rotary(q_vec, query_index, sin_table, cos_table)
    k_indices = np.arange(0, max_num_tokens, step)
    rotated_k_vecs = rotary(k_vec, k_indices, sin_table, cos_table)
    attn_scores = (rotated_k_vecs @ rotated_q_vec) / np.sqrt(head_size)

    plt_obj.plot(k_indices, attn_scores)
    plt_obj.set_title(f"Figure {pic_index}: query_index={query_index}, head_size={head_size}")
    plt_obj.set_xlabel("key index")
    plt_obj.set_ylabel("attention score")
    

plt.rcParams.update({
    "font.sans-serif": ["Times New Roman", ],
    "font.size": 10
})

_, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
plot(axes[0, 0], 1, query_index=0, max_num_tokens=512)
plot(axes[0, 1], 2, query_index=256, max_num_tokens=512)
plot(axes[1, 0], 3, query_index=0, max_num_tokens=65535)
plot(axes[1, 1], 4, query_index=0, head_size=8, max_num_tokens=65535)

# %%
