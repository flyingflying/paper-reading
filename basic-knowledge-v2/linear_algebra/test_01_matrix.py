# -*- coding:utf-8 -*-
# Author: lqxu

#%%

import numpy as np 
import matplotlib.pyplot as plt 

#%%

plt.figure(figsize=(8, 4))

x_values = np.linspace(start=-5, stop=5, num=10000)

matrix = np.array([[1.2, -1.6], [0.8, 0.6]])

for y_value in range(-5, 6, 1):

    points = np.stack([
        x_values,
        np.full_like(x_values, fill_value=y_value)
    ])

    plt.subplot(1, 2, 1)
    plt.plot(points[0], points[1], "r:", )
    
    points = matrix @ points
    plt.subplot(1, 2, 2)
    plt.plot(points[0], points[1], "r:", )

y_values = np.linspace(start=-5, stop=5, num=10000)

for x_value in range(-5, 6, 1):

    points = np.stack([
        np.full_like(y_values, fill_value=x_value),
        y_values
    ])

    plt.subplot(1, 2, 1)
    plt.plot(points[0], points[1], "r:", )
    
    points = matrix @ points
    plt.subplot(1, 2, 2)
    plt.plot(points[0], points[1], "r:", )

plt.subplot(1, 2, 2)
plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.show()

# %%


def simple_inv(matrix: np.ndarray):
    
    if matrix.ndim != 2:
        raise ValueError("传入的必须是矩阵")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("传入的必须是方阵")
    
    # step1: 求取每一个行向量的模长, 结果是 [n, 1]
    row_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    # step2: 转换成基向量
    inv_matirx = matrix / (row_norm ** 2)
    # step3: 转置
    inv_matirx = inv_matirx.T
    
    return inv_matirx


def is_same_tensor(t1: np.ndarray, t2: np.ndarray, eps: float = 1e-8) -> bool:
    return np.all(np.abs(t1 - t2) < eps)


def check(matrix):
    return is_same_tensor(
        simple_inv(matrix), np.linalg.inv(matrix)
    )


case_1 = np.array([[0.6, 0.8], [-0.8, 0.6]])
assert check(case_1)
print(simple_inv(case_1))

case_2 = np.array([[1, 2], [3, 4]])
print(simple_inv(case_2))
print(np.linalg.inv(case_2))
assert check(case_2)

#%%

np.set_printoptions(precision=4, suppress=True)

A = np.array([
    [1, 2, 5],
    [2, 2, -4],
    [3, -2, 1]
])

print(A[:, 0] @ A[:, 1], A[:, 1] @ A[:, 2], A[:, 0] @ A[:, 2])

print(A[0] @ A[1], A[1] @ A[2], A[0] @ A[2])

print(A.T @ A)

A = A / np.linalg.norm(A, axis=0, keepdims=True)

print(A[:, 0] @ A[:, 1], A[:, 1] @ A[:, 2], A[:, 0] @ A[:, 2])

print(A[0] @ A[1], A[1] @ A[2], A[0] @ A[2])

print(A.T @ A)

# %%
