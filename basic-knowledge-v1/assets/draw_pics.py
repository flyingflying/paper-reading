# -*- coding:utf-8 -*-
# Author: lqxu

import numpy as np
import matplotlib.pyplot as plt  # noqa


def unit_step_and_sigmoid_function():

    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    # 画 sigmoid 函数
    a = np.linspace(-8, 8, 200)
    plt.plot(a, sigmoid(a))
    # 画 unit step 函数
    plt.scatter([0, 0], [1, 0], c="w", s=40, marker="o", edgecolors="r")
    plt.scatter([0, ], [0.5, ], c="r", s=40, marker="o", edgecolors="r")
    plt.plot(np.linspace(-8, -0.20, 100), [0., ] * 100, c="r")
    plt.plot(np.linspace(0.20, 8, 100), [1., ] * 100, c="r")
    plt.plot([0., ] * 100, np.linspace(0.02, 0.98, 100), c="r", linestyle="dashed")
    plt.show()


def focal_loss_weight_function():

    def weight_func(x, gamma):
        return (1 - x) ** gamma

    a = np.linspace(0, 1, 500)
    gamma_list = [0., 0.5, 1., 2., 5.]

    for gamma_ in gamma_list:
        plt.plot(a, weight_func(a, gamma_), label=f"gamma={gamma_}")

    plt.legend(loc="best")
    plt.xlabel("probability of target class")
    plt.ylabel("weight")
    plt.title("weight of focal loss")
    plt.show()
