# -*- coding:utf-8 -*-
# Author: lqxu

#%% pre-perpare
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

#%% 正类 loss 函数

def plot_pos_loss_func():
    x = np.linspace(start=-5, stop=5, num=10000)
    
    def loss_func_hard_version(scores):
        scores = np.stack([-x, np.zeros_like(x)])
        return np.max(scores, axis=0)
    
    def loss_func_soft_version(scores):
        probs = 1. / (1. + np.exp(-scores))
        return -np.log(probs)
    
    plt.plot(x, loss_func_hard_version(x), label="hard version")
    plt.plot(x, loss_func_soft_version(x), label="soft version")
    plt.legend()
    
    plt.xlabel("线性函数值")
    plt.ylabel("损失值")
    plt.title("逻辑回归中正类的目标函数")
    
    plt.show()


plot_pos_loss_func()

#%% 负类 loss 函数

def plot_neg_loss_func():
    x = np.linspace(start=-5, stop=5, num=10000)
    
    def loss_func_hard_version(scores):
        scores = np.stack([x, np.zeros_like(x)])
        return np.max(scores, axis=0)
    
    def loss_func_soft_version(scores):
        probs = 1. / (1. + np.exp(-scores))
        return -np.log(1 - probs)
    
    plt.plot(x, loss_func_hard_version(x), label="hard version")
    plt.plot(x, loss_func_soft_version(x), label="soft version")
    plt.legend()
    
    plt.xlabel("线性函数值")
    plt.ylabel("损失值")
    plt.title("逻辑回归中负类的目标函数")
    
    plt.show()


plot_neg_loss_func()

# %% perpare data

# iris 是鸢尾花分类的数据集
# 其输入有四个特征: 花萼 (sepal) 长和宽; 花瓣 (petal) 长和宽
# 其有四个类别: setosa 山鸢尾花; versicolor 变色鸢尾花; virginica 弗吉尼亚鸢尾花

iris = datasets.load_iris()

print(iris.keys())

# 这里, 我们只用 花瓣 (petal) 的长和宽

inputs = iris["data"][:, -2:]
labels = iris["target"]

#%% 二分类的决策边界

def plot_binary_decision_boundary():

    plt.scatter(
        x=inputs[labels == 0][:, 0], y=inputs[labels == 0][:, 1], 
        c="red", label="是山鸢尾花"
    )
    plt.scatter(
        x=inputs[labels != 0][:, 0], y=inputs[labels != 0][:, 1], 
        c="green", label="不是山鸢尾花"
    )
    plt.xlabel("花瓣长度")
    plt.ylabel("花瓣宽度")

    model = LogisticRegression(penalty=None, multi_class="ovr")
    model.fit(inputs, labels == 0)
    a, b = model.coef_[0]
    c = model.intercept_[0]
    
    x = np.linspace(start=0, stop=8, num=10000)
    y = (-a / b) * x + (-c / b)
    plt.plot(x, y, c="blue", label="决策边界")
    
    plt.legend()
    plt.show()


plot_binary_decision_boundary()


# %%


def plot_multi_decision_boundary():

    plt.scatter(
        x=inputs[labels == 0][:, 0], y=inputs[labels == 0][:, 1], 
        c="red", label="山鸢尾花"
    )
    plt.scatter(
        x=inputs[labels == 1][:, 0], y=inputs[labels == 1][:, 1], 
        c="green", label="变色鸢尾花"
    )
    plt.scatter(
        x=inputs[labels == 2][:, 0], y=inputs[labels == 2][:, 1], 
        c="blue", label="弗吉尼亚鸢尾花"
    )
    plt.xlabel("花瓣长度")
    plt.ylabel("花瓣宽度")
    
    model = LogisticRegression(penalty=None, multi_class="multinomial")
    model.fit(inputs, labels)
    
    a0, b0 = model.coef_[0]
    c0 = model.intercept_[0]

    a1, b1 = model.coef_[1]
    c1 = model.intercept_[1]

    a2, b2 = model.coef_[2]
    c2 = model.intercept_[2]
    
    x = np.linspace(start=1, stop=7, num=10000)
    plt.plot(x, (a0-a1) / (b1-b0) * x + (c0-c1) / (b1-b0), c="orange", label="红绿点决策边界")
    plt.plot(x, (a0-a2) / (b2-b0) * x + (c0-c2) / (b2-b0), c="purple", label="红蓝点决策边界")
    plt.plot(x, (a1-a2) / (b2-b1) * x + (c1-c2) / (b2-b1), c="yellow", label="蓝绿点决策边界")
    
    plt.legend()
    plt.show()


plot_multi_decision_boundary()

#%%

def plot_decision_boundary(model, axis):
    
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
