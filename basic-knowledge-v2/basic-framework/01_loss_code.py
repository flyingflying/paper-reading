# -*- coding:utf-8 -*-
# Author: lqxu

#%%

import torch 
from torch import nn 
from torch.nn import functional as F

#%% 测试例子

theta = nn.Parameter(torch.tensor(5.))
x, y = torch.tensor(2.), torch.tensor(9.)

y_hat = theta * x
loss = (y - y_hat) ** 2

loss.backward()
theta.grad

#%% 测试 L2 Loss

y_hat = nn.Parameter(torch.tensor(10.))
y = torch.tensor(9.)

loss = nn.MSELoss()(y_hat, y)
loss.backward()

print(loss)
print(y_hat.grad)

#%% 测试 L1 Loss

y_hat = nn.Parameter(torch.tensor(5.))
y = torch.tensor(10.)

loss = nn.L1Loss()(y_hat, y)
loss.backward()

print(loss)
print(y_hat.grad)

#%% 测试逻辑回归 loss 

print("测试正类的梯度:")

logit = torch.tensor(0.5, requires_grad=True)
loss = nn.BCEWithLogitsLoss()(logit, torch.tensor(1).float())
loss.backward()

print(loss)
print(logit.grad)

with torch.no_grad():
    print(torch.sigmoid(logit) - 1)

print("测试负类的梯度:")

logit = torch.tensor(0.5, requires_grad=True)
loss = nn.BCEWithLogitsLoss()(logit, torch.tensor(0).float())
loss.backward()

print(loss)
print(logit.grad)

with torch.no_grad():
    print(torch.sigmoid(logit))

#%% 测试 softmax 回归

logit = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
target = torch.tensor(0).long()
loss = nn.CrossEntropyLoss()(logit.unsqueeze(0), target.unsqueeze(0))
loss.backward()

print(loss)
print(logit.grad)

with torch.no_grad():
    print(torch.softmax(logit, dim=0) - F.one_hot(target, logit.size(-1)))

#%% 测试 softmax 求导

logit = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
probs = torch.softmax(logit, dim=0)
probs[0].backward()
print(logit.grad)

logit = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
probs = torch.softmax(logit, dim=0)
probs[1].backward()
print(logit.grad)

logit = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
probs = torch.softmax(logit, dim=0)
probs[2].backward()
print(logit.grad)

logit = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
probs = torch.softmax(logit, dim=0)
probs.backward(torch.ones_like(logit))
print(logit.grad)

with torch.no_grad():
    result = 0
    for i in range(3):
        temp = probs * (F.one_hot(torch.tensor(i), logit.size(-1)) - probs[i])
        print(temp)
        result += temp 
    print(result)

#%%


def softmax_grad(x, dy):
    
    output = torch.softmax(x, dim=0)
    
    jacobian_matrix = (torch.eye(x.size(0)) - output) * output.unsqueeze(-1)
    
    return jacobian_matrix.T @ dy


logits = torch.randn(10, requires_grad=True)
grad = torch.randn(10)

torch.softmax(logits, dim=0).backward(grad)
print(logits.grad)

with torch.no_grad():
    print(softmax_grad(logits, grad))

# %%

import matplotlib.pyplot as plt 

layer_1 = torch.nn.Linear(1, 10)
layer_2 = torch.nn.Linear(10, 20)
layer_3 = torch.nn.Linear(20, 10)
layer_4 = torch.nn.Linear(10, 1)

with torch.no_grad():
    input = torch.linspace(-1, 1, 1000).unsqueeze(-1)
    output = layer_4(layer_3(layer_2(layer_1(input)))).squeeze()

    plt.plot(input.numpy(), output.numpy())

#%%

act_func = nn.ReLU()

with torch.no_grad():
    input = torch.linspace(-1, 1, 1000).unsqueeze(-1)
    output = act_func(layer_1(input))
    output = act_func(layer_2(output))
    output = act_func(layer_3(output))
    output = layer_4(output).flatten()

    plt.plot(input.numpy(), output.numpy())

# %%

torch.manual_seed(4)  # 4 / 11 / 14

act_func = nn.Sigmoid()

layer_1 = torch.nn.Linear(1, 10)
layer_2 = torch.nn.Linear(10, 20)
layer_3 = torch.nn.Linear(20, 10)
layer_4 = torch.nn.Linear(10, 1)

with torch.no_grad():
    input = torch.linspace(-1, 1, 1000).unsqueeze(-1)
    output = act_func(layer_1(input))
    output = act_func(layer_2(output))
    output = act_func(layer_3(output))
    output = layer_4(output).flatten()

    plt.plot(input.numpy(), output.numpy())

# %%
