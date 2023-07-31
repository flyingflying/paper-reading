# -*- coding:utf-8 -*-
# Author: lqxu
# reference: https://github.com/liuyubobobo/Play-with-Machine-Learning-Algorithms/tree/master/07-PCA-and-Gradient-Ascent 

#%%

from tqdm import tqdm

import numpy as np 
from numpy import ndarray

import torch 
from torch import nn, Tensor

#%%

class PCA:
    def __init__(self, n_components: int, lr: float = 0.001, n_epoches: int = 1000):
        self.lr = lr
        self.n_epoches = n_epoches
        self.n_components = n_components

        self.mean_ = None
        self.components_ = None
    
    def _get_first_component(self, input: Tensor):
        """ 使用梯度上升法求解第一主成分 """
        # input: [n_samples, n_features]
        n_features = input.size(1)
        
        param = nn.Parameter(torch.rand(n_features, ))
        optimizer = torch.optim.SGD([param, ], self.lr, maximize=True)
        
        for _ in range(self.n_epoches):
            optimizer.zero_grad()
            
            # part1: 前向传播
            output = input.matmul(param)
            variance = (output ** 2).mean()
            
            # part2: 反向传播
            variance.backward()
            
            # part3: 更新参数
            optimizer.step()
            param.data /= torch.linalg.norm(param.data)  # 强制单位向量
        
        return param.data
    
    def fit(self, samples: ndarray):
        # samples: [n_samples, n_features]
        assert samples.ndim == 2
        
        n_features = samples.shape[1]
        assert n_features >= self.n_components

        self.mean_ = samples.mean(axis=0)
        samples = samples - self.mean_

        t_samples: Tensor = torch.from_numpy(samples.astype(np.float32))  # 转换为 tensor
        
        components = []
        
        for _ in range(self.n_components):
            component = self._get_first_component(t_samples)
            components.append(component)
            
            # samples @ component 得到的结果是每一个样本在 component 上的坐标值
            # 这个坐标值在乘以 component 即可以得到投影向量
            samples_projection = t_samples.matmul(component).reshape(-1, 1).mul(component)
            t_samples = t_samples - samples_projection  # 和 component 应该是相互垂直的
        
        self.components_ = torch.stack(components).numpy()  # [n_components, n_features]
        
        return self 

    def transform(self, samples: ndarray):
        # samples: [n_samples, n_features]
        return (samples - self.mean_).dot(self.components_.T)
    
    def inverse_transform(self, samples: ndarray):
        # samples: [n_samples, n_components]
        return samples.dot(self.components_) + self.mean_

# %%

import numpy as np 
from sklearn.decomposition import PCA as sklearn_PCA

X = np.empty((100, 2), dtype=np.float32)
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0, 10., size=100)

model = PCA(n_components=2).fit(X)

sklearn_model = sklearn_PCA(n_components=2).fit(X)

print(model.components_)
print(sklearn_model.components_)

# %%

print(model.transform(X)[0])

print(sklearn_model.transform(X)[0])

# %%

print(model.inverse_transform(model.transform(X))[0])
print(sklearn_model.inverse_transform(sklearn_model.transform(X))[0])


# %%

from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people()

#%%

import matplotlib.pyplot as plt 

from torchvision.utils import make_grid


def plot_faces(face_images):
    face_images = face_images[:, None, :, :]
    picture: ndarray = make_grid(
        torch.from_numpy(face_images), nrow=6, normalize=True,
        padding=6, pad_value=1., 
    ).numpy()
    picture = picture.transpose(1, 2, 0)
    
    plt.imshow(picture)
    plt.axis("off")


plot_faces(faces["images"][:36])

#%%

rand_idx = np.random.permutation(len(faces.data))[:36]

plot_faces(faces["images"][rand_idx])

#%%

face_model = sklearn_PCA(svd_solver='randomized').fit(faces["data"])

#%%

def plot_faces(faces):
    
    fig, axes = plt.subplots(
        6, 6, figsize=(10, 10),
        subplot_kw={'xticks':[], 'yticks':[]},
        gridspec_kw={"hspace": 0.1, "wspace": 0.1}
    )
 
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')

plot_faces(
    face_model.components_[:36].reshape(-1, 62, 47)
)

# %%

plot_faces(
    face_model.components_[-36:].reshape(-1, 62, 47)
)


# %%
