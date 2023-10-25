# -*- coding:utf-8 -*-
# Author: lqxu

#%% basic settings

import os 

os.environ["TRANSFORMERS_OFFLINE"] = "1"  # avoid Internet error!

from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling

_MODEL_NAME = "openai/imagegpt-small"

image_processor: ImageGPTImageProcessor = ImageGPTImageProcessor.from_pretrained(_MODEL_NAME)
model: ImageGPTForCausalImageModeling = ImageGPTForCausalImageModeling.from_pretrained(_MODEL_NAME).eval()

import torch 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 

from torch import Tensor 
from numpy import ndarray 

image = Image.open("transformer/code_analysis/demo_cat.jpg")  # 原图是 480 x 640


def is_same_tensor(t1: Tensor, t2: Tensor, eps: float = 1e-6) -> bool:
    return torch.all(torch.abs(t1 - t2) < eps).item()

#%% image preprocessor 

from transformers.models.imagegpt import ImageGPTForImageClassification

cls_model: ImageGPTForImageClassification = ImageGPTForImageClassification.from_pretrained(_MODEL_NAME).eval()

input_ids = image_processor.preprocess(image)["input_ids"]

if isinstance(input_ids, list):
    input_ids = torch.from_numpy(np.array(input_ids))
elif isinstance(input_ids, ndarray):
    input_ids = torch.from_numpy(input_ids)
elif not isinstance(input_ids, Tensor):
    raise ValueError(f"unknown type of {type(input_ids)}.")

input_ids: Tensor  # [batch_size, num_tokens]

assert input_ids.ndim == 2

with torch.no_grad():
    output = cls_model(input_ids=input_ids)

print(output.logits)

#%% 图片预处理的过程

from sklearn.metrics import pairwise_distances_argmin


def sf_preprocessor(pil_image_):
    # step1: resize 成 32 x 32 大小的图片
    pil_image_ = pil_image_.resize(
        size=(image_processor.size["width"], image_processor.size["height"]), 
        resample=Image.BILINEAR
    )

    # step2: 标准化到 -1 至 1 之间
    image_ = np.array(pil_image_)  # [img_height, img_width, n_channels]
    image_ = image_ / 127.5 - 1
    image_ = image_.reshape(-1, 3)  # [img_height * img_width, n_channels]

    # step3: 进一步离散化
    clusters_ = np.array(image_processor.clusters)
    input_ids = pairwise_distances_argmin(image_, clusters_)
    
    return torch.from_numpy(input_ids)


assert (sf_preprocessor(image) == input_ids[0]).all()

#%% Color Quantization using K-Means

# reference: https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html 

def count_colors(np_image_: ndarray):
    return np.unique(np_image_.reshape(-1, 3), axis=0).shape[0]

import numpy as np 
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

np_image = np.array(image) / 255.
print("原始图片一共有 {} 种颜色".format(count_colors(np_image)))
color_samples = shuffle(np_image.reshape(-1, 3))

kmeans_model = KMeans(n_clusters=10, n_init="auto").fit(color_samples)
color_labels = kmeans_model.predict(np_image.reshape(-1, 3))

reconstructed_image = kmeans_model.cluster_centers_[color_labels].reshape(np_image.shape)
print("重构后的图片一共有 {} 种颜色".format(count_colors(reconstructed_image)))

plt.subplot(1, 2, 1)
plt.imshow(np_image)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image)
plt.axis("off")

#%% image generation

input_ids = torch.full(  # 可以是图片的某一部分, 这里直接使用 BOS (begin of sequence)
    size=(1, 1),  # [batch_size, num_pixels]
    fill_value=model.config.vocab_size - 1
)
output_ids = model.generate(  # 和 GPT 模型的生成方式是一样的
    input_ids=input_ids, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True, top_k=40
).cpu().detach().numpy()[0, 1:]

clusters = np.array(image_processor.clusters)
height = image_processor.size["height"]
width = image_processor.size["width"]

sample_image = clusters[output_ids]
sample_image: ndarray = (sample_image + 1.0) * 127.5
sample_image = np.rint(sample_image).reshape(height, width, 3).astype(np.uint8)

plt.imshow(sample_image)

# %%
