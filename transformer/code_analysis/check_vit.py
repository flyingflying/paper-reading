# -*- coding:utf-8 -*-
# Author: lqxu

"""
paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
paper link: https://arxiv.org/abs/2010.11929

HF API Page: https://huggingface.co/docs/transformers/model_doc/vit
"""


#%% 加载模型
import os 
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 加速

import math 

from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib.axes import Axes

import torch 
from torch import Tensor

import numpy as np 

from transformers.models.vit import ViTForImageClassification, ViTConfig, ViTImageProcessor, ViTModel

model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
whole_model = ViTForImageClassification.from_pretrained(model_name).eval()
image = Image.open("transformer/code_analysis/demo_cat.jpg")  # 原图是 480 x 640


def is_same_tensor(t1: Tensor, t2: Tensor, eps: float = 1e-6) -> bool:
    return torch.all(torch.abs(t1 - t2) < eps).item()


#%% 测试 ViT 模型的运行方式

image = Image.open("transformer/code_analysis/demo_cat.jpg")  # 原图是 480 x 640
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]  # shape: [batch_size, n_channels, img_height, img_width]

with torch.no_grad():
    outputs = whole_model.forward(pixel_values=inputs["pixel_values"])
logits = outputs.logits
y_pred = logits.argmax(-1)

print(f"predicted class: {whole_model.config.id2label[y_pred.item()]}")


# %% 测试 ViT 模型预处理的方式
def simple_processor(image_: Image) -> Tensor:

    # ## step 1: 用 Pillow 对图片进行 resize
    image_ = image_.resize((224, 224), resample=Image.BILINEAR)
    
    """
    Pillow 库 Image 类实现了 array interface, 可以直接通过 np.array 创建 ndarray 对象。
    array interface 的相关内容, 请参考:
        1. [Interoperability with NumPy](https://numpy.org/doc/stable/user/basics.interoperability.html)
        2. [The array interface protocol](https://numpy.org/doc/stable/reference/arrays.interface.html)
        3. Image.__array_interface__ 方法
    """

    # ## step 2: 转换成 ndarray 对象, 并 rescale 到 0 - 1 之间
    image_ = np.array(image_) / 255.
    
    # ## step3: 减去 mean 除以 std, 取值范围变成 -1 到 1 之间
    image_ = (image_ - 0.5) / 0.5
    
    return torch.from_numpy(image_).permute(2, 0, 1).float()


# 验证一致性
print(is_same_tensor(
    processor(images=image, return_tensors="pt")["pixel_values"], 
    simple_processor(image)
))

   
# %% 验证卷积的合理性

input_image_for_conv = torch.randn(3, 224, 224)
input_image_for_linear = input_image_for_conv.reshape(3, 14, 16, 14, 16).contiguous()
input_image_for_linear = input_image_for_linear.permute(1, 3, 0, 2, 4).contiguous()
input_image_for_linear = input_image_for_linear.reshape(14 * 14, 3 * 16 * 16).contiguous()  # [1, 196, 768]

with torch.no_grad():
    conv_layer = whole_model.vit.embeddings.patch_embeddings.projection
    
    linear_result = (input_image_for_linear @ conv_layer.weight.reshape(768, 768).T) + conv_layer.bias
    
    conv_result = conv_layer(input_image_for_conv).reshape(768, 196).transpose(0, 1)

print(is_same_tensor(
    conv_result,
    linear_result,
    eps=1e-4
))

# %% 卷积核的图片

ncols, total = 5, 196
nrows = int(math.ceil(total / ncols))
weight = whole_model.vit.embeddings.patch_embeddings.projection.weight.detach().permute(0, 2, 3, 1)

weight = (weight - weight.min()) / (weight.max() - weight.min())
weight = (weight * 255).clip(0, 255).numpy().astype(np.uint8)

_, plt_objs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

plt_objs: list[Axes] = plt_objs.reshape(-1)

for idx in range(total):
    plt_obj: Axes = plt_objs[idx]
    
    plt_obj.imshow(weight[idx])
    plt_obj.axis("off")


for idx in range(nrows * ncols - total, 0, -1):
    plt_objs[-idx].axis("off")

# %%

pe = whole_model.vit.embeddings.position_embeddings.detach()[0, 1:]
pe = torch.nn.functional.normalize(pe, dim=1)
cos_result = (pe @ pe.T).reshape(196, 14, 14)

ncols, total = 5, 196
nrows = int(math.ceil(total / ncols))

_, plt_objs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

plt_objs: list[Axes] = plt_objs.reshape(-1)

for idx in range(total):
    plt_obj: Axes = plt_objs[idx]
    
    plt_obj.imshow(cos_result[idx])
    plt_obj.axis("off")


for idx in range(nrows * ncols - total, 0, -1):
    plt_objs[-idx].axis("off")

# %% 计算 attention distance

# reference: https://www.zhihu.com/question/492429589

inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]

with torch.no_grad():
    outputs = whole_model.forward(pixel_values=inputs["pixel_values"], output_attentions=True)


# reference: https://gist.github.com/simonster/155894d48aef2bd36bd2dd8267e62391 
def compute_distance_matrix(config: ViTConfig):
    image_size = config.image_size  # 图片的 高 或者 宽, 224
    patch_size = config.patch_size  # 每一个 patch 的 高 或者 宽, 16
    n_patches_per_axis = image_size // patch_size
    n_patches = n_patches_per_axis ** 2  # 14 * 14 = 196

    distance_matrix = np.zeros((n_patches, n_patches))

    for i in range(n_patches):
        for j in range(n_patches):
            if i == j:
                continue

            xi, yi = i // n_patches_per_axis, i % n_patches_per_axis
            xj, yj = j // n_patches_per_axis, j % n_patches_per_axis

            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])
  
    return distance_matrix


dist_matrix = torch.from_numpy(compute_distance_matrix(whole_model.config))

# reference: https://zhuanlan.zhihu.com/p/65220518 
colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 
    'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 
    'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 
    'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 
    'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 
    'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 
    'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

for idx, attention_matrix in enumerate(outputs.attentions):
    attention_matrix = attention_matrix.squeeze(0)[:, 1:, 1:]  # [n_heads, n_patches_query, n_patches_key]
    # 在 n_patches_key 维度上求和, 在 n_patches_query 维度上求平均
    mean_attn_dist_list = (dist_matrix * attention_matrix).sum(dim=-1).mean(dim=-1).tolist()
    
    for idj, mean_attn_dist in enumerate(mean_attn_dist_list):
        plt.scatter(idx, mean_attn_dist, c=colors[idj])


# %%

with torch.no_grad():
    
    pixel_values = torch.randn(1, 3, 224 - 16, 224 - 16)
    
    # patch_embeddings 的作用: 分块, 线性变换
    embeddings = whole_model.vit.embeddings.patch_embeddings.projection(pixel_values)
    embeddings = embeddings.flatten(2).transpose(1, 2)
    # embeddings
    embeddings = torch.cat([
        whole_model.vit.embeddings.cls_token, 
        embeddings
    ], dim=1)
    embeddings += whole_model.vit.embeddings.interpolate_pos_encoding(embeddings, 224 - 16, 224 - 16)
    
    print(whole_model.vit.embeddings.interpolate_pos_encoding(embeddings, 224 + 15, 224 + 15).shape)
    print(is_same_tensor(
        whole_model.vit.embeddings.interpolate_pos_encoding(embeddings, 224 + 15, 224 + 15),
        whole_model.vit.embeddings.position_embeddings, eps=1e-4
    ))
    print(whole_model.vit.embeddings.interpolate_pos_encoding(embeddings, 224 + 16, 224 + 16).shape)
    
    outputs = whole_model.forward(
        pixel_values=pixel_values, interpolate_pos_encoding=True, output_hidden_states=True
    )

    print(outputs.hidden_states[0].shape)

# %%
