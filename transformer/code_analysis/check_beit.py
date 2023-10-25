# -*- coding:utf-8 -*-
# Author: lqxu

#%% 

import os 

os.environ["TRANSFORMERS_OFFLINE"] = "1"  # avoid Internet error!

from transformers.models.beit import BeitImageProcessor, BeitForMaskedImageModeling

model_name = "microsoft/beit-base-patch16-224-pt22k"

image_processor: BeitImageProcessor = BeitImageProcessor.from_pretrained(model_name)
model: BeitForMaskedImageModeling = BeitForMaskedImageModeling.from_pretrained(model_name)

import torch 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 

from torch import Tensor 
from numpy import ndarray 

image = Image.open("transformer/code_analysis/demo_cat.jpg")  # 原图是 480 x 640


def is_same_tensor(t1: Tensor, t2: Tensor, eps: float = 1e-6) -> bool:
    return torch.all(torch.abs(t1 - t2) < eps).item()


# %%
