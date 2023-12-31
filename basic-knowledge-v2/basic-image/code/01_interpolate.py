# -*- coding:utf-8 -*-
# Author: lqxu

#%%

import decimal

import torch 
from torch import nn, Tensor 
from torch.nn import functional as F

# %%


def is_same_tensor(t1: Tensor, t2: Tensor, eps: float = 1e-6, print_info: bool = False) -> bool:
    
    if t1.shape != t2.shape:
        return False
    
    diff = (t1 - t2).abs()
    result = diff.max().item() < eps 
    
    if not result and print_info:
        correct_num = (diff < eps).sum().item()
        total_num = t1.numel()
        print(f"有 {round(correct_num / total_num * 100, 2)}% 的数字是一样的")
    
    return result


def round_half_up(num: float) -> int:
    """ 标准的四舍五入!!! """
    return int(
        decimal.Decimal(num).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP)
    )


def get_map_idx(old_num: int, new_num: int, check_mode: bool = False, debug_mode: bool = False) -> Tensor:
    
    if check_mode or debug_mode:
        """
        神奇的 bug: 不知道为什么, 在 interpolate 中, 63.0 向下取整是 62, 不是 63!
        二进制精度的问题吗? 但是我用 numpy 和 pytorch 转换是没有问题的, 就很神奇, 可能是计算方式不同导致的吧
        """
        idx = F.interpolate(
            torch.arange(old_num).float()[None, None], 
            size=(new_num, ), mode="nearest"
        )[0][0].long()
        
        if check_mode:
            return idx 
        if debug_mode:
            print(idx)
    
    if old_num > new_num:
        """
        插值后的长度小于插值前:
        将 [0, new_num) 的范围映射到 [0, old_num) 范围内, 然后向下取整即可
        """
        idx = torch.arange(new_num).double() / (new_num / old_num)
        if debug_mode:
            print(idx)
        return idx.long()
    
    if old_num < new_num:
        """
        插值后的长度大于插值前:
            将 [0, new_num - 1] 的范围映射到 [-0.5, old_num - 0.5] 范围内, 然后四舍五入即可
        注意:
            对于所有小数部分是 0.5 的数字, 默认的 round 方式是近似到最近的偶数, 比方说 3.5 和 4.5 都会近似到 4 
            参考: https://numpy.org/doc/stable/reference/generated/numpy.rint.html 
            我们需要的是标准的 四舍五入!!! 在 decimal 库中被称为 round_half_up
        """
        idx = torch.arange(new_num).double() / (new_num / old_num)
        idx = (idx - 0.49999)
        if debug_mode:
            print(idx)
        return idx.round().long().clamp_max(old_num - 1).clamp_min(0)
        # idx = (idx - 0.5).tolist()
        # idx = [round_half_up(i) for i in idx]
        # idx = torch.tensor(idx).clamp_max(old_num - 1).clamp_min(0)

    """ 插值前后的长度相等, 那么就是一一对应的关系 """
    return torch.arange(new_num)


def nearest_interpolate(image: Tensor, new_size: tuple[int, int], check_mode: bool = False):
    # ## step1: 获取图片的维度信息
    image = image.permute(1, 2, 0)
    old_height, old_width, _ = image.shape
    new_height, new_width = new_size

    # ## step2: 计算新的索引对应的坐标值
    h_idx = get_map_idx(old_height, new_height, check_mode)
    w_idx = get_map_idx(old_width, new_width, check_mode)

    # ## step3: 最近邻 / pixel replication
    new_image = image[h_idx.unsqueeze(-1), w_idx.unsqueeze(0)]
    new_image = new_image.permute(2, 0, 1)

    return new_image


def test_nearest_interpolate():
    image_tensor = torch.rand(3, 224, 224)

    r1 = F.interpolate(image_tensor.unsqueeze(0), (300, 400), mode="nearest")[0]
    r2 = nearest_interpolate(image_tensor, (300, 400))
    print(is_same_tensor(r1, r2, print_info=True))

    r1 = F.interpolate(image_tensor.unsqueeze(0), (100, 200), mode="nearest")[0]
    r2 = nearest_interpolate(image_tensor, (100, 200))
    print(is_same_tensor(r1, r2, print_info=True))

    r1 = F.interpolate(image_tensor.unsqueeze(0), (500, 50), mode="nearest")[0]
    r2 = nearest_interpolate(image_tensor, (500, 50))
    print(is_same_tensor(r1, r2, print_info=True))

    r1 = F.interpolate(image_tensor.unsqueeze(0), (224, 224), mode="nearest")[0]
    r2 = nearest_interpolate(image_tensor, (224, 224))
    print(is_same_tensor(r1, r2, print_info=True))

    r1 = F.interpolate(image_tensor.unsqueeze(0), (96, 96), mode="nearest")[0]
    r2 = nearest_interpolate(image_tensor, (96, 96))
    print(is_same_tensor(r1, r2, print_info=True))
    
    r1 = F.interpolate(image_tensor.unsqueeze(0), (96, 96), mode="nearest")[0]
    r2 = nearest_interpolate(image_tensor, (96, 96), check_mode=True)
    print(is_same_tensor(r1, r2, print_info=True))
    
    print(get_map_idx(224, 96, debug_mode=True))


test_nearest_interpolate()

# %%
