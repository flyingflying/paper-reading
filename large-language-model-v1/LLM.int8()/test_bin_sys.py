# -*- coding:utf-8 -*-
# Author: lqxu

""" 测试二进制和十进制互转 """

# Decimal 类是 Python 的原生库, 支持十进制运算
# 在写代码时需要注意, 为了保证精度, Decimal 的对象只和 int 进行运算, 不和 float 进行运算
from decimal import Decimal

from contextlib import suppress


def number_2_to_10(bin_str: str) -> str:
    """ 目前只支持正数 """

    point_idx = len(bin_str)

    with suppress(ValueError):
        point_idx = bin_str.index(".")
    
    int_part = bin_str[:point_idx]
    # fraction 指的是小数, 不是分数
    frac_part = bin_str[point_idx+1:]
    
    buffer = []
    for e, n in enumerate(reversed(int_part)):
        buffer.append(Decimal(n) * (2 ** e))
    
    for e, n in enumerate(frac_part, start=1):
        buffer.append(Decimal(n) / (2 ** e))
    
    return str(sum(buffer))


def number_10_to_2(dec_str: str, precision: int = 8) -> str:
    """ 目前只支持正数 """
    
    point_idx = len(dec_str)

    with suppress(ValueError):
        point_idx = dec_str.index(".")

    final_number = Decimal("0.0")

    int_part = Decimal(dec_str[:point_idx]) if dec_str[:point_idx] else final_number
    # fraction 指的是小数, 不是分数
    frac_part = Decimal(dec_str[point_idx:]) if dec_str[point_idx:] else final_number
    
    # 整数除以二取余倒序排列
    int_buffer = []
    while int_part != final_number:
        int_buffer.append(str(int_part % 2))
        int_part = int_part // 2
    int_buffer = list(reversed(int_buffer))

    # 小数乘二取整顺序排列
    frac_buffer = []
    for _ in range(precision):
        if frac_part == final_number:
            break
        frac_part = frac_part * 2

        if frac_part >= 1.0:
            frac_buffer.append("1")
            frac_part = frac_part - 1
        else:
            frac_buffer.append("0")
    
    if len(frac_buffer) == 0:
        return "".join(int_buffer)
    return "".join(int_buffer) + "." + "".join(frac_buffer)


if __name__ == "__main__":
    # 110.101 ==> 6.625
    print(number_2_to_10("110.101"))
    # 6.625 ==> 110.101
    print(number_10_to_2("6.625"))
    print(number_10_to_2("0.3333333333333333333333333333333333333333333333333"))

    # 0.3 转换成二进制是无限小数 !!!
    print(Decimal.from_float(0.3))
    print(number_10_to_2("0.3", precision=100))
