# -*- coding:utf-8 -*-
# Author: lqxu

"""
简单复现一些功能

测试环境: torch==1.13.0
"""

from typing import List


def span_decode(
        start_logits: List[float], end_logits: List[float], 
        seq_len: int = None, threshold: float = 0.
    ):
    """
    基于 PLMEE 3.3 节的算法实现

    注意这里传入的是 logits 值, 不是 prob 值, 因此不需要经过 softmax 函数转换。

    threshold 的值可以通过 torch.logit(torch.tensor(0.5)).item() 的方式获得
    """

    # region 参数检查
    if len(start_logits) != len(end_logits):
        raise ValueError("start_logits 和 end_logits 的长度应当是一致的")

    max_seq_len = len(start_logits)
    if seq_len is None:
        seq_len = max_seq_len
    if not 0 < seq_len <= len(start_logits):
        raise ValueError("seq_len 值超过了 start_logits 序列的长度")
    # endregion 

    results = []
    cur_start = cur_end = -1
    state = 1

    for idx in range(seq_len):
        if state == 1 and start_logits[idx] > threshold:
            cur_start = idx
            state = 2
        if state == 2:
            if start_logits[idx] > threshold and cur_start != -1:
                if start_logits[idx] > start_logits[cur_start]:
                    cur_start = idx
            if end_logits[idx] > threshold:
                cur_end = idx
                state = 3
        if state == 3:
            if end_logits[idx] > threshold and cur_end != -1:
                if end_logits[idx] > end_logits[cur_end]:
                    cur_end = idx 
            # 不太能理解这里的 new start 是什么意思
            # if start_logits[idx] > threshold and cur_start != -1:
            if cur_start != -1:
                results.append((cur_start, cur_end))
                cur_end = -1
                cur_start = idx  # ??? 这里真的没有问题吗 ?
                state = 2
    
    return results


if __name__ == "__main__":
    
    test_cases = [
        {
            "start_logits": [-0.5, -0.5, 0.5, -0.5, -0.5, ],
            "end_logits": [-0.5, -0.5, 0.5, -0.5, -0.5]
        }, 
        {
            "start_logits": [-0.5, -0.5, 0.5, -0.5, -0.5, ],
            "end_logits": [-0.5, -0.5, -0.5, 0.5, -0.5]
        },
        {
            "start_logits": [-0.5, 0.4, 0.5, -0.5, -0.5, ],
            "end_logits": [-0.5, -0.5, -0.5, 0.5, 0.4]
        }
    ]
    
    for test_case in test_cases:
        print(span_decode(**test_case))
