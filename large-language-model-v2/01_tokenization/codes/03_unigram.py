# -*- coding:utf-8 -*-
# Author: lqxu
# Reference: https://huggingface.co/learn/nlp-course/chapter6/7 

#%%

import os 

os.environ["TRANSFORMERS_OFFLINE"] = "1"  # avoid Internet error!

#%%

import math 
from typing import *
from collections import Counter
from itertools import combinations

from transformers.models.xlnet import XLNetTokenizerFast

#%%

xlnet_tokenizer: XLNetTokenizerFast = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")


def pre_tokenize(text: str) -> List[str]:
    basic_tokenizer = xlnet_tokenizer.backend_tokenizer.pre_tokenizer
    word_with_offset_list = basic_tokenizer.pre_tokenize_str(text)
    word_list = [word for word, offset in word_with_offset_list]
    return word_list


print(pre_tokenize("hello world! 你好世界"))

#%%

def word_tokenize(word: str, model: Dict[str, float], return_score: bool) -> Union[float, List[str]]:
    # 使用 维特比算法 寻找最佳概率路径
    nodes = [{"from_idx": None, "score": None} for _ in word]
    nodes.insert(0, {"from_idx": 0, "score": 1})

    # 前向计算到达每一个结点信息量最小的路径
    for cur_idx in range(len(word)):
        cur_node = nodes[cur_idx]

        # 站在 cur_node 的位置, 向后看, 假设只走一步, 计算到达每一个 node 信息量的增量
        # 更新到达每一个 node 信息量最小的路径
        for next_idx in range(cur_idx + 1, len(word) + 1):
            subword = word[cur_idx:next_idx]
            next_node = nodes[next_idx]
            
            if subword not in model:  # 路径不成立!
                continue
            
            score = model[subword] + cur_node["score"]
            
            if next_node["score"] is None or next_node["score"] > score:
                next_node["from_idx"] = cur_idx
                next_node["score"] = score
    
    if return_score:
        return nodes[-1]["score"]
    
    # 回溯
    results = []
    prev_idx, cur_idx = nodes[-1]["from_idx"], -1

    while True:
        results.insert(0, word[prev_idx:cur_idx])
        
        if prev_idx == 0:
            break 
        
        prev_idx, cur_idx = nodes[prev_idx]["from_idx"], prev_idx

    return results


def unigram_train(text_iterator: Iterator[str], vocab_size: int):
    word_counter = Counter(word for text in text_iterator for word in pre_tokenize(text))
    
    subword_counter, cand_counter = Counter(), Counter()
    for word, count in word_counter.items():
        for start_idx, end_idx in combinations(range(len(word) + 1), 2):
            subword = word[start_idx:end_idx]
            
            if len(subword) == 1:  # 如果 subword 是单个 char, 一定要加入 counter 中
                subword_counter[subword] += count
            else:  # 如果 subword 不是单个 char, 则需要根据数量进行筛选, 去掉低频的
                cand_counter[subword] += count 
    
    # 最终 subword_counter 中仅有 vocab_size * 3 的数量
    cand_keep_number = vocab_size * 3 - len(subword_counter)
    subword_counter.update(dict(cand_counter.most_common(cand_keep_number)))
    
    total_count = sum(subword_counter.items())
    model = {
        subword: -math.log(count / total_count)
        for subword, count in subword_counter.items()
    }
    
    def _compute_model_loss(model_):
        return sum(
            count * word_tokenize(word, model_, True)
            for word, count in word_counter.items()
        )
    
    def _compute_subword_score(model_):
        basic_loss = _compute_model_loss(model_)
        
        subword_score_dict = {}
        
        for subword, score in model_.items():
            if len(subword) == 1:
                continue
            model_.pop(subword)
            subword_score_dict[subword] = _compute_model_loss(model_) - basic_loss
            model_[subword] = score

        return subword_score_dict
    
    sum(count * word_tokenize(word, model, True) for word, count in word_counter.items())
    
    