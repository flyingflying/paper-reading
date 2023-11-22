# -*- coding:utf-8 -*-
# Author: lqxu
# Reference: https://huggingface.co/learn/nlp-course/chapter6/5 

#%%

import os 

os.environ["TRANSFORMERS_OFFLINE"] = "1"  # avoid Internet error!

#%%

import random 
from typing import *
from itertools import pairwise
from collections import Counter

from transformers.models.gpt2 import GPT2TokenizerFast

#%%

gpt2_tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")


def pre_tokenize(text: str) -> List[str]:
    basic_tokenizer = gpt2_tokenizer.backend_tokenizer.pre_tokenizer
    word_with_offset_list = basic_tokenizer.pre_tokenize_str(text)
    word_list = [word for word, offset in word_with_offset_list]
    return word_list


print(pre_tokenize("Hello, World! 你好世界"))

#%%

def bpe_train(text_iterator: Iterable[str], vocab_size: int) -> Tuple[Dict[str, str], List[str]]:
    # 符号含义: sw - subword; sws - subword 列表, 并用 空格 拼成字符串 

    # ## 1. 统计 word 的频数
    word_counter = Counter([word for text in text_iterator for word in pre_tokenize(text)])
    
    # ## 2. 统计所有的字符, 初始化词表
    vocab = list(set(char for word in word_counter for char in word))
    
    # ## 3. 初始化每一个 word 分词结果
    # 预分词中的 空格 都被转义了, 这里可以直接用空格作为分隔符
    word_sws_dict = {word: " ".join(list(word)) for word in word_counter}
    
    # ## 4. 构建合并规则
    merge_dict = {}
    
    while len(vocab) < vocab_size:
        # ## 4.1 统计 sw_pair 出现的频数
        sw_pair_counter = Counter()
        for word, count in word_counter.items():
            for sw_pair in pairwise(word_sws_dict[word].split(" ")):
                sw_pair_counter[sw_pair] += count

        # ## 4.2 获取频数最高的那一组
        (sw1, sw2), _ = sw_pair_counter.most_common(1)[0]
        
        # ## 4.3 合并, 并加入结果列表中
        old_pat, new_pat = f"{sw1} {sw2}", f"{sw1}{sw2}"

        word_sws_dict = {
            word: sws.replace(old_pat, new_pat)
            for word, sws in word_sws_dict.items()
        }
        
        merge_dict[old_pat] = new_pat  # insertion order
        vocab.append(new_pat)
    
    return merge_dict, vocab

#%%

def bpe_tokenize(text: str, merge_dict: Dict[str, str], vocab: List[str] = None) -> List[str]:
    results = []

    # 预分词: sentence -> words
    for word in pre_tokenize(text):
        # 一个字符一个 subword
        word = " ".join(list(word))

        # 合并 subword, 注意遍历 dict 的顺序是插入顺序, 这很重要
        for old_pat, new_pat in merge_dict.items():
            word = word.replace(old_pat, new_pat)
        
        # 分词结果存入 results 列表中
        results.extend(word.split(" "))
    
    if vocab is None:
        return results

    return [vocab.index(result) for result in results]  

#%%

check_text_corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

check_merge_dict, check_vocab = bpe_train(check_text_corpus, 49)

print(check_merge_dict)
print(check_vocab)
print(bpe_tokenize("This is not a token.", check_merge_dict))
print(bpe_tokenize("This is not a token.", check_merge_dict, check_vocab))

#%% 

def replace_with_dropout(word: str, old_pat: str, new_pat: str, dropout: float):
    start_idx = 0
    len_word, len_old_pat = len(word), len(old_pat)

    while start_idx <= len_word - len_old_pat:
        end_idx = start_idx + len_old_pat

        if word[start_idx:end_idx] == old_pat and random.random() > dropout:
            word = word[:start_idx] + new_pat + word[end_idx:]
            len_word = len(word)
            start_idx = end_idx
        
        start_idx += 1

    return word 


print(replace_with_dropout("a a a b", "a b", "x", 0.0))

# %%