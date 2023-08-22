# -*- coding:utf-8 -*-
# Author: lqxu

#%% prepare

import os 
os.environ["TRANSFORMERS_OFFLINE"] = "1"

#%% 加载 bert 模型
from transformers.models.bert import BertTokenizerFast
from transformers.models.bert import BertForMaskedLM

PRETRAINED_NAME = "hfl/chinese-bert-wwm-ext"

tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_NAME)
model = BertForMaskedLM.from_pretrained(PRETRAINED_NAME).eval()

# %%

import torch 

model.tie_weights

print(model.config.tie_word_embeddings)

input_embeddings = model.bert.embeddings.word_embeddings.weight

output_embeddings = model.cls.predictions.decoder.weight

print(input_embeddings is output_embeddings)  # input_embeddings 和 output_embeddings 是同一个张量 ！！！

model.cls.predictions.decoder.bias

# model.cls.predictions.decoder.bias

#%% 加载 GPT2 模型

from transformers.models.gpt2 import GPT2Tokenizer
from transformers.models.gpt2 import GPT2LMHeadModel

PRETRAINED_NAME = "IDEA-CCNL/Wenzhong-GPT2-110M"

tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_NAME)
model = GPT2LMHeadModel.from_pretrained(PRETRAINED_NAME).eval()

# %%

print(model.config.tie_word_embeddings)

input_embeddings = model.transformer.wte.weight

output_embeddings = model.lm_head.weight

print(input_embeddings is output_embeddings)

model.lm_head.bias
