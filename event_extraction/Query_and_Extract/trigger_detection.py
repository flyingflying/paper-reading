# -*- coding:utf-8 -*-
# Author: lqxu

"""
触发词检测的模型代码

简称说明:
    1. attn=attention
    2. sentn=sentence

代码测试的环境:
torch==1.13.0
transformers==4.24.0
(不一定要严格按照版本来, 能跑通就行)
"""

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from transformers.models.bert import BertConfig, BertModel


class TestTriggerDetection(nn.Module):
    def __init__(self):
        super(TestTriggerDetection, self).__init__()

        # ## step1: 初始化 bert 模型
        self.bert_config = BertConfig(
            vocab_size=500, max_position_embeddings=32, type_vocab_size=2,
            hidden_size=384, num_hidden_layers=4, num_attention_heads=12, intermediate_size=768, 
            classifier_dropout=0.1
        )
        self.bert = BertModel(self.bert_config)

        # ## step2: 基础的配置参数
        pos_size = 16  # 词性类型的数量
        last_k_layers = 3  # 需要收集 attention_probs 的层数
        
        self.pos_size = pos_size + 1
        self.last_k_layers = last_k_layers
        self.hidden_size = self.bert_config.hidden_size

        # ## step3: 句子 token 的线性变换
        self.sentn_transform = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=False 
        )

        # ## step4: 分类层 (这里标签也是输入的一部分, 分类任务变成了二分类)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size * 3 + self.pos_size, 2),
        )

        # ## step5: 模型状态转换
        self.cpu()
        self.eval()

    def forward(self, input_ids: Tensor, sentn_mask: Tensor, event_mask: Tensor, pos_ids: Tensor):
        """
        输入句子的格式: [CLS] [EVENT] [SEP] sentn_token1 sentn_token2 ... [SEP] event_type event_trigger1 event_trigger2 ... [SEP] [PAD] ...

        这里我们将输入句子的 token 分成三种类型:
            1. 特殊 token, 比方说: [CLS], [SEP], [EVENT], [PAD] 这种
            2. 句子 token, 句子分词出来的 token 
            3. event token, 用于表示当前事件的 token, 包括 event_type 和 event_trigger

        input_ids, sentn_mask, event_mask, pos_ids 的 shape 都是 [batch_size, num_tokens], 他们表示的含义不同:
            1. input_ids: token 的 id 值, 和 HuggingFace Transformers 框架里面的 BertModel 是一致的
            2. sentn_mask: token 是否是句子中的 token, 是为 1, 不是为 0
            3. event_mask: token 是否是 event 的 token, 是为 1, 不是为 0
            4. pos_ids: token 的词性 (part-of-speech), 特殊 token 和 event token 都用 -1 来表示 
        """

        batch_size = input_ids.shape[0] 

        # ## step1: 构建 attention_mask (1 表示参与计算, 0 表示不参与计算)
        attention_mask = input_ids.ne(0).float()

        # ## step2: 构建 token_type_ids
        # [EVENT] 和 sen_tokens 一起作为第一个句子, event_type 和 event triggers 一起作为第二个句子
        token_type_ids = attention_mask.clone().long()
        sep_idx = (input_ids == 102).nonzero(as_tuple=True)[1].reshape(batch_size, -1)[:, 1]
        for i in range(batch_size):
            token_type_ids[i, :sep_idx[i]+1] = 0

        # ## step3: 用 bert 对句向量进行编码
        # all_embeddings: [batch_size, num_tokens, hidden_size]
        # all_attention_probs 是一个 tuple 对象, 里面有每一层 attention_probs 的输出
        # 每一个 attention_probs 的 shape 是 [batch_size, n_heads, n_tokens, n_tokens]
        all_embeddings, _, all_attn_probs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            return_dict=False, output_attentions=True
        )
        all_embeddings = all_embeddings * attention_mask.unsqueeze(-1)  # 这一步我觉得是没有必要的

        # ## step4: 分别拿到句子的 token 编码和事件的 token 编码
        sentn_embeddings = self.get_sub_embeddings(all_embeddings, sentn_mask)  # [batch_size, n_sentn_tokens, hidden_size]
        event_embeddings = self.get_sub_embeddings(all_embeddings, event_mask)  # [batch_size, n_event_tokens, hidden_size]

        # ## step5: 计算 context embeddings 
        # 在这里, 将 sentn_embeddings 作为 value, 用 sentn_attn_probs 计算线性组合的系数, 得到 context embeddings
        sentn_attn_probs = self.get_sentn_attn_probs(all_attn_probs, sentn_mask)  # [batch_size, n_sentn_tokens, n_sentn_tokens]
        sentn_context_embeddings = torch.matmul(sentn_attn_probs, sentn_embeddings)  # [batch_size, n_sentn_tokens, hidden_size]

        # ## step6: 计算 event type aware contextual representation
        # 用 sentn_token 和每一个 event_token 的 cos 相似度作为线性组合的系数, 所有的 event_token 作为向量组, 得到包含 event type 信息的 sentn_token 向量
        cos_sim = F.cosine_similarity(
            self.sentn_transform(sentn_embeddings).unsqueeze(2), 
            event_embeddings.unsqueeze(1),
            dim=-1
        )  # [n_sentn_tokens, n_event_tokens]
        sentn_event_embeddings = torch.matmul(cos_sim, event_embeddings)  # [batch_size, n_sentn_tokens, hidden_size]
        
        # ## step7: 融入 pos one hot 向量的信息, 这里用的是 one-hot encoding 的方式, 不是 embedding 的方式
        pos_ids[pos_ids<0] = self.pos_size - 1
        pos_vectors = F.one_hot(pos_ids, num_classes=self.pos_size)
        
        # ## step8: 分类
        # print(sentn_embeddings.shape, sentn_context_embeddings.shape, sentn_event_embeddings.shape, pos_vectors.shape)
        logits = self.classifier(
            torch.concat([
                sentn_embeddings, 
                sentn_context_embeddings, 
                sentn_event_embeddings, 
                self.get_sub_embeddings(pos_vectors, sentn_mask)
            ], dim=-1)
        )
        
        return logits

    def get_sub_embeddings(self, all_vectors, mask):
        """
        获取 sentn_embeddings 和 event_embeddings
        
        函数有待优化, 包含过多的 for 循环 (优化可能需要整理优化, 局部优化可能解决不了问题)
        """

        batch_size, _, hidden_size = all_vectors.shape

        sub_embeddings = []

        # 遍历每一个样本, 将句子和事件的 token 单独拿出来
        for batch_idx in range(batch_size):
            token_indices = torch.nonzero(mask[batch_idx], as_tuple=False).squeeze(-1)
            sub_embeddings.append(all_vectors[batch_idx, token_indices])

        max_len = torch.max(torch.sum(mask, dim=-1))

        # 将句子的 token 和事件的 token 补零对齐
        for i in range(batch_size):
            sub_embeddings[i] = torch.cat([
                sub_embeddings[i],
                torch.zeros(max_len - len(sub_embeddings[i]), hidden_size)
            ])

        sub_embeddings = torch.stack(sub_embeddings)

        return sub_embeddings

    def get_sub_attn_probs(self, attn_probs: Tensor, mask: Tensor):
        """
        从 attn_probs 中将 sentn_tokens 部分取出来作为 attn_scores, 用概率标准化的方式得到 attn_probs 
        
        attn_probs 的 shape: [batch_size, head_size, n_tokens_query, n_tokens_key]
            对于 attn_probs 来说, attn_probs.sum(axis=-1) 的值应该是 1.0
        mask 的 shape: [batch_size, n_tokens, hidden_size]
        """

        # ## step1: 获取基本信息
        batch_size = attn_probs.shape[0]
        max_len = torch.max(torch.sum(mask, dim=-1))

        # ## step2: 对所有 head 的 attention_probs 取平均值 (思考: 保留 head 的设置是否更好)
        mean_attn_probs = torch.mean(attn_probs, dim=1)  # [batch_size, n_tokens_query, n_tokens_key]

        # ## step3: 计算 attn_scores
        sub_attn_scores = torch.zeros(batch_size, max_len, max_len)
        for batch_idx in range(batch_size):
            sub_idx = torch.nonzero(mask[batch_idx], as_tuple=False).squeeze(-1)  # [n_sentn_tokens, ]
            n_sentn_tokens = sub_idx.shape[0]
            sub_attn_scores[batch_idx, :n_sentn_tokens, :n_sentn_tokens] = mean_attn_probs[batch_idx, sub_idx, :][:, sub_idx]

        # ## step4: 基础的概率标准化: 分数 ==> 概率
        sub_attn_probs = sub_attn_scores / (torch.sum(sub_attn_scores, dim=-1, keepdim=True) + 1e-9)

        return sub_attn_probs

    def get_sentn_attn_probs(self, all_attn_probs, sentn_mask):
        """ 生成 sentn_tokens 部分的 attn_probs """

        attn_probs = 0

        for idx in range(self.last_k_layers):
            # 原代码的 idx 没有减 1, 属于 bug, 我不想说什么了, 如果你实验的效果不好, 可以考虑也不 -1
            # 这样就变成了取 bert 的第一层 attn_probs 了
            attn_probs += self.get_sub_attn_probs(all_attn_probs[-idx-1], sentn_mask)

        attn_probs /= self.last_k_layers
        return attn_probs


if __name__ == "__main__":
    detector = TestTriggerDetection()

    model_inputs = {
        "input_ids": torch.LongTensor([
            [101, 100, 102, 123, 124, 125, 102, 234, 235, 236, 102,   0,   0], 
            [101, 100, 102, 123, 124, 125, 124, 124, 102, 234, 235, 236, 102], 
        ]), 
        "sentn_mask": torch.LongTensor([
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        ]),
        "event_mask": torch.LongTensor([
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        ]),
        "pos_ids": torch.LongTensor([
            [-1, -1, -1,  5,  6,  7, -1, -1, -1, -1, -1,  0,  0],
            [-1, -1, -1,  5,  6,  7,  8,  9, -1, -1, -1, -1, -1],
        ])
    }

    print(detector.forward(**model_inputs))
