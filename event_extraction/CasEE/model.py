# -*- coding:utf-8 -*-
# Author: lqxu

"""
CasEE 模型部分的代码整理

简称说明:
    attn=attention
    cond=condition
    pos=relative_position 相对位置
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EventPredictor(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float = 0.0):
        super(EventPredictor, self).__init__()

        self.v = nn.Linear(hidden_size * 4, 1)
        self.w = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def sigma_func(self, input1: Tensor, input2: Tensor):
        """ 实现论文 3.2 节的公式 (2) """

        # 我们可以假设 input1 和 input2 都是 [batch_size, hidden_size]
        features = torch.concat([  # 三种常规的向量融合方式都用上了 (concat, 逐位相减取绝对值, 逐位相乘)
            input1, input2,
            (input1 - input2).abs(),
            input1 * input2
        ], dim=-1)  # [batch_size, hidden_size * 4]

        return self.v(torch.tanh(self.w(features))).squeeze(-1)  # [batch_size, ]

    def forward(self, token_vectors: Tensor, event_vectors: Tensor, attn_mask: Tensor) -> Tensor:
        event_size = event_vectors.size(0)  # [event_size, hidden_size]
        batch_size, n_tokens, hidden_size = token_vectors.shape  # [batch_size, n_tokens, hidden_size]
        # concat 是没有广播机制的; 在 torch 中, expand 等价于广播机制, 和 broadcast_to 一样
        token_vectors = token_vectors.unsqueeze(1).expand(batch_size, event_size, n_tokens, hidden_size)

        # 这里其实是 attention 机制, 我们可以人为 event 向量是 query 向量, token 向量是 key 向量和 value 向量
        # 按照论文中的说法, sigma_func 可以看作是计算两个向量相关性的函数,
        # 我们可以人为在 attention 中, 计算 query 和 key 向量相关性的方式从 点乘 变成了 sigma_func
        attn_scores = self.sigma_func(  # [batch_size, event_size, n_tokens]
            event_vectors[None, :, None, :].expand_as(token_vectors), token_vectors)
        attn_scores = self.dropout(attn_scores)
        attn_scores = attn_mask.unsqueeze(1) + attn_scores
        attn_probs = torch.softmax(attn_scores, dim=-1)
        new_event_vectors = (attn_probs.unsqueeze(-1) * token_vectors).sum(dim=2)  # [batch_size, event_size, hidden_size]

        # 上面用 sigma_func 衡量 event 向量 和 token 向量 之间的相关性
        # 这里用 sigma_func 衡量 event 向量 和 new_event 向量 之间的相关性
        # 之所以两者共享参数, 我猜测是因为 new_event 向量从本质上来说, 是 token 向量的线性组合
        logits = self.sigma_func(  # [batch_size, event_size]
            event_vectors.unsqueeze(0).expand_as(new_event_vectors), new_event_vectors)
        return logits


class TriggerRecognizer(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout_rate: float = 0.0):
        super(TriggerRecognizer, self).__init__()

        self.self_attn = MultiHeadSelfAttention(hidden_size=hidden_size, n_heads=n_heads, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)

    def forward(self, token_vectors: Tensor, attn_mask: Tensor):
        # token_vectors: [batch_size, n_tokens, hidden_size]

        # ## step1: 自注意力机制
        token_vectors = self.dropout(token_vectors)
        attn_token_vectors = self.dropout(self.self_attn(token_vectors, attn_mask))  # [batch_size, n_tokens, hidden_size]
        token_vectors = self.layer_norm(token_vectors + attn_token_vectors)

        # ## step2: 分类
        token_vectors = self.dropout(F.gelu(self.dense(token_vectors)))

        start_logits = self.start_classifier(token_vectors).squeeze(dim=-1)  # [batch_size, n_tokens]
        end_logits = self.end_classifier(token_vectors).squeeze(dim=-1)  # [batch_size, n_tokens]

        return torch.sigmoid(start_logits), torch.sigmoid(end_logits)


class ArgumentRecognizer(nn.Module):
    def __init__(
            self, n_arguments: int, hidden_size: int, n_heads: int, dropout_rate: float = 0.0, 
            max_position_size: int = 512, pos_hidden_size: int = 64
        ) -> None:
        super(ArgumentRecognizer, self).__init__()

        self.max_position_size = max_position_size

        self.trigger_fusion = CondLayerNorm(hidden_size, hidden_size)
        self.self_attn = MultiHeadSelfAttention(hidden_size, n_heads, dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.pos_embeddings = nn.Embedding(max_position_size * 2, pos_hidden_size)
        self.pos_dense = nn.Linear(hidden_size + pos_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.event_constrain = nn.Linear(hidden_size, n_arguments)
        self.start_classifier = nn.Linear(hidden_size, n_arguments)
        self.end_classifier = nn.Linear(hidden_size, n_arguments)

    def forward(self, token_vectors: Tensor, trigger_mask: Tensor, attn_mask: Tensor, pos_ids: Tensor, event_vectors: Tensor):
        # ## step1: 计算 trigger 向量
        # trigger 部分的 token 向量取平均值 
        # 这里只取了 trigger 的首尾 token, 也就是 2 个
        numerator = torch.sum(token_vectors * trigger_mask.unsqueeze(-1), dim=1)  # 分子
        denominator = torch.sum(trigger_mask, dim=1, keepdim=True)  # 分母
        trigger_vector = numerator / denominator  # [batch_size, hidden_size]
        
        # ## step2: 将触发词信息融入 token_vectors 中
        token_vectors = self.trigger_fusion(token_vectors, trigger_vector)

        # ## step3: 自注意力机制
        token_vectors = self.dropout(token_vectors)
        attn_token_vectors = self.dropout(self.self_attn(token_vectors, attn_mask))
        token_vectors = self.layer_norm(token_vectors + attn_token_vectors)

        # ## step4: 融入到触发词的相对位置信息
        pos_info = self.pos_embeddings(pos_ids + self.max_position_size)  # [batch_size, n_tokens, pos_hidden_size]
        token_vectors = self.pos_dense(torch.cat([token_vectors, pos_info], dim=-1))  # [batch_size, n_tokens, hidden_size]

        # ## step5: 计算分类概率, 注意这里是 条件概率
        event_probs = torch.sigmoid(self.event_constrain(event_vectors)).unsqueeze(1)  # [batch_size, 1, n_arguments]

        token_vectors = self.dropout(F.gelu(token_vectors))
        start_probs = torch.sigmoid(self.start_classifier(token_vectors)) * event_probs  # [batch_size, n_tokens, n_arguments]
        end_probs = torch.sigmoid(self.end_classifier(token_vectors)) * event_probs  # [batch_size, n_tokens, n_arguments]
        
        return start_probs, end_probs


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout_rate: float = 0.1):
        if hidden_size % n_heads != 0:
            raise ValueError("head_size 应该是 hidden_size 的倍数")

        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        self.norm_value = math.sqrt(self.head_size)

        super(MultiHeadSelfAttention, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key   = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input: Tensor, attn_mask: Tensor) -> Tensor:
        # input: shape 都是 [batch_size, n_tokens, hidden_size]
        # attn_mask: 0 参与计算, -10000 不参与计算, shape 是 [batch_size, n_tokens]

        batch_size, n_tokens, _ = input.shape 
        
        def transpose(x):
            x = x.view(batch_size, n_tokens, self.n_heads, self.head_size)
            return x.permute(0, 2, 1, 3)
        
        def untranspose(x):
            x = x.permute(0, 2, 1, 3).contiguous()
            return x.view(batch_size, n_tokens, self.n_heads * self.head_size)

        query_matrix = transpose(self.query(input))  # [batch_size, n_heads, n_tokens, head_size]
        key_matrix   = transpose(self.key(input))
        value_matrix = transpose(self.value(input))

        """
        从本质上来说, 这里的 attention 就是对 query 向量的重新编码, 编码后的向量是 Value 矩阵的线性组合, 线性组合的系数由 query 向量和 Key 矩阵点乘得到。
        因此, Key 矩阵和 Value 矩阵需要保证 "向量" 的数量是一致的, Query 矩阵和 Key 举证需要保证 "特征" 的数量是一致的。
        """
        attn_scores = torch.matmul(query_matrix, key_matrix.transpose(-1, -2))  # [batch_size, n_heads, n_tokens_query, n_tokens_key]
        attn_scores = attn_scores / self.norm_value
        attn_scores = attn_scores + attn_mask[:, None, :, None]
        attn_probs = torch.softmax(attn_scores, dim=-1)  # [batch_size, n_heads, n_tokens_query, n_tokens_key]
        attn_probs = self.dropout(attn_probs)

        context_matrix = torch.matmul(attn_probs, value_matrix)  # [batch_size, n_heads, n_tokens, head_size]
        context_matrix = untranspose(context_matrix)  # [batch_size, n_tokens, hidden_size]

        return context_matrix


class CondLayerNorm(nn.Module):

    def __init__(self, input_size: int, cond_size: int, epsilon: float = 1e-5):
        super(CondLayerNorm, self).__init__()
        self.epsilon = epsilon

        self.weight_transform = nn.Linear(cond_size, input_size)  # 对应 标准差 / gamma
        self.bias_transform = nn.Linear(cond_size, input_size)  # 对应 均值 / beta

        self.reset_parameters()

    def reset_parameters(self):
        # 默认情况下, 转换后的 标准差 是 1
        torch.nn.init.constant_(self.weight_transform.weight, 0)
        torch.nn.init.constant_(self.weight_transform.bias, 1)

        # 默认情况下, 转换后的 均值 是 0
        torch.nn.init.constant_(self.bias_transform.weight, 0)
        torch.nn.init.constant_(self.bias_transform.bias, 0)

    def forward(self, input: Tensor, condition: Tensor) -> Tensor:
        # input: [batch_size, n_tokens, hidden_size]
        # condition: [batch_size, hidden_size]

        # ## step1: 标准化
        # layer normalization 本身就是在 hidden_size 维度上进行归一化
        mean = torch.mean(input, dim=-1, keepdim=True)  # [batch_size, n_tokens, 1]
        variance = torch.var(input, dim=-1, keepdim=True, unbiased=False)  # [batch_size, n_tokens, 1]
        std = torch.sqrt(variance + self.epsilon)  # [batch_size, n_tokens, 1]
        input = (input - mean) / std  # [batch_size, n_tokens, hidden_size]

        # ## step2: 反标准化
        # 每一个样本, 根据 事件/触发词 进行反标准化
        weight = torch.unsqueeze(self.weight_transform(condition), dim=1)  # [batch_size, 1, input_size]
        bias = torch.unsqueeze(self.bias_transform(condition), dim=1)      # [batch_size, 1, input_size]
        output = (input * weight) + bias  # [batch_size, n_tokens, input_size]

        return output


if __name__ == '__main__':
    
    def gen_attn_mask(batch_size, n_tokens):
        # 0: 不参与计算, 1: 参与计算
        attn_mask = torch.randint(low=0, high=2, size=(batch_size, n_tokens), dtype=torch.float32)
        # -10000: 不参与计算, 0: 参与计算
        attn_mask = -10000. * (1 - attn_mask) 
        
        return attn_mask
    
    def test_event_predictor():
        batch_size, n_tokens, hidden_size, event_size = 1, 32, 768, 40

        predictor = EventPredictor(hidden_size=hidden_size, dropout_rate=0.5)

        logits = predictor(
            token_vectors=torch.randn(batch_size, n_tokens, hidden_size),
            event_vectors=torch.randn(event_size, hidden_size),
            attn_mask=gen_attn_mask(batch_size, n_tokens)
        )

        print(logits.shape)

        assert logits.shape == (batch_size, event_size)

    def test_trigger_recognizer():
        batch_size, n_tokens, hidden_size, n_heads = 1, 32, 768, 12

        recognizer = TriggerRecognizer(hidden_size, n_heads)

        start_logits, end_logits = recognizer(
            token_vectors=torch.randn(batch_size, n_tokens, hidden_size),
            attn_mask=gen_attn_mask(batch_size, n_tokens)
        )

        print(start_logits.shape)
        print(end_logits.shape)

        assert start_logits.shape == (batch_size, n_tokens, )
        assert end_logits.shape == (batch_size, n_tokens, )

    def test_argument_recognizer():
        batch_size, n_tokens, hidden_size, n_heads, n_arguments = 1, 32, 768, 12, 50
        
        recognizer = ArgumentRecognizer(n_arguments, hidden_size, n_heads)
        
        start_logits, end_logits = recognizer.forward(
            token_vectors=torch.randn(batch_size, n_tokens, hidden_size), 
            trigger_mask=torch.randint(low=0, high=2, size=(batch_size, n_tokens), dtype=torch.float32), 
            attn_mask=gen_attn_mask(batch_size, n_tokens),
            pos_ids=torch.randint(low=-512, high=512, size=(batch_size, n_tokens), dtype=torch.int64),
            event_vectors=torch.randn(batch_size, hidden_size)
        )
        
        print(start_logits.shape)
        print(end_logits.shape)
        
        assert start_logits.shape == (batch_size, n_tokens, n_arguments)
        assert end_logits.shape == start_logits.shape

    test_event_predictor()
    test_trigger_recognizer()
    test_argument_recognizer()
