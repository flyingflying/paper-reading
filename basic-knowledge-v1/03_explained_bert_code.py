"""
copied from https://github.com/huggingface/transformers/blob/v4.20.0/src/transformers/models/bert/modeling_bert.py
at 2022-08-30, modified by lqxu
注意: 这个文件是用来读代码的, 和原版的相比, 删了很多内容, 没有经过测试, 不要直接引用这个文件中的内容 !!!
"""

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import LongTensor, Tensor

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import ModelOutput
from transformers.models.bert.configuration_bert import BertConfig


class BertEmbeddings(nn.Module):
    """ 嵌入层 """
    position_ids: LongTensor    # [1, config.max_position_embeddings]
    token_type_ids: LongTensor  # [1, config.max_position_embeddings]

    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.position_embeddings = None

        # LayerNorm 用驼峰命名法, 是为了和 tensorflow 中的命名法保持一致
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)
        token_type_ids = torch.zeros(self.position_ids.size()).long()
        # 这里 token_type_ids 用 non-persistent buffer 的原因是其创建没有用到 config 中的参数
        self.register_buffer("token_type_ids", token_type_ids, persistent=False)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # type: (LongTensor, LongTensor, LongTensor, Tensor) -> Tensor
        """\
        :param input_ids:      [batch_size, seq_len]
        :param token_type_ids: [batch_size, seq_len]
        :param position_ids:   [batch_size, seq_len]
        :param inputs_embeds:  [batch_size, seq_len, hidden_size] custom word embeddings
        :return:               [batch_size, seq_len, hidden_size]
        """

        # input_ids 和 inputs_embeds 两者应当是一个为 None, 一个不为 None
        # 如果不想用 BERT 的 word embeddings, 则可以直接传入 inputs_embeds, 作为词嵌入
        if input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)
        seq_length = inputs_embeds.size(1)

        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """ Attention 层:  包含 self attention, masked self attention 和 cross attention 层 """
    position_ids: Optional[LongTensor]  # [config.max_position_embeddings, ]

    def __init__(self, config: BertConfig, position_embedding_type: str = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 剪枝的时候会用到

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type in ["relative_key", "relative_key_query"]:
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
            self.register_buffer("position_ids", torch.arange(self.max_position_embeddings))
        else:
            self.max_position_embeddings = self.distance_embedding = self.position_ids = None

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_len, hidden_size] -> [batch_size, num_heads, seq_len, head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None,
                output_attentions=False):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tuple[Tensor, Tensor], bool) -> Tuple[Tensor, ...]
        """\
        :param hidden_states:          [batch_size, seq_len, hidden_size]
        :param attention_mask:         [batch_size, 1, 1, seq_len_key]
        :param head_mask:              [1, num_heads, 1, 1]
        :param encoder_hidden_states:  [batch_size, encoder_seq_len, hidden_size]
        :param encoder_attention_mask: [batch_size, 1, seq_len_query, seq_len_key]
        :param past_key_value:         [batch_size, num_heads, seq_len, head_size], 两个
        :param output_attentions: 是否返回 attention_probs
        :return: [batch_size, seq_len, hidden_size], [batch_size, num_heads, seq_len_query, seq_len_key]
        """

        # query_layer, key_layer 和 value_layer 的 shape 都是 [batch_size, num_heads, seq_len, head_size]
        # 如果进行了剪枝, 只会影响 num_heads, 其它维度不影响
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if encoder_hidden_states is not None:  # 是 cross attention
            """
            对于 cross attention 来说, 是将编码器的编码后的句子 (encoder_hidden_states) 进行线性变换后作为 KEY 矩阵和 VALUE 矩阵, 
            解码器输入的词向量 (hidden_states) 进行线性变换后作为 QUERY 矩阵, 进行 attention 计算。需要注意:
            1. mask 用的是 encoder 的 mask, 不是 decoder 的 mask (一个 query 和 所有 key 向量点乘, mask, 然后 softmax 归一化)
            2. 在生产过程中, 由于 decoder 输入的句子是一个一个词递增的, 那么会导致大量 KEY 矩阵和 VALUE 矩阵的重复计算。
               因此, 第一次计算完成后, 可以将 KEY 矩阵和 VALUE 矩阵保存起来, 以加速后面的运算。
            """
            attention_mask = encoder_attention_mask
            if past_key_value is None:  # 生产时第一次计算, 或者是训练 decoder 时
                key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            else:  # 生产时非第一次计算
                key_layer, value_layer = past_key_value
        elif past_key_value is not None:  # 是 masked self attention
            """
            对于 masked self attention 来说, 在计算时, 每一个词只能用其左边的词向量进行编码,不能用其右边的词向量。(标准的语言模型)
            在生产过程中, 在第二次输入时, 有两个词, 但是第一个词向量的编码依然只能用自己, 不能用第二个词向量的信息。
            这样的话, 无论后面输入的词有几个, 前面输入的词编码一直是不变的。
            总上所说, 我们可以简化计算, 每一步只输入一个词, 即上一步输出的词, 这样的话:
            1. 所有的线性层和 LayerNorm 没有问题, 因为都是对每一个词向量进行操作的;
            2. cross attention 层没有问题, KEY 矩阵和 VALUE 矩阵是确定的, 只要计算当前词语的 query 向量即可
            3. masked self attention 层需要将上一次的 KEY 矩阵和 VALUE 矩阵保存下来, 和当前词语的 key, value 向量 concat 在一起就行
            """
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:  # self attention
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 将 "query" 和 "key" 进行矩阵点乘得到 raw attention scores
        # attention_scores: [batch_size, num_heads, seq_len_query, seq_len_key]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.distance_embedding is not None:
            seq_length = hidden_states.size(1)
            position_ids = self.position_ids[:seq_length]
            # distance: [seq_len_query, seq_len_key], 每一个 query 向量到每一个 key 向量之间的相对距离
            # query 向量和 key 向量之间的相对距离计算公式为 query_index - key_index
            # 相对位置的值域为 [-max_seq_len+1, max_seq_len-1]=[-511, 511]
            distance = position_ids.view(-1, 1) - position_ids.view(1, -1)
            # positional_embedding: [seq_len_query, seq_len_key, head_size]
            positional_embedding = self.distance_embedding(distance + (self.max_position_embeddings - 1))
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                """
                https://arxiv.org/pdf/1803.02155.pdf 论文中的 公式 5
                relative_position_scores: [batch_size, num_heads, seq_len_query, seq_len_key]
                这里的代码虽然是和 query_layer 进行点乘操作, 但是本质上来说还是和 key_layer 相加, 是往 key 向量中融入相对位置信息
                代码的含义: 每一个 query 向量都要和其对应的所有 query-key 相对位置向量进行点乘, 非 einsum 写法:
                relative_position_scores = torch.sum(query_layer.unsqueeze(3) * positional_embedding, dim=-1)
                """
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                # ## https://arxiv.org/pdf/2009.13658.pdf 论文中的 3.5.4 小节 (也就是论文中的 method 4)
                # relative_position_scores_query 和上面的 relative_position_scores 是一致的, 是往 key 向量中融入相对位置信息
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                # relative_position_scores_key 是往 query 向量中融入相对位置信息
                # 代码含义: 每一个 key 向量都要和其对应的所有 query-key 相对位置向量进行点乘
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                # relative_position_scores_key = torch.sum(key_layer.unsqueeze(2) * positional_embedding, dim=-1)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # 关于 attention_mask 的计算方法, 参考: PreTrainedModel.get_extended_attention_mask 方法
            attention_scores = attention_scores + attention_mask

        # 将 attention score 标准化为概率
        # attention_probs: [batch_size, num_heads, seq_len_query, seq_len_key]
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # 这里的 dropout 会导致部分 token 不参与计算
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            # head_mask 的 shape 是 [1, num_heads, 1, 1], 值为 0 或者 1 (0 不参与计算, mask 掉; 1 参与计算, 不 mask 掉)
            attention_probs = attention_probs * head_mask

        # context_layer: [batch_size, seq_len, hidden_size]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    """ BertSelfAttention 层后的 Add & Norm 层 """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """
        :param hidden_states: [batch_size, seq_len, hidden_size]
        :param input_tensor: [batch_size, seq_len, hidden_size]
        :return: [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """ 将 BertSelfAttention 和 BertSelfOutput 层组合在一起 """
    def __init__(self, config: BertConfig, position_embedding_type: str = None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: List[int]):
        # heads 参数表示的是裁剪的 head 索引值, 一般 BERT 的 BertSelfAttention 层有 12 个 head, 索引值范围在 [0, 11]
        # 无论经过了怎么裁剪, 索引值一直是按照原始 head 的索引来, 如果希望按照新的 head 索引值来, 需要清空 pruned_heads 集合, 重新保存权重值
        # 关于如何选择要裁剪的 head, 参考论文: https://arxiv.org/abs/1905.09418
        if len(heads) == 0:
            return
        # find_pruneable_heads_and_indices 函数返回的 heads 表示要裁剪的 head 索引值, 和参数 heads 表示的意思一致
        # 其返回的 index 表示要保留的特征索引, 而不是要删除的特征索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        """" 参数剪枝
        线性层的 weight 维度是 [out_features, in_features], bias 维度是 [out_features, ]
        裁剪后的线性层:
            query, key 和 value 层的 weight 维度是 [pruned_out_features, in_features], bias 维度是 [pruned_out_features, ]
        相当于输入的特征数量不变, 减少输出的特征数量, 也就是减少线性回归数 [每一个输出的特征对应一个线性回归]
        """
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        # dense 层的 weight 维度是 [out_features, pruned_in_features], bias 维度不变, 还是 [out_features, ]
        # 相当于输入的特征数量减少, 输出的特征数量不变, 也就是每一个线性回归的变量数减少了, 如果原本的输入是 [x1, x2, x3], 剪枝掉 x2 后输入就变成了 [x1, x3]
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # ## 更新参数值
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)  # 如果清空这个集合, 一定要重新保存参数值

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        """\
        :param hidden_states:  [batch_size, seq_len, hidden_size]
        :param attention_mask: [batch_size, 1, 1, seq_len_key] 或者 [batch_size, 1, seq_len_query, seq_len_key]
        :param head_mask:      [1, num_heads, 1, 1]
        :return:               [batch_size, seq_len, hidden_size]
        """
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions=False)
        attention_output = self.output(hidden_states=self_outputs[0], input_tensor=hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    """ BERT: feed forward 层 """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        :param hidden_states: [batch_size, seq_len, hidden_size]
        :return: [batch_size, seq_len, intermediate_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """ BertIntermediate 层后的 Add & Norm 层, 和 BertSelfOutput 层是相似的 """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """
        :param hidden_states: [batch_size, seq_len, intermediate_size]
        :param input_tensor: [batch_size, seq_len, hidden_size]
        :return: [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """ Bert Layer: 将 BertAttention, BertIntermediate 和 BertOutput 层组合在一起 """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # type: (Tensor, Tensor, Tensor) -> Tensor

        """\
        :param hidden_states:  [batch_size, seq_len, hidden_size]
        :param attention_mask: [batch_size, 1, 1, seq_len_key] 或者 [batch_size, 1, seq_len_query, seq_len_key]
        :param head_mask:      [1, num_heads, 1, 1]
        :return:               [batch_size, seq_len, hidden_size]
        """

        attention_output = self.attention.forward(
            hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask)
        """
        apply_chunking_to_forward 的含义是将 input_tensors 中的所有 tensor 在 chunk_dim 维度上分块, 每一块的维度是 chunk_size,
        然后每一小块分别输入到 forward_fn 中, 然后将所有的输出 concat 在 chunk_dim 上拼接在一起
        reference: https://huggingface.co/docs/transformers/glossary#feed-forward-chunking
        其目的是防止 intermediate_size 过大导致占用过多的内存, 不过相应的, 计算速度会慢很多
        """
        # noinspection PyTypeChecker
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        # 上面的代码等价于: layer_output = self.feed_forward_chunk(attention_output)
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """ 将 BertLayer 组合在一起 """
    layer: List[BertLayer]

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])  # noqa
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        """\
        :param hidden_states:  [batch_size, seq_len, hidden_size]
        :param attention_mask: [batch_size, 1, 1, seq_len_key]
        :param head_mask:      [num_layers, 1, num_heads, 1, 1]
        :return:               [batch_size, seq_len, hidden_size]
        """

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                # gradient checkpoint 是一种节省显存的计算方式
                # reference: http://www.manongjc.com/detail/27-npfegllifhhciob.html
                # reference: https://pytorch.org/docs/stable/checkpoint.html
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer_module,  # function
                    hidden_states, attention_mask, layer_head_mask  # args
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask)

            hidden_states = layer_outputs

        return hidden_states


class BertPooler(nn.Module):
    """ 将 seq_len 个词向量池化成一个向量 """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 这里的 "池化" 很简单, 直接用 [CLS] 所对应的词向量
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    """ 这就是一个 feed forward 层, 只是 dim_feedforward=hidden_size """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 注意这里使用了 LayerNorm
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """ Language Model 预测层: MLM 和 CLM 都使用 """
    def __init__(self, config):
        super().__init__()
        # BertPredictionHeadTransform 中有 dense 层, 激活层 和 layer norm 层
        # 越靠近输出的层越 task-specified, 也就是说加这个 dense 层让 pre-train 和 fine-tune 的任务有一定性质的区别
        self.transform = BertPredictionHeadTransform(config)

        """
        这里使用了 weight tying 的技巧, 让 input embedding 和 output embedding 的权重保持一致:
            1. 引用 https://paperswithcode.com/method/weight-tying 中的说法:
               Weight tying improves the performance of language models by tying (sharing)
               the weights of the embedding and softmax layer.
            2. 原始论文: https://arxiv.org/pdf/1608.05859v3.pdf 
            3. 我的理解: 这里是做词表级别的分类, 也就是说这里的每一个词的 "分类向量" 就是词嵌入向量, 他们是共享参数的
               对于 MLM 来说, 分类的任务是将 [MASK] 映射到文本中原来的词上面, 也就是说词嵌入中包含了一定的文本映射的能力
               对于 CLM 来说, 分类的任务是在只有上文的前提下去预测下文, 也就是说词嵌入中包含了一定的文本预测的能力
               更多的理解还需要阅读更多关于语言模型的论文, 后面也有论文对这种方法提出了质疑, 认为在数据量足够多的情况下, 不用的效果会更好
            4. 其它引用: 
               a. https://blog.csdn.net/xiaoxu1025/article/details/111623629
               b. https://www.zhihu.com/question/492121451
               c. https://kexue.fm/archives/8747
        """
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # output-only bias, 我认为是为了适配分类任务而设定的
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    """ 只有 MLM 的情况 """
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    """ 只有 NSP 的情况 """
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    """ 综合 NSP 和 MLM 任务"""
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# noinspection PyAbstractClass
class BertPreTrainedModel(PreTrainedModel):
    """ 修改 PreTrainedModel 基类 """

    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _set_gradient_checkpointing(module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# noinspection PyAbstractClass
class BertModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):  # weight tying 中需要的方法
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):  # weight tying 中需要的方法
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        # 详细见: transformers.modeling_utils.PreTrainedModel.prune_heads 方法
        # heads_to_prune 是一个字典对象, key 值是 BertEncoder 中 BertLayer 的索引值, value 值是 BertLayer 中 BertAttention 中 head 的索引值
        # 例子: {1: [0, 2]} 会将第二个 BertLayer 中 BertAttention 的第一个和第三个 head 给剪枝掉
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    # noinspection PyMethodOverriding
    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int, ...]) -> Tensor:
        """\
        attention_mask: 1 (True) 参与计算; 0(False) 不参与计算 \n
        一共有三种 attention 层:\n
            self attention: 只需要掩掉行 padding 即可
            masked self attention: 除了掩掉行 padding 外, 还要掩掉上三角
            cross attention: mask 来源于 encoder, 不需要调用这个函数
        """
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]  # [batch_size, 1, seq_len_query, seq_len_key]
        elif attention_mask.ndim == 2:
            if self.config.is_decoder:
                # 仅仅针对 masked self attention; cross attention 的 mask 来源于 encoder
                batch_size, seq_len = input_shape
                seq_ids = torch.arange(seq_len, device=attention_mask.device)
                # [seq_len_query, seq_len_key] 和 [seq_len_query, 1] 进行比较
                causal_mask: Tensor = seq_ids[None, :].repeat(repeats=(seq_len, 1)) <= seq_ids[:, None]  # noqa
                causal_mask = causal_mask.to(attention_mask.dtype)
                # input 的长度可能小于 mask 的长度 (在有 past_key_value 的情况下), 此时在前面补 1
                actual_length = attention_mask.size(1)
                if seq_len < actual_length:
                    prefix_mask = torch.ones(
                        size=(seq_len, actual_length),
                        device=causal_mask.device, dtype=causal_mask.dtype)
                    causal_mask = torch.cat([prefix_mask, causal_mask], dim=-1)
                # 两者都是 True 时才是 True, 参与计算; 有一个是 False 就不参与计算
                # [batch_size, 1, seq_len_query, seq_len_key]
                extended_attention_mask = causal_mask[None, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len_key]
        else:
            raise ValueError(f"The dimensions of the attention mask should be 2 or 3, but got {attention_mask.ndim}!")

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # 1 ==> 0 and 0 ==> -10000.0
        return extended_attention_mask


# noinspection PyAbstractClass
class BertForPreTraining(BertPreTrainedModel):
    """ BERT 预训练中使用的方法 """
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()  # 里面包含 weight tying

    def get_output_embeddings(self):  # weight tying 中使用的方法
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):  # weight tying 中使用的方法
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            # 注意 MLM 任务中加了额外的 dense 层, 但是 NSP 任务中没有加额外的 dense 层
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss  # 两个 loss 直接相加 !!!

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# noinspection PyAbstractClass
class BertLMHeadModel(BertPreTrainedModel):  # CLM 模型, 吐槽一下, 你为啥不将 CLM 写在类名中呢 ...

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config: BertConfig):
        super().__init__(config)

        if not config.is_decoder:
            warnings.warn("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)  # 吐槽一下, 明明是 CLM 模型 ...

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


# noinspection PyAbstractClass
class BertForMaskedLM(BertPreTrainedModel):  # 只用 MLM

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config: BertConfig):
        super().__init__(config)

        if config.is_decoder:
            warnings.warn(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


# noinspection PyAbstractClass
class BertForNextSentencePrediction(BertPreTrainedModel):  # 只有 NSP
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# noinspection PyAbstractClass
class BertForSequenceClassification(BertPreTrainedModel):  # 文本分类
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# noinspection PyAbstractClass
class BertForMultipleChoice(BertPreTrainedModel):  # 单选
    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,       # [batch_size, num_choices, seq_len], 问题文本 + 选项文本
        attention_mask: Optional[torch.Tensor] = None,  # [batch_size, num_choices, seq_len]
        token_type_ids: Optional[torch.Tensor] = None,  # [batch_size, num_choices, seq_len]
        position_ids: Optional[torch.Tensor] = None,    # [batch_size, num_choices, seq_len]
        head_mask: Optional[torch.Tensor] = None,       # [num_layers, num_heads]
        inputs_embeds: Optional[torch.Tensor] = None,   # [batch_size, num_choices, seq_len, hidden_size]
        labels: Optional[torch.Tensor] = None,          # [batch_size, ]
        output_attentions: Optional[bool] = None,       # bool
        output_hidden_states: Optional[bool] = None,    # bool
        return_dict: Optional[bool] = None,             # bool
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 合并所有输入张量的 batch_size 和 num_choices 维度
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 用 bert 进行编码, 取池化层结果
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]  # [batch_size * num_choices, hidden_size]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size * num_choices, 1]
        reshaped_logits = logits.view(-1, num_choices)  # [batch_size, num_choices]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 就是正常的单标签分类问题
            loss = loss_fct(reshaped_logits, labels)
            # 需要注意的是, 在 MultipleChoice 中, 所有选项都是用同一个线性层变成分数的, 这样就和位置无关了

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# noinspection PyAbstractClass
class BertForTokenClassification(BertPreTrainedModel):  # token 级别的分类

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# noinspection PyAbstractClass
class BertForQuestionAnswering(BertPreTrainedModel):  # span 分类

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # 这里的 num_labels 恒定为 2

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,        # [batch_size, seq_len], 问题文本 + 段落文本
        attention_mask: Optional[torch.Tensor] = None,   # [batch_size, seq_len]
        token_type_ids: Optional[torch.Tensor] = None,   # [batch_size, seq_len]
        position_ids: Optional[torch.Tensor] = None,     # [batch_size, seq_len]
        head_mask: Optional[torch.Tensor] = None,        # [num_layers, num_heads]
        inputs_embeds: Optional[torch.Tensor] = None,    # [batch_size, seq_len, hidden_size]
        start_positions: Optional[torch.Tensor] = None,  # [batch_size, ]
        end_positions: Optional[torch.Tensor] = None,    # [batch_size, ]
        output_attentions: Optional[bool] = None,        # bool
        output_hidden_states: Optional[bool] = None,     # bool
        return_dict: Optional[bool] = None,              # bool
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForQuestionAnswering
        # 用 bert 对词向量重新编码
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, 2]
        start_logits, end_logits = logits.split(split_size=1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # [batch_size, seq_len]
        end_logits = end_logits.squeeze(-1).contiguous()  # [batch_size, seq_len]

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)  # [batch_size, ]
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)  # [batch_size, ]
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)  # seq_len
            start_positions = start_positions.clamp(0, ignored_index)  # 小于 0 的变成 0, 大于 0 的变成 ignored_index
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)  # ignore_index 不再是 -100 了
            start_loss = loss_fct(start_logits, start_positions)  # 进行 seq_len 个数的分类
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


raise RuntimeError("不要使用这个文件")
