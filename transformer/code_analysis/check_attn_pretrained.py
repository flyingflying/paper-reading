# -*- coding:utf-8 -*-
# Author: lqxu

#%%
import os 
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 加速

import math 
import torch 
from torch import Tensor

#%%

from transformers.models.bert import BertForMaskedLM, BertTokenizer

_MODEL_NAME = "hfl/chinese-bert-wwm-ext"
model = BertForMaskedLM.from_pretrained(_MODEL_NAME).eval()
tokenizer = BertTokenizer.from_pretrained(_MODEL_NAME)

# 超过 512 个 tokens
_LONG_SENTENCE = """\
2010年，全球瞩目中国，中国聚焦上海。世博会的盛大开幕，为这座国际化大都市注入了全新的活力。作为中国餐饮业领跑者的肯德基，也抓住了这个百年难遇的契机。6月1日，伴随着中国大陆第3000家餐厅在上海开业，并同步启用全新品牌口号“生活如此多娇”，肯德基以独有的方式与世博共襄盛举，再次用实际行动诠释“立足中国、融入生活”的总策略。
6月1日，上海漕宝路星星广场，锣鼓喧天，鞭炮齐鸣，肯德基莘漕餐厅喜庆开业。这标志着中国肯德基再次打破了自己创造的开店记录，以全国3000家餐厅的数量继续领跑业界。
距离去年2600家花落郑州，时隔不到1年，肯德基就兑现了当时的承诺，完成了3000家的目标。1987年到2004年，中国肯德基差不多花费了17年的时间苦练基本功，开出了1000家；而随后的6年时间，厚积薄发，一口气就开出了2000家。这一组组光鲜数字的背后，是无数肯德基人付出的辛劳和努力，当然最重要的还是广大消费者对这个品牌的不离不弃。
2009年，金融危机席卷全球，中国肯德基经受住了考验，并逆势前进，年开店数远超年初不少于300家的目标，首次突破400家，新进城市30多个。今年依然保持强劲势头。一年过半，已经交上了200多家的骄人业绩。如今3000家餐厅遍及中国大陆除西藏以外的30个省、市、自治区的500余座城市，从北京、上海、广州、深圳等大都市，到省会城市，再到县级市，甚至乡镇等，老百姓都可以很容易地找到肯德基这个熟悉的标识，享受到他带来的美味和便捷。
"""

tokenizer_kwargs = {"max_length": 512, "truncation": True, "padding": "max_length"}

# %%

with torch.no_grad():
    
    sentences = [
        "今天天气不错", 
        "如果您还有什么疑惑或建议，欢迎在下方评论区继续讨论。如果您觉得本文还不错，欢迎分享/打赏本文。打赏并非要从中获得收益，而是希望知道科学空间获得了多少读者的真心关注。当然，如果你无视它，也不会影响你的阅读。再次表示欢迎和感谢！",
        _LONG_SENTENCE
    ]

    inputs = tokenizer(sentences, return_tensors="pt", **tokenizer_kwargs)
    lengths = inputs["attention_mask"].sum(dim=1).int().tolist()
    outputs = model.forward(**inputs, output_attentions=True)

    attn_matrix = torch.concat(outputs.attentions, dim=1).to(torch.float64)
    query_entropy = -(attn_matrix * (attn_matrix + 1e-8).log()).sum(dim=-1)
    results = query_entropy.mean(dim=[1, 2]).tolist()
    
    for length, result in zip(lengths, results):
        print(f"{length}: {round(result, 4)} vs {round(math.log(length), 4)}")

# %%

@torch.no_grad()
def mlm_accuracy(
        input_length: int = 64, 
        batch_size: int = 1, 
        run_all: bool = False, 
        num_sentences: int = 10000,
        device: str = "cpu"
    ):

    from tqdm import tqdm
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers.data import DataCollatorForLanguageModeling
    
    model.to(device)

    dataset = load_dataset(path="graelo/wikipedia", name="20230901.zh")
    # dataset = load_dataset(path="wiki.zh")
    tokenizer_kwargs = {"max_length": input_length, "truncation": True, "padding": "max_length"}
    
    if run_all:
        dataset = dataset["train"].map(
            lambda batch: tokenizer(batch["text"], **tokenizer_kwargs), 
            batched=True, batch_size=1000, remove_columns=dataset["train"].column_names
        )
    else:
        dataset = [
            tokenizer(sentence, **tokenizer_kwargs)
            for sentence in dataset["train"][:num_sentences]["text"]
        ]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    total_num, correct_num = 0, 0
    
    for inputs in tqdm(dataloader):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)

        labels = inputs["labels"]
        output_ids = outputs.logits.argmax(dim=-1)
        
        total_num += (labels != -100).sum()
        correct_num += ((output_ids == labels) & (labels != -100)).sum()
    
    accuracy =  (correct_num / total_num).item() * 100
    return accuracy


mlm_accuracy(64)

# ## accuracy: 
# mlm_accuracy(64, batch_size=1000, num_sentences=None, device="cuda:0", run_all=True)

#%%

import einops
import pandas as pd 

with torch.no_grad():

    inputs = tokenizer(_LONG_SENTENCE, return_tensors="pt", **tokenizer_kwargs)
    
    outputs = model.forward(**inputs, output_attentions=True)

    attention_matrix = einops.rearrange(
        torch.cat(outputs["attentions"], dim=1), 
        "n_batchs n_heads n_query_tokens n_key_tokens -> (n_batchs n_heads) n_query_tokens n_key_tokens"
    )
    
    _, sigma_vectors, _ = torch.linalg.svd(attention_matrix)
    
    results = torch.cumsum(sigma_vectors, dim=1) / torch.sum(sigma_vectors, dim=1, keepdim=True)

    positions = [torch.where(result > 0.80)[0][0].item() for result in results]
    
    print(positions)
    print("平均位置:", round(sum(positions) / len(positions), 4))
    
    pd.Series(positions).hist()

    # [num_heads, num_queries, num_keys]


# %%

from transformers.models.bert import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertSelfAttention

_MODEL_NAME = "hfl/chinese-bert-wwm-ext"


def forward(
        self, hidden_states, attention_mask = None, head_mask = None, 
        encoder_hidden_states = None, encoder_attention_mask = None, past_key_value = None, 
        output_attentions = False
    ):

    query_layer = self.transpose_for_scores(self.query(hidden_states))
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)
    outputs = (context_layer, torch.stack([attention_scores, attention_probs])) if output_attentions else (context_layer,)

    return outputs


BertSelfAttention.forward = forward

model = BertForMaskedLM.from_pretrained(_MODEL_NAME).eval()
tokenizer = BertTokenizer.from_pretrained(_MODEL_NAME)


def matrix_sparsity(tensor: Tensor, eps: float = 1e-8):
    num_zero_elements = (tensor.abs() < eps).sum()
    total_elements = tensor.numel()
    return num_zero_elements / total_elements


with torch.no_grad():

    inputs = tokenizer(_LONG_SENTENCE, return_tensors="pt", **tokenizer_kwargs, return_attention_mask=False)
    outputs = model.forward(**inputs, output_attentions=True)

    # [num_heads, num_queries, num_keys]
    attn_results = torch.concat(outputs.attentions, dim=2).squeeze()
    attn_score_matrices = attn_results[0]
    attn_probs_matrices = attn_results[1]
    
    for attn_matrix in attn_score_matrices:
    # for attn_matrix in attn_probs_matrices:
        
        attn_matrix[attn_matrix.abs() < 1e-4] = 0.0
        
        length = attn_matrix.size(0)
        rank = torch.linalg.matrix_rank(attn_matrix)
        sparsity = round(matrix_sparsity(attn_matrix).item() * 100, 2)
        
        print(f"序列长度: {length}, 矩阵的秩: {rank}, 矩阵稀疏度: {sparsity}")

# %%
