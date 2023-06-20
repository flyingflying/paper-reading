# -*- coding:utf-8 -*-
# Author: lqxu

import os 

import torch

os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 加速

bert_name = "hfl/chinese-bert-wwm-ext"


@torch.no_grad()
def test_mlm_pipeline():
    from transformers.pipelines import FillMaskPipeline

    from transformers.models.bert import BertForMaskedLM, BertTokenizerFast

    model = BertForMaskedLM.from_pretrained(bert_name).eval()
    tokenizer = BertTokenizerFast.from_pretrained(bert_name)

    fill_masker = FillMaskPipeline(model=model, tokenizer=tokenizer)

    results = fill_masker("生活的真谛是[MASK]。", top_k=5)
    # results = fill_masker("生活的真谛是[MASK]。", top_k=5, targets=["真", "善", "美"])
    # results = fill_masker("生[MASK]的真谛是[MASK]。", top_k=1)

    for result in results:
        print(result)


@torch.no_grad()
def test_mlm_for_inference():
    from transformers.models.bert import BertForMaskedLM, BertTokenizerFast

    model = BertForMaskedLM.from_pretrained(bert_name).eval()
    tokenizer = BertTokenizerFast.from_pretrained(bert_name)

    inputs = tokenizer("生活的真谛是[MASK]。", return_tensors="pt")
    logits = model(**inputs).logits.detach().cpu().numpy()

    print(logits.shape)

    print(tokenizer.decode(inputs["input_ids"][0].tolist()))
    print(tokenizer.decode(logits.argmax(axis=-1)[0].tolist()))
    
    # 在 HuggingFace 的模型主页上, 关于 Fill-Mask 的测试, 如果序列中的 [MASK] 数量超过一个, 就会有 undefined 错误 !!!


@torch.no_grad()
def test_mlm_for_train():
    import math 
    from transformers.data import DataCollatorForLanguageModeling
    from transformers.models.bert import BertTokenizerFast, BertForMaskedLM

    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(bert_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    sentences = ["生活的真谛是懒。"]
    inputs = [dict(tokenizer(sentence)) for sentence in sentences]
    while True:
        # 对于 transformers 中的 data collator 来说, 每一个 样本 都是一个字典 !!!
        # data collator 返回的是一个 字典 !!! 注意数据的 "嵌入" 关系
        batch_inputs = data_collator(inputs)
        
        # 如果生成的标签都是 -100, 那么就再次生成
        if not torch.all(batch_inputs["labels"] == -100).item():
            break

    print("token 和 label 的对应关系表")
    # 从 input_ids 中随机选择 mlm_probability 比例的 token, 其中 80% 替换成 [MASK], 10% 随机替换, 10% 不变; 未被选中的 token 标签都是 -100
    # data_collator.torch_mask_tokens(tokenizer(sentences, return_tensors="pt")["input_ids"])
    for input_id, label in zip(batch_inputs["input_ids"][0], batch_inputs["labels"][0]):
        print(input_id.item(), label.item())
    print("===" * 20)
    
    model = BertForMaskedLM.from_pretrained(bert_name).eval()
    batch_outputs = model(**batch_inputs)
    
    # 对于 MLM 模型来说, preplexity 的定义并不明确, 直接借鉴 CLM 的方式 (https://huggingface.co/docs/transformers/tasks/language_modeling#train)
    # 对于 CLM 模型来说, perplexity is defined as the exponentiated average negative log-likelihood of a sequence.
    # reference: https://huggingface.co/docs/transformers/perplexity 
    perplexity = math.exp(batch_outputs.loss.item())
    print("preplexity 的值是: ", round(perplexity, 4))  # 不唯一, 和输入的关系很大


if __name__ == "__main__":
    
    """ HuggingFace 的 MLM 教程文档: https://huggingface.co/docs/transformers/tasks/masked_language_modeling """
    
    from io import StringIO
    from contextlib import redirect_stderr
    
    empty_stream = StringIO()
    
    with redirect_stderr(empty_stream):
        # test_mlm_pipeline()
        # test_mlm_for_inference()
        test_mlm_for_train()
