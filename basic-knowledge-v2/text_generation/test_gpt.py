# -*- coding:utf-8 -*-
# Author: lqxu

import os 
import torch 

# ## GPT1 对应的是 OpenAIGPTModel, 位于 openai 文件夹下
# from transformers.models.openai import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from transformers.models.gpt2 import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2DoubleHeadsModel

os.environ["TRANSFORMERS_OFFLINE"] = "1"

gpt_name = "IDEA-CCNL/Wenzhong-GPT2-110M"


def test_basic():
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
    model: GPT2Model = torch.no_grad()(GPT2Model.from_pretrained(gpt_name).eval())

    # ## 测试输入
    inputs = tokenizer("openai 是一家什么样的公司?", return_tensors="pt")  # 只有两个, input_ids 和 attention_mask    
    print("输入的 key 值有: ", list(inputs.keys()))

    ## 测试分词
    # GPT1 使用的是 Byte-Pair-Encoding 分词算法, BERT 使用的是 WordPiece 分词算法
    # GPT2 使用的是 byte-level Byte-Pair-Encoding 分词算法 (之后需要去了解相关的内容)
    input_ids = inputs["input_ids"][0].tolist()
    print(tokenizer.decode(input_ids))  # GPT2 不会在中文字符之间加空格, 并且会保留空格
    print([tokenizer.decode(input_id) for input_id in input_ids])

    # ## 测试模型输出
    outputs = model(**inputs)
    print("输出的 key 值有: ", list(outputs.keys()))
    
    # ## 测试输出的向量
    # GPT 就是将 encoder 中的 self attention 模块变成 masked self attention 模块; 或者说是将 decoder 中的 cross attention 模块去掉
    # 也就是说, 在编码每一个 token 时, 只考虑在其之前的 token 向量, 不考虑其之后的 token 向量
    # 换言之, 每一层的某一个 token 编码, 用到的是上一层这个 token 之前的 向量
    print("最终输出的词向量 shape 是: ", list(outputs.last_hidden_state.shape))  # 输出形式和 bert 没有区别, 但是计算方式有区别
    # past_key_values 用于加速文本生成
    print("past_key_values 信息:", len(outputs.past_key_values), len(outputs.past_key_values[0]), list(outputs.past_key_values[0][0].shape))


def test_clm_for_inference():
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
    model: GPT2LMHeadModel = torch.no_grad()(GPT2LMHeadModel.from_pretrained(gpt_name).eval())
    
    batch_inputs = tokenizer("openai 是一家什么样的公司?", return_tensors="pt")  # 只有两个, input_ids 和 attention_mask    
    outputs = model(**batch_inputs)
    
    print("输出的 key 值有: ", list(outputs.keys()))
    
    logits = outputs.logits
    print("计算出来的 logits 的 shape 是: ", list(logits.shape))  # [batch_size, n_tokens, vocab_size]


def test_clm_for_train():
    # https://huggingface.co/docs/transformers/tasks/language_modeling 
    from transformers import DataCollatorForLanguageModeling

    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
    # pad token 和 eos token 可以是一致的, 生成 pad token 和 eos token 效果是一样的
    # 不过 CLM 任务中不需要 eos token 就是了
    tokenizer.pad_token = tokenizer.eos_token
    model: GPT2LMHeadModel = torch.no_grad()(GPT2LMHeadModel.from_pretrained(gpt_name).eval())
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sentences = ["openai 是一家什么样的公司?", "今天的天气如何?"]
    inputs = [dict(tokenizer(sentence)) for sentence in sentences]
    batch_inputs = data_collator(inputs)
    print(batch_inputs)
    # GPT 中的 CLM 任务和 seq2seq 任务还是有差别的
    # GPT 的 CLM 预训练任务就是预测下一个字符, 没有 终止符! 你可以将其理解为 `小说续写` 任务, 只要不中断, 就会一直生成下去
    # 在计算 loss 时, 最后一个 token 预测的结果会忽略
    print("input_ids 和 labels 是否一致: ", torch.all(batch_inputs["input_ids"][0] == batch_inputs["labels"][0]).item())
    
    # 计算 loss, 移动一下位置即可
    batch_outputs = model(**batch_inputs)
    print(torch.nn.functional.cross_entropy(
        batch_outputs.logits[:, :-1].transpose(-1, -2), 
        batch_inputs["labels"][:, 1:])
    )
    print(batch_outputs.loss)


def test_pipeline():
    from transformers.pipelines import TextGenerationPipeline
    
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(gpt_name)
    
    generator = TextGenerationPipeline(model, tokenizer)

    # greedy search
    print("贪婪搜索测试")
    print(generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=False))
    # print(generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=False, no_repeat_ngram_size=3))
    print("===" * 20, end="\n\n")

    # # beam search
    # print("束搜索测试")
    # print("没有加 n-gram penalty")
    # for result in generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=False, num_beams=5, num_return_sequences=5):
    #     print(result)

    # print("加入 n-grams penalty")
    # for result in generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=False, num_beams=5, no_repeat_ngram_size=3, num_return_sequences=5):
    #     print(result)
    # print("===" * 20, end="\n\n")
    
    # # sampling
    # print("文本生成 sampling 测试")
    # for _ in range(10):
    #     print(generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=True, top_k=0))
    # print("===" * 20, end="\n\n")
    
    # print(generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=True, top_k=0, temperature=0.7))
    # print("===" * 20, end="\n\n")
    
    # print(generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=True, top_k=0, temperature=0.0001))
    # print("===" * 20, end="\n\n")
    
    # print(generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=True, top_k=5))
    # print("===" * 20, end="\n\n")
    
    # print(generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=True, top_k=0, top_p=0.95))
    # print("===" * 20, end="\n\n")

    # print(generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=True, top_k=50, top_p=0.95))
    # print("===" * 20, end="\n\n")

    print("Contrastive Search 测试")
    print("alpha=0.6:", generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=False, top_k=5, penalty_alpha=0.6))
    print("alpha=0.0:", generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=False, top_k=5, penalty_alpha=0.0001))  # 如果设置成 0.0, transformers 会识别成 greedy search
    print("===" * 20, end="\n\n")


def draw_pics():
    """ 画 contrastive search 博客 5.3 节中的图 """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers.pipelines import TextGenerationPipeline
    
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(gpt_name).eval()
    generator = TextGenerationPipeline(model, tokenizer)
    
    gs_result = generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=False)[0]["generated_text"]
    cs_result = generator("北京是中国的", pad_token_id=0, max_length=150, do_sample=False, top_k=5, penalty_alpha=0.6)[0]["generated_text"]
    
    print(gs_result)
    print(cs_result)
    
    model = torch.no_grad()(model)
    gs_token_vectors = model(**tokenizer(gs_result, return_tensors="pt"), output_hidden_states=True).hidden_states[-1][0].detach().cpu().numpy()
    cs_token_vectors = model(**tokenizer(cs_result, return_tensors="pt"), output_hidden_states=True).hidden_states[-1][0].detach().cpu().numpy()
    
    gs_sim = cosine_similarity(gs_token_vectors)
    cs_sim = cosine_similarity(cs_token_vectors)
    
    plt.subplot(1, 2, 1)
    sns.heatmap(gs_sim, cmap=plt.get_cmap('Greens'), xticklabels=False, yticklabels=False, square=True)
    plt.subplot(1, 2, 2)
    sns.heatmap(cs_sim, cmap=plt.get_cmap('Greens'), xticklabels=False, yticklabels=False, square=True)
    plt.show()
    
    print(gs_token_vectors.shape, cs_token_vectors.shape)

    
def test_generate():
    
    from transformers.modeling_utils import GenerationMixin
    
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
    model: GPT2LMHeadModel | GenerationMixin = GPT2LMHeadModel.from_pretrained(gpt_name)

    question = "北京是中国的"
    inputs = tokenizer(question, return_tensors='pt')
    generation_output = model.generate(
        **inputs, eos_token_id=50256, pad_token_id=0,  # 基础参数
        return_dict_in_generate=True, output_scores=True,  # 输出形式 参数
        max_length=150, do_sample=True, top_p = 0.6, num_return_sequences = 5  # top-p 采样参数
    )

    for idx, sentence in enumerate(generation_output.sequences):
        print(f"序列 {idx}: ")
        print(tokenizer.decode(sentence).split("<|endoftext|>")[0])
        print("*" * 40)


def test_double_heads():
    # 官方的例子有点抽象, 其实就是加了一个 next sentence predictions 任务 (不知道为什么用 multiple-choice classification 这种说法)
    # reference: https://zhuanlan.zhihu.com/p/452665587 
    
    # 关于 ROCStories, 参考:
    # https://cs.rochester.edu/nlp/rocstories/ 
    # https://github.com/darr/gpt/blob/master/rocstories_dataset.py 
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
    model: GPT2DoubleHeadsModel = GPT2DoubleHeadsModel.from_pretrained(gpt_name)

    # Add a [CLS] to the vocabulary (we should train it also!)
    num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    # Update the model embeddings with the new vocabulary size
    embedding_layer = model.resize_token_embeddings(len(tokenizer))
    
    model = torch.no_grad()(model.eval())

    choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]  # 这里压根就没有构成句子对, 应该还是要加 分割 token 的
    encoded_choices = [tokenizer.encode(s) for s in choices]
    cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

    input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
    mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

    outputs = model(input_ids, mc_token_ids=mc_token_ids)
    lm_logits = outputs.logits
    mc_logits = outputs.mc_logits
    
    print(mc_logits)


def test_cls():
    from transformers.models.gpt2 import GPT2ForSequenceClassification
    
    num_labels = 5
    sentence = "北京是中国的"
    
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
    
    # 个人觉得, 这里的模型架构不太好, 分类层只有简单的 `nn.Linear`, 还没有 bias 参数
    # 正常情况下, 应该是 线性 -> 激活 -> 线性 的架构
    # 需要注意的是, 和 BERT 模型不同, 这里的 句向量 是 最后一个词向量 !!! 不能选第一个词向量 !!!
    model: GPT2ForSequenceClassification = GPT2ForSequenceClassification.from_pretrained(gpt_name, num_labels=num_labels)
    inputs = tokenizer(sentence, return_tensors="pt")    
    outputs = model(**inputs)

    print(outputs.logits.shape)
    print(outputs.logits.shape[1] == num_labels)


def test_token_cls():
    from transformers.models.gpt2 import GPT2ForTokenClassification
    
    num_labels = 5
    sentence = "北京是中国的"
    
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_name)
    
    # 这里还是只有单一的 线性层, 但是有 bias 了
    model: GPT2ForTokenClassification = GPT2ForTokenClassification.from_pretrained(gpt_name, num_labels=num_labels)
    inputs = tokenizer(sentence, return_tensors="pt")    
    outputs = model(**inputs)

    print(outputs.logits.shape)


if __name__ == "__main__":
    
    import logging 
    from transformers.modeling_utils import logger
    logger.setLevel(logging.CRITICAL)
    
    from io import StringIO
    from contextlib import redirect_stderr
    
    empty_stream = StringIO()
    with redirect_stderr(empty_stream):
        test_token_cls()
