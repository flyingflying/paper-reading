# -*- coding:utf-8 -*-
# Author: lqxu
# reference: https://github.com/huggingface/peft/blob/main/examples/sequence_classification

#%% 防止网络错误

import os 
os.environ["TRANSFORMERS_OFFLINE"] = "1"

#%% 加载模型

from transformers.models.bert import BertTokenizerFast
from transformers.models.bert import BertForSequenceClassification

PRETRAINED_NAME = "hfl/chinese-bert-wwm-ext"

tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_NAME)
model = BertForSequenceClassification.from_pretrained(PRETRAINED_NAME, num_labels=2)

#%% peft 模型

from peft import get_peft_model, PeftType, TaskType, PeftModelForSequenceClassification

from peft import PromptEncoderConfig, PrefixTuningConfig, PromptTuningConfig

# ## p-tuning
peft_config = PromptEncoderConfig(
    task_type=TaskType.SEQ_CLS, num_virtual_tokens=20, encoder_hidden_size=128, 
)
# ## prefix-tuning
peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=20)
# ## prompt-tuning
peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=10)

peft_model: PeftModelForSequenceClassification = get_peft_model(model, peft_config)

#%% 测试

from torch import no_grad

peft_model.eval()

peft_model.save_pretrained

peft_model.get_prompt

model.bert.forward

with no_grad():
    inputs = tokenizer("今天天气真好", return_tensors="pt")
    print(inputs.keys())
    
    outputs = peft_model.forward(**inputs, output_attentions=True, output_hidden_states=True)
    
    print("logits:", outputs.logits)
    
    print(outputs.attentions[0].shape)
    print(outputs.attentions[-1].shape)
    
    print(outputs.hidden_states[-1].shape)
