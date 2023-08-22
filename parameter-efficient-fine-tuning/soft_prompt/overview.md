
# soft prompt 简介

soft prompt 在 PEFT 系列方法中属于 [additive 加性](https://arxiv.org/abs/2303.15647) 的方法。其核心思想在 [AutoPrompt](autoprompt.md) 博客中已经说明了。

[Prompt-tuning](https://arxiv.org/abs/2104.08691) 就是在句子前面加上若干 伪 (pseudo) token, 希望这些 伪 token 可以用来代替 prompt。这些 伪 token 可以是随机初始化, 也可以是一个人为写好的句子。模型的训练过程就是迭代更新这些 伪 token。

[Prefix-tuning](https://arxiv.org/abs/2101.00190) 不在输入部分增加 伪 token, 而在每一个 attention 的 KEY 矩阵和 VALUE 矩阵后面增加 伪 token。方式和 encoder-decoder 方式很像, 只是现在 attention 中新增的 token 不是 encoder 的输出, 而是一些可学习的参数。模型训练时只更新加在 KEY 矩阵和 VALUE 矩阵中的参数。[PEFT](https://github.com/huggingface/peft) 实现时使用的是 `past_key_value` 参数。(虽然论文中的故事不是这么说的)

[P-tuning](https://arxiv.org/abs/2103.10385) 和 prompt-tuning 差不多, 都是直接在输入处添加 伪 token。不一样的是, 其还让这些 伪 token 之间通过一个双向 LSTM 层, 让这些 伪 token 间构建 "关联"。更多可以参考苏剑林博客: [P-tuning：自动构建模版，释放语言模型潜能](https://spaces.ac.cn/archives/8295)。

[P-tuning v2](https://arxiv.org/abs/2110.07602) 和 prefix-tuning 是一致的。区别在于, prefix-tuning 是针对 NLG 任务进行的实验, 而它是针对 NLU 任务进行的实验。
