
# 大模型时代的分词算法

[TOC]

## 一、前言

在自然语言处理领域, **词语** 和 **句子** 的划分一直是一大难题。**句子** 是模型输入的最小单元, 而 **词语** 则是模型处理的最小单元。

在目前 Transformer + 大语言模型的时代, 科研界对 **分句** 的研究几乎没有了, 转而变成了对 **文本向量检索** (Dense Passage Retrieval) 和 **长度外推性** (Length Extrapolation) 的研究。而对于部分工业界的任务来说, 这属于落地的难点之一, 需要结合具体的任务, 配合大量的人工规则来处理。

同时, 在大模型和大数据的加持下, **分词** 算法的准确率对模型性能的影响也大大降低了。即使分词不合理或者分错了, 模型也能够 "理解" 句子的含义, 因此相关领域的研究也不是特别多。本文就来聊一聊大模型时代的分词算法。

按照语言体系, 分词算法大体上可以分成两种:

+ 中文分词: 句子中词与词之间没有分隔符, 比方说 汉语, 日语, 韩语
+ 英文分词: 句子中用 **空格** 作为词与词之间的分隔符, 比方说 英语, 法语, 德语

下面, 我们先分别了解一下两种体系的分词方式。

## 二、中文分词

中文分词 是一个独立的 NLP 任务, 也是入门 NLP 很好的方式之一, 主要的方式有:

+ 基于 **词表匹配** 的方式 (前缀字典树, AC 自动机)
+ 基于 n-gram 的 **概率路径** 方式 (维特比算法)
+ 转化为 **序列标注** 任务, 使用 HMM, CRF 或者 token classification 等模型

其中, 最后一种方式需要人工标注数据, 然后训练模型。前两种方式除了人工标注数据外, 还可以通过 无监督学习 的方式生成所需要 **词典**。为了和序列标注任务相互对应, 我们将生成 **词典** 的过程也称为 "训练"。

对于第一种方式, "训练" 的过程只需要构建一个 **词表** 即可, 这种算法也被称为 **新词发现**, 具体的可以去看 Matrix67 的博客: [互联网时代的社会语言学：基于SNS的文本数据挖掘](http://www.matrix67.com/blog/archives/5044)。而对于第二种方式, "训练" 的过程除了构建 **词表** 外, 还要计算相对应的 **概率**。当然, 不同的 匹配 方式和 概率路径 计算方式, "训练" 过程是不一样的。

为了方便理解和说明, 在后文中, 中文分词算法统称为 **文本切分算法**, 而前两种方式的无监督训练方法称为 **无监督的文本切分算法**。

## 三、英文分词

对于英文分词来说, 一般情况下, 基于 空格 + 标点分词即可, 用正则表达式就能解决。

但是, 对于文本生成任务来说, 我们需要 **词表级别** 的分类任务, 即 softmax 回归的类别数和词表的数量是一致的。此时, 主要会产生以下一些问题:

首先, 是词表过大的问题。一般情况下, 每一种语言常用词在 40 到 50 万之间, 这就意味着最后的分类任务类别数在 40 到 50 万之间, 计算的开销非常之大, 但也不是不能接受。在 [word2vec](https://zhuanlan.zhihu.com/p/653414844) 中也有类似的问题, 可以使用 **负采样** (negative sampling) 的方式来近似计算。

其次, 是 OOV (Out-of-vocabulary) 的问题。对于那些不常见的, 被排除在词表之外的词语, 模型永远不能生成。在通用模型迁移到特定领域时, 这个问题会特别明显。

为了解决这个问题, [论文](https://arxiv.org/abs/1508.07909) 中对 **英文分词** 进行了优化, 提出了 **subword 分词** 方式。

其核心思想很简单: 在英文中, 一些英语单词还可以继续拆分, 拆成 前缀(prefix), 词根(stem), 后缀(suffix) 组合的形式。这些拆分出来的部分一般具有通用的含义, 同时单词的含义也与之有关。

那么, 我们可以对每一个英文单词执行 **文本切分算法**, 将其拆分成更小的 subword, 以减少英文词表的数量。这样, 常见的词直接在词表中, 不常见的词则拆成 subword 组合的形式。同时, 这些 subword 可以组合出训练集中没有英文单词, 那么就能解决 OOV 的问题了。

此时, 分词分为两个阶段: (1) 对句子使用空格 + 标点分词, 分成一个一个 word; (2) 对每一个 word 使用 **文本切分算法**, 分成一个一个 subword, 作为最终的结果。

在这种情况下, 使用 序列标注 的切分算法很困难, 因为其需要 人工标注, 我们很难去定义切分方式。一般都采用 **无监督的文本切分算法**, 也就是使用 匹配 或者 概率路径 的方式, 并通过无监督算法生成分词所需要的 词表 以及 概率。常见的算法包括: BPE, WordPiece, Unigram 等等。

## 四、多语言分词框架

在 LLM 时代, 我们希望训练出来的模型是 **语言无关** 的, 也就意味着支持所有的语言。此时, 上一部分提到的两个问题会更加突出: 全球有 300 多种文字系统, 如果每一种语言有 5 万个常用词, 那么词表就有 1,500 万之多, 此时 softmax 回归计算开销很难接受, 同时 OOV 问题也十分严重。一般情况下, 我们期望的 **词表** 大小是在 4 万到 10 万之间。

一种可行的方式是直接按照 **字符** 来分, 一个字符作为一个词语, 同时英文中的 **空格** 也当作一个词语, 词表就是 [unicode](https://en.wikipedia.org/wiki/Unicode) 中定义的 15 万种字符。这种方式对于中文来说比较友好, 但是对于英文来说效率太低了: 对于自回归模型来说, 推理时是一个一个字符蹦出来的, 这显然不能接受。不仅如此, **词表** 的大小也不符合我们的期望。

那么应该怎么办呢? 目前主流的做法是上一部分 subword 分词方式的扩展, 流程也是分为两步:

第一步, 将句子按照 **空格** 分词, 同时将 **空格** 作为词语的一部分, 我们称其为 word。如果是中英混合的文本, 那么 word 可能是英文单词, 也可能是中文片段。举例来说, `"Hello, World 你好世界"` 分成了 `"Hello,"`, `" World"`, `" 你好世界"` 三个 word。

需要强调一点, 这里将英文中的 **空格** 当作词语的一部分, 这意味着英文编码和解码具有可逆性, 即英文的解码不需要再额外添加 **空格** 了。否则, 如果模型生成的文本是中英混合的, 处理起来会非常麻烦。

第二步, 对每一个 word 采用 **文本切分算法**, 变成更小的 subword:

+ 如果是英文单词, 会被拆成更小的字母组合, 比方说, `"Hello,"` 可以被分成 `"He"`, `"llo"`, `","` 三个 subword
+ 如果是中文片段, 则会被拆成更小的词语, 比方说, `" 你好世界"` 可以分成 `" "`, `"你好"`, `"世界"` 三个 subword

一般情况下, 我们会将 语料库 中所有的字符都添加到 词表 中。那么就一定有 unicode 支持的, 但是 **词表** 中没有的字符。为了解决这样的问题, 我们需要引入 **字节**, 主要的方式有两种:

第一种: [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 使用的方式: 将 word 直接用 utf-8 编码成 **字节流**, 然后在字节流上采用 **文本切分算法**。由于 **字节** 的种类数是固定的 256 种, 对 **词表** 的影响就非常小了。

第二种: [LLaMA](https://arxiv.org/abs/2302.13971) 和 [sentencepiece](https://github.com/google/sentencepiece) 使用的方式: 还是在 word 上面采用 **文本切分算法**, 同时将 256 种 **字节** 加入 **词表** 中。在分词时, 如果出现了未知字符, 那么将未知字符用 utf-8 编码成 **字节流**, 一个字节作为一个 subword 加入分词结果中, 这样就解决 OOV 的问题了。

由于分词的结果中包含 **字节**, 现在的论文中都会将其称为 **byte-level**。

这样的分词结果肯定会有不合理的地方。比方说, 两个汉字可能会被分成三个 subword, 每一个 subword 包含两个字节。一般情况下, 在 utf-8 中一个汉字占三个字节, 这就意味着中间的 subword 是由第一个汉字最后一个字节 和 第二个汉字第一个字节构成, 这显然是不合理的。但是在大模型和大数据的加持下, 这种不合理的问题并不大。

不仅如此, 生成的文本也可能不合理, 有可能会组合出 unicode 中不存在的字符。这个时候, 我们只需要忽略掉错误的部分即可 (在 Python 中, 将 `bytes` 对象中 `encode` 方法 的 `errors` 参数设置为 `"ignore"` 就能实现了)。如果这种情况经常出现, 你需要评估一下模型训练的是否充分。

至此, 分词的框架就搭建完成了, 剩下的就是如何设计 **无监督的文本切分算法**。常见的有 BPE 和 Unigram 算法, 常用的工具有 [sentencepiece](https://github.com/google/sentencepiece) 和 [tokenizers](https://github.com/huggingface/tokenizers)。

下面, 就让我们看看具体的 **无监督的文本切分算法**。为了叙述方便, 在下文中, 我将 文本 统称为 word, 其可以是 英文单词, 中文文本片段, 一个完整的句子, 字节流等等。具体的要看框架怎么设计了。

## 五、BPE 分词算法

BPE 算法的全称是 Byte-Pair Encoding, 最早是 1994 年提出的, 应用于 [数据压缩](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) 任务。在 2015 年, 有人提出将其延申至 [文本分词](https://arxiv.org/abs/1508.07909) 任务, 同时没有改变算法的名称。如果你不了解之前的数据压缩算法, 可能会好奇为什么是这个名字。

OpenAI 关于 GPT 的一系列工作, 以及 Meta 关于 LLaMA 的一系列工作都是使用这种分词方式。OpenAI 虽然没有开源 GPT-3 模型的参数权重, 但是开源了分词库 [tiktoken](https://github.com/openai/tiktoken), 帮助用户在调用接口前计算 token 数。

下面就让我们看看这个算法。首先来看看 "训练" 过程:

第一步, 将 语料库 中所有的 word 以字符 (最小颗粒) 为单位进行拆分, 每一个字符作为一个 subword。比方说, `"best"` 被拆分成 `"b e s t"` 四个 subword (用 **空格** 作为分隔符)。

第二步, 初始化 **词表** 和 **合并规则列表**。其中, **词表** 就是所有可能的 subword 集合, 也就是初始化成 语料库 中所有可能的字符集合。

第三步, 遍历 word 中所有可能的 **subword pair**, 统计他们出现的频数。然后将数据集中 频数最高 的那一个 **subword pair** 合并在一起, 合并后的 subword 加入 **词表** 中, 合并规则加入 **合并规则列表** 中。

举例来说, `"b e s t"` 现在有三种可能的 **subword pair**: `"b e"`, `"e s"` 和 `"s t"`。统计完成后, 发现在语料库中, `"s t"` 出现的频数最高。那么, 我们就将整个数据集中的 `"s t"` 合并起来, 变成 `"st"`。此时 `best` 的拆分结果就是 `"b e st"`, 从四个 subword 变成三个 subword。

同时, 将 `"st"` 这个 subword 加入 **词表** 中, 将 `("s t", "st")` 加入 **合并规则列表** 中。

第四步, 一致重复 第三步 的操作, 直至 **词表** 达到预设的大小即可。

整个训练过程就是不断地去寻找语料库中频数最高的 **subword pair**, 然后合并成一个 **subword**, 和 [数据压缩](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) 算法中的核心思想是一致的。

需要注意的是, **subword pair** 的合并次数并不一定等于其频数。举例来说, 对于 `x x x` 来说, 其中 `x x` 出现了两次, 但是我们合并时只能合并一个, 即 `xx x`。

那么, 我们怎么对 word 进行分词呢? 首先, 将 word 以字符为单位进行拆分, 一个字符作为一个 subword。然后按照 **合并规则列表** 中的顺序, 如果出现了 **subword pair**, 就进行合并。最后剩下的 subword 列表就是最终的分词结果。

整体上的方案就是这样, 可以说非常巧妙。这个算法开源在 [subword-nmt](https://github.com/rsennrich/subword-nmt) 项目中。

现在, 很多论文中都说其使用了 Byte-Level BPE (BBPE), 就是将第四部分所说的 byte-level 方案和 BPE 结合起来, 最初是 OpenAI 在 [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 工作中提出。网上很多博客在介绍 BBPE 时, 引用的论文是 Meta 的 [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/abs/1909.03341), 其发布比 GPT-2 晚半年, 不要弄错关系了。

## 六、WordPiece 分词算法

WordPiece 是谷歌于 2016 年发布的 [工作](https://arxiv.org/abs/1609.08144)。其代码没有开源, 出名的原因是谷歌发布的 [BERT](https://arxiv.org/abs/1810.04805) 模型使用了这种分词方式。这意味着, WordPiece 的 "训练" 代码没有开源, 但是 "分词" 代码开源了。从方法上来看, 其是 BPE 方法的改进, 但是由于没有开源, 除了谷歌外, 没有其它公司使用。

相较于 BPE, 其改进主要体现在下面两个方面:

首先, 在训练过程的第三步中, 不仅仅计算 **subword pair** 的频数, 而是计算两者之间的 score 值, 计算公式如下:

$$
\mathrm{score} (sw_1, sw_2) = \frac{\mathrm{count}(sw_1, sw_2)}{\mathrm{count}(sw_1) \times \mathrm{count}(sw_2)}
$$

其中, $sw_1$ 和 $sw_2$ 表示两个 subword, 他们的频数是 $\mathrm{count}(sw_1)$ 和 $\mathrm{count}(sw_2)$。而 $\mathrm{count}(sw_1, sw_2)$ 表示 **subword pair** 出现的频数。如果将所有的 **频数** 除以总频数, 转化成概率, 再对比值取对数, 那不就是 PMI (Pointwise Mutual Information) 吗? 对此有疑问的可以参考我之前的 [文章](https://zhuanlan.zhihu.com/p/612898756)。

也就是说, 现在的训练过程是不断地去寻找语料库中 PMI 最高的 **subword pair**, 然后合并成一个 **subword**。这样做的好处是什么呢? 比方说, `"un"` 和 `"able"` 在语料库中的频数很高, 如果是 BPE 算法, 两者就会合并成一个 subword。但是, `"un"` 作为词语前缀的情况很多, `"able"` 作为词语后缀的情况很多, 那么他们相关性就没有那么高, PMI 值就会偏低, 那么在 WorldPiece 算法中, 两者就不会合并成一个 subword。

其次, 是分词过程的不同。这里不使用 **合并规则列表** 来进行分词, 而是直接使用 **词表** 分词, 采用 **正向最长匹配** 的策略。也就是说, 不断寻找 word 中的最长前缀 subword。用一个具体的例子来说明:

词表中有 `"研究"`, `"生命"`, `"起源"`, `"研究生"`, `生` 等 subword, 现在对 `"研究生命起源"` 这样一个 word 进行分词:

+ 从 "研" 开始, 匹配到词表中最长的前缀 subword 是 `"研究生"`
+ 从 "命" 开始, 匹配到词表中最长的前缀 subword 是 `"命"`
+ 从 "起" 开始, 匹配到词表中最长的前缀 subword 是 `"起源"`

那么分词结果就是 `["研究生", "命", "起源"]`。这显然是不合理的, 合理的分词结果应该是 `["研究", "生命", "起源"]`。

虽然直接举反面的例子不太好, 但这是 **正向最长匹配** 中存在的问题。如果是 英文词语, 在使用 PMI 策略的情况下, 问题不是很大。但是对于 中文片段 来说, 就会有很大问题了。因此, 在 [BERT](https://arxiv.org/abs/1810.04805) 模型中, 中文都是单字成 subword, 直接切成最小颗粒。

## 七、Unigram 分词算法

unigram 分词方式属于 n-gram 概率路径 分词方式的一种, 中文名称是 **一元分词**。在中文分词中, 比较常见的是 **二元分词** (bigram), 比方说 [HanLP V1](https://github.com/hankcs/HanLP/tree/v1.8.4) 中默认的分词方式就是这种。

n-gram 的本质就是 标准语言模型 的简化。而 unigram 是最强的假设, 即每一个词语的概率是完全独立的, 和之前的文本没有关系的。那么, 我们只要统计出每一个词语在语料库中出现的概率 (词频 除以 总词频), 就可以估算出一个句子一种切分方式的概率。接下来, 我们遍历一个句子所有可能的切分方式, 找到概率最大的那一种方式, 作为分词结果。一个一个遍历的成本太高了, 一般使用 维特比算法。

对此不了解的可以去网上搜索相关的资料, 确保理解上面的内容。那么应该如何根据语料库来生成词典呢? 目前主流的是用 [Subword Regularization](https://arxiv.org/abs/1804.10959) 论文提出的方案:

第一步, 初始化 **subword 概率词典**: 对于每一个 word, 统计所有可能的 subword, 以及在 语料库 种出现的频数, 然后保留频数最高的 $S$ 个, 转化成信息量, 构成词典。

举例来说: `"good"` 一词一共有 9 种可能的 subword: `"g"`, `"o"`, `"d"`, `"go"`, `"oo"`, `"od"`, `"goo"`, `"ood"` 和 `"good"` (一般还会限制 subword 的长度, 这里没有体现)。每一个 word 都这样处理, 然后统计频数, 去除低频的, 除以 总频数 转化成 **概率**, 再取 负对数 转化成 **信息量**。

现在, 我们可以用 **subword 概率词典** 对 word 进行分词, 同时可以得到分词方式的 **信息量**。那么, 我们可以用 **subword 概率词典** 对语料库中所有的 word 进行分词, 然后将得到的 **信息量** 求和, 记作当前 **subword 概率词典** 的 likelihood 值。

第二步, 对于 **subword 概率词典** 中的每一个 subword, 我们计算在去除它的情况下的 likelihood 值, 记作 loss 值。这样, 每一个 subword 都有一个 loss 值, loss 值越小表示在分词过程中重要程度越高。然后只保留  **subword 概率词典** 中 loss 值最低的 $\eta \%$ subword, 其它都去除掉。需要注意的是, 为了避免 OOV 的问题, 所有单个字符的 subword 都是保留的。

第三步, 一直重复 第二步 的操作, 直到 **subword 概率词典** 的大小和预先设置的 **词表** 大小一致即可。

整体上的方案就是这样, 这个算法开源在 [sentencepiece](https://github.com/google/sentencepiece) 项目中。

## 八、随机分词

我们知道, 对于 文本生成 任务来说, 不能完全按照 概率最高 的方式来生成, 而只要是在合理区间内的都可以。同理, 对于分词来说也是类似: 分词方式需要一定的 "随机性"!

举例来说, 如果词表中有 `"白云"`, `"机场"`, `"白云机场"` 等词语。在分词时, `"广州的白云机场"` 会被 固定 的分成: `["广州", "的", "白云机场"]`。那么会出什么样的问题呢?

对于 续写 任务来说, 如果用户输入的是 `"广州的白云"`, 虽然 `"机场"` 一词在词表中, 但是模型不会在后面接 `"机场"` 这样的词语, 而会接对 云朵 的描述内容, 比方说 `"很白"` 等等。这是为什么呢?

因为在训练模型时, `"白云机场"` 是一个固定的 token, 不会被分割开。也就是说, `"白云"` 后面不会出现 `"机场"` 这个 token。那么应该怎么解决呢? 答案是 **随机分词**, 即给予一定的概率分成 `"白云"` 和 `"机场"` 两个 token, 让模型知道 `"白云"` 后面是有可能出现 `"机场"` 这个 token 的。

最早提出这一想法的是 [Subword Regularization](https://arxiv.org/abs/1804.10959) 这篇论文, 也就是开源 [sentencepiece](https://github.com/google/sentencepiece) 的工作。其将上述方式称为 subword sampling。

对于第七部分的 unigram 分词算法, 我们可以采用 n-best 维特比算法, 找出最佳的 n 个分词路径, 并计算出路径的概率分数。然后根据路径的概率分数进行 **采样**, 选择其中的一个即可。

而对于 BPE 算法, 有人提出 [BPE-Dropout](https://arxiv.org/abs/1910.13267) 方案, 即在分词时, 有 $p\%$ 的概率不进行合并。

沿用第五部分的例子, 对于 `"x x x"` 来说, 现在要将 `"x x"` 合并成 `"xx"`。按照原来的方案, 合并结果应该是 `"xx x"`。而按照现在的方案, 还有可能分成 `"x xx"` (第一次合并时触发 dropout, 没有进行合并); 也有可能保持原状, 还是 `x x x` (两次合并都触发 dropout)。

个人观点, 这种随机分词方式在 LLM 模型训练 时起的作用比在 LLM 模型推理 时要大, 可以增加模型的鲁棒性, 是一种非常好的 预训练 策略。

## 九、总结

至此, 大语言模型常用的分词方式都介绍完成了。这属于非常宏观的介绍, 对于细节的介绍很少。主要体现在两个方面。

首先, 不同的模型在分词时会添加不同的标记符号:

+ GPT-2 在将 word 编码成字节流后, 会转义回字符串, 同时确保不会转义成 空白字符
+ sentencepiece 中, 英文词语的空格添加在开头, 并转义成 `"\u2581"`
+ BPE 原始论文中, 英文词语的空格添加在结尾, 并转义成 `"</w>"`
+ bert 开源的 wordpiece 中, 非起始 subword 都会添加 `"##"` 标识, 间接的将空格添加到 word 中

其次, 实现的细节不同。如果完全按照上面所说的来, 运行效率会非常低, 需要一些字符串搜索算法进行优化。

总之, 整体的分词框架就是这样。如果之后有时间, 可以自己实现一个版本。

最后聊一点题外话, 关于中文语料的事情。

如果你关注 LLM 开源项目, 应该知道哈工大崔一鸣大佬的 [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) 项目。这个项目是针对 LLaMA 模型进行中文微调的, 其中做了一件事情, 那就是 **中文词表扩充**。

读过 [LLaMA](https://arxiv.org/abs/2302.13971) 论文应该知道, 其使用的是 [sentencepiece](https://github.com/google/sentencepiece) 中的 BPE 算法训练的, 同时将 `byte_fallback` 参数设置成 `True`, 具体可以参考 [issue 621](https://github.com/google/sentencepiece/issues/621) 和 [issue 1218](https://github.com/huggingface/tokenizers/issues/1218)。

训练完成后, 词表中包含 32,000 个词语。其中 中文词汇 占比很少, 甚至连 `"气"` 这个字都不在词表中。此时, 你或许会好奇, 怎么可能呢? 就算训练语料中的中文很少, 2000 个常用字应该是包含的啊。

后来, 我发现 [sentencepiece](https://github.com/google/sentencepiece) 的训练参数中有一个是 `character_coverage`, 其含义是随机删除语料库中一定比例的字符, 具体可以参考 [issue 412](https://github.com/google/sentencepiece/issues/412)。而没有出现在词表中的汉字大概率是被删除了。其中, 默认的删除比例是万分之五。不知道 LLaMA 在训练时有没有调大。~~所以中文语料占比是有多低啊。~~

## 十、引用

论文:

+ [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
+ [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)
+ [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959)
+ [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/abs/1910.13267)

代码和开源项目:

+ OpenAI 开源的 [tiktoken](https://github.com/openai/tiktoken)
+ [BPE 论文](https://arxiv.org/abs/1508.07909) 开源的 [subword-nmt](https://github.com/rsennrich/subword-nmt)
+ [Subword Regularization 论文](https://arxiv.org/abs/1804.10959) 开源的 [sentencepiece](https://github.com/google/sentencepiece)
+ [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)
+ [HuggingFace GPT-1 分词源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/openai/tokenization_openai.py)
+ [HuggingFace GPT-2 分词源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py)

博客和文档:

+ [HuggingFace Transformers 分词介绍文档](https://huggingface.co/docs/transformers/tokenizer_summary)
+ [HuggingFace NLP Course 分词介绍文档](https://huggingface.co/learn/nlp-course/chapter6/1)
+ [大词表语言模型在续写任务上的一个问题及对策](https://kexue.fm/archives/9762)
+ [BytePiece：更纯粹、更高压缩率的Tokenizer](https://kexue.fm/archives/9752)
+ [随机分词浅探：从Viterbi Decoding到Viterbi Sampling](https://kexue.fm/archives/9768)

其它:

+ [Languages of the World](https://www.nationsonline.org/oneworld/languages.htm)
