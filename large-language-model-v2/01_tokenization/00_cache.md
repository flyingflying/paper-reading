
# 分词算法总结

[TOC]

## 新词发现

+ Matrix67 的博客: [互联网时代的社会语言学：基于SNS的文本数据挖掘](http://www.matrix67.com/blog/archives/5044)
  + 自由程度: 统计 **候选词** 左右字的 **信息熵**
  + 凝固程度: 统计 **候选词** 内部的 **互信息**

和苏剑林的博客: [重新写了之前的新词发现算法：更快更好的新词发现](https://spaces.ac.cn/archives/6920)。

对于第二种方式, "训练" 的过程不仅包含 **词典** 的构建, 还需要统计词语的概率, 具体可以参考苏剑林的博客: [【中文分词系列】 5. 基于语言模型的无监督分词](https://spaces.ac.cn/archives/3956)。

## 第三方库

英文分词: [nltk](https://github.com/nltk/nltk) 中的 `wordpunct_tokenize`, [spaCy](https://github.com/explosion/spaCy) 等等。

中文分词: [jieba](https://github.com/fxsjy/jieba), [pkuseg](https://github.com/lancopku/pkuseg-python), [THULAC](http://thulac.thunlp.org/), [LTP](https://github.com/HIT-SCIR/ltp) [HanLP V1](https://github.com/hankcs/HanLP/tree/v1.8.4) 和 [HanLP V2](https://github.com/hankcs/HanLP), [LAC](https://github.com/baidu/lac), [ik-analyzer](https://code.google.com/archive/p/ik-analyzer/) 等等。
