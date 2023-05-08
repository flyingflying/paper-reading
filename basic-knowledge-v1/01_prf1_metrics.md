
# Precision, Recall & F1 Score

[TOC]

## 简介

对于分类问题, 最直接的测试指标就是 accuracy (正确率), 用预测正确的数目除以总样本数。然而当某一类的样本数远远大于其它类时, 这样的评价指标就会出现问题, 这时往往会用 Precision, Recall 和 F1 score 这些评价标准。

## 二分类问题

我们假设两个类别的名字是 "正样本" 和 "负样本", 并且按照一般的约定, 样本数量少的是 "正样本", 样本数量多的是 "负样本"。我们可以将整个测试集分成四类: true positive, true negative, false positive 和 false negative。这四类的命名规则如下:

+ 名称中的 true 和 false 表示的是预测是否正确
+ 名称中的 positive 和 negative 表示的是预测的类别: 正样本和负样本

根据上述的命名规则, 我们可以知道这四类的含义分别是:

+ true positive (tp): 预测为正样本且预测正确 ==> 预测是正样本实际也是正样本
+ true negative (tn): 预测为负样本且预测正确 ==> 预测是负样本实际也是负样本
+ false positive (fp): 预测为正样本且预测错误 ==> 预测是正样本但实际上是负样本
+ false negative (fn): 预测为负样本且预测错误 ==> 预测是负样本但实际上是正样本

我们可以将 false positive 称为 **假阳性**, 将 false negative 称为 **假阴性** 。

由于负样本的数量远远大于正样本, 模型预测的结果自然会倾向于负样本, 因此 true negative 集合中占据了绝大部分的样本数量。为了更合理的评价分类模型, 我们应该将这一部分去掉, 来进行计算。那么应该怎么去掉这一部分呢?

我们的做法是只计算 "正样本" 的正确率, precision 和 recall 的计算方式如下:

+ precision (预测的)精确率, 即预测是正样本集合中的正确率, 公式为 tp / (tp + fp)
+ recall (实际的)召回率, 即实际是正样本集合中的正确率, 公式为 tp / (tp + fn)

形象化的理解, precision 高表示 "宁缺毋滥", 希望预测出来的 "正样本" 都是正确的, "假阳性" 尽可能地少。在信息抽取领域, 如果是给客户展示结果, 那么你应该希望 precision 高, 因为这样展示的效果好。

recall 高表示 "宁滥毋缺", 希望所有的 "正样本" 都被预测出来 (召回), "假阴性" 尽可能地少。在信息抽取领域, 我们一般希望 recall 高, 因为这样方便对结果进行人工筛选。

在实际比较模型优劣时, 我们肯定希望用一个数值来比较, 而不是两个数值, 那么我们需要将两个指标合并成一个指标, 最简单的办法当然是取平均。在统计学中, 有三种取平均的方式: 算术平均数, 几何平均数和调和平均数。三种平均数中, 调和平均数受极端值的影响最大, 那么我们就选用调和平均数。换言之, 我们选用调和平均数的原因是我们希望的是 precision 和 recall 的值都尽可能地高。我们将 precision 和 recall 的调和平均数称为 F1 score。

在很多实际问题中, "负样本" 的统计并不方便, 我们只会去统计预测为正样本的集合以及实际为正样本的集合, 两个集合交集的样本数量就是 true positive, 用公式表示如下:

设集合 $\mathbf{A}$ 表示预测为正样本的集合, 集合 $\mathbf{B}$ 表示实际为正样本的集合, 那么 $precision$, $recall$ 和 $F1$ 的计算公式如下:

$$
precision = \frac{|\mathbf{A} \cup \mathbf{B}|}{|\mathbf{A}|}
$$

$$
recall = \frac{|\mathbf{A} \cup \mathbf{B}|}{|\mathbf{B}|}
$$

$$
F1 = \frac{2 * |\mathbf{A} \cup \mathbf{B}|}{|\mathbf{A}| + |\mathbf{B}|}
$$

其中, $|\mathbf{A}|$, $|\mathbf{B}|$ 和 $|\mathbf{A} \cup \mathbf{B}|$ 分别表示集合 $\mathbf{A}$ 元素的数量, 集合 $\mathbf{B}$ 元素的数量, 以及两个集合并集中元素的数量。

对于实际的问题来说, 我们可能更看重 precision 或者 recall 一些, 比方说对于信息检索来说, 应该尽可能将正样本检索出来, 给使用者来筛选, 那么此时更看重 recall。此时用 F1 来综合两个指标就不合适了, 可以用 Fbeta 指标, 这个指标中的 beta 值表示 precision 和 recall 所占的权重, 当 beta=1 时, 就是 F1 指标。

在使用 sklearn 的 classification_report 方法时, 会发现还有一个指标值是 support, 这个值就是实际为正样本的数量。

## 多分类问题

如果分类的标签有多个, 应该怎么去计算 precision, recall 和 F1 score 呢? 对于某一确定的标签 a, 我们定义:

+ 正样本: 所有标签是 a 的样本
+ 负样本: 所有标签不是 a 的样本

这样我们就可以计算每一个标签三个指标值, 那么如何将这一组指标值综合起来呢? 一般有两种办法: macro (宏观) 和 micro (微观)

+ 宏观指标就是综合考虑每一个标签, 也就是说将所有类别的 precision, recall 和 F1 score 分别求算术平均数, 得到 macro-precision, macro-recall 和 macro-F1
+ 加权宏观指标和宏观指标相似, 就是求所有类别的加权平均数, 权重值是 (tp + fn), 也就是实际为正样本的数量, 对应 sklearn 中地 support
+ 微观指标就是综合考虑每一个正样本, 将所有类别的 tp, fp 和 fn 加在一起求 precision, recall 和 F1 score, 得到 micro-precision, micro-recall 和 micro-F1

如果是 **多标签单分类问题**, 也就是一个样本只能有一个类别, 那么会出现一个问题, 那就是: micro-F1 = accuracy = micro-Precision = micro-Recall, 四者的值是相等的。原因很简单, 因为某一类的 fp 样本一定是其它类的 fn 样本。

如果是 **多标签多分类问题**, 也就是一个样本可以有多个类别, 那么就没有上述的问题, 这个要注意一下。

在很多实际问题中, 实际上是 "正样本" 有多个类别, "负样本" 只有一个类别 (表示这个样本不属于任何一个 "正样本")。不仅如此, 负样本的数量也远远大于所有正样本的类别之和。注意, 在这种情况下, micro 指标, macro 指标和 weighted macro 指标都只统计 "正样本" 的情况, 不统计 "负样本" 的情况, 同时 **多标签单分类问题** 也没有上面所说地特殊情况了。

## NER 问题

对于 NER 问题来说, 负样本就是所有的候选实体个数, 正样本就是命名实体个数, 如果文本的序列长度是 $n$, 那么候选实体的个数就是 $\frac{n(n+1)}{2}$, 正样本只是其中很少的一部分, 因此用 precision, recall 和 F1 score 来评价模型的效果肯定更加合理一些。如果命名实体的类别有多个, 也就是说正样本的类别有多个, 那么我们在计算 macro, micro 和 weighted macro 时也是只考虑正样本的情况, 不考虑负样本的情况。

## References

+ [为何选用F1值（调和平均数）衡量P与R？](https://blog.csdn.net/weixin_39490983/article/details/88297899) (Accessed on 2022-08-26)
+ [一文解释Micro-F1, Macro-F1，Weighted-F1](https://blog.csdn.net/qq_27668313/article/details/125570210) (Accessed on 2022-08-26)
