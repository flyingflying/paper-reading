# torchscript

[TOC]

PyTorch 默认是 **动态图模式 (eager mode)**, 这使得其非常的灵活, 很容易写代码。在部署时, 我们肯定希望使用 **静态图模式 (graph mode)**, 这样运行效率更高。

PyTorch 中有 `JIT` (just-in-time compilation) 模块, 可以将模型, 代码和参数统一转换成 `torchscript` 文件。按照官方文档中所说, `torchscript` 是 PyTorch 模型的一种中间表示 (intermediate representation), 你可以在其它语言中直接使用 (eg. C++, Java, etc), 不需要写额外的模型代码了, 可以说是非常方便。

`torchscript` 提供了两种导出模型的方式: `tracing` 和 `script`。

## tracing 模式

`tracing` 模式需要给模块提供样例输入, `JIT` 模块会根据样例输入来运行代码, 保留生成的计算图, 并将和计算图有关的代码保存下来, 其它的代码都会删除, 效率是非常高的。

我们可以通过下面的代码来学习怎么使用 `tracing` 模式:

```python
import torch
from torch import jit
from torch.utils import benchmark
from transformers.models.bert import BertConfig, BertModel

# 初始化模型
bert_config = BertConfig.from_pretrained("hfl/chinese-bert-wwm-ext", cache_dir="../huggingface_cache", torchscript=True)
bert_model = BertModel(bert_config, add_pooling_layer=True)
bert_model.eval()  # 必须是 eval 模式, 不然会报错: Tensor-likes are not close!
print("initialize successfully!!!")

# 用 JIT 生成 torchscript 模型
input_ids = torch.randint(low=0, high=bert_config.vocab_size, size=(1, bert_config.max_position_embeddings, ))
attention_mask = torch.randint(low=0, high=2, size=(1, bert_config.max_position_embeddings, ))
script_model = jit.trace(bert_model, example_inputs=(input_ids, attention_mask))
print("generate successfully!!!")

# 保存 torchscript 模型
jit.save(script_model, "script_bert_model.pt")
# 加载 torchscript 模型
script_model = jit.load("script_bert_model.pt")
print("save and load successfully!!!")

# 测试结果
test_input_ids = torch.randint(low=0, high=bert_config.vocab_size, size=(5, bert_config.max_position_embeddings))
test_attention_mask = torch.randint(low=0, high=2, size=(5, bert_config.max_position_embeddings))
gold_last_hidden_state, gold_pooler_output = bert_model(test_input_ids, test_attention_mask)
test_last_hidden_state, test_pooler_output = script_model(test_input_ids, test_attention_mask)
assert torch.all(gold_last_hidden_state == test_last_hidden_state)  # noqa
assert torch.all(gold_pooler_output == test_pooler_output)  # noqa
print("test successfully!!!")
```

`tracing` 模式的使用有以下限制:

+ 模块输入的参数和输出的参数必须全部是 `torch.Tensor` 对象:
  + 输入的参数仅仅接受 positional arguments, 不接受 keyword arguments
  + 如果输出的参数有多个, 必须是 `Tuple[torch.Tensor]`, 中间不能包含 `None`
+ 所有和计算图无关的代码不会保存下来:
  + 如果代码中的 `if` 语句和 `for` 语句只和配置文件相关, 那么没事, 你可以大胆的使用 `tracing` 模式
  + 如果代码中的 `if` 语句和 `for` 语句和输入有关系, 那么你就需要根据实际情况考虑能否使用 `tracing` 模式了
+ 关于张量的维度需要注意:
  + 实际使用时张量的维度是没有限制的, 只要代码能够跑成功就行, 和样例输入的维度没有关系
  + `torchscript` 是会保存计算图的, 因此样例输入张量的 `ndims` 和实际使用张量的 `ndims` 尽量保持一致, 并且 `shape` 按照输入的最大值进行, 否则在运行时, 程序要修改计算图的 `shape`, 这样会很大程度地影响计算效率。

现在有如下代码:

```python
import torch
class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.linear = torch.nn.Linear(in_features=4, out_features=5)

    def forward(self, input_ids):
        ret = 0
        batch_size, seq_len = input_ids.shape
        for batch_idx in range(batch_size):  # 和计算图无关
            ret += self.linear(input_ids[batch_idx]).sum()
        return ret


test = Test().eval()
model = torch.jit.trace(test, torch.rand(2, 4))
print(model.code)
```

我们可以通过 `.code` 查看转化后的代码, 通过 `.graph` 查看转化后的计算图。下面是转化后的代码:

```python
# noinspection PyUnresolvedReferences, PyShadowingBuiltins
def forward(self, input_ids: Tensor) -> Tensor:
    linear = self.linear
    seq_len = ops.prim.NumToTensor(torch.size(input_ids, 1))
    _0 = int(seq_len)
    _1 = int(seq_len)
    input = torch.slice(torch.select(input_ids, 0, 0), 0, 0, _1)
    ret = torch.add(torch.sum((linear).forward(input, )), CONSTANTS.c0)  # noqa 
    input0 = torch.slice(torch.select(input_ids, 0, 1), 0, 0, _0)
    _2 = torch.sum((linear).forward1(input0, ))  # noqa 
    return torch.add_(ret, _2)
```

观察上面的代码, 我们注意到, 由于 `tracing` 模式仅仅保存和计算图相关的代码, `for` 语句和 `if` 语句都会失效:

+ 如果你的需求是 `batch_size` 一直为 `2`, 那么用 `tracing` 是没有问题的
+ 如果你的需求是只关注前两个样本, 后面的全部都忽略掉, 那么用 `tracing` 也是没有问题的
+ 如果 `batch_size` 等于 `1`, 则会报错
+ 如果你的需求是所有的样本都要运行, 且会出现 `batch_size` 大于 `2` 的情况, 此时虽然程序正常运行, 但肯定不符合要求, 属于 bug

综上所述, 有 `if` 语句和 `for` 语句的代码需要根据实际情况自行判断能否用 `torch.jit.trace` 进行转换, 不能一概而论。

## script 模式

`script` 模式和 `tracing` 模式不一样在于 `script` 会保留所有的代码, 不仅仅是和计算图操作有关的代码。这样的话, `for` 语句和 `if` 语句都会被保存下来。使用这种模式就不需要样例输入了, 对函数的参数和返回的类型也没有了限制, 但是需要注意的是:

+ 部分 Python 代码 `script` 模式是不支持的, 比方说 Generator Expression
+ 需要通过 `.code` 方式确定 `script` 解析的函数参数类型是否正确, 不正确的话需要用 `typing` 自行添加类型注解

示例代码很简单: `torch.jit.script(model)`, 保存和加载都和之前的一样。`script` 模式也会保存计算图, 可以通过 `.graph` 的方式查看。对于不涉及到张量的运算也可以使用 `script` 进行编译, 会有一定性质的加速。

## 总结

关于 `torchscript` 还有以下需要说明的:

+ 如果我们想要编译一个 PyTorch 中的模块, 也可以通过修改继承对象的方式: 原本是继承 `nn.Module`, 现在是继承 `jit.ScriptModule`
+ `jit.script` 不仅能传入 PyTorch 中的模块, 还可以传入函数, 使用方式一样 (PyTorch 中的模块本身也是 `Callable` 的对象)
+ `jit.trace` 会运行多次函数或者模块, 如果每一次返回的张量不一致, 会有警告: `Tensor-likes are not close!`, 意思是两个张量不一致, 如果你的模块中有 `Dropout`， 那么需要先调用模块的 `.eval()` 方法
+ 在默认情况下, `torch.jit` 只会导出模块的 `.forward` 方法, 导出对象的 `.code` 和 `.graph` 属性返回的其实是 `.forward` 方法的这些属性
+ 如果你希望 PyTorch 中模块的其它方法也要被导出, 用 `torch.jit.export` 装饰器修饰即可, 导出后, 就可以使用这些方法了, 并且这些方法也有 `.code` 和 `.graph` 属性
+ 在 PyTorch-Lightning 中, `LightningModule` 直接提供了 `to_torchscript` 方法, 相当于是将 `torch.jit.script`, `torch.jit.trace` 和 `torch.jit.save`方法封装到模块中了, 并没有什么特别的。

经过测试, 正常情况下, `torchscript` 的速度比原来的要快 5% ~ 30%, 还是很香的。

值得一提的是, 截至 4.20 版本, HuggingFace Transformers 对 `torchscript` 的支持非常的不好:

+ 如果用 `script` 模式根本无法导出, 因为这个模式下会去解析大量库中的代码, 而库中有些代码在 `script` 模式下是不支持的, 比方说 Generator Expression, 因此官方给出的导出代码也不是用 `script` 模式
+ 如果在 `BertModel` 中设置 `add_pooling_layer=False`, 则用 `tracing` 模式导出会失败, 因为返回值中有 `None`
+ 如果想要在 `BertModel` 中设置 `add_pooling_layer=False`, 并用 `tracing` 模式导出, 那么必须要在 `BertModel` 外面封装一层 PyTorch 模块

总结以下, `tracing` 模式只会保存和计算图相关的代码, 结果线性执行, 编译时需要样例输入, 会运行多次检测输出是否一致, 对 Python 代码要求低;

`script` 模式会保留所有的 Python 代码, 支持流程语句, 编译时不需要样例输入, 会生成计算图, 对 Python 代码要求高。

## References

+ [Features](https://pytorch.org/features/) (Accessed on 2022-09-15, PyTorch Version: 1.12.1)
+ [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#basics-of-torchscript) (Accessed on 2022-09-15, PyTorch Version: 1.12.1)
+ [Export 🤗 Transformers Models (huggingface.co)](https://huggingface.co/docs/transformers/main/en/serialization#torchscript) (Accessed on 2022-09-15, Transformers Version: 4.20.0)
+ [Deploy models into production (advanced)](https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_advanced_2.html) (Accessed on 2022-09-16, PyTorch-Lightning Version: 1.7.6)
