
# DALL·E 2

+ 根据文本生成图片
+ 根据文本对已有的图片进行编辑和修改
+ 根据已有的图片生成相似的图片

DALL·E 2 生成图片的分辨率是 DALL·E 1 的 4 倍, 看起来更加清晰逼真。经过调查, 有 71.7% 的人认为, DALL·E 2 生成的图片比 DALL·E 1 更加贴和 文本 (caption); 有 88.8% 的人认为, DALL·E 2 生成的图片比 DALL·E 1 更加真实 (photorealism)。

+ DALL·E 1 (2021-01, OpenAI): [paper](https://arxiv.org/abs/2102.12092), [code](https://github.com/openai/DALL-E), [blog](https://openai.com/research/dall-e)
+ CogView (2021-05, 清华大学): [paper](https://arxiv.org/abs/2105.13290), [code](https://github.com/THUDM/CogView)
+ NUWA (2021-11, 微软亚洲研究院): [paper](https://arxiv.org/abs/2111.12417), [code](https://github.com/microsoft/NUWA)
+ GLIDE (2021-12, OpenAI): [paper](https://arxiv.org/abs/2112.10741), [code](https://github.com/openai/glide-text2im)
+ ERNIE-ViLG (2021-12, 百度): [paper](https://arxiv.org/abs/2112.15283)
+ DALL·E 2 (2022-04, OpenAI): [paper](https://arxiv.org/abs/2204.06125), [blog](https://openai.com/dall-e-2)
+ CogView 2 (2022-04, 清华大学): [paper](https://arxiv.org/abs/2204.14217), [code](https://github.com/THUDM/CogView2)
+ CogVideo (2022-05, 清华大学): [paper](https://arxiv.org/abs/2205.15868), [code](https://github.com/THUDM/CogVideo)
+ Imagen (2022-05, Google): [paper](https://arxiv.org/abs/2205.11487), [blog](https://imagen.research.google/)

[CLIP](https://arxiv.org/abs/2103.00020)

unCLIP

DALL·E 1 是 [VQ-VAE](https://arxiv.org/abs/1711.00937) 和 [VQ-VAE v2](https://arxiv.org/abs/1906.00446) 的改进, 之后再去了解。而 DALL·E 2 是基于 [CLIP](https://arxiv.org/abs/2103.00020) 和 [DDPM](https://arxiv.org/abs/2006.11239) 的工作

+ [DDPM](https://arxiv.org/abs/2006.11239)
  + 原先是直接从 $x_{t}$ 预测 $x_{t-1}$ 时刻的图像, 现在是从 $x_{t}$ 预测 $x_0$ 时刻的噪声
  + 预测 正态分布 需要预测 均值和方差, 在图像生成任务中, 我们可以只预测 均值, 方差变成定值即可
+ [improved DDPM](https://arxiv.org/abs/2102.09672)
  + 还是要预测 正态分布 的方差
  + 扩散率改成 cosine 变化方式
+ [Diffusion Models beats GAN](https://arxiv.org/abs/2105.05233)
  + 大模型: multi-scale attention
  + adaptive group normalization
  + classifier guidance (IS score/FID score 太低)
+ classifier free guidance → [GLIDE](https://arxiv.org/abs/2112.10741) → DALL·E 2

级联生成方式: 64x64 → 256x256 → 1024x1024

训练数据集: 图文匹配的数据集

先用 CLIP 文本编码器将文本编码成 向量, 再用 prior 模型根据文本向量生成图片向量 (CLIP 图片编码器编码的向量), 最后再根据图片向量通过 decoder 模型生成图片
