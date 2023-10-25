# -*- coding:utf-8 -*-
# Author: lqxu

#%%
import os 
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 加速

import torch 
from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib.axes import Axes
from transformers.models.vit import ViTImageProcessor
from transformers.models.auto import AutoImageProcessor
from transformers.models.vit_mae import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining

model_id = "facebook/vit-mae-base"
demo_img_path = "transformer/code_analysis/demo_cat.jpg"

image = Image.open(demo_img_path)  # .crop([320, 140, 544, 364])
processor: ViTImageProcessor = AutoImageProcessor.from_pretrained(model_id)
whole_model: ViTMAEForPreTraining = ViTMAEForPreTraining.from_pretrained(model_id).eval()

#%%
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]
noise = torch.randn(1, 196)

with torch.no_grad():
    # step1: 构建 patch 向量 [batch_size, n_patches, hidden_size]
    patch_vectors = whole_model.vit.embeddings.patch_embeddings.projection(pixel_values).flatten(2).transpose(1, 2)  # 使用卷积层来模拟线性层
    # step2: 添加 position 位置信息
    patch_vectors += whole_model.vit.embeddings.position_embeddings[:, 1:, :]
    # step3: 随机掩码掉一些 patch [batch_size, n_unmasked_patches, hidden_size]
    patch_vectors, mask, ids_restore = whole_model.vit.embeddings.random_masking(patch_vectors, noise=noise)
    # step4: 添加 cls patch [batch_size, n_unmasked_patches + 1, hidden_size]
    cls_tokens = whole_model.vit.embeddings.cls_token + whole_model.vit.embeddings.position_embeddings[:, :1, :]
    patch_vectors = torch.cat([cls_tokens, patch_vectors], dim=1)

    # step5: 使用编码器编码
    encoder_outputs = whole_model.vit.encoder.forward(patch_vectors, output_attentions=True, output_hidden_states=True, return_dict=True)
    patch_vectors, hidden_states, attentions = encoder_outputs.last_hidden_state, encoder_outputs.hidden_states, encoder_outputs.attentions
    patch_vectors = whole_model.vit.layernorm(patch_vectors)
    
    # step6: 使用解码器解码
    batch_size, n_patches = ids_restore.shape
    n_unmasked_patches = patch_vectors.size(1)
    decoder_patch_vectors = whole_model.decoder.decoder_embed(patch_vectors)  # 就是一个线性层
    decoder_hidden_size = decoder_patch_vectors.size(-1)
    mask_tokens = whole_model.decoder.mask_token.repeat(batch_size, n_patches + 1 - n_unmasked_patches, 1)
    
    decoder_patch_vectors = torch.cat([
        decoder_patch_vectors[:, :1, :],  # cls tokens
        torch.gather(
            torch.cat([decoder_patch_vectors[:, 1:, :], mask_tokens], dim=1),
            index=ids_restore.unsqueeze(-1).repeat(1, 1, decoder_hidden_size), dim=1
        )
    ], dim=1)
    
    decoder_patch_vectors += whole_model.decoder.decoder_pos_embed

    for layer_module in whole_model.decoder.decoder_layers:
        layer_outputs = layer_module(decoder_patch_vectors, head_mask=None)
        decoder_patch_vectors = layer_outputs[0]
    
    decoder_patch_vectors = whole_model.decoder.decoder_norm(decoder_patch_vectors)
    logits = whole_model.decoder.decoder_pred(decoder_patch_vectors)
    logits = logits[:, 1:, :]


with torch.no_grad():
    model_outputs = whole_model.forward(pixel_values=pixel_values, noise=noise)

print((logits == model_outputs.logits).all().item())

# %%

inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]

with torch.no_grad():
    model_outputs = whole_model.forward(pixel_values=pixel_values)

nrows, ncols = 2, 4

_, plt_objs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
plt_objs: list[Axes] = plt_objs.reshape(-1)


def show_image(
        img: torch.Tensor, img_mask: torch.Tensor = None,
        need_unpatchify: bool = False, show_grid_line: bool = False,
        plt_obj: Axes = None, plt_title: str = None
    ):
    
    if need_unpatchify:
        img = whole_model.unpatchify(img.unsqueeze(0))[0]
    
    img = img.detach().cpu().permute(1, 2, 0)
    img = img * torch.tensor([0.229, 0.224, 0.225])
    img = img + torch.tensor([0.485, 0.456, 0.406])
    img = (img * 255.).clip_(0, 255)
    img = img.to(torch.uint8)

    if img_mask is not None:
        img_mask = img_mask.reshape(14, 14)[:, None, :, None, None].repeat(1, 16, 1, 16, 3).reshape(224, 224, 3).to(torch.uint8)
        img = img * (1 - img_mask)

    if show_grid_line:
        from torchvision.utils import make_grid

        img = img.reshape(14, 16, 14, 16, 3).permute(0, 2, 4, 1, 3).reshape(-1, 3, 16, 16)
        img = make_grid(img, nrow=14, normalize=False, pad_value=255)
        img = img.permute(1, 2, 0)

    if plt_obj is None:
        plt_obj = plt
    
    plt_obj.imshow(img)
    plt_obj.axis("off")
    
    if plt_title is not None:
        if hasattr(plt_obj, "set_title"):
            plt_obj.set_title(plt_title)
        else:
            plt_obj.title(plt_title)

    return img 


show_image(
    pixel_values[0], show_grid_line=True, plt_obj=plt_objs[0], plt_title="original image"
)
show_image(
    model_outputs.logits[0], need_unpatchify=True,
    show_grid_line=True, plt_obj=plt_objs[1], plt_title="output image"
)
show_image(
    pixel_values[0], img_mask=model_outputs.mask[0],
    show_grid_line=True, plt_obj=plt_objs[2], plt_title="masked image"
)
show_image(
    model_outputs.logits[0], img_mask=1 - model_outputs.mask[0], need_unpatchify=True,
    show_grid_line=True, plt_obj=plt_objs[3], plt_title="masked output image"
)

show_image(
    pixel_values[0], plt_obj=plt_objs[4], plt_title="original image"
)
show_image(
    model_outputs.logits[0], need_unpatchify=True,
    plt_obj=plt_objs[5], plt_title="output image"
)
show_image(
    pixel_values[0], img_mask=model_outputs.mask[0],
    plt_obj=plt_objs[6], plt_title="masked image"
)
show_image(
    model_outputs.logits[0], img_mask=1 - model_outputs.mask[0], need_unpatchify=True,
    plt_obj=plt_objs[7], plt_title="masked output image"
)
# %%
