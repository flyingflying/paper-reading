# -*- coding:utf-8 -*-
# Author: lqxu
# Reference: https://github.com/openai/CLIP
# Paper: Learning Transferable Visual Models From Natural Language Supervision
# Paper Link: https://arxiv.org/abs/2103.00020

import math 

import torch 
from torch import nn, Tensor, LongTensor

from torchvision.models import resnet18
from transformers.models.bert import BertModel, BertConfig


class PseudoCLIP(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.config = config = BertConfig(
            vocab_size=1000, num_hidden_layers=1, max_position_embeddings=128
        )

        # 简易的图片编码器
        self.image_encoder = resnet18(num_classes=config.hidden_size)
        # task-specific 层
        # self.image_projector = nn.Linear(config.hidden_size, config.hidden_size)
        # 简易的文本编码器
        self.text_encoder = BertModel(config, add_pooling_layer=True)
        # task-specific 层
        # self.text_projector = nn.Linear(config.hidden_size, config.hidden_size)
        # 将 temperature 设置成可学习参数
        temperature = math.log(1 / 0.07)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def encode_image(self, batch_images: Tensor) -> Tensor:
        """ 编码图片, batch_images: [batch_size, n_channels, img_height, img_width] """
        image_vectors = self.image_encoder(batch_images)
        # image_projector 层是针对 图文匹配 任务设置的, 这里的向量编码也是为了 图文匹配 任务编码的
        # 如果是其它下游任务, 使用 linear probe 的方式, 则 image_projector 层要去除掉
        # image_vectors = self.image_projector(image_vectors)
        image_vectors = torch.nn.functional.normalize(image_vectors, dim=1)
        return image_vectors  # [batch_size, hidden_size]
    
    def encode_text(self, batch_input_ids: LongTensor) -> Tensor:
        """ 编码文字, batch_input_ids: [batch_size, num_tokens] """
        attn_mask = batch_input_ids.ne(0).float()

        # ## 方式 1: 取最后一个 token 作为句向量
        # token_vectors = self.text_encoder(text_input_ids, attn_mask)[0]  # [batch_size, num_tokens, hidden_size]
        # text_vectors = token_vectors[torch.where(text_input_ids == 102)]  # [batch_size, hidden_size]

        # ## 方式 2: 直接用 pooler 的结果作为句向量
        text_vectors = self.text_encoder(batch_input_ids, attn_mask)[1]  # [batch_size, hidden_size]

        # text_vectors = self.text_projector(text_vectors)
        text_vectors = torch.nn.functional.normalize(text_vectors, dim=1)
        return text_vectors

    def forward(self, batch_images: Tensor, batch_input_ids: LongTensor) -> Tensor:
        """
        batch_images: [batch_size, n_channels, img_height, img_width]
        batch_input_ids: [batch_size, num_tokens]
        :ret contrastive loss (in-batch negatives)
        """

        # ## step1: 编码图片和文本
        image_vectors = self.encode_image(batch_images)
        text_vectors = self.encode_text(batch_input_ids)

        # ## step2: 计算 loss 值
        image_logits = image_vectors @ text_vectors.t() / self.temperature.exp()
        text_logits = image_logits.t()
        
        labels = torch.arange(image_logits.size(0))
        image_loss = torch.nn.functional.cross_entropy(image_logits, labels)
        text_loss = torch.nn.functional.cross_entropy(text_logits, labels)
        
        loss = (image_loss + text_loss) / 2
        return loss 


def test_forward(model: PseudoCLIP):
    images = torch.randint(low=0, high=255, size=(10, 3, 224, 224)).float() / 255.
    
    input_ids = torch.randint(
        low=1, high=model.config.vocab_size, 
        size=(10, model.config.max_position_embeddings)
    )
    
    loss = model(input_ids, images)
    
    loss.backward()
    
    print(loss)
    
    print(model.temperature.grad)


def test_inference(model: PseudoCLIP):
    # reference: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb 

    from torch.utils.data import Dataset, DataLoader

    class TestDataset(Dataset):
        def __init__(self, num_images: int = 100, num_labels: int = 2):
            super().__init__()
            self.total_images = torch.randint(low=0, high=256, size=(num_images, 3, 224, 224), dtype=torch.float) / 255.
            self.total_labels = torch.randint(low=0, high=num_labels, size=(num_images, ))
            self.num_images = num_images
        
        def __getitem__(self, index) -> Tensor:
            return self.total_images[index], self.total_labels[index]
        
        def __len__(self) -> int:
            return self.num_images

    def test_tokenize(texts: list[str]):
        return torch.randint(
            low=1, high=model.config.vocab_size, 
            size=(len(texts), model.config.max_position_embeddings)
        )

    @torch.no_grad()
    def create_cls_weights(label_names: list[str], templates: list[str]) -> Tensor:
        cls_vectors = []

        for label_name in label_names:
            # 根据 template 构建分类文本
            texts = [template.format(label_name) for template in templates]
            # 对文本进行分词
            input_ids = test_tokenize(texts)  # [batch_size, num_tokens]
            # 编码成向量
            text_vectors = model.encode_text(input_ids)  # [batch_size, hidden_size]
            # 将所有的向量取平均
            cls_vector = text_vectors.mean(dim=0)  # [hidden_size, ]
            # 标准化成单位向量
            cls_vector = torch.nn.functional.normalize(cls_vector, dim=0)  # [hidden_size]
            cls_vectors.append(cls_vector)

        cls_vectors = torch.stack(cls_vectors, dim=0)  # [num_labels, hidden_size]
        return cls_vectors
    
    def accuracy(logits: Tensor, y_true: Tensor, top_k_list: list[int] = None) -> list[int]:
        # logits: [batch_size, num_labels]
        # y_true: [batch_size, ]

        if top_k_list is None:
            top_k_list = [1, ]
        
        y_pred = torch.topk(logits, k=max(top_k_list), dim=1, largest=True, sorted=True).indices  # [batch_size, top_k]
        correct_matrix: Tensor = y_true.unsqueeze(-1) == y_pred
        
        return [correct_matrix[:, :k].flatten().sum().item() for k in top_k_list]  # 正确的数量
    
    test_templates = [
        "A photo of a {}.", 
        "A bad photo of a {}.",
        "A sculpture of a {}."
    ]
    
    test_labels = ["apple", "banana", "peef", "pear", "peach", "orange"]
    
    test_dataloader = DataLoader(TestDataset(num_labels=len(test_labels)), batch_size=10, shuffle=False)
    
    test_cls_vectors = create_cls_weights(test_labels, test_templates)  # [num_labels, hidden_size]
    
    top1_correct_num, top6_correct_num, total_num = 0., 0., 0.
    for test_images, test_targets in test_dataloader:
        test_image_vectors = model.encode_image(test_images)  # [batch_size, hidden_size]
        
        logits = test_image_vectors @ test_cls_vectors.t()  # [batch_size, num_labels]
        # probs = torch.softmax(logits, dim=1)
        
        top1, top6 = accuracy(logits, test_targets, [1, 6])
        
        top1_correct_num += top1
        top6_correct_num += top6 
        total_num += test_images.size(0)
    
    print(f"Top-1 accuracy: {top1_correct_num / total_num * 100}%")
    print(f"Top-6 accuracy: {top6_correct_num / total_num * 100}%")


if __name__ == "__main__":
    model_ = PseudoCLIP().eval()
    
    test_inference(model_)
