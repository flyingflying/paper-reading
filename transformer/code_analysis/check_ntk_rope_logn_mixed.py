# -*- coding:utf-8 -*-
# copied from transformers.models.llama.modeling_llama and https://github.com/bojone/rerope/blob/main/ntk_patch.py 
# revised by lqxu 

import math 
import torch 

training_length = 4096


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, ntk_mixed_version: bool = False):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

        # ############################################################################
        # new code 
        if ntk_mixed_version:
            k, b = 12, 0.75
            max_position_embeddings = training_length * k
            a = math.log(k) / (dim / 2)**b
            inv_freq = base**(-torch.arange(0, dim, 2).float().to(device) / dim)
            inv_freq *= (-a * torch.arange(1, dim // 2 + 1).float().to(device)**b).exp()
            self.register_buffer('inv_freq', inv_freq)
            self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.get_default_dtype())
        # ############################################################################

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    """
    我理解的使用方法是: 如果原始的 max_position_embeddings=2048, 现在改为 max_position_embeddings=4096, scaling_factor=2.0
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_and_logn_scale(q, k, cos, sin, position_ids):
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    scale = ((position_ids + 1)[:, None, :, None].log() / math.log(training_length)).clip(1)
    return q_embed * scale.to(q_embed.dtype), k_embed


if __name__ == "__main__":

    from transformers.models.llama import LlamaConfig
    
    config = LlamaConfig()
    
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    batch_size, num_tokens = 2, 1024

    shape = (batch_size, num_heads, num_tokens, head_dim)
    query_states = torch.randn(size=shape)
    key_states = torch.randn(size=shape)
    position_ids = torch.arange(num_tokens).unsqueeze(0).repeat(batch_size, 1)

    rotary_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings=config.max_position_embeddings)
    cos, sin = rotary_emb(query_states, seq_len=num_tokens)
    query_states_v0, key_states_v0 = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    print(query_states_v0.shape)
    print(key_states_v0.shape)

    rotary_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings=config.max_position_embeddings, ntk_mixed_version=True)
    cos, sin = rotary_emb(query_states, seq_len=num_tokens)
    query_states_v1, key_states_v1 = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    print(query_states_v1.shape)
    print(key_states_v1.shape)
