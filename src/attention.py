import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, d_embed*3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:

        batch_size, seq_len, d_embed = x.shape

        intermidiate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(intermidiate_shape).transpose(1,2)
        k = k.view(intermidiate_shape).transpose(1,2)
        v = v.view(intermidiate_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2) # / (self.d_head ** 0.5)

        if causal_mask:
            mask = torch.ones_like(weight, d_type=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= (self.d_head ** 0.5)

        weight = F.softmax(weight, dim=-1)

        out = weight @ v

        out = out.transpose(1,2).reshape(batch_size, seq_len, d_embed)
        out = self.out_proj(out)

        return out

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, embedding_dim: int, n_cross: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.k_proj = nn.Linear(n_cross, embedding_dim, bias=in_proj_bias)
        self.v_proj = nn.Linear(n_cross, embedding_dim, bias=in_proj_bias)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = embedding_dim // n_heads

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor: # x is latent, context is text
        batch_size, seq_len, d_embed = x.shape

        intermidiate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q = self.q_proj(x).view(intermidiate_shape).transpose(1,2)
        k = self.k_proj(context).view(intermidiate_shape).transpose(1,2)
        v = self.v_proj(context).view(intermidiate_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2) / (self.d_head ** 0.5)

        weight = F.softmax(weight, dim=-1)

        out = weight @ v

        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, d_embed)
        out = self.out_proj(out)

        return out
