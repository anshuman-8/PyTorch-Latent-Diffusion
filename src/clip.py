import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, context_size:int): # context_size is the number of tokens in the context
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # for positional embedding learnable parameters are used 
        self.positional_embedding = nn.Parameter(torch.zeros(context_size, embedding_dim)) # try random instead of zeros

    def forward(self, tokens):
        output = self.token_embedding(tokens) + self.positional_embedding
        return output
    

class CLIPLayer(nn.Module):
    def __init__(self, num_heads:int, embedding_dim:int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.attention = SelfAttention(num_heads, embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim*4)
        self.linear2 = nn.Linear(embedding_dim*4, embedding_dim)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        residue1 = x
        x = self.attention(x)

        x = self.layer_norm1(x)
        x = x + residue1

        residue2 = x

        x = self.layer_norm2(x)
        x = self.linear1(x)
        x *= torch.sigmoid(x*1.702) # Quick GELU Activation
        x = self.linear2(x)
        
        x = x + residue2

        output = x

        return output


class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens:torch.LongTensor)->torch.FloatTensor:
        token = tokens.type(torch.long)
        state = self.embedding(token)

        for layer in self.layers:
            state = layer(state)

        output = self.layer_norm(state)

        return output
    
        