import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np

# Basic decoder block of Transformer
class TransformerBlock(nn.Module):
    def __init__(self, max_len:int, emb_dim: int, ffn_dim: int, num_heads: int, dropout: float=0.1, act: nn.Module=nn.ReLU()):
        super(TransformerBlock, self).__init__()
        assert emb_dim % num_heads == 0, f"Embedding dim {emb_dim} must be multiple of number of heads {num_heads}"
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.max_len = max_len
        
        self.v = nn.Linear(emb_dim, emb_dim)
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        
        self.mlp = nn.Sequential(nn.Linear(emb_dim, ffn_dim),
                                 nn.Dropout(dropout),
                                 nn.LayerNorm(ffn_dim),
                                 act,
                                 nn.Linear(ffn_dim,emb_dim)
                                 )
        self.dropout = nn.Dropout(dropout)
        
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        # parameters, which are not optimized for but saved in dictionary
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(self.max_len, self.max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        
    def attention(self, x: torch.tensor) -> torch.tensor:
        
        # x: B N L = B N (num_heads emb_dim)
        q = einops.rearrange(self.q(x), "B N (H E) -> B H N E", H=self.num_heads)
        k = einops.rearrange(self.k(x), "B N (H E) -> B H E N", H=self.num_heads)
        v = einops.rearrange(self.v(x), "B N (H E) -> B H N E", H=self.num_heads)
        
        # B H N N 
        score = (q @ k ) / np.sqrt(1/self.emb_dim)
        score = score.masked_fill(self.mask == 0, float("-inf"))
        weight = F.softmax(score, dim=-1)
        
        # B H N N x B H N E -> B H N E
        attn = einops.rearrange(weight @ v, "B H N E -> B N (H E)")
        attn = self.attn_dropout(attn)
        
        
        return attn
     
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.dropout(self.attention(self.ln1(x)))
        return self.mlp(self.ln2(x))

# Constructing a Transformer stack from decoder blocks
class Transformer(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, max_len: int, vocab_sz: int, emb_dim: int, ffn_dim: int, dropout: float=0.1, act: nn.Module=nn.ReLU()):
        super(Transformer, self).__init__()

        # The first section comprises learnable positional embeddings, the second part the actual token embeddings
        self.emb = nn.Linear(max_len+vocab_sz, emb_dim)
        
        trans_stack = []
        for i in range(num_layers):
            trans_stack.append(TransformerBlock(emb_dim, ffn_dim, num_heads, dropout,act))
        self.transformer = nn.Sequential(*trans_stack)
        
        self.final = nn.Linear(emb_dim, vocab_sz)
        
        self.max_len = max_len
        self.vocab_sz = vocab_sz
    
    def forward(self, x):
        
        one_hot = torch.zeros((x.shape[0], self.max_len, self.max_len+self.vocab_sz))
        
        for sample in range(x.shape[0]):
            for token in range(self.max_len):
                one_hot[sample,token,token] = 1
                one_hot[sample,token,self.max_len+x[sample,token]] = 1
        emb = self.emb(one_hot)
        emb = self.transformer(emb)
        emb = self.final(emb)
        
        return emb
        
