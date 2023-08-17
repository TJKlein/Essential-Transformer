import numpy as np
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

# Basic encoder block of Transformer
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int, dropout: float=0.1, act: nn.Module=nn.ReLU()):
        super(TransformerBlock,self).__init__()
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        
        self.q = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        
        self.mlp = nn.Sequential(nn.Linear(emb_dim, ffn_dim),
                                 nn.Dropout(dropout),
                                 nn.LayerNorm(ffn_dim),
                                 act,
                                 nn.Linear(ffn_dim,emb_dim)
                                )
        
    def attention(self, x: torch.tensor) -> torch.tensor:
        
        q = einops.rearrange(self.q(x), "B N (H E) -> B H N E", H=self.num_heads)
        k = einops.rearrange(self.k(x), "B N (H E) -> B H E N", H=self.num_heads)
        v = einops.rearrange(self.v(x), "B N (H E) -> B H N E", H=self.num_heads)
        
        # B H N E x B H E N -> B H N N
        score = q @ k / np.sqrt(1/self.emb_dim)
        weight = F.softmax(score, dim=-1)
        # B H N N x B H N E -> B H N E
        attn = weight @ v
        attn = self.dropout(attn)
        attn = einops.rearrange(attn, "B H N E -> B N (H E)")
        return attn
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = self.mlp(self.ln2(x))
        return x

# Constructing a Transformer stack from decoder blocks
class Transformer(nn.Module):
    def __init__(self, num_layers:int, vocab_sz: int, emb_dim: int, max_len:int, num_heads: int, ffn_dim: int, dropout: float=0.1, act: nn.Module=nn.ReLU()):
        super(Transformer,self).__init__()
        
        trans_stack = []
        for i in range(num_layers):
            trans_stack.append(TransformerBlock(emb_dim, num_heads, ffn_dim, dropout, act))
            
        self.transformer = nn.Sequential(*trans_stack)
        
        self.ln1 = nn.LayerNorm(emb_dim)
        self.final = nn.Linear(emb_dim, vocab_sz)

        # The first section comprises learnable positional embeddings, the second part the actual token embeddings
        self.emb = nn.Linear(max_len+vocab_sz, emb_dim)
        
        self.max_len = max_len
        self.vocab_sz = vocab_sz
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        BS = x.shape[0]
        # B N (N +  max_len)
        one_hot = torch.zeros((BS,self.max_len, self.max_len+self.vocab_sz))
        for data in range(BS):
            for i in range(self.max_len):
                one_hot[data,i,i]=1
                one_hot[data,i,self.max_len + x[data,i]] = 1
        # B N (N + max_len) -> B N (H E)
        emb = self.emb(one_hot)
        emb = self.transformer(emb)
        return self.final(self.ln1(emb))
        
