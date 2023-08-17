# Essential-Transformer


This GitHub repository presents a minimalist yet comprehensive implementation of the Transformer architecture's encoder and decoder components, aimed at providing an intuitive understanding of the core concepts underlying this powerful model. The Transformer architecture has revolutionized natural language processing and machine translation. The implementations serve as a didactic resource for enthusiasts, researchers, and learners who wish to grasp its fundamental principles. Among the simplifications is a trainable positional embedding that is added to the token embeddings.


1. Toy example of instantiating a decoder block:

```python
import torch
from decoder import Transformer, TransformerBlock

# Some toy parameters
num_heads = 16
emb_dim = 768
ffn_dim = 1024
num_layers = 12
max_len = 128
vocab_sz = 10000
batch_sz = 10

# Toy input data corresponding to embeddings
x = torch.randn((batch_sz,max_len,emb_dim))

tb = TransformerBlock(max_len, emb_dim, ffn_dim, num_heads)
tb(x)
```
2. Toy example of instantiating a transformer block:
```python
import torch
from decoder import Transformer, TransformerBlock

# Some toy parameters
num_heads = 16
emb_dim = 768
ffn_dim = 1024
num_layers = 12
max_len = 128
vocab_sz = 10000
batch_sz = 10

# Toy input data corresponding to random tokens
x = torch.randint(0,vocab_sz,(batch_sz, max_len))

trans = Transformer(num_layers, num_heads, max_len, vocab_sz, emb_dim, ffn_dim)
trans(x)
```
