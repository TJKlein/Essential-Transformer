# Essential-Transformer


This GitHub repository presents a minimalist yet comprehensive implementation of the Transformer architecture's encoder and decoder components, aimed at providing an intuitive understanding of the core concepts underlying this powerful model. The Transformer architecture has revolutionized natural language processing and machine translation. The implementations serve as a didactic resource for enthusiasts, researchers, and learners who wish to grasp its fundamental principles. Among the simplifications is a trainable positional embedding that is added to the token embeddings.


Toy example of instantiating a decoder block:

```shell
import torch
from decoder import Transformer, TransformerBlock

# some toy parameters
num_heads = 16
emb_dim = 768
ffn_dim = 1024
num_layers = 12
max_len = 128
vocab_sz = 10000
BS = 10

x = torch.randn((BS,max_len,emb_dim))
tb = TransformerBlock(max_len, emb_dim, ffn_dim, num_heads)
tb(x)
```
