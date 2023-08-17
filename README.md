# Essential-Transformer
## Understanding the backbone of encoders and decoders in 45-minutes

This GitHub repository provides a minimalist yet comprehensive implementation of the Transformer architecture's encoder and decoder components, aimed at providing an intuitive understanding of the core concepts underlying this powerful model. The Transformer architecture has revolutionized natural language processing and machine translation. The implementations serve as a didactic resource for enthusiasts, researchers, and learners who wish to grasp its fundamental principles. 

To keep things simple a couple of assumptions are made:
* positional embeddings are treated as trainable that are added to the token embeddings
* the embedding dimensionality must be a multiple of the number of heads (the joint embedding is reshaped before softmax normalization)


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
2. Toy example of instantiating a transformer decoder:
```python
import torch
from decoder import Transformer

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

3. Toy example of instantiating a transformer decoder with multi-query attention:
```python
import torch
from decoder_multi_query_attention import Transformer

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

4. Toy example of instantiating a transformer encoder:
```python
import torch
from encoder import Transformer

num_heads = 16
emb_dim = 768
ffn_dim = 1024
num_layers = 12
max_len = 128
vocab_sz = 10000
batch_sz = 10
# Toy input data corresponding to random tokens
x = torch.randint(0,vocab_sz,(batch_sz, max_len))

trans = Transformer(num_layers, vocab_sz, emb_dim, max_len, num_heads, ffn_dim)
trans(x)
```
