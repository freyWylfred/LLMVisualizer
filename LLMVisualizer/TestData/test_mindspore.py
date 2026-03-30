import mindspore
import mindspore.nn as nn
from mindspore import Tensor

vocab_size = 32000
hidden_size = 768
num_heads = 12
num_layers = 8

class MindSporeAttentionBlock(nn.Cell):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm([hidden_size])
        self.ffn = nn.Dense(hidden_size, hidden_size * 4)
        self.out_proj = nn.Dense(hidden_size * 4, hidden_size)
        self.norm2 = nn.LayerNorm([hidden_size])

    def construct(self, x):
        h = self.attn(x, x, x)[0]
        x = self.norm1(x + h)
        x = self.norm2(x + self.out_proj(nn.GELU()(self.ffn(x))))
        return x


class MindSporeTransformer(nn.Cell):
    def __init__(self, vocab_size=32000, hidden_size=768, num_heads=12, num_layers=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.CellList([
            MindSporeAttentionBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm([hidden_size])
        self.head = nn.Dense(hidden_size, vocab_size)

    def construct(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.head(x)
        return x
