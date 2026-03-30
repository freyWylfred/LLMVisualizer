import paddle
import paddle.nn as nn

vocab_size = 30000
hidden_size = 512
num_heads = 8
num_layers = 6

class PaddleTransformerBlock(nn.Layer):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiHeadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Linear(hidden_size, hidden_size * 4)
        self.out_proj = nn.Linear(hidden_size * 4, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        h = self.attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.out_proj(paddle.nn.functional.gelu(self.ffn(x))))
        return x


class PaddleGPT(nn.Layer):
    def __init__(self, vocab_size=30000, hidden_size=512, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.LayerList([
            PaddleTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.head(x)
        return x
