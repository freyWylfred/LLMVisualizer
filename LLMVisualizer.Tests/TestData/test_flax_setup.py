import jax
import jax.numpy as jnp
import flax.linen as nn

vocab_size = 32000
hidden_size = 512
num_heads = 8
num_layers = 6

class FlaxAttentionBlock(nn.Module):
    hidden_size: int
    num_heads: int

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.norm = nn.LayerNorm()
        self.ffn = nn.Dense(features=self.hidden_size * 4)
        self.out_proj = nn.Dense(features=self.hidden_size)

    def __call__(self, x):
        h = self.norm(x)
        h = self.attn(h, h)
        x = x + h
        x = x + self.out_proj(nn.relu(self.ffn(x)))
        return x


class FlaxTransformer(nn.Module):
    vocab_size: int = 32000
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size)
        self.blocks = [FlaxAttentionBlock(self.hidden_size, self.num_heads) for _ in range(self.num_layers)]
        self.ln_f = nn.LayerNorm()
        self.head = nn.Dense(features=self.vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.head(x)
        return x
