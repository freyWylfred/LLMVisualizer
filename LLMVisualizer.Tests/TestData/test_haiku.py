import jax
import jax.numpy as jnp
import haiku as hk

vocab_size = 32000
hidden_size = 768
num_heads = 12

class HaikuTransformer(hk.Module):
    def __init__(self, vocab_size=32000, hidden_size=768, num_heads=12, name=None):
        super().__init__(name=name)
        self.embedding = hk.Embed(vocab_size, hidden_size)
        self.linear1 = hk.Linear(hidden_size * 4)
        self.linear2 = hk.Linear(hidden_size)
        self.ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.head = hk.Linear(vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        x = self.ln(x)
        x = self.linear1(x)
        x = jax.nn.gelu(x)
        x = self.linear2(x)
        x = self.head(x)
        return x
