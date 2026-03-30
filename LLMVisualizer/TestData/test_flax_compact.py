import jax
import jax.numpy as jnp
import flax.linen as nn

vocab_size = 16384
hidden_size = 256
num_heads = 4

class CompactTransformer(nn.Module):
    vocab_size: int = 16384
    hidden_size: int = 256
    num_heads: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(num_heads=self.num_heads)(x)
        x = nn.Dense(features=self.hidden_size * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.LayerNorm()(x)
        logits = nn.Dense(features=self.vocab_size)(x)
        return logits
