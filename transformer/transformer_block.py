from torch import nn

from transformer.self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads)

        # The layer normalization is applied over the embedding dimension only.
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            nn.ReLU(),
            nn.Linear(4 * emb, emb)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        ff = self.ff(x)
        res = self.norm2(ff + x)
        return res
