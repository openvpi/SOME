import torch
import torch.nn as nn


class BoundaryEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, p_invert=0):
        super().__init__()
        self.p_invert = p_invert
        self.embedding = nn.Embedding(2, embedding_dim)

    def forward(self, x):
        x = x.cumsum(dim=-1).fmod(2)
        invert_flags = torch.rand(*x.shape[:-1], device=x.device).unsqueeze(-1) < self.p_invert
        x = x ^ invert_flags
        x = self.embedding(x)
        return x
