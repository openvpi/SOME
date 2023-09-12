import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, conditiondim=None):
        super().__init__()
        if conditiondim is None:
            conditiondim = dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(conditiondim, hidden_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1),
                                    )

    def forward(self, q, kv=None):
        # b, c, h, w = x.shape
        if kv is None:
            kv = q
        q, kv = map(
            lambda t: rearrange(t, "b c t -> b t c", ), (q, kv)
        )

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=2)

        q, k, v = map(
            lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.heads), (q, k, v)
        )

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(out, "b h t c -> b (h c) t", h=self.heads, )
        return self.to_out(out)
