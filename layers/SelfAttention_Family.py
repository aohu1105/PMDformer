import torch
import torch.nn as nn
from math import sqrt
from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        _, S, _ = x.shape
        H = self.n_heads

        
        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, S, H, -1)
        values = self.value_projection(x).view(B, S, H, -1)

        scale = 1. / sqrt(queries.size(-1))
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        out = rearrange(V, 'b l h d -> b l (h d)')
        out = self.out_projection(out)

        return out, A
    

class TrendRestorationAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(TrendRestorationAttention, self).__init__()
        
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, means):
        B, L, _ = x.shape
        _, S, _ = x.shape
        H = self.n_heads

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, S, H, -1)
        values = self.value_projection(x).view(B, S, H, -1) + means.unsqueeze(2)

        scale = 1. / sqrt(queries.size(-1))
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        out = rearrange(V, 'b l h d -> b l (h d)')
        out = self.out_projection(out)

        return out, A
