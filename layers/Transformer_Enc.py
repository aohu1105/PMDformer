import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class EncoderLayer_PVA(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(EncoderLayer_PVA, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        new_x, attn = self.attention(
            x
        )
        x = x + self.dropout(new_x)
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y), attn


class Encoder_PVA(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder_PVA, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
    

    def forward(self, x):
        # x [B, L, D]
        attns = []
        
        for attn_layer in (self.attn_layers):
            x, attn = attn_layer(x)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderLayer_TRA(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(EncoderLayer_TRA, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, means):
        new_x, attn = self.attention(
            x, means
        )
        x = x + self.dropout(new_x)
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y), attn


class Encoder_TRA(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder_TRA, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
    

    def forward(self, x, means):
        # x [B, L, D]
        attns = []
        
        for attn_layer in (self.attn_layers):
            x, attn = attn_layer(x, means)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)

        return x, attns
