import pandas as pd #1.5.3
import numpy as np #1.20.3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class SnpEmbedding(nn.Module):
    """
    Snp embeddings - use default nn.Embedding. Created class for potential custom functionality / encapsulation
    """
    def __init__(self, snp_encoding_size, embed_size):
        super(SnpEmbedding, self).__init__()
        self.embedding = nn.Embedding(snp_encoding_size, embed_size)  # Create an embedding layer
        
    def forward(self, x):
        x = x.long()
        return self.embedding(x)  # Forward pass to get embeddings
        
class PosEmbedding(nn.Module):
    """
    Pos embeddings - sine-cosine encoding of absolute snp positions. Enables positional information to be
    captured and the model to learn positional contexts between SNPs. 
    """
    def __init__(self, max_pos_length, embed_size):
        super(PosEmbedding, self).__init__()
        self.max_pos_length = max_pos_length
        self.embed_size = embed_size        
        
        # Create a positional encoding matrix with shape (max_position, embedding_dim). Sine + cosine values calculated in
        # embedding space. Relative positions and attension can be learned.
        position = torch.arange(max_pos_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        positional_encoding = torch.zeros(max_pos_length, embed_size)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Register this matrix as a buffer that is not a model parameter
        self.register_buffer('positional_encoding', positional_encoding)
        
    def forward(self, x):
        """
        Inputs:
            x: A tensor of shape (batch_size, sequence_length) containing the SNP positions.
        Returns:
            A tensor of shape (batch_size, sequence_length, embedding_dim) with added positional encodings.
        """
        # Retrieve the positional encodings based on the SNP positions in x
        # Ensure the positions in x do not exceed max_position and is int (scaled positions are float)
        x = x.clamp(0, self.max_pos_length - 1)
        x = x.round().long()
        return self.positional_encoding[x]

# helper functions
def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class LinformerSelfAttention(nn.Module):
    def __init__(self, embed_size, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        
        dim = embed_size
        
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len =  seq_len
        self.k = k
        

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context = None, **kwargs):
        # x shape is [batch, seq_len, embed_size]
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len <= self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # allow for variable sequence lengths (less than maximum sequence length) by slicing projections

        if kv_len < self.seq_len:
            kv_projs = map(lambda t: t[:kv_len], kv_projs)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))
        
        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
        
class TransformerLayer(nn.Module):
    def __init__(self, embed_size, seq_len, heads, dropout, k, forward_expansion=4):
        super(TransformerLayer, self).__init__()
        self.attention = LinformerSelfAttention(embed_size, seq_len, k, heads, 
                                            dim_head = None, one_kv_head = False, share_kv = False, 
                                            dropout=dropout) 
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Values, Keys and Queries have size: (batch_size, query_len, embedding_size)
        attention = self.attention(x) # attention shape: (batch_size, query_len, embedding_size)
        # Add skip connection, run through normalization and finally dropout
        norm_out = self.dropout(self.norm1(attention + x)) # x shape: (batch_size, query_len, embedding_size)
        forward = self.feed_forward(norm_out) # forward shape: (batch_size, query_len, embedding_size)
        out = self.dropout(self.norm2(forward + x)) # out shape: (batch_size, query_len, embedding_size)
        return out

class Encoder(nn.Module):
    def __init__(self, snp_encoding_size, embed_size, seq_len, num_layers, heads,
        device, forward_expansion, dropout, k, max_pos_length): 
        super(Encoder, self).__init__()
        self.embed_size = embed_size # size of the input embedding
        self.device = device # either "cuda" or "cpu"
        # Lookup table with an embedding for each word in the vocabulary
        self.snp_embedding = SnpEmbedding(snp_encoding_size, embed_size)
        # Lookup table with a positional embedding for each word in the sequence
        self.position_embedding = PosEmbedding(max_pos_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_size,
                    seq_len,
                    heads,
                    dropout,
                    k,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        """
        Forward pass.
        :param x: source sequence. Shape: (batch_size, source_sequence_len).
        :param positions: source positions. Shape: (batch_size, source_sequence_len).
        :return output: torch tensor of shape (batch_size, src_sequence_length, embedding_size)
        """
        batch_size, seq_length = x.shape
        
        out = self.dropout(
            (self.snp_embedding(x) # Shape (batch_size, snps_total, embed_size) e.g. (200000,5,128)
            + self.position_embedding(positions)) # Shape (batch_size, snps_total, embed_size) e.g. (200000,5,128)
        )
        # Final shape should be [batch_size, snp_total, embed_size]

        # In the Encoder the query, key, value are all the same
        for layer in self.layers:
            out = layer(out)

        return out

class Transformer(nn.Module):
    def __init__(self, snp_encoding_size, src_pad_idx, embed_size, seq_len,
                 num_layers, forward_expansion, heads, dropout, k, device, max_pos_length):

        super(Transformer, self).__init__()
        # === Encoder ===
        self.encoder = Encoder(snp_encoding_size, embed_size, seq_len, num_layers, heads,
                               device, forward_expansion, dropout, k, max_pos_length )
        self.src_pad_idx = src_pad_idx
        self.device = device
        
        # === Regression Out ===
        self.fc_out = nn.Linear(embed_size, 1) # Single regression target value


    def forward(self, snp, pos, y=[]):

        enc_out = self.encoder(snp, pos) 
        
        # Aggregate layers output e.g. mean
        aggregated_out = enc_out.mean(dim=1)  # [batch_size, embed_size]
        
        out = self.fc_out(aggregated_out) # [batch_size, 1]
        return out