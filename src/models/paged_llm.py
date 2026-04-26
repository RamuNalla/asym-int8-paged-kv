import torch
import torch.nn as nn
import math
from src.cache.paged_kv import PagedKVCache

class PagedMinimalAttention(nn.Module):
    """
    Attention layer integrated with the Block-based Memory Manager.
    """
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, paged_cache: PagedKVCache = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if paged_cache is not None:
            # Route new keys/values into the block table and retrieve the reconstructed sequence
            k, v = paged_cache.update(k, v)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if paged_cache is None and seq_len > 1:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)