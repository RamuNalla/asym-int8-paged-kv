import torch
import torch.nn as nn
import math
from src.cache.standard_kv import StandardKVCache

class MinimalAttention(nn.Module):
    """
    A stripped-down Multi-Head Attention layer designed strictly 
    to benchmark compute complexity and KV-cache routing.
    """
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, kv_cache: StandardKVCache = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # 1. Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. KV-Cache Routing
        if kv_cache is not None:
            # Standard Cache active: append new K, V and fetch full history
            k, v = kv_cache.update(k, v)
        
        # 3. Core Attention Math: Softmax(Q * K^T / sqrt(d)) * V
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask logic (simplified for benchmarking full vs single step)
        if kv_cache is None and seq_len > 1:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # 4. Reassemble and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)