import torch
from typing import Tuple

class PreAllocatedKVCache:
    """
    Implements a static KV-Cache. 
    Memory is pre-allocated for the maximum sequence length to avoid torch.cat overhead.
    """
    def __init__(self, batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, device: torch.device):
        self.max_seq_len = max_seq_len
        self.current_seq_len = 0
        
        # Pre-allocate the massive contiguous tensors upfront
        # Shape: (batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)
        self.v_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inserts new keys and values into the pre-allocated block and returns the active view."""
        new_tokens_len = k_new.size(2)
        
        if self.current_seq_len + new_tokens_len > self.max_seq_len:
            raise ValueError("Exceeded maximum sequence length!")

        # Insert the new token into the exact reserved slot
        self.k_cache[:, :, self.current_seq_len : self.current_seq_len + new_tokens_len, :] = k_new
        self.v_cache[:, :, self.current_seq_len : self.current_seq_len + new_tokens_len, :] = v_new
        
        self.current_seq_len += new_tokens_len
        
        # Return ONLY the active, "filled" portion of the cache for attention math
        k_active = self.k_cache[:, :, :self.current_seq_len, :]
        v_active = self.v_cache[:, :, :self.current_seq_len, :]
        
        return k_active, v_active