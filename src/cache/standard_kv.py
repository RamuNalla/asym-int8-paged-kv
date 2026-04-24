import torch
from typing import Tuple, Optional

class StandardKVCache:
    """
    Implements a standard, contiguous KV-Cache.
    Tensors grow linearly with each generation step.
    """
    def __init__(self):
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new keys and values to the contiguous cache."""
        if self.k_cache is None or self.v_cache is None:
            self.k_cache = k_new
            self.v_cache = v_new
        else:
            # Concatenate along the sequence length dimension (dim=2)
            # Shape expectation: (batch_size, num_heads, seq_len, head_dim)
            self.k_cache = torch.cat([self.k_cache, k_new], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=2)
            
        return self.k_cache, self.v_cache

    def reset(self):
        self.k_cache = None
        self.v_cache = None