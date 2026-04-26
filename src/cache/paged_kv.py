import torch
from typing import Tuple, List

class PagedKVCache:
    """
    Implements PagedAttention "Lite".
    Memory is managed in non-contiguous, fixed-size blocks to eliminate fragmentation.
    """
    def __init__(self, batch_size: int, num_heads: int, head_dim: int, block_size: int = 16, device: torch.device = 'cpu'):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device
        
        self.current_seq_len = 0
        
        # The "Physical Memory Pool": A list storing individual, non-contiguous block tensors
        self.k_blocks: List[torch.Tensor] = []
        self.v_blocks: List[torch.Tensor] = []
        
        # The "Page Table": Maps the logical sequence to physical block indices
        self.block_table: List[int] = []

    def allocate_new_block(self):
        """Creates a new physical block and updates the page table."""
        new_k_block = torch.zeros(self.batch_size, self.num_heads, self.block_size, self.head_dim, device=self.device)
        new_v_block = torch.zeros(self.batch_size, self.num_heads, self.block_size, self.head_dim, device=self.device)
        
        block_idx = len(self.k_blocks)
        self.k_blocks.append(new_k_block)
        self.v_blocks.append(new_v_block)
        self.block_table.append(block_idx)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Routes new tokens into the correct physical block."""
        new_tokens_len = k_new.size(2)
        
        for i in range(new_tokens_len):
            # Calculate which block and which slot inside the block this token belongs to
            logical_position = self.current_seq_len + i
            block_idx_in_table = logical_position // self.block_size
            slot_in_block = logical_position % self.block_size
            
            # If we've exceeded mapped blocks, allocate a new page from the OS
            if block_idx_in_table >= len(self.block_table):
                self.allocate_new_block()
                
            physical_block_idx = self.block_table[block_idx_in_table]
            
            # Insert the token into its exact slot in the non-contiguous physical block
            self.k_blocks[physical_block_idx][:, :, slot_in_block:slot_in_block+1, :] = k_new[:, :, i:i+1, :]
            self.v_blocks[physical_block_idx][:, :, slot_in_block:slot_in_block+1, :] = v_new[:, :, i:i+1, :]

        self.current_seq_len += new_tokens_len
        
        # For this pure PyTorch "Lite" implementation, we dynamically reconstruct the contiguous 
        # sequence just for the attention matmul to avoid writing custom CUDA kernels yet.
        k_active = torch.cat(self.k_blocks, dim=2)[:, :, :self.current_seq_len, :]
        v_active = torch.cat(self.v_blocks, dim=2)[:, :, :self.current_seq_len, :]
        
        return k_active, v_active