import torch
import matplotlib.pyplot as plt
from src.cache.paged_kv import PagedKVCache

class PagedMemoryProfiler:
    def __init__(self, dtype_bytes=4):
        self.dtype_bytes = dtype_bytes

    def calculate_allocated_mb(self, cache: PagedKVCache) -> float:
        """Calculates VRAM based on the number of physical blocks actually allocated."""
        if len(cache.block_table) == 0: return 0.0
        
        # Elements per block = batch * num_heads * block_size * head_dim
        k_elements = cache.k_blocks[0].numel() * len(cache.block_table)
        v_elements = cache.v_blocks[0].numel() * len(cache.block_table)
        return ((k_elements + v_elements) * self.dtype_bytes) / (1024 ** 2)

    def calculate_active_mb(self, cache: PagedKVCache) -> float:
        """Calculates VRAM holding exactly the active tokens."""
        if cache.current_seq_len == 0: return 0.0
        
        active_k_elements = cache.batch_size * cache.num_heads * cache.current_seq_len * cache.head_dim
        active_v_elements = cache.batch_size * cache.num_heads * cache.current_seq_len * cache.head_dim
        return ((active_k_elements + active_v_elements) * self.dtype_bytes) / (1024 ** 2)

def main():
    batch_size = 1
    max_seq_len = 2048
    num_heads = 32
    head_dim = 128
    block_size = 16
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Initializing Paged Cache (Block Size: {block_size}) on {device}...")
    
    cache = PagedKVCache(batch_size, num_heads, head_dim, block_size, device)
    profiler = PagedMemoryProfiler()
    
    allocated_memory_history = []
    active_memory_history = []
    
    print("Simulating Autoregressive Generation...")
    
    for step in range(1, max_seq_len + 1):
        k_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
        v_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
        
        cache.update(k_new, v_new)
        
        allocated_memory_history.append(profiler.calculate_allocated_mb(cache))
        active_memory_history.append(profiler.calculate_active_mb(cache))

    # Deliverable: The Paged Memory Graph
    plt.figure(figsize=(10, 6))
    
    plt.plot(allocated_memory_history, label='Paged VRAM (Allocated Blocks)', color='red', linewidth=2.5)
    plt.plot(active_memory_history, label='Actually Used VRAM (Active Tokens)', color='blue', linewidth=2.5, linestyle='--')
    
    # Shade the wasted area (which will now be microscopic)
    plt.fill_between(range(max_seq_len), active_memory_history, allocated_memory_history, color='red', alpha=0.3, label='Wasted VRAM (Minimal)')
    
    plt.title("Memory Fragmentation Eliminated: PagedAttention KV Cache")
    plt.xlabel("Generation Step (Sequence Length)")
    plt.ylabel("KV Cache VRAM (Megabytes)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("stage3_paged_memory.png", dpi=300)
    print("Graph saved as 'stage3_paged_memory.png'")

if __name__ == "__main__":
    main()