import torch
import matplotlib.pyplot as plt
from src.cache.preallocated_kv import PreAllocatedKVCache
from src.profiler.memory_tracker import KVCacheMemoryProfiler

def main():
    # Set maximum constraints for the experiment (Task 2.1)
    batch_size = 1
    max_seq_len = 2048
    num_heads = 32
    head_dim = 128
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Initializing Pre-Allocated Cache on {device}...")
    
    # Initialize the cache and profiler
    cache = PreAllocatedKVCache(batch_size, max_seq_len, num_heads, head_dim, device)
    profiler = KVCacheMemoryProfiler(dtype_bytes=4) # Using float32
    
    allocated_memory_history = []
    active_memory_history = []
    wasted_ratio_history = []
    
    print("Simulating Autoregressive Generation...")
    
    # Simulate generating tokens from 1 up to 2048 (Task 2.2)
    for step in range(1, max_seq_len + 1):
        # Create a dummy token projection (1 token at a time)
        k_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
        v_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
        
        # Update cache
        cache.update(k_new, v_new)
        
        # Capture memory states (Task 2.3)
        allocated_mb = profiler.calculate_allocated_mb(cache)
        active_mb = profiler.calculate_active_mb(cache)
        wasted_pct = profiler.get_wasted_ratio(cache)
        
        allocated_memory_history.append(allocated_mb)
        active_memory_history.append(active_mb)
        wasted_ratio_history.append(wasted_pct)

    print(f"Final Step - Allocated: {allocated_memory_history[-1]:.2f} MB")
    print(f"Final Step - Active: {active_memory_history[-1]:.2f} MB")
    print(f"Average VRAM Wasted Across Generation: {sum(wasted_ratio_history)/len(wasted_ratio_history):.2f}%")

    # Deliverable: The Fragmentation Report Graph
    plt.figure(figsize=(10, 6))
    
    plt.plot(allocated_memory_history, label='Pre-Allocated VRAM (The Ceiling)', color='red', linewidth=2.5)
    plt.plot(active_memory_history, label='Actually Used VRAM (Active Tokens)', color='blue', linewidth=2.5)
    
    # Shade the wasted area
    plt.fill_between(range(max_seq_len), active_memory_history, allocated_memory_history, color='red', alpha=0.15, label='Wasted VRAM (Fragmentation)')
    
    plt.title("Memory Fragmentation in Pre-Allocated KV Caches")
    plt.xlabel("Generation Step (Sequence Length)")
    plt.ylabel("KV Cache VRAM (Megabytes)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("stage2_fragmentation_report.png", dpi=300)
    print("Graph saved as 'stage2_fragmentation_report.png'")

if __name__ == "__main__":
    main()