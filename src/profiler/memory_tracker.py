class KVCacheMemoryProfiler:
    """
    Calculates the actual memory footprint of the KV Cache in Megabytes (MB).
    """
    def __init__(self, dtype_bytes=4): # 4 bytes for float32
        self.dtype_bytes = dtype_bytes

    def calculate_allocated_mb(self, cache) -> float:
        """Calculates the total VRAM physically allocated to the cache."""
        # Total elements = batch * num_heads * max_seq_len * head_dim
        k_elements = cache.k_cache.numel()
        v_elements = cache.v_cache.numel()
        total_bytes = (k_elements + v_elements) * self.dtype_bytes
        return total_bytes / (1024 ** 2)

    def calculate_active_mb(self, cache) -> float:
        """Calculates the VRAM actually holding meaningful token data."""
        # Active elements = batch * num_heads * current_seq_len * head_dim
        active_k_elements = cache.k_cache[:, :, :cache.current_seq_len, :].numel()
        active_v_elements = cache.v_cache[:, :, :cache.current_seq_len, :].numel()
        total_active_bytes = (active_k_elements + active_v_elements) * self.dtype_bytes
        return total_active_bytes / (1024 ** 2)
        
    def get_wasted_ratio(self, cache) -> float:
        allocated = self.calculate_allocated_mb(cache)
        active = self.calculate_active_mb(cache)
        if allocated == 0: return 0.0
        return ((allocated - active) / allocated) * 100