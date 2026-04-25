import sys
from pathlib import Path

# Repo root on sys.path so `python scripts/01_run_baseline.py` finds `src`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import time
import matplotlib.pyplot as plt
from src.models.baseline_llm import MinimalAttention
from src.cache.standard_kv import StandardKVCache

def simulate_inference(attention_layer, prefix_len, gen_steps, use_cache=False, device='cpu', embed_dim=256):
    latencies = []
    
    # Context Phase (Prefill)
    prefix_tokens = torch.randn(1, prefix_len, embed_dim, device=device)
    cache = StandardKVCache() if use_cache else None
    
    start_time = time.perf_counter()
    _ = attention_layer(prefix_tokens, kv_cache=cache)
    ttft = time.perf_counter() - start_time
    
    # Generation Phase (Autoregressive loop)
    current_seq_len = prefix_len
    
    for _ in range(gen_steps):
        step_start = time.perf_counter()
        
        if use_cache:
            # Only pass the "newly generated" token (simulated here as a random vector)
            next_token = torch.randn(1, 1, embed_dim, device=device)
            _ = attention_layer(next_token, kv_cache=cache)
        else:
            # Must re-pass the entire sequence history + the new token
            current_seq_len += 1
            full_sequence = torch.randn(1, current_seq_len, embed_dim, device=device)
            _ = attention_layer(full_sequence, kv_cache=None)
            
        step_time = time.perf_counter() - step_start
        latencies.append(step_time * 1000) # Convert to ms
        
    return ttft * 1000, latencies

def main():
    # Detect Apple Silicon or fallback to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    embed_dim = 2048
    attention = MinimalAttention(embed_dim=embed_dim).to(device)
    attention.eval()
    
    prefix_len = 1024
    gen_steps = 1000
    
    print("Running No Cache baseline...")
    ttft_no, latencies_no = simulate_inference(
        attention, prefix_len, gen_steps, use_cache=False, device=device, embed_dim=embed_dim
    )
    
    print("Running Standard Cache baseline...")
    ttft_cache, latencies_cache = simulate_inference(
        attention, prefix_len, gen_steps, use_cache=True, device=device, embed_dim=embed_dim
    )
    
    print(f"No Cache   -> TTFT: {ttft_no:.2f}ms | Avg TPOT: {sum(latencies_no)/len(latencies_no):.2f}ms")
    print(f"With Cache -> TTFT: {ttft_cache:.2f}ms | Avg TPOT: {sum(latencies_cache)/len(latencies_cache):.2f}ms")
    
    # Generate the Deliverable Graph
    plt.figure(figsize=(10, 6))
    plt.plot(latencies_no, label='No Cache (Recompute)', color='red', linewidth=2)
    plt.plot(latencies_cache, label='Standard KV-Cache', color='blue', linewidth=2)
    
    plt.title("Autoregressive Generation Latency: Recompute vs Standard KV-Cache")
    plt.xlabel("Generation Step")
    plt.ylabel("Time Per Output Token (TPOT) - milliseconds")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("stage1_baseline_latency.png", dpi=300)
    print("Graph saved as 'stage1_baseline_latency.png'")

if __name__ == "__main__":
    main()