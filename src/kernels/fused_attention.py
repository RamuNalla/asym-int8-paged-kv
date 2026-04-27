import torch
import triton
import triton.language as tl

@triton.jit
def int8_decode_attention_kernel(
    q_ptr, k_ptr, v_ptr, k_scale_ptr, v_scale_ptr, out_ptr,
    seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for autoregressive decoding (1 Query token vs N Cached tokens).
    Dequantizes INT8 Keys and Values on the fly using asymmetric scales.
    """
    # Identify the specific head this program block is processing
    pid = tl.program_id(axis=0)
    
    # Offsets for the Q vector (1 token, full head_dim)
    offs_d = tl.arange(0, BLOCK_SIZE)
    q_ptrs = q_ptr + pid * head_dim + offs_d
    q = tl.load(q_ptrs, mask=offs_d < head_dim, other=0.0)
    
    # Initialize accumulator for the attention scores
    scores = tl.zeros([seq_len], dtype=tl.float32)
    
    # Loop over the sequence to compute Q * K^T
    for i in range(seq_len):
        # Load INT8 Keys and the Per-Channel scales
        k_ptrs = k_ptr + pid * (seq_len * head_dim) + i * head_dim + offs_d
        k_int8 = tl.load(k_ptrs, mask=offs_d < head_dim, other=0)
        k_scale = tl.load(k_scale_ptr + pid * head_dim + offs_d, mask=offs_d < head_dim, other=1.0)
        
        # Dequantize Key on the fly: FP32_K = INT8_K * Scale
        k_fp32 = k_int8 * k_scale
        
        # Dot product
        score = tl.sum(q * k_fp32)
        scores[i] = score

    # Scale by 1/sqrt(d)
    scores = scores / 11.313 # assuming head_dim = 128
    
    # Softmax
    max_score = tl.max(scores)
    exp_scores = tl.exp(scores - max_score)
    sum_exp = tl.sum(exp_scores)
    attn_weights = exp_scores / sum_exp
    
    # Compute Weighted Sum of Values
    out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(seq_len):
        weight = attn_weights[i]
        
        # Load INT8 Values and the Per-Token scale
        v_ptrs = v_ptr + pid * (seq_len * head_dim) + i * head_dim + offs_d
        v_int8 = tl.load(v_ptrs, mask=offs_d < head_dim, other=0)
        v_scale = tl.load(v_scale_ptr + pid * seq_len + i) # 1 scale per token
        
        # Dequantize Value on the fly
        v_fp32 = v_int8 * v_scale
        
        out += weight * v_fp32
        
    # Store output
    out_ptrs = out_ptr + pid * head_dim + offs_d
    tl.store(out_ptrs, out, mask=offs_d < head_dim)