import torch

class AsymmetricKVQuantizer:
    """
    Handles Per-Channel quantization for Keys and Per-Token quantization for Values.
    Compresses FP32/FP16 tensors into INT8 (1 byte) representations.
    """
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.q_max = (1 << (num_bits - 1)) - 1  # 127 for INT8
        self.q_min = -(1 << (num_bits - 1))     # -128 for INT8

    def quantize_keys(self, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Per-Channel Quantization for Keys.
        Calculates a scale for each hidden dimension channel across the sequence.
        Input shape: (batch, num_heads, seq_len, head_dim)
        """
        # Find absolute max along the sequence length dimension (dim=2)
        # Resulting scale shape: (batch, num_heads, 1, head_dim)
        k_abs_max = torch.amax(torch.abs(k), dim=2, keepdim=True)
        k_abs_max = torch.clamp(k_abs_max, min=1e-5)
        
        k_scales = k_abs_max / self.q_max
        
        # Quantize and cast to INT8
        k_quantized = torch.round(k / k_scales).clamp(self.q_min, self.q_max).to(torch.int8)
        return k_quantized, k_scales

    def quantize_values(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Per-Token Quantization for Values.
        Calculates a scale for each individual token across its hidden dimensions.
        Input shape: (batch, num_heads, seq_len, head_dim)
        """
        # Find absolute max along the head dimension (dim=3)
        # Resulting scale shape: (batch, num_heads, seq_len, 1)
        v_abs_max = torch.amax(torch.abs(v), dim=3, keepdim=True)
        v_abs_max = torch.clamp(v_abs_max, min=1e-5)
        
        v_scales = v_abs_max / self.q_max
        
        # Quantize and cast to INT8
        v_quantized = torch.round(v / v_scales).clamp(self.q_min, self.q_max).to(torch.int8)
        return v_quantized, v_scales