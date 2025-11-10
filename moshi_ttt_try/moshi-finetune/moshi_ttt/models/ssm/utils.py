from typing import Tuple

import torch
from einops import rearrange


def precompute_audio_rope_1d(
    head_dim: int, 
    seq_len: int, 
    base_freq: float = 10000.0,
    audio_scaling: bool = True,
) -> torch.Tensor:
    """
    Precompute 1D Audio RoPE values for temporal sequences.
    
    Replaces CogVideo's 3D spatial-temporal RoPE with audio-appropriate 1D temporal encoding.
    
    Args:
        head_dim: Attention head dimension
        seq_len: Sequence length
        base_freq: Base frequency for RoPE
        audio_scaling: Apply audio-specific frequency scaling
        
    Returns:
        torch.Tensor: Complex exponentials [seq_len, head_dim//2]
    """
    # Compute inverse frequencies
    freqs = 1.0 / (base_freq ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    if audio_scaling:
        # Audio-specific frequency scaling for better temporal modeling
        audio_scale = torch.exp(-torch.arange(0, head_dim, 2).float() / (head_dim * 2))
        freqs = freqs * (0.5 + 1.5 * audio_scale)
    
    # Compute position encodings
    positions = torch.arange(seq_len, dtype=torch.float)
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # [seq_len, head_dim//2]
    
    # Return as complex exponentials (compatible with apply_rotary_emb)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    
    return freqs_cis


def apply_audio_rotary_emb(
    Q: torch.Tensor,
    K: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply audio rotary embeddings to Q and K tensors.
    
    Drop-in replacement for apply_rotary_emb with proper 1D temporal encoding.
    
    Args:
        Q: Query tensor [batch, seq_len, num_heads, head_dim]
        K: Key tensor [batch, seq_len, num_heads, head_dim]
        freqs_cis: Complex exponentials [seq_len, head_dim//2]
        
    Returns:
        tuple: (Q_rope, K_rope) with audio RoPE applied
    """
    # Convert to complex representation
    def to_complex(x):
        """Convert real tensor to complex by grouping adjacent dimensions."""
        # torch.complex doesn't support bfloat16, convert to float32 first
        original_dtype = x.dtype
        if original_dtype == torch.bfloat16:
            x = x.float()
        x_reshaped = x.view(*x.shape[:-1], -1, 2)  # [..., head_dim//2, 2]
        result = torch.complex(x_reshaped[..., 0], x_reshaped[..., 1])
        return result
    
    def from_complex(x_complex, target_dtype=None):
        """Convert complex tensor back to real."""
        real_part = torch.real(x_complex)
        imag_part = torch.imag(x_complex)
        result = torch.stack([real_part, imag_part], dim=-1).flatten(-2)
        # Convert back to original dtype if it was bfloat16
        if target_dtype == torch.bfloat16:
            result = result.bfloat16()
        return result
    
    # Store original dtype for restoration
    original_dtype = Q.dtype
    
    # Convert Q, K to complex
    Q_complex = to_complex(Q)  # [batch, seq_len, num_heads, head_dim//2]
    K_complex = to_complex(K)  # [batch, seq_len, num_heads, head_dim//2]
    
    # Apply rotation
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    
    Q_rope_complex = Q_complex * freqs_cis
    K_rope_complex = K_complex * freqs_cis
    
    # Convert back to real, restoring original dtype
    Q_rope = from_complex(Q_rope_complex, target_dtype=original_dtype)
    K_rope = from_complex(K_rope_complex, target_dtype=original_dtype)
    
    return Q_rope, K_rope


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def precompute_freqs_cis_3d(
    dim: int, height: int, width: int, compressed_num_frames: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.
    Args:
        dim (int): Dimension of the frequency tensor.
        height (int): Height of the latents feature map.
        width (int): Width of the latents feature map.
        compressed_number_frames (int): Number of frames of the latents feature map.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    dim_t = dim // 4
    dim_h = dim // 8 * 3
    dim_w = dim // 8 * 3
    freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: dim_t // 2].float() / dim_t))
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: dim_h // 2].float() / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: dim_w // 2].float() / dim_w))

    grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
    grid_h = torch.arange(height, dtype=torch.float32)
    grid_w = torch.arange(width, dtype=torch.float32)

    freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
    freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
    freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

    freqs = broadcat(
        (
            freqs_t[:, None, None, :],
            freqs_h[None, :, None, :],
            freqs_w[None, None, :, :],
        ),
        dim=-1,
    )

    freqs = rearrange(freqs, "t h w d -> (t h w) d")
    freqs = freqs.contiguous()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1]), f"{freqs_cis.shape} != {seqlen, x.shape[-1]}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def scan(f, init, xs, checkpoint_group=0):
    """Mimic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        sub_out_list = []
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            sub_out_list.append(y)
        sub_out = torch.stack(sub_out_list)
        return carry, sub_out

    if checkpoint_group > 0:
        out_list = []
        for k in range(0, num_items, checkpoint_group):
            carry, sub_out = torch.utils.checkpoint.checkpoint(
                scan_fn,
                carry,
                k,
                min(k + checkpoint_group, num_items),
                use_reentrant=False,
            )
            out_list.append(sub_out)
        out = torch.concatenate(out_list, dim=0)
    else:
        carry, out = scan_fn(carry, 0, num_items)

    return carry, out
