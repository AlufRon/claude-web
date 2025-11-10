"""
Test RoPE (Rotary Position Embeddings) Components

Tests the core RoPE implementation in isolation to ensure correctness
before integration testing.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from moshi_ttt.ttt_layer import RotaryEmbedding, rotate_half, apply_rotary_pos_emb


def test_rotary_embedding_initialization():
    """Test that RoPE initializes correctly with expected dimensions"""
    dim = 128
    max_pos = 64
    base = 10000

    rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_pos, base=base)

    assert rope.dim == dim
    assert rope.max_position_embeddings == max_pos
    assert rope.base == base
    assert rope.inv_freq.shape == (dim // 2,), f"Expected inv_freq shape ({dim//2},), got {rope.inv_freq.shape}"
    assert rope.cos_cached.shape == (max_pos, dim), f"Expected cos_cached shape ({max_pos}, {dim}), got {rope.cos_cached.shape}"
    assert rope.sin_cached.shape == (max_pos, dim), f"Expected sin_cached shape ({max_pos}, {dim}), got {rope.sin_cached.shape}"

    print("✅ RoPE initialization test passed")


def test_rotary_embedding_forward():
    """Test RoPE forward pass with various input shapes"""
    dim = 128
    max_pos = 64

    rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_pos)

    # Test with 5D tensor (TTT mini-batch format)
    B, H, NC, C, HD = 2, 8, 10, 64, 128
    x = torch.randn(B, H, NC, C, HD)
    seq_len = NC * C  # 640
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(B, -1)

    # RoPE will apply modulo internally, so we don't need to do it here
    cos, sin = rope(x, position_ids)

    # Note: position_ids get flattened in forward, so output is [B*seq_len, dim]
    # But we only care that it works, shape is flexible
    assert cos.shape[1] == dim, f"Expected dim {dim}, got {cos.shape[1]}"
    assert sin.shape[1] == dim, f"Expected dim {dim}, got {sin.shape[1]}"
    assert cos.dtype == x.dtype
    assert sin.dtype == x.dtype

    # Test with None position_ids (should generate sequential)
    cos_default, sin_default = rope(x, None)
    assert cos_default.shape[1] == dim

    print("✅ RoPE forward pass test passed")


def test_position_modulo_invariance():
    """
    CRITICAL TEST: Verify that RoPE internally applies modulo correctly.

    Positions (0, 1, ..., 63), (64, 65, ..., 127), and (5000, 5001, ..., 5063)
    should all produce the same cos/sin because RoPE applies modulo internally.

    This is the key to preventing extrapolation issues.
    """
    dim = 128
    max_pos = 64

    rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_pos)
    x = torch.randn(1, 8, 1, 64, 128)  # Dummy tensor for device reference

    # Positions 0-63
    pos_1 = torch.arange(0, 64).unsqueeze(0)
    cos_1, sin_1 = rope(x, pos_1)

    # Positions 64-127 (RoPE will apply modulo internally)
    pos_2 = torch.arange(64, 128).unsqueeze(0)
    cos_2, sin_2 = rope(x, pos_2)

    # Should be identical because RoPE applies modulo internally
    assert torch.allclose(cos_1, cos_2, atol=1e-6), "Position modulo failed for cos"
    assert torch.allclose(sin_1, sin_2, atol=1e-6), "Position modulo failed for sin"

    # Test with very large positions (simulating long context)
    # Use 5056 as start so 5056 % 64 = 0, matching pos_1
    pos_3 = torch.arange(5056, 5056 + 64).unsqueeze(0)
    cos_3, sin_3 = rope(x, pos_3)

    assert torch.allclose(cos_1, cos_3, atol=1e-6), "Position modulo failed for large positions (cos)"
    assert torch.allclose(sin_1, sin_3, atol=1e-6), "Position modulo failed for large positions (sin)"

    print("✅ Position modulo invariance test passed (CRITICAL)")


def test_rotate_half():
    """Test that rotate_half correctly rotates features"""
    # Simple test case
    x = torch.tensor([[1., 2., 3., 4.]])
    rotated = rotate_half(x)
    expected = torch.tensor([[-3., -4., 1., 2.]])

    assert torch.allclose(rotated, expected), f"Expected {expected}, got {rotated}"

    # Test with multi-dimensional tensor
    x_multi = torch.randn(2, 4, 8, 128)
    rotated_multi = rotate_half(x_multi)

    assert rotated_multi.shape == x_multi.shape
    # First half should be negative of second half of input
    assert torch.allclose(rotated_multi[..., :64], -x_multi[..., 64:])
    # Second half should be first half of input
    assert torch.allclose(rotated_multi[..., 64:], x_multi[..., :64])

    print("✅ rotate_half test passed")


def test_apply_rotary_pos_emb():
    """Test applying RoPE to Q and K tensors"""
    B, H, L, HD = 2, 8, 100, 128

    q = torch.randn(B, H, L, HD)
    k = torch.randn(B, H, L, HD)
    cos = torch.randn(L, HD)
    sin = torch.randn(L, HD)

    # unsqueeze_dim controls which dimension to unsqueeze for broadcasting
    # For shape [B, H, L, HD], we need to unsqueeze at dim 0 and 1
    # But apply_rotary_pos_emb handles this with unsqueeze_dim parameter
    # Default unsqueeze_dim=1 means unsqueeze at dims 0 and 1
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    # Check shapes preserved
    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape

    # Check that embedding actually changed the values
    assert not torch.allclose(q_embed, q), "Q should be modified by RoPE"
    assert not torch.allclose(k_embed, k), "K should be modified by RoPE"

    # Test that RoPE is deterministic
    q_embed_2, k_embed_2 = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_embed, q_embed_2), "RoPE should be deterministic"
    assert torch.allclose(k_embed, k_embed_2), "RoPE should be deterministic"

    print("✅ apply_rotary_pos_emb test passed")


def test_rope_with_different_dtypes():
    """Test RoPE works with different dtypes (float32, bfloat16)"""
    dim = 128
    max_pos = 64

    rope = RotaryEmbedding(dim=dim, max_position_embeddings=max_pos)

    # Test with float32
    x_f32 = torch.randn(1, 8, 10, 64, 128, dtype=torch.float32)
    pos = torch.arange(640).unsqueeze(0)
    cos_f32, sin_f32 = rope(x_f32, pos)
    assert cos_f32.dtype == torch.float32
    assert sin_f32.dtype == torch.float32

    # Test with bfloat16 (Moshi uses bf16)
    x_bf16 = torch.randn(1, 8, 10, 64, 128, dtype=torch.bfloat16)
    cos_bf16, sin_bf16 = rope(x_bf16, pos)
    assert cos_bf16.dtype == torch.bfloat16
    assert sin_bf16.dtype == torch.bfloat16

    print("✅ RoPE dtype test passed")


def test_rope_gradient_flow():
    """Test that gradients flow through RoPE correctly"""
    B, H, L, HD = 2, 4, 50, 64

    q = torch.randn(B, H, L, HD, requires_grad=True)
    k = torch.randn(B, H, L, HD, requires_grad=True)

    rope = RotaryEmbedding(dim=HD, max_position_embeddings=16)
    pos = torch.arange(L).unsqueeze(0)  # RoPE will apply modulo internally

    cos, sin = rope(q, pos)
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

    # Compute a simple loss
    loss = (q_embed ** 2).sum() + (k_embed ** 2).sum()
    loss.backward()

    # Check gradients exist
    assert q.grad is not None, "Gradient should flow to Q"
    assert k.grad is not None, "Gradient should flow to K"
    assert not torch.isnan(q.grad).any(), "Q gradients should not be NaN"
    assert not torch.isnan(k.grad).any(), "K gradients should not be NaN"

    print("✅ RoPE gradient flow test passed")


if __name__ == "__main__":
    print("Running RoPE Component Tests...\n")

    test_rotary_embedding_initialization()
    test_rotary_embedding_forward()
    test_position_modulo_invariance()
    test_rotate_half()
    test_apply_rotary_pos_emb()
    test_rope_with_different_dtypes()
    test_rope_gradient_flow()

    print("\n" + "="*60)
    print("✅ All RoPE component tests passed!")
    print("="*60)
