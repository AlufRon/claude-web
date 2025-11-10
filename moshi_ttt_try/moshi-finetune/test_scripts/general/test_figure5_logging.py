"""
Test Figure 5 logging functionality
"""
import torch
from moshi_ttt.models.ssm.ops.ttt_mlp import (
    fig5_set_logging,
    fig5_clear,
    fig5_get,
    ttt_mlp
)

def test_figure5_logging():
    """Test that Figure 5 logging collects three losses per position."""
    print("=" * 60)
    print("Testing Figure 5 Logging")
    print("=" * 60)
    
    # Enable Figure 5 logging
    fig5_set_logging(True, max_T=100)
    fig5_clear()
    
    # Create dummy data for a small sequence
    B, num_heads, L, head_dim = 1, 4, 10, 32
    mini_batch_size = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize TTT weights
    W1_init = torch.randn(num_heads, head_dim, head_dim, device=device) * 0.01
    b1_init = torch.zeros(num_heads, 1, head_dim, device=device)
    W2_init = torch.randn(num_heads, head_dim, head_dim, device=device) * 0.01
    b2_init = torch.zeros(num_heads, 1, head_dim, device=device)
    
    # Layer norm params
    ttt_norm_weight = torch.ones(num_heads, head_dim, device=device)
    ttt_norm_bias = torch.zeros(num_heads, head_dim, device=device)
    
    # Input tensors
    XK = torch.randn(B, num_heads, L, mini_batch_size, head_dim, device=device)
    XQ = torch.randn(B, num_heads, L, mini_batch_size, head_dim, device=device)
    XV = torch.randn(B, num_heads, L, mini_batch_size, head_dim, device=device)
    eta = torch.ones(B, num_heads, L, mini_batch_size, 1, device=device) * 0.01
    
    # Run TTT-MLP with Figure 5 logging
    print(f"\nProcessing sequence of length {L} with layer_id=0")
    output = ttt_mlp(
        XK, XQ, XV, eta,
        ttt_norm_weight, ttt_norm_bias,
        W1_init, b1_init, W2_init, b2_init,
        checkpoint_group_size=1,
        log_losses=False,
        layer_id=0,
        stream_pos_base=0
    )
    
    # Check results
    fig5_data = fig5_get()
    
    print(f"\n✅ Figure 5 data collected for layers: {list(fig5_data.keys())}")
    
    if 0 in fig5_data:
        layer_data = fig5_data[0]
        counts = layer_data['cnt']
        non_zero_positions = sum(1 for c in counts if c > 0)
        
        print(f"✅ Layer 0: Logged {non_zero_positions} positions")
        
        # Show first few positions
        for pos in range(min(5, non_zero_positions)):
            if counts[pos] > 0:
                l0 = layer_data['l0'][pos] / counts[pos]
                lprev = layer_data['lprev'][pos] / counts[pos]
                lafter = layer_data['lafter'][pos] / counts[pos]
                print(f"   Position {pos}: l0={l0:.4f}, lprev={lprev:.4f}, lafter={lafter:.4f}")
        
        # Verify the ordering (should typically have l0 > lprev > lafter)
        valid_orderings = 0
        for pos in range(non_zero_positions):
            if counts[pos] > 0:
                l0 = layer_data['l0'][pos] / counts[pos]
                lprev = layer_data['lprev'][pos] / counts[pos]
                lafter = layer_data['lafter'][pos] / counts[pos]
                if l0 >= lprev >= lafter:
                    valid_orderings += 1
        
        print(f"\n📊 Validation: {valid_orderings}/{non_zero_positions} positions follow l0 ≥ lprev ≥ lafter")
    else:
        print("❌ No data collected for layer 0")
    
    # Disable logging
    fig5_set_logging(False)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_figure5_logging()
