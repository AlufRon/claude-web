#!/usr/bin/env python3
"""
Debug why ttt_mlp is never called even though _apply_ttt_processing runs
"""

import torch
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def debug_apply_ttt_processing():
    """Step through _apply_ttt_processing to find where it fails"""
    print("🔍 DEBUGGING _apply_ttt_processing EXECUTION")
    print("=" * 60)
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer, HybridSeqModelingBlock
        from moshi_ttt.config import TTTConfig
        
        # Create test setup
        d_model = 512
        num_heads = 8
        batch_size = 1
        seq_len = 32
        
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model * 4,
            causal=True,
            context=100,
            norm='rms_norm'
        )
        
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        seq_modeling_block = hybrid_layer.seq_modeling_block
        
        print("✅ Created test setup")
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model) * 0.1
        
        # Step 1: Get attention output
        print(f"\n🔄 STEP 1: Attention processing")
        attn_output = seq_modeling_block._attn_forward(x)
        print(f"   Attention output shape: {attn_output.shape}")
        print(f"   Attention output finite: {torch.isfinite(attn_output).all()}")
        
        # Step 2: Start TTT processing 
        print(f"\n🔄 STEP 2: TTT processing setup")
        B, seq_len, d_model = attn_output.shape
        print(f"   Input shape: [{B}, {seq_len}, {d_model}]")
        
        # Convert to TTT format
        from moshi_ttt.format_utils import moshi_to_ttt_format
        x_ttt, conversion_metadata = moshi_to_ttt_format(attn_output, ttt_config)
        B, H, NC, C, HD = x_ttt.shape
        print(f"   TTT format shape: [{B}, {H}, {NC}, {C}, {HD}]")
        print(f"   NC (num chunks): {NC}")
        print(f"   C (chunk size): {C}")
        
        # Step 3: Manual _apply_ttt_processing with detailed logging
        print(f"\n🔄 STEP 3: Detailed _apply_ttt_processing")
        
        try:
            # Get Q, K, V projections
            print("   3.1: Getting QKV projections...")
            XQ = seq_modeling_block.wq(attn_output)
            XK = seq_modeling_block.wk(attn_output) 
            XV = seq_modeling_block.wv(attn_output)
            print(f"       XQ shape: {XQ.shape}")
            print(f"       XK shape: {XK.shape}")
            print(f"       XV shape: {XV.shape}")
            
            # Reshape projections
            print("   3.2: Reshaping projections...")
            XQ = XQ.view(B, seq_len, H, HD)
            XK = XK.view(B, seq_len, H, HD)
            XV = XV.view(B, seq_len, H, HD)
            print(f"       Reshaped to: [{B}, {seq_len}, {H}, {HD}]")
            
            # Apply L2 normalization
            print("   3.3: Applying L2 normalization...")
            XQ = torch.nn.functional.normalize(XQ, p=2, dim=-1)
            XK = torch.nn.functional.normalize(XK, p=2, dim=-1)
            print(f"       XQ after norm finite: {torch.isfinite(XQ).all()}")
            print(f"       XK after norm finite: {torch.isfinite(XK).all()}")
            
            # Apply layer norm reconstruction target
            print("   3.4: Layer norm reconstruction target...")
            XV = seq_modeling_block.ln_reconstruction_target(XV, XK)
            print(f"       XV after ln_target finite: {torch.isfinite(XV).all()}")
            
            # Convert to TTT format
            print("   3.5: Converting to TTT format...")
            XQ_ttt, _ = moshi_to_ttt_format(XQ.view(B, seq_len, H*HD), ttt_config)
            XK_ttt, _ = moshi_to_ttt_format(XK.view(B, seq_len, H*HD), ttt_config)
            XV_ttt, _ = moshi_to_ttt_format(XV.view(B, seq_len, H*HD), ttt_config)
            
            # Reshape to separate heads
            XQ_ttt = XQ_ttt.view(B, H, NC, C, HD)
            XK_ttt = XK_ttt.view(B, H, NC, C, HD) 
            XV_ttt = XV_ttt.view(B, H, NC, C, HD)
            print(f"       XQ_ttt shape: {XQ_ttt.shape}")
            print(f"       All TTT tensors finite: {torch.isfinite(XQ_ttt).all() and torch.isfinite(XK_ttt).all() and torch.isfinite(XV_ttt).all()}")
            
            # Compute learning rate
            print("   3.6: Computing TTT learning rate...")
            padded_len = NC * C
            x_padded = attn_output
            if attn_output.shape[1] < padded_len:
                padding = torch.zeros(B, padded_len - seq_len, ttt_config.model_dim, 
                                    device=attn_output.device, dtype=attn_output.dtype)
                x_padded = torch.cat([attn_output, padding], dim=1)
                print(f"       Added padding: {seq_len} -> {padded_len}")
                
            x_chunked = x_padded.view(B, NC, C, ttt_config.model_dim)
            ttt_lr_eta = seq_modeling_block.get_eta(x_chunked)
            eta = 1.0 / C * ttt_lr_eta
            print(f"       eta shape: {eta.shape}")
            print(f"       eta finite: {torch.isfinite(eta).all()}")
            print(f"       eta range: [{eta.min().item():.6f}, {eta.max().item():.6f}]")
            
            # Prepare parameters
            print("   3.7: Preparing TTT parameters...")
            W1_states = seq_modeling_block.W1.unsqueeze(0).expand(B, -1, -1, -1)
            b1_states = seq_modeling_block.b1.unsqueeze(0).expand(B, -1, -1, -1)
            W2_states = seq_modeling_block.W2.unsqueeze(0).expand(B, -1, -1, -1)
            b2_states = seq_modeling_block.b2.unsqueeze(0).expand(B, -1, -1, -1)
            print(f"       W1_states shape: {W1_states.shape}")
            print(f"       All parameter states finite: {all(torch.isfinite(s).all() for s in [W1_states, b1_states, W2_states, b2_states])}")
            
            # The critical moment: calling ttt_mlp
            print("   3.8: Calling ttt_mlp...")
            checkpoint_group_size = 0
            print(f"       checkpoint_group_size: {checkpoint_group_size}")
            
            from moshi_ttt.models.ssm.ops.ttt_mlp import ttt_mlp
            
            print("       🎯 About to call ttt_mlp...")
            
            XQW_batch = ttt_mlp(
                XK_ttt,     # K
                XQ_ttt,     # Q  
                XV_ttt,     # V
                eta,        # learning rate
                seq_modeling_block.ttt_norm_weight,
                seq_modeling_block.ttt_norm_bias,
                W1_states,
                b1_states,
                W2_states, 
                b2_states,
                checkpoint_group_size
            )
            
            print("       ✅ ttt_mlp completed successfully!")
            print(f"       XQW_batch shape: {XQW_batch.shape}")
            print(f"       XQW_batch finite: {torch.isfinite(XQW_batch).all()}")
            
            return True
            
        except Exception as e:
            print(f"       ❌ _apply_ttt_processing failed at step 3: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 TTT MLP FAILURE DEBUG")
    print("=" * 60)
    
    success = debug_apply_ttt_processing()
    
    if success:
        print(f"\n✅ ttt_mlp CAN be called successfully in isolation!")
        print(f"   → The problem must be in the integration or calling context")
    else:
        print(f"\n❌ ttt_mlp fails even in isolation")
        print(f"   → Found the exact failure point")

if __name__ == "__main__":
    main()