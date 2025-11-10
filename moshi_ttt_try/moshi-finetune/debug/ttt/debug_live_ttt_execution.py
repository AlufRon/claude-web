#!/usr/bin/env python3
"""
Patch the live _apply_ttt_processing method to add detailed logging
and find exactly where ttt_mlp call is missing
"""

import torch
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def patch_apply_ttt_with_logging():
    """Replace _apply_ttt_processing with a logged version"""
    print("🔧 PATCHING _apply_ttt_processing WITH DETAILED LOGGING")
    print("=" * 60)
    
    try:
        from moshi_ttt.hybrid_layer import HybridSeqModelingBlock
        
        # Save original method
        original_apply_ttt = HybridSeqModelingBlock._apply_ttt_processing
        
        def logged_apply_ttt_processing(self, x_ttt, x_original):
            """Logged version of _apply_ttt_processing"""
            print(f"\n   🎯 [LOGGED] _apply_ttt_processing called")
            
            try:
                B, H, NC, C, HD = x_ttt.shape
                print(f"   📏 Input shapes: x_ttt={x_ttt.shape}, x_original={x_original.shape}")
                
                # Get Q, K, V projections from original input
                print(f"   🔧 Getting QKV projections...")
                XQ = self.wq(x_original)  # [B, seq_len, H*HD]
                XK = self.wk(x_original)  # [B, seq_len, H*HD] 
                XV = self.wv(x_original)  # [B, seq_len, H*HD]
                print(f"   ✅ QKV projections: XQ={XQ.shape}, XK={XK.shape}, XV={XV.shape}")
                
                # Reshape projections: [B, seq_len, H*HD] -> [B, seq_len, H, HD]
                seq_len = x_original.shape[1]
                XQ = XQ.view(B, seq_len, H, HD)
                XK = XK.view(B, seq_len, H, HD)
                XV = XV.view(B, seq_len, H, HD)
                print(f"   ✅ Reshaped to: [{B}, {seq_len}, {H}, {HD}]")
                
                # Apply L2 normalization (following Video-DiT)
                print(f"   🔧 Applying L2 normalization...")
                XQ = torch.nn.functional.normalize(XQ, p=2, dim=-1)
                XK = torch.nn.functional.normalize(XK, p=2, dim=-1)
                print(f"   ✅ L2 normalization complete")
                
                # Apply layer norm reconstruction target (Video-DiT pattern)
                print(f"   🔧 Applying layer norm reconstruction target...")
                XV = self.ln_reconstruction_target(XV, XK)
                print(f"   ✅ Layer norm reconstruction target complete")
                
                # Convert to TTT format: [B, seq_len, H, HD] -> [B, H, NC, C, HD]
                print(f"   🔧 Converting to TTT format...")
                from moshi_ttt.format_utils import moshi_to_ttt_format
                XQ_ttt, _ = moshi_to_ttt_format(XQ.view(B, seq_len, H*HD), self.ttt_config)
                XK_ttt, _ = moshi_to_ttt_format(XK.view(B, seq_len, H*HD), self.ttt_config)
                XV_ttt, _ = moshi_to_ttt_format(XV.view(B, seq_len, H*HD), self.ttt_config)
                print(f"   ✅ TTT format conversion complete")
                
                # Reshape to separate heads properly: [B, H, NC, C, HD] -> [B, H, NC, C, HD]
                XQ_ttt = XQ_ttt.view(B, H, NC, C, HD)
                XK_ttt = XK_ttt.view(B, H, NC, C, HD) 
                XV_ttt = XV_ttt.view(B, H, NC, C, HD)
                print(f"   ✅ Head separation complete: {XQ_ttt.shape}")
                
                # Compute TTT learning rate (eta) following Video-DiT pattern
                print(f"   🔧 Computing TTT learning rate...")
                padded_len = NC * C
                x_padded = x_original
                if x_original.shape[1] < padded_len:
                    padding = torch.zeros(B, padded_len - seq_len, self.d_model, device=x_original.device, dtype=x_original.dtype)
                    x_padded = torch.cat([x_original, padding], dim=1)
                    print(f"   📏 Added padding: {seq_len} -> {padded_len}")
                    
                x_chunked = x_padded.view(B, NC, C, self.d_model)
                
                # Compute learning rate using Video-DiT approach
                ttt_lr_eta = self.get_eta(x_chunked)  # [B, H, NC, 1, C] 
                eta = 1.0 / C * ttt_lr_eta  # Video-DiT uses 1/mini_batch_size
                print(f"   ✅ Learning rate computed: eta shape={eta.shape}")
                
                # Prepare TTT-MLP parameters for batch processing
                print(f"   🔧 Preparing TTT parameters...")
                W1_states = self.W1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, HD, 4*HD]
                b1_states = self.b1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 1, 4*HD]
                W2_states = self.W2.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 4*HD, HD]
                b2_states = self.b2.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 1, HD]
                print(f"   ✅ TTT parameters prepared")
                
                # Apply TTT-MLP processing (using our copied ttt_mlp function)
                # CRITICAL FIX: Disable checkpointing to avoid PyTorch device state error
                # Following TTT-Video-DiT's evaluation approach (scan_checkpoint_group_size = 1e6)
                checkpoint_group_size = 0  # 0 = no checkpointing, allows TTT gradients to flow
                print(f"   🎯 About to call ttt_mlp with checkpoint_group_size={checkpoint_group_size}")
                
                from moshi_ttt.models.ssm.ops.ttt_mlp import ttt_mlp
                XQW_batch = ttt_mlp(
                    XK_ttt,     # K
                    XQ_ttt,     # Q  
                    XV_ttt,     # V
                    eta,        # learning rate
                    self.ttt_norm_weight,
                    self.ttt_norm_bias,
                    W1_states,
                    b1_states,
                    W2_states, 
                    b2_states,
                    checkpoint_group_size
                )
                print(f"   🎉 ttt_mlp SUCCESS! Output shape: {XQW_batch.shape}")
                
                # Reshape output back to expected format
                L = NC * C
                XQW_batch = XQW_batch.reshape(B, L, self.d_model)
                
                # Apply post normalization and output projection
                XQW_batch = self.post_norm(XQW_batch)
                XQW_batch = self.wo(XQW_batch)
                
                # Remove padding if we added any
                if seq_len < padded_len:
                    XQW_batch = XQW_batch[:, :seq_len, :]
                
                print(f"   ✅ _apply_ttt_processing completed successfully!")
                return XQW_batch
                
            except Exception as e:
                print(f"   ❌ [LOGGED] _apply_ttt_processing FAILED: {e}")
                import traceback
                traceback.print_exc()
                # Return the original input to continue execution
                return x_original
        
        # Apply the patch
        HybridSeqModelingBlock._apply_ttt_processing = logged_apply_ttt_processing
        print("✅ Patch applied successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to apply patch: {e}")
        return False

def test_with_patched_method():
    """Test with the patched _apply_ttt_processing method"""
    print(f"\n🔍 TESTING WITH PATCHED METHOD")
    print("-" * 50)
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Small test setup
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
        hybrid_layer.train()
        
        print("✅ Created hybrid layer")
        
        # Test forward pass
        x = torch.randn(batch_size, seq_len, d_model) * 0.1
        
        print(f"\n🔄 FORWARD PASS WITH PATCHED LOGGING:")
        output = hybrid_layer(x)
        
        print(f"\n✅ Forward pass completed!")
        print(f"Output shape: {output.shape}")
        print(f"Output finite: {torch.isfinite(output).all()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 LIVE TTT EXECUTION DEBUG WITH PATCHING")
    print("=" * 60)
    
    # Apply patch
    patch_success = patch_apply_ttt_with_logging()
    
    if patch_success:
        # Test with patch
        test_success = test_with_patched_method()
        
        if test_success:
            print(f"\n🎯 DIAGNOSIS COMPLETE")
            print(f"Check the logs above to see exactly where ttt_mlp execution happens or fails")
        else:
            print(f"\n❌ Test failed even with patch")
    else:
        print(f"\n❌ Could not apply patch")

if __name__ == "__main__":
    main()