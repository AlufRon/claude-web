#!/usr/bin/env python3
"""
Add detailed step-by-step logging to _apply_ttt_processing to find exact failure point
"""

import torch
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def patch_apply_ttt_with_detailed_logging():
    """Replace _apply_ttt_processing with detailed step logging"""
    print("🔧 PATCHING _apply_ttt_processing WITH STEP-BY-STEP LOGGING")
    print("=" * 60)
    
    try:
        from moshi_ttt.hybrid_layer import HybridSeqModelingBlock
        
        # Save original method
        original_apply_ttt = HybridSeqModelingBlock._apply_ttt_processing
        
        def logged_apply_ttt_processing(self, x_ttt, x_original):
            """Step-by-step logged version of _apply_ttt_processing"""
            print(f"\n   🎯 [DETAILED] _apply_ttt_processing starting")
            
            try:
                B, H, NC, C, HD = x_ttt.shape
                print(f"   STEP 1: Input shapes - x_ttt={x_ttt.shape}, x_original={x_original.shape}")
                
                # Get Q, K, V projections from original input
                print(f"   STEP 2: Getting QKV projections...")
                XQ = self.wq(x_original)  # [B, seq_len, H*HD]
                XK = self.wk(x_original)  # [B, seq_len, H*HD] 
                XV = self.wv(x_original)  # [B, seq_len, H*HD]
                print(f"   STEP 2: ✅ QKV shapes: {XQ.shape}, {XK.shape}, {XV.shape}")
                
                # Reshape projections: [B, seq_len, H*HD] -> [B, seq_len, H, HD]
                print(f"   STEP 3: Reshaping projections...")
                seq_len = x_original.shape[1]
                XQ = XQ.view(B, seq_len, H, HD)
                XK = XK.view(B, seq_len, H, HD)
                XV = XV.view(B, seq_len, H, HD)
                print(f"   STEP 3: ✅ Reshaped to [{B}, {seq_len}, {H}, {HD}]")
                
                # Apply L2 normalization (following Video-DiT)
                print(f"   STEP 4: L2 normalization...")
                XQ = torch.nn.functional.normalize(XQ, p=2, dim=-1)
                XK = torch.nn.functional.normalize(XK, p=2, dim=-1)
                print(f"   STEP 4: ✅ L2 normalization complete")
                
                # Apply layer norm reconstruction target (Video-DiT pattern)
                print(f"   STEP 5: Layer norm reconstruction target...")
                XV = self.ln_reconstruction_target(XV, XK)
                print(f"   STEP 5: ✅ Layer norm reconstruction complete")
                
                # Convert to TTT format: [B, seq_len, H, HD] -> [B, H, NC, C, HD]
                print(f"   STEP 6: Converting to TTT format...")
                from moshi_ttt.format_utils import moshi_to_ttt_format
                XQ_ttt, _ = moshi_to_ttt_format(XQ.view(B, seq_len, H*HD), self.ttt_config)
                XK_ttt, _ = moshi_to_ttt_format(XK.view(B, seq_len, H*HD), self.ttt_config)
                XV_ttt, _ = moshi_to_ttt_format(XV.view(B, seq_len, H*HD), self.ttt_config)
                print(f"   STEP 6: ✅ TTT format conversion complete")
                
                # Reshape to separate heads properly: [B, H, NC, C, HD] -> [B, H, NC, C, HD]
                print(f"   STEP 7: Separating heads...")
                XQ_ttt = XQ_ttt.view(B, H, NC, C, HD)
                XK_ttt = XK_ttt.view(B, H, NC, C, HD) 
                XV_ttt = XV_ttt.view(B, H, NC, C, HD)
                print(f"   STEP 7: ✅ Head separation complete: {XQ_ttt.shape}")
                
                # Compute TTT learning rate (eta) following Video-DiT pattern
                print(f"   STEP 8: Computing learning rate...")
                padded_len = NC * C
                x_padded = x_original
                if x_original.shape[1] < padded_len:
                    padding = torch.zeros(B, padded_len - seq_len, self.d_model, device=x_original.device, dtype=x_original.dtype)
                    x_padded = torch.cat([x_original, padding], dim=1)
                    print(f"   STEP 8: Added padding: {seq_len} -> {padded_len}")
                    
                x_chunked = x_padded.view(B, NC, C, self.d_model)
                
                # Compute learning rate using Video-DiT approach
                ttt_lr_eta = self.get_eta(x_chunked)  # [B, H, NC, 1, C] 
                eta = 1.0 / C * ttt_lr_eta  # Video-DiT uses 1/mini_batch_size
                print(f"   STEP 8: ✅ Learning rate computed: eta shape={eta.shape}")
                
                # Prepare TTT-MLP parameters for batch processing
                print(f"   STEP 9: Preparing TTT parameters...")
                W1_states = self.W1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, HD, 4*HD]
                b1_states = self.b1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 1, 4*HD]
                W2_states = self.W2.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 4*HD, HD]
                b2_states = self.b2.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 1, HD]
                print(f"   STEP 9: ✅ TTT parameters prepared")
                
                # Apply TTT-MLP processing (using our copied ttt_mlp function)
                # CRITICAL FIX: Disable checkpointing to avoid PyTorch device state error
                # Following TTT-Video-DiT's evaluation approach (scan_checkpoint_group_size = 1e6)
                checkpoint_group_size = 0  # 0 = no checkpointing, allows TTT gradients to flow
                print(f"   STEP 10: About to call ttt_mlp with checkpoint_group_size={checkpoint_group_size}")
                
                # Import and call ttt_mlp
                from moshi_ttt.models.ssm.ops.ttt_mlp import ttt_mlp
                print(f"   STEP 10: ttt_mlp imported successfully")
                
                print(f"   STEP 10: Calling ttt_mlp with args:")
                print(f"     XK_ttt: {XK_ttt.shape}")
                print(f"     XQ_ttt: {XQ_ttt.shape}")
                print(f"     XV_ttt: {XV_ttt.shape}")
                print(f"     eta: {eta.shape}")
                print(f"     checkpoint_group_size: {checkpoint_group_size}")
                
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
                print(f"   STEP 10: 🎉 ttt_mlp SUCCESS! Output shape: {XQW_batch.shape}")
                
                # Reshape output back to expected format
                print(f"   STEP 11: Reshaping output...")
                L = NC * C
                XQW_batch = XQW_batch.reshape(B, L, self.d_model)
                print(f"   STEP 11: ✅ Reshaped to: {XQW_batch.shape}")
                
                # Apply post normalization and output projection
                print(f"   STEP 12: Post processing...")
                XQW_batch = self.post_norm(XQW_batch)
                XQW_batch = self.wo(XQW_batch)
                print(f"   STEP 12: ✅ Post processing complete")
                
                # Remove padding if we added any
                if seq_len < padded_len:
                    XQW_batch = XQW_batch[:, :seq_len, :]
                    print(f"   STEP 13: ✅ Padding removed")
                
                print(f"   🎉 _apply_ttt_processing COMPLETED SUCCESSFULLY!")
                return XQW_batch
                
            except Exception as e:
                print(f"   ❌ [DETAILED] _apply_ttt_processing FAILED at some step: {e}")
                import traceback
                traceback.print_exc()
                # Return original input to continue execution
                print(f"   🔄 Returning original input as fallback")
                return x_original
        
        # Apply the patch
        HybridSeqModelingBlock._apply_ttt_processing = logged_apply_ttt_processing
        print("✅ Detailed logging patch applied")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to apply patch: {e}")
        return False

def test_with_detailed_logging():
    """Test with detailed step-by-step logging"""
    print(f"\n🔍 TESTING WITH DETAILED STEP-BY-STEP LOGGING")
    print("-" * 60)
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Create setup
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
        
        print("✅ Test setup created")
        
        # Test forward pass
        x = torch.randn(batch_size, seq_len, d_model) * 0.1
        
        print(f"\n🔄 FORWARD PASS WITH DETAILED LOGGING:")
        output = hybrid_layer(x)
        
        print(f"\n✅ Forward pass completed! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 DETAILED TTT EXECUTION DEBUG")
    print("=" * 60)
    
    # Apply detailed logging patch
    patch_success = patch_apply_ttt_with_detailed_logging()
    
    if patch_success:
        # Test with detailed logging
        test_success = test_with_detailed_logging()
        
        if test_success:
            print(f"\n🎯 DETAILED LOGGING COMPLETE")
            print(f"Check the step-by-step output above to find where ttt_mlp execution happens or fails")
        else:
            print(f"\n❌ Test failed even with detailed logging")
    else:
        print(f"\n❌ Could not apply detailed logging patch")

if __name__ == "__main__":
    main()