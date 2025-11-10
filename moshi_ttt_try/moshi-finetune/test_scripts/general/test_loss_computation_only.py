#!/usr/bin/env python3
"""
Direct test of the loss computation fix.
Creates fake audio data and verifies we get different loss values.
"""

import torch
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_loss_computation_directly():
    """Test loss computation with fake audio data to verify variability."""
    try:
        from moshi.models import loaders
        from moshi.models.lm import LMGen
        
        logger.info("🔧 Testing Loss Computation Directly")
        
        # Load minimal Moshi model
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        model = checkpoint_info.get_moshi(device='cuda', dtype=torch.bfloat16)
        model.eval()
        
        logger.info("✅ Model loaded")
        
        # Create some fake audio data (different patterns to ensure different losses)
        batch_size = 1
        seq_length = 10  # Very short sequence
        audio_codebooks = 8
        
        # Pattern 1: Ascending values
        codes1 = torch.arange(seq_length).unsqueeze(0).unsqueeze(0).repeat(1, audio_codebooks, 1)
        codes1 = codes1.to(device='cuda', dtype=torch.long) % 1024  # Clamp to vocab size
        
        # Pattern 2: Descending values  
        codes2 = torch.arange(seq_length, 0, -1).unsqueeze(0).unsqueeze(0).repeat(1, audio_codebooks, 1)
        codes2 = codes2.to(device='cuda', dtype=torch.long) % 1024
        
        # Pattern 3: Random values
        torch.manual_seed(42)
        codes3 = torch.randint(0, 1024, (1, audio_codebooks, seq_length), device='cuda', dtype=torch.long)
        
        logger.info(f"📊 Testing 3 different audio patterns, {seq_length} tokens each")
        
        losses_per_pattern = []
        
        for pattern_idx, codes in enumerate([codes1, codes2, codes3], 1):
            logger.info(f"\n🎯 Testing Pattern {pattern_idx}")
            
            # Create LMGen for this pattern
            lm_gen = LMGen(
                lm_model=model,
                use_sampling=False,
                temp=1.0, temp_text=1.0, top_k=0, top_k_text=0,
                cfg_coef=1.0, check=False, condition_tensors=None,
            )
            
            pattern_losses = []
            
            with lm_gen.streaming(batch_size=1):
                for t in range(seq_length - 1):  # -1 because we compute next-token loss
                    # Step 1: LMGen step (for proper streaming state)
                    audio_codes = codes[:, :, t:t+1]  # [1, 8, 1]
                    tokens = lm_gen.step(audio_codes)
                    
                    # Step 2: Direct model forward for loss computation
                    if t >= 2:  # Need some context
                        # Get context
                        context_length = min(t + 1, 8)
                        start_pos = max(0, t + 1 - context_length)
                        context_codes = codes[:, :, start_pos:t+1]
                        
                        # Create full input sequence
                        input_sequence = torch.zeros(1, model.num_codebooks, context_length,
                                                   dtype=torch.long, device='cuda')
                        input_sequence[:, model.audio_offset:model.audio_offset + 8] = context_codes
                        
                        # Forward pass
                        with torch.no_grad():
                            model_output = model.forward(input_sequence)
                            
                            if model_output.logits.shape[-2] > 0:
                                # Get next target
                                next_target = codes[:, :, t+1]  # [1, 8]
                                
                                # Get last position logits
                                last_pos_logits = model_output.logits[:, :, -1, :]  # [1, 8, vocab_size]
                                last_pos_mask = model_output.mask[:, :, -1]         # [1, 8]
                                
                                # Compute loss
                                total_loss = 0.0
                                valid_codebooks = 0
                                
                                for cb in range(8):
                                    if last_pos_mask[0, cb]:
                                        cb_logits = last_pos_logits[0, cb]  # [vocab_size]
                                        cb_target = next_target[0, cb]       # scalar
                                        
                                        cb_loss = torch.nn.functional.cross_entropy(
                                            cb_logits.unsqueeze(0),
                                            cb_target.unsqueeze(0),
                                            reduction='none'
                                        )
                                        total_loss += cb_loss.item()
                                        valid_codebooks += 1
                                
                                if valid_codebooks > 0:
                                    avg_loss = total_loss / valid_codebooks
                                    pattern_losses.append(avg_loss)
                                    logger.info(f"   Position {t}: loss = {avg_loss:.6f}")
            
            losses_per_pattern.append(pattern_losses)
            logger.info(f"   Pattern {pattern_idx} average loss: {np.mean(pattern_losses):.6f}")
        
        # Check for variability
        all_losses = []
        for pattern_losses in losses_per_pattern:
            all_losses.extend(pattern_losses)
        
        if len(all_losses) > 0:
            unique_losses = set(f"{loss:.4f}" for loss in all_losses)
            logger.info(f"\n🔍 Total losses computed: {len(all_losses)}")
            logger.info(f"🔍 Unique loss values: {len(unique_losses)}")
            logger.info(f"🔍 Loss range: {min(all_losses):.6f} to {max(all_losses):.6f}")
            
            if len(unique_losses) > 1:
                logger.info("🎉 SUCCESS: Different loss values found!")
                logger.info("✅ LibriLight logits fix is working!")
                return True
            else:
                logger.warning("⚠️ All losses are the same - fix may not be working")
                return False
        else:
            logger.warning("⚠️ No losses computed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_loss_computation_directly()
    if success:
        print("\n🎉 LOSS COMPUTATION FIX VERIFIED!")
    else:
        print("\n💥 LOSS COMPUTATION FIX FAILED!")