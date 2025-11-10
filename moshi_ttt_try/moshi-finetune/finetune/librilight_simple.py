# Simplified LibriLight Evaluation - AUTOREGRESSIVE STREAMING WITH TTT SUPPORT

"""
This is the CORRECT LibriLight evaluation for autoregressive audio prediction!

Uses depformer_replace_tokens to enable autoregressive evaluation on Moshi's own audio stream,
matching the TTT paper methodology.

Key differences from conversation mode:
1. ❌ OLD (WRONG): Pass LibriLight as user audio → Moshi responds (conversation mode)
2. ✅ NEW (CORRECT): Pass silence as user audio, use depformer_replace_tokens for Moshi stream
   → Autoregressive prediction: P(audio[t+1] | audio[0:t])

This matches training and the TTT paper:
- Model predicts its own audio continuation (not response to user)
- TTT adapts to sequence patterns, content, and style
- Loss should decrease over long sequences as TTT learns
- Proper teacher forcing prevents error accumulation

Technical implementation:
1. ✅ User stream (codebooks 9-16): Silence (not in conversation mode)
2. ✅ Moshi stream (codebooks 1-8): LibriLight audio via depformer_replace_tokens
3. ✅ Gets logits from TTT-updated state via step_with_logits()
4. ✅ Computes ONLY audio loss (LibriLight has no text labels)
5. ✅ Processes one token at a time (proper streaming evaluation)
"""

import torch
import torch.nn.functional as F
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def evaluate_librilight_simple(
    model,
    lm_gen,
    audio_codes: torch.Tensor,
    audio_targets: torch.Tensor,
    max_length: int = 25000,
    first_codebook_weight_multiplier: float = 100.0
) -> List[float]:
    """
    LibriLight evaluation using STREAMING with TTT support.
    
    Uses LMGen.step_with_logits() to get logits from TTT-updated streaming state.
    
    Args:
        model: LMModel with TTT layers
        lm_gen: LMGen instance
        audio_codes: Input audio codes [1, 8, seq_len]
        audio_targets: Target audio codes [1, 8, seq_len]
        max_length: Maximum sequence length to process
        
    Returns:
        List of per-position audio losses
    """
    model.eval()
    
    seq_length = min(audio_codes.shape[-1], max_length)
    position_losses = []
    
    # Check if TTT is actually enabled (matches paper_metrics.py detection)
    has_ttt = any(hasattr(layer, 'ttt_layer') or 'ttt' in str(type(layer)).lower() 
                  for layer in model.transformer.layers)
    ttt_status = "TTT enabled" if has_ttt else "TTT disabled"
    
    logger.info(f"Starting LibriLight evaluation: {seq_length} tokens")
    logger.info(f"Using STREAMING mode ({ttt_status})")
    
    # Prepare input: full 17 codebooks
    # [0] = text (silence)
    # [1-8] = Moshi audio (our data)
    # [9-16] = User audio (silence)
    B = 1
    device = audio_codes.device
    
    # For LibriLight autoregressive evaluation:
    # - User stream: silence/zeros (not in conversation mode)
    # - Moshi stream: LibriLight audio (autoregressive prediction with teacher forcing)
    num_user_codebooks = 8  # User audio codebooks (9-16 in full 17-codebook model)
    silence_input = torch.zeros(B, num_user_codebooks, 1, device=device, dtype=audio_codes.dtype)
    
    with torch.no_grad(), lm_gen.streaming(batch_size=B):
        for t in range(seq_length - 1):
            # Current Moshi audio at position t [1, 8]
            current_moshi_audio = audio_codes[:, :, t:t+1]  # [1, 8, 1]
            
            # Next target at position t+1 [1, 8]
            next_target = audio_targets[:, :, t+1]
            
            # Use depformer_replace_tokens for teacher forcing on Moshi stream
            # This makes the model predict autoregressively: given audio[0:t] -> predict audio[t+1]
            # NOT user->moshi response (which is what passing audio as input does)
            result = lm_gen.step_with_logits(
                silence_input,  # User stream: silence (not conversation mode)
                depformer_replace_tokens=current_moshi_audio,  # Moshi stream: ground truth for teacher forcing
                teacher_forcing_tokens=next_target  # Depformer internal: ground truth for autoregressive codebooks
            )
            
            if result is None:
                # Not ready to output yet (initial delay)
                continue
            
            # Get audio logits [B, dep_q=8, vocab_size]
            audio_logits = result['audio_logits']  # [1, 8, vocab_size]
            
            # Compute loss for this position (matching moshi-finetune training loss)
            loss = _compute_position_audio_loss(
                audio_logits, 
                next_target,
                first_codebook_weight_multiplier=first_codebook_weight_multiplier
            )
            
            if loss is not None:
                position_losses.append(loss)
            
            # Progress logging
            if t % 1000 == 0 and t > 0:
                avg_loss = sum(position_losses) / len(position_losses) if position_losses else 0
                logger.info(f"Position {t}/{seq_length}: avg_loss={avg_loss:.4f}")
    
    logger.info(f"LibriLight evaluation complete: {len(position_losses)} positions")
    return position_losses


def _compute_position_audio_loss(
    audio_logits: torch.Tensor,
    target: torch.Tensor,
    first_codebook_weight_multiplier: float = 100.0
) -> float:
    """
    Compute audio loss for a single position - EXACTLY like moshi-finetune training.
    
    This matches the compute_loss_with_mask function in finetune/loss.py:
    1. Compute cross-entropy per codebook
    2. Multiply by weights
    3. Sum and divide by total weight
    
    Args:
        audio_logits: [1, 8, vocab_size] (squeezed from forward_depformer)
        target: [1, 8]
        first_codebook_weight_multiplier: Weight multiplier for first codebook (default: 100.0)
    
    Returns:
        Weighted average loss across audio codebooks (matching training loss calculation)
    """
    try:
        # Build weights: first codebook gets multiplier, rest get 1.0
        # Shape: [1, 8]
        weights = torch.ones_like(target, dtype=torch.float32)
        weights[:, 0] *= first_codebook_weight_multiplier  # First audio codebook
        
        # Flatten everything EXACTLY like training code
        # audio_logits: [1, 8, vocab_size] -> [8, vocab_size]
        # target: [1, 8] -> [8]
        # weights: [1, 8] -> [8]
        logits_flat = audio_logits.view(-1, audio_logits.size(-1)).float()  # [8, vocab_size]
        target_flat = target.view(-1)  # [8]
        weights_flat = weights.view(-1)  # [8]
        
        # EXACTLY like training: cross_entropy then multiply by weights
        losses = F.cross_entropy(logits_flat, target_flat, reduction="none")  # [8]
        weighted_losses = losses * weights_flat  # [8]
        
        # Sum and divide by total weight (EXACTLY like training)
        total_loss = torch.sum(weighted_losses)
        total_weight = torch.sum(weights_flat)
        final_loss = total_loss / total_weight
        
        # DEBUG: Log first time
        if not hasattr(_compute_position_audio_loss, '_logged'):
            logger.warning(f"🔍 DEBUG Loss Computation (FIXED):") 
            logger.warning(f"  audio_logits.shape: {audio_logits.shape}")
            logger.warning(f"  target.shape: {target.shape}")
            logger.warning(f"  logits min/max/mean: {logits_flat.min().item():.2f} / {logits_flat.max().item():.2f} / {logits_flat.mean().item():.2f}")
            logger.warning(f"  logits std: {logits_flat.std().item():.2f}")
            logger.warning(f"  target values: {target_flat.tolist()}")
            logger.warning(f"  weights_flat: {weights_flat.tolist()}")
            logger.warning(f"  losses (CE): {losses.tolist()}")
            logger.warning(f"  weighted_losses: {weighted_losses.tolist()}")
            logger.warning(f"  total_loss: {total_loss.item():.4f}")
            logger.warning(f"  total_weight: {total_weight.item():.4f}")
            logger.warning(f"  final_loss: {final_loss.item():.4f}")
            _compute_position_audio_loss._logged = True
        
        return final_loss.item()
        
    except Exception as e:
        logger.warning(f"Error computing loss: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None
        
    except Exception as e:
        logger.warning(f"Error computing loss: {e}")
        return None


def aggregate_librilight_results(position_losses: List[float]) -> Dict[str, float]:
    """
    Aggregate position-wise losses into metrics.
    
    Returns losses at 8k, 16k, 24k positions and slope (TTT improvement).
    """
    if not position_losses:
        return {
            'librilight_loss_8k': 0.0,
            'librilight_loss_16k': 0.0,
            'librilight_loss_24k': 0.0,
            'librilight_slope': 0.0,
            'librilight_samples': 0
        }
    
    # Get losses at key positions
    loss_8k = position_losses[8000] if len(position_losses) > 8000 else 0.0
    loss_16k = position_losses[16000] if len(position_losses) > 16000 else 0.0
    loss_24k = position_losses[24000] if len(position_losses) > 24000 else 0.0
    
    # Compute slope (improvement from start to end)
    # Negative slope = improvement (loss decreasing)
    if len(position_losses) > 1000:
        early_avg = sum(position_losses[:1000]) / 1000
        late_avg = sum(position_losses[-1000:]) / 1000
        slope = (late_avg - early_avg) / len(position_losses)
    else:
        slope = 0.0
    
    return {
        'librilight_loss_8k': loss_8k,
        'librilight_loss_16k': loss_16k,
        'librilight_loss_24k': loss_24k,
        'librilight_slope': slope,
        'librilight_samples': len(position_losses)
    }


# NOTE: This is still a workaround!
# The PROPER fix is to add this method to LMGen in lm.py:
#
# def step_with_logits(self, input_tokens):
#     """Like step() but also returns logits for evaluation."""
#     # ... same logic as _step() ...
#     transformer_out, text_logits = state.graphed_main(...)
#     
#     # Get audio logits
#     audio_logits_list = []
#     for cb_idx in range(self.lm_model.dep_q):
#         logits = self.lm_model.forward_depformer(cb_idx, text_token_input, transformer_out)
#         audio_logits_list.append(logits)
#     audio_logits = torch.stack(audio_logits_list, dim=1)
#     
#     return {
#         'tokens': output_tokens,
#         'text_logits': text_logits,
#         'audio_logits': audio_logits
#     }
#
# Then evaluation becomes trivial:
#
# def evaluate_librilight_correct(model, lm_gen, codes, targets):
#     losses = []
#     with lm_gen.streaming(batch_size=1):
#         for t in range(len(codes) - 1):
#             result = lm_gen.step_with_logits(codes[:, :, t:t+1])
#             loss = F.cross_entropy(result['audio_logits'].view(-1, vocab_size),
#                                   targets[:, :, t+1].view(-1))
#             losses.append(loss.item())
#     return losses
