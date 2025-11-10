"""
Streamlined Paper Metrics Evaluation for TTT-Moshi Training
Simplified version of comprehensive evaluation system for integration into training pipeline.
"""

import json
import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import random
import pandas as pd
import numpy as np
import gc

try:
    import torchaudio
except ImportError:
    torchaudio = None
    
try:
    import librosa
except ImportError:
    librosa = None

try:
    from .librilight_loader import LibriLightLoader
except ImportError:
    LibriLightLoader = None

try:
    from .librilight_plotting import create_librilight_plots, extract_position_losses_from_results, determine_model_type
except ImportError:
    create_librilight_plots = None
    extract_position_losses_from_results = None 
    determine_model_type = None

# Essential import for proper Moshi streaming evaluation
from moshi.models.lm import LMGen

logger = logging.getLogger(__name__)


def _compute_paper_exact_loss(logits: torch.Tensor, target: torch.Tensor, 
                             valid_mask: torch.Tensor, semantic_weight: float = 100.0) -> torch.Tensor:
    """
    Compute paper-exact loss with semantic token weighting.
    Following eval_paper methodology for consistent evaluation.
    """
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, target, reduction="none")
    
    # Apply semantic weighting (first codebook gets higher weight)
    # This assumes target shape allows codebook identification
    if len(target.shape) > 1:
        # If target has codebook dimension, apply weighting
        weights = torch.ones_like(target, dtype=torch.float32)
        weights[:, 0] *= semantic_weight  # First codebook (semantic) gets higher weight
        weights = weights.view(-1)
    else:
        # Flat target - apply uniform weighting
        weights = torch.ones_like(target, dtype=torch.float32)
    
    # Apply weights and mask
    valid_mask = valid_mask.view(-1)
    loss = loss * weights * valid_mask.float()
    
    # Return mean loss over valid positions
    return loss.sum() / valid_mask.float().sum()


class PaperMetricsEvaluator:
    """
    Lightweight evaluator for paper metrics during training.
    Supports sBLIMP, sWUGGY, tStoryCloze, and sStoryCloze benchmarks.
    """
    
    def __init__(self, mimi_encoder, interleaved_tokenizer, device: str = "cuda", config=None):
        self.mimi = mimi_encoder
        self.tokenizer = interleaved_tokenizer  
        self.device = device
        self.config = config or {}
        
        # Paper-exact loss configuration
        self.first_codebook_weight_multiplier = 100.0
        
        # Stream configuration options (from eval_paper)
        self.use_user_stream = self.config.get('paper_metrics_use_user_stream', False)
        self.use_silence_codes = self.config.get('paper_metrics_use_silence', False)
        
        # Streaming evaluation configuration options
        self.librilight_streaming_enabled = self.config.get('librilight_streaming', {}).get('enabled', True)
        self.memory_check_enabled = self.config.get('librilight_streaming', {}).get('memory_check', True)
        self.cache_clear_interval = self.config.get('librilight_streaming', {}).get('cache_clear_interval', 3000)
        self.max_sequence_length = self.config.get('librilight_streaming', {}).get('max_sequence_length', 50000)
        self.ttt_verification_enabled = self.config.get('librilight_streaming', {}).get('ttt_verification', True)
        self.memory_log_interval = self.config.get('librilight_streaming', {}).get('memory_log_interval', 1000)
        
        # TTT-Optimized chunking configuration
        self.ttt_config = self.config.get('ttt', {})
        self.ttt_optimize_chunk_size = self.ttt_config.get('optimize_chunk_size', True)
        self.ttt_chunk_size_override = self.ttt_config.get('chunk_size', None)
        self.ttt_max_chunk_size = self.ttt_config.get('max_chunk_size', 50)
        self.ttt_prefer_efficiency = self.ttt_config.get('prefer_efficiency', True)
        self.ttt_mini_batch_size = self.ttt_config.get('mini_batch_size', 16)
        
        # Fixed streaming evaluation configuration
        self.use_fixed_streaming = self.config.get('librilight_streaming', {}).get('use_fixed_method', True)
        self.verify_loss_computation = self.config.get('librilight_streaming', {}).get('verify_loss_computation', True)
        
        # Initialize audio cache for massive speedup
        from .audio_cache import get_audio_cache, get_silence_cache
        self.audio_cache = get_audio_cache() if mimi_encoder else None
        self.silence_cache = get_silence_cache() if mimi_encoder else None
        
        # Display stream configuration like reference implementation
        if self.use_user_stream and self.use_silence_codes:
            stream_name = "User Audio Stream (9-16) + Natural Silence (1-8)"
            interpretation = "Cross-stream with natural silence (experimental mode)"
        elif self.use_user_stream:
            stream_name = "User Audio Stream (9-16)"
            interpretation = "Cross-stream evaluation (architecture research mode)"
        elif self.use_silence_codes:
            stream_name = "Moshi Audio Stream (1-8) + Natural Silence (9-16)"
            interpretation = "Optimal silence mode (+2-6% performance boost)"
        else:
            stream_name = "Moshi Audio Stream (1-8)"
            interpretation = "Standard evaluation mode"
        
        logger.info(f"Paper metrics evaluator initialized with audio cache system")
        logger.info(f"Streaming config: enabled={self.librilight_streaming_enabled}, "
                   f"memory_check={self.memory_check_enabled}, "
                   f"cache_clear_interval={self.cache_clear_interval}, "
                   f"ttt_verification={self.ttt_verification_enabled}")
        logger.info(f"🔧 Loss computation: {'FIXED (TTT-aware)' if self.use_fixed_streaming else 'LEGACY (separate forward pass)'}, "
                   f"verification={self.verify_loss_computation}")
        logger.info(f"🎵 Stream Configuration: {stream_name}")
        logger.info(f"   Interpretation: {interpretation}")
        
        # Log TTT chunking configuration
        if self.ttt_optimize_chunk_size:
            optimal_chunk = self.get_optimal_ttt_chunk_size()
            efficiency_info = self.calculate_ttt_efficiency(optimal_chunk)
            logger.info(f"🧠 TTT-Optimized Chunking: chunk_size={optimal_chunk}, "
                       f"efficiency={efficiency_info['efficiency_percent']:.1f}%, "
                       f"mini_batches={efficiency_info['num_mini_batches']}")
        else:
            logger.info(f"🧠 TTT Chunking: Legacy mode (chunk_size={self.ttt_max_chunk_size})")
    
    def get_optimal_ttt_chunk_size(self) -> int:
        """Calculate optimal chunk size for TTT evaluation based on configuration."""
        if not self.ttt_optimize_chunk_size:
            return self.ttt_max_chunk_size  # Use legacy behavior
        
        if self.ttt_chunk_size_override is not None:
            return min(self.ttt_chunk_size_override, self.ttt_max_chunk_size)  # Use explicit override
        
        # Calculate optimal chunk size based on mini_batch_size
        mini_batch = self.ttt_mini_batch_size
        max_chunk = self.ttt_max_chunk_size
        
        if self.ttt_prefer_efficiency:
            # Find largest divisor of max_chunk_size that's >= mini_batch_size
            # This gives 100% efficiency with largest possible chunks
            for chunk_size in range(max_chunk, mini_batch - 1, -1):
                if max_chunk % chunk_size == 0:
                    return chunk_size
            # Fallback: use mini_batch_size for perfect efficiency
            return mini_batch
        else:
            # Use mini_batch_size directly for maximum adaptation granularity
            return min(mini_batch, max_chunk)
    
    def calculate_ttt_efficiency(self, chunk_size: int) -> dict:
        """Calculate TTT processing efficiency metrics for a given chunk size."""
        mini_batch = self.ttt_mini_batch_size
        num_complete_batches = chunk_size // mini_batch
        remaining_tokens = chunk_size % mini_batch
        
        if remaining_tokens > 0:
            # Last mini-batch needs padding
            padding_tokens = mini_batch - remaining_tokens
            total_processed = chunk_size + padding_tokens
            efficiency = chunk_size / total_processed
            num_batches = num_complete_batches + 1
        else:
            # Perfect alignment
            efficiency = 1.0
            num_batches = num_complete_batches
            padding_tokens = 0
        
        return {
            "chunk_size": chunk_size,
            "mini_batch_size": mini_batch,
            "efficiency_percent": efficiency * 100,
            "padding_tokens": padding_tokens,
            "num_mini_batches": num_batches,
            "tokens_per_batch": chunk_size / num_batches if num_batches > 0 else 0,
            "total_processed_tokens": total_processed if remaining_tokens > 0 else chunk_size,
            "is_perfectly_aligned": remaining_tokens == 0
        }
    
    def _evaluate_librilight_simple(self, model, codes, targets):
        """
        Simplified LibriLight evaluation using proper streaming API.
        
        FIXES:
        1. Uses model's predicted text token (not silence) for audio-only evaluation
        2. Computes ONLY audio loss (LibriLight has no text labels)  
        3. Simple, ~50 lines instead of 600+
        
        NOTE: This is still a workaround. The proper fix is to add step_with_logits() to LMGen API.
        """
        from .librilight_simple import evaluate_librilight_simple
        
        # Create LMGen instance
        lm_gen = LMGen(
            lm_model=model,
            use_sampling=False,  # Deterministic for evaluation
            temp=1.0,
            top_k=0,
            cfg_coef=1.0,
            check=False,
            condition_tensors=None,
        )
        
        # Use the simplified implementation with same weight multiplier as training
        return evaluate_librilight_simple(
            model, 
            lm_gen, 
            codes, 
            targets,
            first_codebook_weight_multiplier=self.first_codebook_weight_multiplier
        )

    def _encode_audio(self, audio_path: str) -> torch.Tensor:
        """Encode audio file to tokens using MIMI with caching for massive speedup."""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            # Use cached encoding for massive speedup (90%+ time reduction)
            if self.audio_cache is not None:
                codes = self.audio_cache.get_encoded_audio(audio_path, self.mimi, self.device)
                if codes is not None:
                    return codes
            
            # Fallback to direct encoding if cache miss
            # Use torchaudio for faster loading than librosa
            if torchaudio is not None:
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample to 24kHz if needed
                if sample_rate != 24000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                    waveform = resampler(waveform)
            elif librosa is not None:
                # Fallback to librosa if torchaudio not available
                waveform, sample_rate = librosa.load(audio_path, sr=24000)
                waveform = torch.from_numpy(waveform).unsqueeze(0)  # Add channel dim
            else:
                raise ImportError("Neither torchaudio nor librosa is available for audio loading")
            
            # Convert to MIMI input format [B, C, T]
            waveform = waveform.to(self.device).unsqueeze(0)  # Add batch dimension
            
            # Encode with MIMI
            with torch.no_grad():
                codes = self.mimi.encode(waveform)
                
            # Validate encoded result
            if codes.numel() == 0:
                raise ValueError(f"MIMI encoding failed - empty result for: {audio_path}")
                
            return codes
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to encode audio {audio_path}: {e}")
            # No fallback - fail loudly to ensure we notice data issues
            raise RuntimeError(f"Audio encoding failed for {audio_path}: {e}") from e
    
    def generate_silence_codes(self, target_shape):
        """Generate silence codes with caching for massive speedup."""
        # Use cached silence codes for instant generation
        if self.silence_cache is not None:
            return self.silence_cache.get_silence_codes(target_shape, self.mimi, self.device)
        
        # Fallback to direct generation if no cache
        B, K, T = target_shape
        
        # Convert temporal length to audio samples for 24kHz
        samples_per_frame = 24000 // 12.5  # Moshi frame rate
        target_samples = int(T * samples_per_frame)
        
        # Create silence tensor [B, channels, samples]
        silence = torch.zeros(
            (B, self.mimi.channels, target_samples),  # Correct channels from MIMI model
            device=self.device,
            dtype=torch.float32
        )
        
        # Encode to get silence codes
        with torch.no_grad():
            silence_codes = self.mimi.encode(silence)
        
        return silence_codes
    
    def _compute_likelihood(self, model, codes: torch.Tensor) -> float:
        """Compute log-likelihood with optimal silence configuration following eval_paper methodology"""
        try:
            with torch.no_grad():
                codes = codes.to(self.device)
                B, K, T = codes.shape
                
                # Prepare input tokens (exact working methodology)
                inp = torch.full(
                    (B, model.num_codebooks, T),
                    model.zero_token_id,
                    device=self.device,
                    dtype=codes.dtype
                )
                
                # Place audio codes in the chosen stream(s) - following eval_paper methodology exactly
                moshi_start = model.audio_offset
                user_start = model.audio_offset + model.dep_q
                
                if self.use_user_stream:
                    # User stream mode: Audio in User stream (9-16)
                    if self.use_silence_codes:
                        # Generate silence codes for Moshi stream
                        silence_codes = self.generate_silence_codes(codes.shape)
                        inp[:, moshi_start:moshi_start+K] = silence_codes
                        inp[:, user_start:user_start+K] = codes
                        logger.debug(f"Natural silence mode - Moshi stream ({moshi_start}-{moshi_start+K-1}) has silence codes, User stream ({user_start}-{user_start+K-1}) has audio")
                    else:
                        # Cross-stream: Audio in User stream, Moshi stream stays as padding
                        inp[:, user_start:user_start+K] = codes
                        logger.debug(f"Cross-stream evaluation - Audio placed in User stream ({user_start}-{user_start+K-1})")
                else:
                    # Moshi stream mode: Audio in Moshi stream (1-8)
                    if self.use_silence_codes:
                        # OPTIMAL MODE: Generate silence codes for User stream (+2-6% performance boost)
                        silence_codes = self.generate_silence_codes(codes.shape)
                        inp[:, moshi_start:moshi_start+K] = codes
                        inp[:, user_start:user_start+K] = silence_codes
                        logger.debug(f"Optimal silence mode - Moshi stream ({moshi_start}-{moshi_start+K-1}) has audio, User stream ({user_start}-{user_start+K-1}) has silence codes")
                    else:
                        # Standard: Place audio in Moshi stream only
                        inp[:, moshi_start:moshi_start+K] = codes
                        logger.debug(f"Standard mode - Audio placed in Moshi stream ({moshi_start}-{moshi_start+K-1})")
                
                # Forward pass
                out = model(inp)
                
                # Compute exactly as in fine-tune loss (exact working methodology)
                target = codes[:, :min(K, model.dep_q), :]
                
                # Use model output mask for proper masking
                mask = out.mask
                logits = out.logits[:, :mask.shape[1]].float()
                
                # BUILD A MANUAL MASK combining model mask and code validity 
                mask_codes = (target != model.zero_token_id)  # [B,K,T]
                mask_logits = ~torch.isnan(logits).any(dim=-1)  # [B,K,T]
                valid_mask = mask & mask_codes & mask_logits  # [B,K,T]
                
                # Apply 100× weighting to semantic tokens (exact working methodology)
                weights = valid_mask.float()
                weights[:, 0] *= 100.0  # Semantic token weighting
                
                # Compute paper-exact loss with proper masking
                logits = logits.view(-1, logits.size(-1))
                targ = torch.where(valid_mask, target, torch.zeros_like(target)).view(-1)
                w_flat = weights.view(-1)
                
                loss = F.cross_entropy(logits, targ, reduction="none")
                loss = torch.where(w_flat > 0, loss * w_flat, torch.zeros_like(loss))
                
                return (loss.sum() / w_flat.sum()).item()
                
        except Exception as e:
            logger.error(f"Error computing likelihood: {e}")
            return float('inf')
    
    @torch.no_grad()
    def evaluate_sblimp(self, model, max_samples: int = None) -> Dict[str, float]:
        """
        Evaluate on sBLIMP using exact working methodology from clean_sblimp_evaluator.py
        """
        try:
            # Get paths from config
            audio_dir = self.config.get('sblimp_audio_dir')
            gold_csv = self.config.get('sblimp_gold_csv')
            max_pairs = max_samples or self.config.get('sblimp_max_pairs', 100)
            logger.info(f"sBLIMP: Target sample count: {max_pairs}")
            
            if not audio_dir or not gold_csv:
                logger.warning("sBLIMP paths not configured, skipping")
                return {'sblimp_accuracy': 0.0, 'sblimp_samples': 0}
            
            if not Path(gold_csv).exists():
                logger.warning(f"sBLIMP gold CSV not found: {gold_csv}")
                return {'sblimp_accuracy': 0.0, 'sblimp_samples': 0}
            
            if not Path(audio_dir).exists():
                logger.warning(f"sBLIMP audio directory not found: {audio_dir}")
                return {'sblimp_accuracy': 0.0, 'sblimp_samples': 0}
            
            # Load CSV using exact working methodology
            df = pd.read_csv(gold_csv)
            logger.info(f"sBLIMP: Loaded CSV with columns: {list(df.columns)}")
            
            # Create sentence pairs using exact working logic
            sentence_pairs = self._create_sblimp_sentence_pairs(df, audio_dir, max_pairs)
            logger.info(f"sBLIMP: Found {len(sentence_pairs)} valid pairs")
            
            if len(sentence_pairs) == 0:
                logger.warning("sBLIMP: No valid pairs found")
                return {'sblimp_accuracy': 0.0, 'sblimp_samples': 0}
            
            correct_predictions = 0
            total_pairs = 0
            failed_pairs = 0
            
            for pair in sentence_pairs:
                try:
                    # Encode both audio files using exact working methodology  
                    good_codes = self._encode_audio(pair['good_audio_path'])
                    bad_codes = self._encode_audio(pair['bad_audio_path'])
                    
                    # Compute likelihoods using paper-exact methodology
                    good_nll = self._compute_sblimp_likelihood(model, good_codes)
                    bad_nll = self._compute_sblimp_likelihood(model, bad_codes)
                    
                    # Model should prefer grammatical sentences (lower NLL)
                    prediction_correct = good_nll < bad_nll
                    
                    if prediction_correct:
                        correct_predictions += 1
                    
                    total_pairs += 1
                    
                    if total_pairs % 10 == 0:
                        logger.info(f"sBLIMP: Processed {total_pairs}/{len(sentence_pairs)} pairs")
                    
                except Exception as e:
                    failed_pairs += 1
                    logger.warning(f"sBLIMP: Error processing pair {pair.get('pair_id', 'unknown')}: {e}")
                    continue
            
            if total_pairs > 0:
                accuracy = correct_predictions / total_pairs
                logger.info(f"sBLIMP: {correct_predictions}/{total_pairs} = {accuracy:.3f} (failed: {failed_pairs})")
                
                return {
                    'sblimp_accuracy': accuracy,
                    'sblimp_samples': total_pairs,
                    'sblimp_correct': correct_predictions
                }
            else:
                logger.warning("sBLIMP: No valid pairs processed")
                return {'sblimp_accuracy': 0.0, 'sblimp_samples': 0}
            
        except Exception as e:
            logger.error(f"sBLIMP evaluation failed: {e}")
            return {'sblimp_accuracy': 0.0, 'sblimp_samples': 0}
    
    def _create_sblimp_sentence_pairs(self, df, audio_dir, max_pairs=None):
        """Create sentence pairs using exact working methodology from clean_sblimp_evaluator.py"""
        sentence_pairs = []
        
        # Group by the complete minimal pair identifier - EXACT SAME AS WORKING CODE
        grouped = df.groupby(['id', 'voice', 'type', 'subtype'])
        
        for (pair_id, voice, pair_type, subtype), group in grouped:
            # Early termination if we have enough pairs
            if max_pairs and len(sentence_pairs) >= max_pairs:
                break
            if len(group) == 2:  # Should have exactly grammatical + ungrammatical
                grammatical = group[group['correct'] == 1]
                ungrammatical = group[group['correct'] == 0]
                
                if len(grammatical) == 1 and len(ungrammatical) == 1:
                    good_row = grammatical.iloc[0]
                    bad_row = ungrammatical.iloc[0]
                    
                    # Build file paths (handle potential .wav duplication) - EXACT SAME AS WORKING CODE
                    good_stem = good_row['filename'].replace('.wav', '')
                    bad_stem = bad_row['filename'].replace('.wav', '')
                    good_audio = Path(audio_dir) / f"{good_stem}.wav"
                    bad_audio = Path(audio_dir) / f"{bad_stem}.wav"
                    
                    if good_audio.exists() and bad_audio.exists():
                        sentence_pairs.append({
                            'pair_id': pair_id,
                            'type': pair_type,
                            'subtype': subtype,
                            'voice': voice,
                            'good_audio_path': str(good_audio),
                            'bad_audio_path': str(bad_audio),
                            'good_transcription': good_row['transcription'],
                            'bad_transcription': bad_row['transcription']
                        })
        
        return sentence_pairs
    
    def _compute_sblimp_likelihood(self, model, audio_codes):
        """Compute likelihood using paper-exact methodology with optimal silence configuration"""
        try:
            B, K, T = audio_codes.shape
            
            # Prepare input for audio-only evaluation
            num_codebooks = model.num_codebooks
            input_codes = torch.full(
                (B, num_codebooks, T),
                model.zero_token_id,
                device=self.device,
                dtype=audio_codes.dtype
            )
            
            # Place audio codes in the chosen stream(s) - following eval_paper methodology exactly
            moshi_start = model.audio_offset
            user_start = model.audio_offset + model.dep_q
            
            if self.use_user_stream:
                # User stream mode: Audio in User stream (9-16)
                if self.use_silence_codes:
                    # Generate silence codes for Moshi stream
                    silence_codes = self.generate_silence_codes(audio_codes.shape)
                    input_codes[:, moshi_start:moshi_start+K] = silence_codes
                    input_codes[:, user_start:user_start+K] = audio_codes
                    logger.debug(f"sBLIMP: Natural silence mode - Moshi stream ({moshi_start}-{moshi_start+K-1}) has silence codes, User stream ({user_start}-{user_start+K-1}) has audio")
                else:
                    # Original: padding tokens in Moshi stream (leave as zero_token_id)
                    input_codes[:, user_start:user_start+K] = audio_codes
                    logger.debug(f"sBLIMP: Cross-stream evaluation - Audio placed in User stream ({user_start}-{user_start+K-1})")
            else:
                # Moshi stream mode: Audio in Moshi stream (1-8)
                if self.use_silence_codes:
                    # OPTIMAL MODE: Generate silence codes for User stream (+2-6% performance boost)
                    silence_codes = self.generate_silence_codes(audio_codes.shape)
                    input_codes[:, moshi_start:moshi_start+K] = audio_codes
                    input_codes[:, user_start:user_start+K] = silence_codes
                    logger.debug(f"sBLIMP: Optimal silence mode - Moshi stream ({moshi_start}-{moshi_start+K-1}) has audio, User stream ({user_start}-{user_start+K-1}) has silence codes")
                else:
                    # Default: Place audio in Moshi stream, User stream stays as padding
                    input_codes[:, moshi_start:moshi_start+K] = audio_codes
                    logger.debug(f"sBLIMP: Standard evaluation - Audio placed in Moshi stream ({moshi_start}-{moshi_start+K-1})")
            
            # Forward pass
            with torch.no_grad():
                output = model.forward(input_codes)
            
            # CRITICAL: Target should always be the actual audio codes we're evaluating
            # Regardless of where we placed them in the input (Moshi stream 1-8 or User stream 9-16)
            # The model always outputs predictions for 8 codebooks, and we compare against the same 8 audio codes
            target = audio_codes[:, :min(K, model.dep_q), :]
            
            # BUILD A MANUAL MASK over the 8 predicted streams,
            # combining "where we actually placed a code" and "where logits are finite":
            mask_codes = (target != model.zero_token_id)  # [B,8,T]
            mask_logits = ~torch.isnan(output.logits).any(dim=-1)  # [B,8,T]
            valid_mask = mask_codes & mask_logits  # [B,8,T]
            
            logger.debug(f"sBLIMP: Valid mask positions: {valid_mask.sum().item()}/{valid_mask.numel()}")
            
            # Paper-exact loss computation with per-timestep normalization
            audio_loss = self._compute_paper_exact_loss_sblimp(
                output.logits,  # Shape: [B, dep_q, T, card]
                target,  # Shape: [B, dep_q, T] 
                valid_mask,  # Shape: [B, dep_q, T] - CORRECTED MASK
                first_codebook_weight_multiplier=100.0
            )
                
            return audio_loss.item()
            
        except Exception as e:
            logger.error(f"Error computing sBLIMP likelihood: {e}")
            return float('inf')
    
    def _compute_paper_exact_loss_sblimp(self, logits, target, target_mask, first_codebook_weight_multiplier=100.0):
        """Official training-exact loss computation: matches eval_paper clean_sblimp_evaluator.py exactly"""
        # Mask invalid positions (line 14 in loss.py from reference)
        target = torch.where(target_mask, target, torch.zeros_like(target))
        
        # Apply 100× weighting to semantic tokens (lines 16-18 in loss.py from reference)
        weights = target_mask.float()
        weights[:, 0] *= first_codebook_weight_multiplier
        
        # Compute cross-entropy loss (lines 24-29 in loss.py from reference) - EXACT MATCH
        logits = logits.view(-1, logits.size(-1)).float()
        target = target.view(-1)
        weights = weights.view(-1)
        mb_loss = F.cross_entropy(logits, target, reduction="none")
        mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
        mb_loss = torch.sum(mb_loss) / torch.sum(weights)
        
        return mb_loss
    
    @torch.no_grad()
    def evaluate_swuggy(self, model, max_samples: int = None) -> Dict[str, float]:
        """
        Evaluate on sWUGGY (phonotactic minimal pairs).
        Returns accuracy comparing words vs non-words.
        """
        try:
            # Get paths from config
            audio_dir = self.config.get('swuggy_audio_dir')
            gold_csv = self.config.get('swuggy_gold_csv')
            max_pairs = max_samples or self.config.get('swuggy_max_pairs', 100)
            
            if not audio_dir or not gold_csv:
                logger.warning("sWUGGY paths not configured, skipping")
                return {'swuggy_accuracy': 0.0, 'swuggy_samples': 0}
            
            if not Path(gold_csv).exists():
                logger.warning(f"sWUGGY gold CSV not found: {gold_csv}")
                return {'swuggy_accuracy': 0.0, 'swuggy_samples': 0}
            
            # Load CSV data
            df = pd.read_csv(gold_csv)
            logger.info(f"sWUGGY: Loaded {len(df)} samples from CSV")
            
            # Create word-nonword pairs using exact eval_paper methodology
            word_nonword_pairs = []
            
            # Group by id to get word-nonword pairs (exact same as eval_paper)
            grouped = df.groupby('id')
            
            for pair_id, group in grouped:
                # Early termination if we have enough pairs
                if max_pairs and len(word_nonword_pairs) >= max_pairs:
                    break
                # Separate words (correct=1) and nonwords (correct=0)
                words = group[group['correct'] == 1]
                nonwords = group[group['correct'] == 0]
                
                if len(words) > 0 and len(nonwords) > 0:
                    # For each voice combination, create word-nonword pairs
                    for _, word_row in words.iterrows():
                        for _, nonword_row in nonwords.iterrows():
                            # Only pair same voice for fair comparison
                            if word_row['voice'] == nonword_row['voice']:
                                word_stem = word_row['filename'].replace('.wav', '')
                                nonword_stem = nonword_row['filename'].replace('.wav', '')
                                word_audio = Path(audio_dir) / f"{word_stem}.wav"
                                nonword_audio = Path(audio_dir) / f"{nonword_stem}.wav"
                                
                                if word_audio.exists() and nonword_audio.exists():
                                    word_nonword_pairs.append({
                                        'pair_id': pair_id,
                                        'voice': word_row['voice'],
                                        'word_audio_path': str(word_audio),
                                        'nonword_audio_path': str(nonword_audio),
                                        'word_text': word_row['word'] if pd.notna(word_row['word']) else 'WORD',
                                        'nonword_text': nonword_row['word'] if pd.notna(nonword_row['word']) else 'NONWORD'
                                    })

            if len(word_nonword_pairs) == 0:
                logger.warning("sWUGGY: No complete pairs found in CSV")
                return {'swuggy_accuracy': 0.0, 'swuggy_samples': 0}
            
            logger.info(f"sWUGGY: Found {len(word_nonword_pairs)} word-nonword pairs")
            
            correct = 0
            total = 0
            failed_pairs = 0
            
            for pair in word_nonword_pairs:
                try:
                    # Encode audio to tokens
                    word_codes = self._encode_audio(pair['word_audio_path'])
                    nonword_codes = self._encode_audio(pair['nonword_audio_path'])
                    
                    # Compute likelihoods (returns NLL - higher is worse)
                    word_nll = self._compute_likelihood(model, word_codes)
                    nonword_nll = self._compute_likelihood(model, nonword_codes)
                    
                    # Model should prefer real word (lower NLL means higher probability)
                    if word_nll < nonword_nll:  # FIXED: Lower NLL = better
                        correct += 1
                    total += 1
                    
                    if total % 10 == 0:
                        logger.info(f"sWUGGY: Processed {total}/{len(word_nonword_pairs)} pairs")
                    
                except Exception as e:
                    failed_pairs += 1
                    logger.error(f"Error processing sWUGGY sample {pair.get('pair_id', 'unknown')}: {e}")
                    continue
            
            accuracy = correct / total if total > 0 else 0.0
            logger.info(f"sWUGGY: {correct}/{total} = {accuracy:.3f} (failed: {failed_pairs})")
            
            return {
                'swuggy_accuracy': accuracy,
                'swuggy_samples': total,
                'swuggy_correct': correct
            }
            
        except Exception as e:
            logger.error(f"sWUGGY evaluation failed: {e}")
            return {'swuggy_accuracy': 0.0, 'swuggy_samples': 0}
    
    @torch.no_grad()
    def evaluate_story_cloze(self, model, dataset: str = 'tstory', max_samples: int = None) -> Dict[str, float]:
        """
        Evaluate on Story Cloze tasks (tStoryCloze or sStoryCloze).
        Returns accuracy for story completion prediction.
        """
        try:
            if dataset not in ['tstory', 'sstory']:
                raise ValueError(f"Invalid dataset: {dataset}")
            
            # Get paths from config
            audio_dir = self.config.get(f'{dataset}_audio_dir')
            max_pairs = max_samples or self.config.get(f'{dataset}_max_pairs', 100)
            
            if not audio_dir:
                logger.warning(f"{dataset} audio directory not configured, skipping")
                return {f'{dataset}_accuracy': 0.0, f'{dataset}_samples': 0}
            
            if not Path(audio_dir).exists():
                logger.warning(f"{dataset} audio directory not found: {audio_dir}")
                return {f'{dataset}_accuracy': 0.0, f'{dataset}_samples': 0}
            
            # Look for structured data files that indicate correct/incorrect pairs
            audio_files = list(Path(audio_dir).rglob("*.wav"))
            if len(audio_files) < 2:
                logger.warning(f"Not enough audio files found in {audio_dir} - need at least 2 files for comparison")
                return {f'{dataset}_accuracy': 0.0, f'{dataset}_samples': 0}
            
            logger.info(f"Found {len(audio_files)} audio files for {dataset}")
            
            # Use exact working pattern from clean_sstory_cloze_evaluator.py
            import re
            story_pairs = {}
            
            matched_files = 0
            for audio_file in audio_files:
                filename = audio_file.name  # Get full filename with .wav
                
                # Use exact regex pattern from working implementation
                # Pattern: (\d+)_([a-f0-9\-]+)_(correct|incorrect)\.wav
                match = re.match(r'(\d+)_([a-f0-9\-]+)_(correct|incorrect)\.wav', filename)
                
                if match:
                    story_id = match.group(1)
                    uuid = match.group(2) 
                    label = match.group(3)
                    
                    # Group by UUID (exact working methodology)
                    if uuid not in story_pairs:
                        story_pairs[uuid] = {}
                    story_pairs[uuid][label] = audio_file
                    matched_files += 1
            
            logger.info(f"{dataset}: Matched {matched_files}/{len(audio_files)} files using exact working pattern")
            
            # Filter to complete pairs only
            complete_pairs = {sid: pair for sid, pair in story_pairs.items() 
                            if 'correct' in pair and 'incorrect' in pair}
            
            if len(complete_pairs) == 0:
                logger.warning(f"No structured story cloze pairs found in {audio_dir}. "
                             f"Expected filenames with pattern suffixes like '_correct/_incorrect', "
                             f"'_ending1/_ending2', etc. Found files: {[f.name for f in audio_files[:5]]}")
                return {f'{dataset}_accuracy': 0.0, f'{dataset}_samples': 0}
            
            logger.info(f"Found {len(complete_pairs)} complete story pairs for {dataset}")
            
            # Limit to max_pairs and take a random sample for variety
            story_ids = list(complete_pairs.keys())
            if len(story_ids) > max_pairs:
                import random
                story_ids = random.sample(story_ids, max_pairs)
            
            correct = 0
            total = 0
            
            for story_id in story_ids:
                try:
                    pair = complete_pairs[story_id]
                    correct_file = pair['correct']
                    incorrect_file = pair['incorrect']
                    
                    # Encode both audio files
                    correct_codes = self._encode_audio(str(correct_file))
                    incorrect_codes = self._encode_audio(str(incorrect_file))
                    
                    # Compute likelihoods (returns NLL - higher is worse)
                    correct_nll = self._compute_likelihood(model, correct_codes)
                    incorrect_nll = self._compute_likelihood(model, incorrect_codes)
                    
                    # Model should prefer correct ending (lower NLL means higher probability)
                    if correct_nll < incorrect_nll:  # FIXED: Lower NLL = better
                        correct += 1
                    total += 1
                    
                    if total % 10 == 0:
                        logger.info(f"{dataset}: processed {total}/{len(story_ids)} pairs")
                    
                except Exception as e:
                    logger.error(f"Error processing {dataset} story {story_id}: {e}")
                    # Continue processing other pairs
                    continue
            
            accuracy = correct / total if total > 0 else 0.0
            logger.info(f"{dataset}: {correct}/{total} = {accuracy:.3f}")
            
            return {
                f'{dataset}_accuracy': accuracy,
                f'{dataset}_samples': total
            }
            
        except Exception as e:
            logger.error(f"{dataset} evaluation failed: {e}")
            return {f'{dataset}_accuracy': 0.0, f'{dataset}_samples': 0}

    @torch.no_grad()
    def evaluate_long_context_loss(self, model, max_files: int = None) -> Dict[str, float]:
        """
        Evaluate per-position loss on long audio files to measure TTT's long-context benefits.
        Key metric: Does loss keep decreasing with longer context (TTT) vs plateau (baseline)?
        """
        try:
            # Get paths from config
            audio_dir = self.config.get('long_context_audio_dir')
            max_files = max_files or self.config.get('long_context_max_files', 10)
            
            if not audio_dir:
                logger.warning("Long context audio directory not configured, skipping")
                return {
                    'long_context_loss_8k': 0.0,
                    'long_context_loss_16k': 0.0, 
                    'long_context_loss_24k': 0.0,
                    'long_context_slope': 0.0,
                    'long_context_samples': 0
                }
            
            if not Path(audio_dir).exists():
                logger.warning(f"Long context audio directory not found: {audio_dir}")
                return {
                    'long_context_loss_8k': 0.0,
                    'long_context_loss_16k': 0.0,
                    'long_context_loss_24k': 0.0, 
                    'long_context_slope': 0.0,
                    'long_context_samples': 0
                }
            
            # Find all "combined" wav files in directory and subdirectories
            wav_files = []
            for wav_file in Path(audio_dir).rglob("*.wav"):
                if "combined" in wav_file.name:
                    wav_files.append(wav_file)
            
            if len(wav_files) == 0:
                logger.warning(f"No 'combined' wav files found in {audio_dir}")
                return {
                    'long_context_loss_8k': 0.0,
                    'long_context_loss_16k': 0.0,
                    'long_context_loss_24k': 0.0,
                    'long_context_slope': 0.0, 
                    'long_context_samples': 0
                }
            
            # Limit number of files for evaluation
            wav_files = wav_files[:max_files]
            logger.info(f"Long context: Evaluating {len(wav_files)} combined files")
            
            # OPTION A: CONCATENATE ALL FILES INTO MEGA-SEQUENCE
            logger.info("OPTION A: Concatenating all files into mega-sequence for true long-context evaluation")
            
            # First, encode all files and collect their codes
            all_codes_list = []
            individual_lengths = []
            
            for wav_file in wav_files:
                try:
                    codes = self._encode_audio(str(wav_file))  # [1, 8, seq_len]
                    
                    if codes.shape[-1] < 1000:  # Skip very short files
                        logger.debug(f"Skipping short file: {wav_file.name} ({codes.shape[-1]} tokens)")
                        continue
                    
                    all_codes_list.append(codes)
                    individual_lengths.append(codes.shape[-1])
                    logger.info(f"Added {wav_file.name}: {codes.shape[-1]} tokens")
                    
                except Exception as e:
                    logger.warning(f"Failed to encode {wav_file.name}: {e}")
                    continue
            
            if len(all_codes_list) == 0:
                logger.warning("No valid audio files could be encoded")
                return {
                    'long_context_loss_8k': 0.0,
                    'long_context_loss_16k': 0.0,
                    'long_context_loss_24k': 0.0,
                    'long_context_slope': 0.0,
                    'long_context_samples': 0
                }
            
            # Concatenate all codes into one mega-sequence
            mega_codes = torch.cat(all_codes_list, dim=2)  # [1, 8, total_seq_len]
            total_length = mega_codes.shape[-1]
            logger.info(f"MEGA-SEQUENCE CREATED: {total_length} tokens from {len(all_codes_list)} files")
            logger.info(f"Individual lengths: {individual_lengths}")
            
            all_position_losses = []
            successful_files = len(all_codes_list)
            
            model.eval()
            
            # Process the mega-sequence as one unit
            try:
                codes = mega_codes
                
                # 2. Create 17-codebook input following paper_metrics pattern
                seq_len = codes.shape[-1]
                batch_size = codes.shape[0]
                K = codes.shape[1]  # Should be 8
                
                # Prepare input and target for next-token prediction
                input_codes = codes[:, :, :-1]  # [1, 8, seq_len-1]
                target_codes = codes[:, :, 1:]   # [1, 8, seq_len-1] 
                
                # Create 17-codebook input tensor
                inp = torch.zeros(batch_size, 17, seq_len - 1, device=codes.device, dtype=codes.dtype)
                
                # Place audio in Moshi stream (codebooks 1-8) - following standard mode
                moshi_start = 1
                inp[:, moshi_start:moshi_start+K] = input_codes
                
                # 3. STREAMING FORWARD PASS (replaces batch processing)
                if not self.librilight_streaming_enabled:
                    logger.warning("Streaming evaluation disabled - this may cause memory crashes with long sequences")
                    # Note: Could implement fallback to batch processing here if needed
                
                # Track TTT state changes for verification (if enabled)
                initial_ttt_info = None
                if self.ttt_verification_enabled:
                    initial_ttt_info = self._track_ttt_state_changes(model)
                else:
                    logger.debug("TTT verification disabled")
                
                # Check memory availability before streaming evaluation (if enabled)
                memory_ok = True
                if self.memory_check_enabled:
                    memory_ok = self._ensure_memory_availability(required_gb=4.0)
                    if not memory_ok:
                        logger.warning("Low memory detected, proceeding with caution")
                else:
                    logger.debug("Memory checking disabled")
                
                try:
                    # Simple streaming evaluation (cleaner implementation)
                    logger.info("Using simplified LibriLight evaluation")
                    loss_per_position = self._evaluate_librilight_simple(
                        model, input_codes, target_codes
                    )
                    
                    # Verify TTT learning occurred (if enabled and tracking was started)
                    final_ttt_info = None
                    if self.ttt_verification_enabled and initial_ttt_info is not None:
                        final_ttt_info = self._verify_ttt_learning(model, initial_ttt_info)
                        # Store TTT info for logging
                        self._last_ttt_info = final_ttt_info
                    else:
                        logger.debug("TTT verification skipped")
                    
                    # Store the streaming sequence loss results
                    all_position_losses.append(loss_per_position)
                    
                    # Log processing results for streaming sequence
                    non_zero_losses = [l for l in loss_per_position if l > 0]
                    logger.info(f"Processed STREAMING-SEQUENCE: {len(loss_per_position)} positions, {len(non_zero_losses)} valid losses")
                    if len(non_zero_losses) > 0:
                        logger.info(f"  Loss range: {min(non_zero_losses):.4f} - {max(non_zero_losses):.4f}")
                        # Log TTT verification results (if available)
                        if final_ttt_info and final_ttt_info.get('has_ttt', False):
                            logger.info(f"  TTT verification: {final_ttt_info['num_ttt_layers']} layers, weights_changed={final_ttt_info['weights_changed']}")
                        elif self.ttt_verification_enabled:
                            logger.debug("  TTT verification: No TTT layers detected in model")
                    else:
                        logger.warning(f"  No valid losses computed for streaming sequence")
                        
                except Exception as streaming_error:
                    logger.error(f"Streaming evaluation failed: {streaming_error}")
                    # Fallback: try to continue with error handling
                    loss_per_position = []
                    all_position_losses.append(loss_per_position)
                        
            except Exception as e:
                logger.error(f"Error processing streaming sequence: {e}")
                return {
                    'long_context_loss_8k': 0.0,
                    'long_context_loss_16k': 0.0,
                    'long_context_loss_24k': 0.0,
                    'long_context_slope': 0.0,
                    'long_context_samples': 0
                }
            
            if successful_files == 0:
                logger.warning("No files processed successfully")
                return {
                    'long_context_loss_8k': 0.0,
                    'long_context_loss_16k': 0.0,
                    'long_context_loss_24k': 0.0,
                    'long_context_slope': 0.0,
                    'long_context_samples': 0
                }
            
            # 5. Aggregate across files
            min_length = min(len(losses) for losses in all_position_losses)
            if min_length < 100:
                logger.warning(f"Sequences too short for meaningful analysis: {min_length}")
                return {
                    'long_context_loss_8k': 0.0,
                    'long_context_loss_16k': 0.0,
                    'long_context_loss_24k': 0.0,
                    'long_context_slope': 0.0,
                    'long_context_samples': successful_files
                }
            
            # Average position losses across all files
            avg_position_losses = np.mean([losses[:min_length] for losses in all_position_losses], axis=0)
            
            # 6. Enhanced Position Sampling - Extract metrics every 1k tokens
            # Determine measurement positions (every 1k tokens)
            measurement_positions = list(range(1000, min_length, 1000))
            
            # Extract position-specific metrics
            position_metrics = {}
            logged_positions = []
            
            for pos in measurement_positions:
                if pos < len(avg_position_losses):
                    position_key = f'librilight_loss_{pos//1000}k'
                    position_metrics[position_key] = float(avg_position_losses[pos])
                    logged_positions.append(f"{pos//1000}k@{pos}")
            
            logger.info(f"Position sampling: {len(measurement_positions)} positions from 1k to {min_length//1000}k tokens")
            logger.info(f"Measured positions: {', '.join(logged_positions[:10])}{'...' if len(logged_positions) > 10 else ''}")
            
            # Backward compatibility - keep original 8k, 16k, 24k metrics
            loss_8k = avg_position_losses[min(8000, min_length-1)] if min_length > 8000 else float('nan')
            loss_16k = avg_position_losses[min(16000, min_length-1)] if min_length > 16000 else float('nan')
            loss_24k = avg_position_losses[min(24000, min_length-1)] if min_length > 24000 else float('nan')
            
            # Enhanced slope analysis - multiple segments
            slopes = {}
            if min_length > 8000:
                # Overall slope from 8k to end
                start_idx = min(8000, min_length // 2)
                end_idx = min_length - 1
                if end_idx > start_idx:
                    slopes['overall'] = (avg_position_losses[end_idx] - avg_position_losses[start_idx]) / (end_idx - start_idx)
                else:
                    slopes['overall'] = 0.0
                    
                # Segment slopes to detect plateau
                if min_length > 16000:
                    slopes['early'] = (avg_position_losses[8000] - avg_position_losses[4000]) / 4000 if min_length > 8000 else 0.0
                    slopes['middle'] = (avg_position_losses[16000] - avg_position_losses[8000]) / 8000 if min_length > 16000 else 0.0
                    if min_length > 24000:
                        slopes['late'] = (avg_position_losses[24000] - avg_position_losses[16000]) / 8000
            else:
                slopes['overall'] = 0.0
            
            slope = slopes.get('overall', 0.0)  # For backward compatibility
            
            # Log comprehensive results
            key_results = [f"{k}: {v:.4f}" for k, v in position_metrics.items() if any(x in k for x in ['8k', '16k', '24k'])]
            logger.info(f"Key position results - {', '.join(key_results)}")
            logger.info(f"Slope analysis - overall: {slopes.get('overall', 0):.6f}, early: {slopes.get('early', 0):.6f}, middle: {slopes.get('middle', 0):.6f}, late: {slopes.get('late', 0):.6f}")
            
            # 7. Enhanced WandB plot data for TTT vs Baseline comparison
            try:
                import matplotlib.pyplot as plt
                
                # Detect model type for comparison
                model_type = "TTT" if hasattr(model, 'ttt_config') or any('ttt' in str(type(m)).lower() for m in model.modules()) else "Baseline"
                
                # Create enhanced plot
                plt.figure(figsize=(14, 8))
                positions = np.arange(len(avg_position_losses))
                
                # Plot main curve with model type styling
                if model_type == "TTT":
                    plt.plot(positions, avg_position_losses, linewidth=3, color='blue', 
                           label=f'{model_type} Model - Average Loss', alpha=0.8)
                else:
                    plt.plot(positions, avg_position_losses, linewidth=3, color='red', 
                           label=f'{model_type} Model - Average Loss', alpha=0.8)
                
                # Mark measurement positions (every 1k)
                for i, pos in enumerate(measurement_positions[:20]):  # Show first 20 positions
                    if pos < len(avg_position_losses) and i % 2 == 0:  # Every other position to avoid clutter
                        plt.axvline(x=pos, color='gray', linestyle=':', alpha=0.4)
                        
                # Highlight key positions
                key_positions = [8000, 16000, 24000, 32000]
                colors = ['red', 'orange', 'green', 'purple']
                for pos, color in zip(key_positions, colors):
                    if pos < min_length:
                        plt.axvline(x=pos, color=color, linestyle='--', alpha=0.7, 
                                  label=f'{pos//1000}k tokens')
                        plt.text(pos, max(avg_position_losses)*0.9, f'{pos//1000}k', 
                               rotation=90, verticalalignment='bottom', fontsize=10)
                
                plt.xlabel('Sequence Position (tokens)', fontsize=12)
                plt.ylabel('Cross-Entropy Loss', fontsize=12)
                plt.title(f'Position-wise Loss Analysis - {model_type} Model\n({successful_files} sequences, avg length: {min_length:,} tokens)', fontsize=14)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # Add text box with key statistics
                stats_text = f"Slope Analysis:\nOverall: {slopes.get('overall', 0):.6f}\nEarly (4k-8k): {slopes.get('early', 0):.6f}\nMiddle (8k-16k): {slopes.get('middle', 0):.6f}"
                if 'late' in slopes:
                    stats_text += f"\nLate (16k-24k): {slopes['late']:.6f}"
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=9)
                
                # Store enhanced plot data for wandb logging
                self._plot_data = {
                    'positions': positions.tolist(),
                    'losses': avg_position_losses.tolist(),
                    'model_type': model_type,
                    'measurement_positions': measurement_positions,
                    'position_metrics': position_metrics,
                    'slopes': slopes,
                    'figure': plt.gcf()
                }
                
                plt.tight_layout()
                plt.close()
                
            except Exception as e:
                logger.warning(f"Could not create enhanced plot: {e}")
                # Fallback to simple plot
                self._plot_data = {
                    'positions': list(range(len(avg_position_losses))),
                    'losses': avg_position_losses.tolist(),
                    'model_type': 'Unknown',
                    'measurement_positions': measurement_positions,
                    'position_metrics': position_metrics,
                    'slopes': slopes if 'slopes' in locals() else {}
                }
            
            # 8. Return comprehensive metrics
            results = {
                # Backward compatibility metrics
                'librilight_loss_8k': float(loss_8k),
                'librilight_loss_16k': float(loss_16k),
                'librilight_loss_24k': float(loss_24k),
                'librilight_slope': float(slope),
                'librilight_samples': successful_files,
                
                # Enhanced slope metrics
                'librilight_slope_overall': float(slopes.get('overall', 0)),
                'librilight_slope_early': float(slopes.get('early', 0)),
                'librilight_slope_middle': float(slopes.get('middle', 0)),
                'librilight_slope_late': float(slopes.get('late', 0)),
                
                # Context analysis
                'librilight_context_length': int(min_length),
                'librilight_positions_measured': len(measurement_positions)
            }
            
            # Add all position-specific metrics
            results.update(position_metrics)
            
            # Generate LibriLight evaluation plots automatically
            try:
                if create_librilight_plots is not None:
                    logger.info("🎨 Generating LibriLight evaluation plots...")
                    
                    # Extract position data for plotting
                    position_losses, measurement_positions = extract_position_losses_from_results(results)
                    
                    # Determine model type for plot labels
                    model_type = determine_model_type(results, self.config)
                    
                    # Create plots with timestamp
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    main_plot, detailed_plot = create_librilight_plots(
                        results=results,
                        position_losses=position_losses,
                        measurement_positions=measurement_positions,
                        model_type=model_type,
                        timestamp=timestamp
                    )
                    
                    if main_plot and detailed_plot:
                        logger.info(f"✅ LibriLight plots saved:")
                        logger.info(f"   📊 Main: {main_plot}")
                        logger.info(f"   📈 Detailed: {detailed_plot}")
                        
                        # Store plot paths in results for logging
                        results['_plot_main_path'] = str(main_plot)
                        results['_plot_detailed_path'] = str(detailed_plot)
                    else:
                        logger.warning("⚠️ LibriLight plot generation failed")
                else:
                    logger.warning("⚠️ LibriLight plotting module not available")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not generate LibriLight plots: {e}")
                # Continue without plots - don't fail the evaluation
            
            return results
            
        except Exception as e:
            logger.error(f"Error in long context evaluation: {e}")
            
            return {
                'long_context_loss_8k': 0.0,
                'long_context_loss_16k': 0.0,
                'long_context_loss_24k': 0.0,
                'long_context_slope': 0.0,
                'long_context_samples': 0
            }
            
        finally:
            # Cleanup after LibriLight evaluation
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✅ LibriLight evaluation cleanup complete")

    def _evaluate_librilight_streaming(self, model, codes, targets):
        """
        Moshi-Native token-by-token streaming evaluation for maximum TTT adaptation.
        
        IMPLEMENTATION: True Moshi streaming (S=1) with TTT online gradient descent:
        - Uses LMModel.forward() in streaming mode (enables KV cache + context window)
        - Processes exactly 1 token per step (matches Moshi's native S=1 constraint)
        - TTT configured for mini_batch_size=1 (online GD from TTT paper)
        - Maximum adaptation: each token gets individual TTT gradient update
        - Respects Moshi's streaming architecture completely
        
        Args:
            model: LMModel instance (training model) 
            codes: Input codes [1, 8, seq_len] 
            targets: Target codes [1, 8, seq_len]
            
        Returns:
            List of per-position losses
        """
        # Ensure model is in eval mode
        model.eval()
        
        # Pre-evaluation memory cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        position_losses = []
        seq_length = codes.shape[-1]
        
        # Apply sequence length limit if configured
        if seq_length > self.max_sequence_length:
            logger.warning(f"Sequence length {seq_length} exceeds limit {self.max_sequence_length}, truncating")
            seq_length = self.max_sequence_length
            codes_truncated = codes[:, :, :seq_length]
            targets_truncated = targets[:, :, :seq_length]
        else:
            codes_truncated = codes
            targets_truncated = targets
        
        logger.info(f"Starting streaming evaluation: {seq_length} tokens using LMModel streaming mode")
        
        # Check and reset any existing streaming state to prevent conflicts
        def reset_streaming_state_if_needed(model):
            """Reset streaming state if any modules are already streaming"""
            streaming_modules = []
            for name, module in model.named_modules():
                if hasattr(module, '_streaming_state') and module._streaming_state is not None:
                    streaming_modules.append(name)
            
            if streaming_modules:
                logger.warning(f"Found {len(streaming_modules)} modules already in streaming mode, resetting...")
                logger.debug(f"Streaming modules: {streaming_modules[:5]}...")  # Log first 5
                
                # Reset all streaming states
                for name, module in model.named_modules():
                    if hasattr(module, '_streaming_state') and module._streaming_state is not None:
                        module._streaming_state = None
                logger.info("All streaming states reset successfully")
                return True
            return False
        
        # Reset any existing streaming state
        had_streaming_state = reset_streaming_state_if_needed(model)
        if had_streaming_state:
            logger.info("Model streaming state was reset before evaluation")
        
        try:
            # Enter streaming mode for LMModel (enables KV cache and context window)
            with model.streaming(batch_size=1):
                logger.debug("Entered LMModel streaming mode")
                
                # Build context incrementally, using TTT-optimized chunk size for maximum efficiency
                optimal_chunk_size = self.get_optimal_ttt_chunk_size()
                chunk_size = min(optimal_chunk_size, seq_length)  # Respect sequence length limits
                
                # Log TTT efficiency metrics for this evaluation
                efficiency_info = self.calculate_ttt_efficiency(chunk_size)
                logger.info(f"🧠 TTT Chunking: chunk_size={chunk_size}, "
                           f"efficiency={efficiency_info['efficiency_percent']:.1f}%, "
                           f"mini_batches={efficiency_info['num_mini_batches']}, "
                           f"padding={efficiency_info['padding_tokens']}")
                
                if not efficiency_info['is_perfectly_aligned']:
                    logger.debug(f"⚠️  TTT Efficiency: {efficiency_info['padding_tokens']} padding tokens "
                                f"({100 - efficiency_info['efficiency_percent']:.1f}% overhead)")
                
                for chunk_start in range(0, seq_length, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, seq_length)
                    chunk_length = chunk_end - chunk_start
                    
                    # Monitor memory periodically
                    self._monitor_streaming_memory(chunk_start, seq_length)
                    
                    # Process this chunk
                    chunk_codes = codes_truncated[:, :, chunk_start:chunk_end]  # [1, 8, chunk_length]
                    chunk_targets = targets_truncated[:, :, chunk_start:chunk_end]  # [1, 8, chunk_length]
                    
                    # Create 17-codebook input (text + audio format)
                    codes_input = torch.zeros(1, 17, chunk_length, device=codes_truncated.device, dtype=codes_truncated.dtype)
                    codes_input[:, 1:9] = chunk_codes  # Place audio in codebooks 1-8
                    
                    with torch.no_grad():
                        # Use LMModel.forward() in streaming mode to get logits
                        out = model(codes=codes_input, condition_tensors=None)
                        # Extract logits for audio codebooks: [1, 8, chunk_length, vocab_size]
                        audio_logits = out.logits[:, :8, :].float()  # [1, 8, chunk_length, vocab_size]
                        
                        # Compute loss for each position in this chunk
                        for t_rel in range(chunk_length):
                            t_abs = chunk_start + t_rel
                            
                            # Get logits and target for this position
                            pos_logits = audio_logits[:, :, t_rel, :]  # [1, 8, vocab_size]
                            pos_target = chunk_targets[:, :, t_rel]   # [1, 8]
                            
                            # Compute loss for this position
                            loss = self._compute_position_loss(pos_logits, pos_target, model)
                            position_losses.append(loss)
                            
                            if t_abs % 1000 == 0:
                                logger.debug(f"Position {t_abs}: loss={loss:.4f}")
                    
                    # Memory management every 3k tokens
                    if chunk_end % 3000 < chunk_size:
                        torch.cuda.empty_cache()
                        logger.debug(f"Memory cleanup at chunk ending {chunk_end}")
            
            logger.info(f"Streaming evaluation completed: {len(position_losses)} positions processed")
            return position_losses
            
        except Exception as e:
            logger.error(f"Error in streaming evaluation: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        
        finally:
            # Final memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _compute_position_loss(self, logits, target, model):
        """
        Compute loss for single position with proper masking.
        Handles invalid tokens and ensures finite results.
        
        Args:
            logits: Model logits [1, 8, vocab_size]
            target: Target tokens [1, 8]
            model: Model (to get zero_token_id)
            
        Returns:
            Average loss for this position (float)
        """
        # Create mask for valid positions (exclude zero tokens and out-of-range)
        valid_mask = (target != model.zero_token_id) & (target >= 0) & (target < logits.size(-1))
        
        if valid_mask.sum() > 0:
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                reduction='none'
            )
            
            # Apply mask and get mean
            masked_loss = loss * valid_mask.view(-1).float()
            valid_count = valid_mask.sum().float()
            
            if valid_count > 0 and torch.isfinite(masked_loss).all():
                avg_loss = masked_loss.sum() / valid_count
                if torch.isfinite(avg_loss):
                    return avg_loss.item()
        
        return 0.0  # Fallback for invalid positions

    def _track_ttt_state_changes(self, model):
        """
        Capture TTT weights before evaluation to verify learning.
        Returns dict with state information for later verification.
        
        Args:
            model: Moshi model (may or may not have TTT layers)
            
        Returns:
            Dict with TTT state information
        """
        ttt_info = {'has_ttt': False, 'weights_changed': False}
        
        try:
            # Check if model has TTT layers
            ttt_layers = []
            for name, module in model.named_modules():
                if 'ttt' in name.lower() or hasattr(module, 'ttt_layer'):
                    ttt_layers.append((name, module))
            
            if ttt_layers:
                ttt_info['has_ttt'] = True
                ttt_info['num_ttt_layers'] = len(ttt_layers)
                
                # Capture initial weights
                initial_weights = {}
                for name, module in ttt_layers:
                    if hasattr(module, 'ttt_layer'):
                        # Store a hash or small subset of weights to detect changes
                        state_dict = module.ttt_layer.state_dict()
                        # Store checksums for memory efficiency
                        checksums = {}
                        for param_name, param_tensor in state_dict.items():
                            checksums[param_name] = torch.sum(param_tensor).item()
                        initial_weights[name] = checksums
                
                ttt_info['initial_weights'] = initial_weights
                logger.info(f"TTT tracking initialized: {len(ttt_layers)} TTT layers found")
            else:
                logger.info("No TTT layers detected in model")
                
        except Exception as e:
            logger.warning(f"Could not track TTT state: {e}")
        
        return ttt_info

    def _verify_ttt_learning(self, model, initial_ttt_info):
        """
        Verify TTT weights changed during evaluation.
        
        Args:
            model: Moshi model after evaluation
            initial_ttt_info: State info from _track_ttt_state_changes()
            
        Returns:
            Updated TTT info dict with verification results
        """
        if not initial_ttt_info['has_ttt']:
            return initial_ttt_info
        
        try:
            changes_detected = False
            changed_layers = []
            
            for name, module in model.named_modules():
                if name in initial_ttt_info['initial_weights']:
                    if hasattr(module, 'ttt_layer'):
                        # Compare checksums to detect changes
                        current_state = module.ttt_layer.state_dict()
                        initial_checksums = initial_ttt_info['initial_weights'][name]
                        
                        layer_changed = False
                        for param_name, param_tensor in current_state.items():
                            if param_name in initial_checksums:
                                current_sum = torch.sum(param_tensor).item()
                                initial_sum = initial_checksums[param_name]
                                if abs(current_sum - initial_sum) > 1e-6:  # Allow for numerical precision
                                    layer_changed = True
                                    break
                        
                        if layer_changed:
                            changes_detected = True
                            changed_layers.append(name)
            
            initial_ttt_info['weights_changed'] = changes_detected
            initial_ttt_info['changed_layers'] = changed_layers
            
            if changes_detected:
                logger.info(f"TTT learning verified: {len(changed_layers)} layers changed weights during evaluation")
            else:
                logger.warning("TTT verification: No weight changes detected - TTT may not be learning")
                
        except Exception as e:
            logger.warning(f"Could not verify TTT learning: {e}")
        
        return initial_ttt_info

    def _ensure_memory_availability(self, required_gb=4.0):
        """
        Ensure sufficient GPU memory for streaming evaluation.
        
        Args:
            required_gb: Minimum required memory in GB
            
        Returns:
            bool: True if sufficient memory available
        """
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Check available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            current_allocated = torch.cuda.memory_allocated() / (1024**3)
            available = total_memory - current_allocated
            
            logger.info(f"Memory status: {available:.1f}GB available of {total_memory:.1f}GB total")
            
            if available < required_gb:
                logger.warning(f"Low memory: {available:.1f}GB < {required_gb:.1f}GB required")
                # Additional cleanup
                gc.collect()
                torch.cuda.empty_cache()
                
                # Recheck after cleanup
                current_allocated = torch.cuda.memory_allocated() / (1024**3)
                available = total_memory - current_allocated
                logger.info(f"After cleanup: {available:.1f}GB available")
                
            return available >= required_gb
            
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
            return True  # Assume OK if check fails

    def _monitor_streaming_memory(self, step, total_steps, log_interval=None):
        """
        Monitor memory usage during streaming evaluation.
        
        Args:
            step: Current processing step
            total_steps: Total steps in sequence
            log_interval: Steps between memory logs
        """
        if log_interval is None:
            log_interval = self.memory_log_interval
            
        if step % log_interval == 0:
            try:
                current_allocated = torch.cuda.memory_allocated() / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                progress = (step / total_steps) * 100 if total_steps > 0 else 0
                
                logger.debug(f"Streaming progress {progress:.1f}% ({step}/{total_steps}): "
                           f"Current={current_allocated:.2f}GB, Peak={max_allocated:.2f}GB")
                
                # Optional: Clear cache periodically during long sequences
                if step > 0 and step % self.cache_clear_interval == 0:
                    torch.cuda.empty_cache()
                    logger.debug(f"Cleared CUDA cache at step {step} (interval={self.cache_clear_interval})")
                    
            except Exception as e:
                logger.warning(f"Memory monitoring failed at step {step}: {e}")

    @torch.no_grad()
    def evaluate_librilight_long_context(self, model) -> Dict[str, float]:
        """
        Evaluate long-context performance using LibriLight audiobook data.
        
        This follows the TTT paper methodology:
        - Uses coherent audiobook sequences (15-60 minutes)
        - Tests on multiple different books/speakers
        - Averages results across sequences for robust evaluation
        - Provides scientifically rigorous comparison
        """
        try:
            
            # Check if LibriLight is configured and available
            librilight_dir = self.config.get('librilight_audio_dir')
            if not librilight_dir or LibriLightLoader is None:
                logger.warning("LibriLight evaluation not configured or loader unavailable")
                return {
                    'librilight_loss_8k': 0.0,
                    'librilight_loss_16k': 0.0,
                    'librilight_loss_24k': 0.0,
                    'librilight_slope': 0.0,
                    'librilight_samples': 0
                }
            
            if not Path(librilight_dir).exists():
                logger.warning(f"LibriLight directory not found: {librilight_dir}")
                return {
                    'librilight_loss_8k': 0.0,
                    'librilight_loss_16k': 0.0,
                    'librilight_loss_24k': 0.0,
                    'librilight_slope': 0.0,
                    'librilight_samples': 0
                }
            
            # Get configuration
            evaluation_mode = self.config.get('librilight_evaluation_mode', 'single_book')
            num_chapters = self.config.get('librilight_max_chapters', 3)
            num_sequences = self.config.get('librilight_num_sequences', 1)
            
            logger.info(f"LibriLight evaluation mode: {evaluation_mode}, {num_chapters} chapters")
            
            # Initialize LibriLight loader
            loader = LibriLightLoader(librilight_dir)
            
            all_position_losses = []
            successful_sequences = 0
            sequence_metadata = []
            
            model.eval()
            
            # Clear GPU memory and TTT states before LibriLight evaluation
            torch.cuda.empty_cache()
            
            # TTT states already saved at function start for proper isolation
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            # Create chapter sequences (no concatenation needed)
            chapter_sequences = []
            
            if evaluation_mode == 'single_book':
                # Single specific book evaluation
                speaker_id = self.config.get('librilight_speaker_id', '100')
                book_name = self.config.get('librilight_book_name', 'emerald_city_librivox_64kb_mp3')
                
                result = loader.get_ttt_evaluation_chapters(
                    speaker_id=speaker_id,
                    book_name=book_name,
                    num_chapters=num_chapters
                )
                
                if result:
                    chapter_paths, metadata = result
                    chapter_sequences = [(chapter_paths, metadata)]
                    logger.info(f"Selected single book: '{metadata['book_title']}' by {metadata['author']}")
                else:
                    logger.error(f"Failed to get chapters for {speaker_id}/{book_name}")
                    chapter_sequences = []
            
            elif evaluation_mode == 'multi_book':
                # TTT paper style: multiple different books/speakers
                suitable_books = loader.get_suitable_books_for_ttt()
                if len(suitable_books) >= num_sequences:
                    import random
                    selected_books = random.sample(suitable_books, num_sequences)
                    
                    for book in selected_books:
                        chapters = book.get_chapters(0, num_chapters)
                        if len(chapters) >= num_chapters:
                            chapter_paths = [ch['path'] for ch in chapters]
                            metadata = {
                                'speaker_id': book.speaker_id,
                                'book_title': book.title,
                                'author': book.author,
                                'genre': book.genre,
                                'num_chapters': len(chapters)
                            }
                            chapter_sequences.append((chapter_paths, metadata))
                    
                    logger.info(f"Selected {len(chapter_sequences)} books for multi-book evaluation")
                else:
                    logger.warning(f"Only {len(suitable_books)} suitable books found, requested {num_sequences}")
                    chapter_sequences = []
            
            else:
                # Random book evaluation
                result = loader.get_random_book_sequence(num_chapters)
                if result:
                    book, chapters = result
                    chapter_paths = [ch['path'] for ch in chapters]
                    metadata = {
                        'speaker_id': book.speaker_id,
                        'book_title': book.title,
                        'author': book.author,
                        'num_chapters': len(chapters)
                    }
                    chapter_sequences = [(chapter_paths, metadata)]
                    logger.info(f"Selected random book: '{book.title}' by {book.author}")
                else:
                    chapter_sequences = []
            
            # Process each sequence
            for chapter_paths, metadata in chapter_sequences:
                try:
                    # Process each chapter and concatenate the encoded sequences
                    all_chapter_codes = []
                    
                    for chapter_path in chapter_paths:
                        # Encode each chapter using MIMI
                        chapter_codes = self._encode_audio(str(chapter_path))  # [1, 8, seq_len]
                        if chapter_codes.shape[-1] > 0:
                            all_chapter_codes.append(chapter_codes)
                        else:
                            logger.warning(f"Empty encoding for chapter: {chapter_path.name}")
                    
                    if not all_chapter_codes:
                        logger.warning(f"No valid chapters found for sequence: {metadata.get('book_title', 'Unknown')}")
                        continue
                    
                    # Concatenate all chapters along sequence dimension
                    codes = torch.cat(all_chapter_codes, dim=-1)  # [1, 8, total_seq_len]
                    
                    if codes.shape[-1] < 1000:  # Skip very short sequences
                        logger.warning(f"Skipping short sequence: {codes.shape[-1]} tokens")
                        continue
                    
                    # Create 17-codebook input following paper_metrics pattern
                    seq_len = codes.shape[-1]
                    batch_size = codes.shape[0]
                    K = codes.shape[1]  # Should be 8
                    
                    # Prepare input and target for next-token prediction
                    input_codes = codes[:, :, :-1]  # [1, 8, seq_len-1]
                    target_codes = codes[:, :, 1:]   # [1, 8, seq_len-1] 
                    
                    # Create 17-codebook input tensor
                    inp = torch.zeros(batch_size, 17, seq_len - 1, device=codes.device, dtype=codes.dtype)
                    
                    # Place audio in Moshi stream (codebooks 1-8)
                    moshi_start = 1
                    inp[:, moshi_start:moshi_start+K] = input_codes
                    
                    # STREAMING FORWARD PASS (replaces batch processing) - SECOND INSTANCE
                    logger.info(f"Processing LibriLight sequence: {codes.shape[-1]} tokens, '{metadata.get('book_title', 'Unknown')}' (STREAMING)")
                    
                    # Track TTT state changes for verification (if enabled)
                    initial_ttt_info = None
                    if self.ttt_verification_enabled:
                        initial_ttt_info = self._track_ttt_state_changes(model)
                        
                    # Check memory availability (if enabled)
                    if self.memory_check_enabled:
                        memory_ok = self._ensure_memory_availability(required_gb=4.0)
                        if not memory_ok:
                            logger.warning("Low memory detected, proceeding with caution")
                    
                    try:
                        # Simple streaming evaluation (no memory crashes)
                        logger.info("Using simplified LibriLight evaluation")
                        loss_per_position = self._evaluate_librilight_simple(
                            model, input_codes, target_codes
                        )
                        
                        # Verify TTT learning occurred (if enabled)
                        if self.ttt_verification_enabled and initial_ttt_info is not None:
                            final_ttt_info = self._verify_ttt_learning(model, initial_ttt_info)
                            # Log TTT results
                            if final_ttt_info and final_ttt_info.get('has_ttt', False):
                                logger.info(f"  TTT verification: {final_ttt_info['num_ttt_layers']} layers, weights_changed={final_ttt_info['weights_changed']}")
                                
                        # Store results only if successful
                        if len(loss_per_position) > 0:
                            all_position_losses.append(loss_per_position)
                            successful_sequences += 1
                            sequence_metadata.append(metadata)
                            logger.info(f"Successfully processed sequence: {len(loss_per_position)} losses")
                        else:
                            logger.warning("Empty loss list returned from streaming evaluation")
                        
                    except Exception as streaming_error:
                        logger.error(f"Streaming evaluation failed: {streaming_error}")
                        # Don't increment counters or store metadata for failed sequences
                        logger.warning("Skipping failed sequence")
                        
                except Exception as e:
                    logger.error(f"Error processing LibriLight sequence '{metadata.get('book_title', 'Unknown')}': {e}")
                    continue
                
                finally:
                    # LibriLight chapters are permanent files, no cleanup needed
                    pass
            
            # Aggregate results across sequences
            if successful_sequences == 0:
                logger.warning("No LibriLight sequences processed successfully")
                return {
                    'librilight_loss_8k': 0.0,
                    'librilight_loss_16k': 0.0,
                    'librilight_loss_24k': 0.0,
                    'librilight_slope': 0.0,
                    'librilight_samples': 0
                }
            
            # Average position losses across all sequences (TTT paper methodology)
            min_length = min(len(losses) for losses in all_position_losses)
            if min_length < 100:
                logger.warning(f"LibriLight sequences too short for meaningful analysis: {min_length}")
                return {
                    'librilight_loss_8k': 0.0,
                    'librilight_loss_16k': 0.0,
                    'librilight_loss_24k': 0.0,
                    'librilight_slope': 0.0,
                    'librilight_samples': successful_sequences
                }
            
            # Average across all sequences  
            import numpy as np
            avg_position_losses = np.mean([losses[:min_length] for losses in all_position_losses], axis=0)
            
            # Extract key metrics
            loss_8k = avg_position_losses[min(8000, min_length-1)] if min_length > 1000 else float('nan')
            loss_16k = avg_position_losses[min(16000, min_length-1)] if min_length > 1000 else float('nan')
            loss_24k = avg_position_losses[min(24000, min_length-1)] if min_length > 1000 else float('nan')
            
            # Calculate slope from 8k onwards
            if min_length > 8000:
                start_idx = min(8000, min_length // 2)
                end_idx = min_length - 1
                if end_idx > start_idx:
                    slope = (avg_position_losses[end_idx] - avg_position_losses[start_idx]) / (end_idx - start_idx)
                else:
                    slope = 0.0
            else:
                slope = 0.0
            
            # Log results
            pos_8k = min(8000, min_length-1) if min_length > 1000 else -1
            pos_16k = min(16000, min_length-1) if min_length > 1000 else -1  
            pos_24k = min(24000, min_length-1) if min_length > 1000 else -1
            
            logger.info(f"LibriLight positions - 8k@{pos_8k}, 16k@{pos_16k}, 24k@{pos_24k}, total_length={min_length}")
            logger.info(f"LibriLight results - 8k: {loss_8k:.4f}, 16k: {loss_16k:.4f}, 24k: {loss_24k:.4f}, slope: {slope:.6f}")
            logger.info(f"LibriLight books evaluated: {[meta.get('book_title', 'Unknown') for meta in sequence_metadata]}")
            
            # Enhanced position metrics (every 1k tokens for comprehensive analysis)
            position_metrics = {}
            measurement_positions = list(range(1000, min_length, 1000))
            
            for pos in measurement_positions:
                if pos < len(avg_position_losses):
                    position_key = f"librilight_loss_{pos}"
                    position_metrics[position_key] = float(avg_position_losses[pos])
            
            logger.info(f"Enhanced LibriLight positions: {len(measurement_positions)} positions measured")
            
            # Prepare return results with enhanced position metrics
            results = {
                'librilight_loss_8k': float(loss_8k),
                'librilight_loss_16k': float(loss_16k),
                'librilight_loss_24k': float(loss_24k),
                'librilight_slope': float(slope),
                'librilight_samples': successful_sequences
            }
            
            # Add enhanced position metrics
            results.update(position_metrics)
            
            # Log enhanced position metrics for visibility
            if position_metrics:
                enhanced_positions = sorted([k for k in position_metrics.keys() if k.startswith('librilight_loss_')])
                logger.info(f"Enhanced position metrics: {len(enhanced_positions)} positions measured")
                sample_metrics = {k: f"{position_metrics[k]:.4f}" for k in enhanced_positions[:5]}
                logger.info(f"Sample enhanced positions: {sample_metrics}")
            
            # Enhanced slope analysis - multiple segments
            slopes = {}
            if min_length > 8000:
                # Overall slope from 8k to end
                start_idx = min(8000, min_length // 2)
                end_idx = min_length - 1
                if end_idx > start_idx:
                    slopes['overall'] = (avg_position_losses[end_idx] - avg_position_losses[start_idx]) / (end_idx - start_idx)
                else:
                    slopes['overall'] = 0.0
                    
                # Segment slopes to detect plateau
                if min_length > 16000:
                    slopes['early'] = (avg_position_losses[8000] - avg_position_losses[4000]) / 4000 if min_length > 8000 else 0.0
                    slopes['middle'] = (avg_position_losses[16000] - avg_position_losses[8000]) / 8000 if min_length > 16000 else 0.0
                    if min_length > 24000:
                        slopes['late'] = (avg_position_losses[24000] - avg_position_losses[16000]) / 8000
            else:
                slopes['overall'] = 0.0
            
            # Enhanced WandB plot data for TTT vs Baseline comparison
            try:
                import matplotlib.pyplot as plt
                
                # Detect model type for comparison
                model_type = "TTT" if hasattr(model, 'ttt_config') or any('ttt' in str(type(m)).lower() for m in model.modules()) else "Baseline"
                
                # Create enhanced plot
                plt.figure(figsize=(14, 8))
                positions = np.arange(len(avg_position_losses))
                
                # Plot main curve with model type styling
                if model_type == "TTT":
                    plt.plot(positions, avg_position_losses, linewidth=3, color='blue', 
                           label=f'{model_type} Model - Average Loss', alpha=0.8)
                else:
                    plt.plot(positions, avg_position_losses, linewidth=3, color='red', 
                           label=f'{model_type} Model - Average Loss', alpha=0.8)
                
                # Mark measurement positions (every 1k)
                for i, pos in enumerate(measurement_positions[:20]):  # Show first 20 positions
                    if pos < len(avg_position_losses) and i % 2 == 0:  # Every other position to avoid clutter
                        plt.axvline(x=pos, color='gray', linestyle=':', alpha=0.4)
                        
                # Highlight key positions
                key_positions = [8000, 16000, 24000, 32000]
                colors = ['red', 'orange', 'green', 'purple']
                for pos, color in zip(key_positions, colors):
                    if pos < min_length:
                        plt.axvline(x=pos, color=color, linestyle='--', alpha=0.7, 
                                  label=f'{pos//1000}k tokens')
                        plt.text(pos, max(avg_position_losses)*0.9, f'{pos//1000}k', 
                               rotation=90, verticalalignment='bottom', fontsize=10)
                
                plt.xlabel('Sequence Position (tokens)', fontsize=12)
                plt.ylabel('Cross-Entropy Loss', fontsize=12)
                plt.title(f'LibriLight Position-wise Loss Analysis - {model_type} Model\n({successful_sequences} sequences, avg length: {min_length:,} tokens)', fontsize=14)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # Add text box with key statistics
                stats_text = f"Slope Analysis:\nOverall: {slopes.get('overall', 0):.6f}\nEarly (4k-8k): {slopes.get('early', 0):.6f}\nMiddle (8k-16k): {slopes.get('middle', 0):.6f}"
                if 'late' in slopes:
                    stats_text += f"\nLate (16k-24k): {slopes['late']:.6f}"
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=9)
                
                # Store enhanced plot data for wandb logging
                self._plot_data = {
                    'positions': positions.tolist(),
                    'losses': avg_position_losses.tolist(),
                    'model_type': model_type,
                    'measurement_positions': measurement_positions,
                    'position_metrics': position_metrics,
                    'slopes': slopes,
                    'figure': plt.gcf()
                }
                
                plt.tight_layout()
                plt.close()
                
                logger.info(f"LibriLight plot created successfully for {model_type} model")
                
            except Exception as e:
                logger.warning(f"Could not create LibriLight plot: {e}")
                # Fallback to simple plot data
                self._plot_data = {
                    'positions': list(range(len(avg_position_losses))),
                    'losses': avg_position_losses.tolist(),
                    'model_type': 'Unknown',
                    'measurement_positions': measurement_positions,
                    'position_metrics': position_metrics,
                    'slopes': slopes if 'slopes' in locals() else {}
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in LibriLight long context evaluation: {e}")
            return {
                'librilight_loss_8k': 0.0,
                'librilight_loss_16k': 0.0,
                'librilight_loss_24k': 0.0,
                'librilight_slope': 0.0,
                'librilight_samples': 0
            }
        
        finally:
            # Simple cleanup - no TTT state save/restore complexity
            logger.info("🧹 LibriLight evaluation cleanup...")
            
            # Clear gradients and free memory
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("✅ LibriLight evaluation cleanup complete")
    
    @torch.no_grad()
    def evaluate_all(self, model) -> Dict[str, float]:
        """
        Run all paper metrics evaluations using config-specified sample counts.
        Returns combined results dictionary.
        """
        model.eval()
        results = {}
        
        # Get max_pairs from config for each benchmark (using config values, not hardcoded defaults)
        sblimp_max = self.config.get('sblimp_max_pairs', 2000)
        swuggy_max = self.config.get('swuggy_max_pairs', 2000)
        tstory_max = self.config.get('tstory_max_pairs', 2000)
        sstory_max = self.config.get('sstory_max_pairs', 2000)
        
        logger.info(f"Running paper metrics evaluation:")
        logger.info(f"  sBLIMP: {sblimp_max} samples")
        logger.info(f"  sWUGGY: {swuggy_max} samples") 
        logger.info(f"  tStory: {tstory_max} samples")
        logger.info(f"  sStory: {sstory_max} samples")
        
        # Run all evaluations with config-specified sample counts
        results.update(self.evaluate_sblimp(model, sblimp_max))
        results.update(self.evaluate_swuggy(model, swuggy_max))
        results.update(self.evaluate_story_cloze(model, 'tstory', tstory_max))
        results.update(self.evaluate_story_cloze(model, 'sstory', sstory_max))
        
        # Long context evaluation is now handled by LibriLight (better methodology)
        
        # Add LibriLight evaluation (if configured)
        if self.config.get('librilight_audio_dir'):
            logger.info(f"  LibriLight: {self.config.get('librilight_evaluation_mode', 'single_book')} mode")
            results.update(self.evaluate_librilight_long_context(model))
        
        # Compute overall average (excluding sample counts)
        accuracy_keys = [k for k in results.keys() if k.endswith('_accuracy')]
        accuracies = [results[k] for k in accuracy_keys if results[k] > 0]
        
        if accuracies:
            results['paper_metrics_avg'] = sum(accuracies) / len(accuracies)
            # Calculate F1-inspired harmonic mean for more sensitive combination
            # Harmonic mean penalizes poor performance on any individual metric
            if all(acc > 0 for acc in accuracies):
                results['paper_metrics_f1'] = len(accuracies) / sum(1/acc for acc in accuracies)
            else:
                results['paper_metrics_f1'] = 0.0
        else:
            results['paper_metrics_avg'] = 0.0
            results['paper_metrics_f1'] = 0.0
        
        logger.info(f"Paper metrics average: {results['paper_metrics_avg']:.3f}, f1: {results['paper_metrics_f1']:.3f}")
        
        return results
    
    @torch.no_grad()
    def evaluate_all_without_librilight(self, model) -> Dict[str, float]:
        """
        Run fast paper metrics evaluations (excluding LibriLight).
        Designed for periodic evaluation during training to avoid computation graph contamination.
        """
        model.eval()
        results = {}
        
        # Get max_pairs from config for each benchmark
        sblimp_max = self.config.get('sblimp_max_pairs', 2000)
        swuggy_max = self.config.get('swuggy_max_pairs', 2000)
        tstory_max = self.config.get('tstory_max_pairs', 2000)
        sstory_max = self.config.get('sstory_max_pairs', 2000)
        
        logger.info(f"Running fast paper metrics evaluation (no LibriLight):")
        logger.info(f"  sBLIMP: {sblimp_max} samples")
        logger.info(f"  sWUGGY: {swuggy_max} samples") 
        logger.info(f"  tStory: {tstory_max} samples")
        logger.info(f"  sStory: {sstory_max} samples")
        
        # Run fast evaluations only
        results.update(self.evaluate_sblimp(model, sblimp_max))
        results.update(self.evaluate_swuggy(model, swuggy_max))
        results.update(self.evaluate_story_cloze(model, 'tstory', tstory_max))
        results.update(self.evaluate_story_cloze(model, 'sstory', sstory_max))
        
        # Compute overall average (excluding sample counts)
        accuracy_keys = [k for k in results.keys() if k.endswith('_accuracy')]
        accuracies = [results[k] for k in accuracy_keys if results[k] > 0]
        
        if accuracies:
            results['paper_metrics_avg'] = sum(accuracies) / len(accuracies)
            # Calculate F1-inspired harmonic mean
            if all(acc > 0 for acc in accuracies):
                results['paper_metrics_f1'] = len(accuracies) / sum(1/acc for acc in accuracies)
            else:
                results['paper_metrics_f1'] = 0.0
        else:
            results['paper_metrics_avg'] = 0.0
            results['paper_metrics_f1'] = 0.0
        
        logger.info(f"Fast paper metrics average: {results['paper_metrics_avg']:.3f}, f1: {results['paper_metrics_f1']:.3f}")
        
        return results
    
    @torch.no_grad()
    def evaluate_librilight_only(self, model) -> Dict[str, float]:
        """
        Run only LibriLight evaluation in isolation.
        Designed for end-of-training evaluation to avoid mid-training computation graph issues.
        """
        model.eval()
        results = {}
        
        # Add LibriLight evaluation (if configured)
        if self.config.get('librilight_audio_dir'):
            logger.info(f"Running LibriLight evaluation: {self.config.get('librilight_evaluation_mode', 'single_book')} mode")
            results.update(self.evaluate_librilight_long_context(model))
        else:
            logger.warning("LibriLight evaluation not configured - returning empty results")
            results = {
                'librilight_loss_8k': 0.0,
                'librilight_loss_16k': 0.0,
                'librilight_loss_24k': 0.0,
                'librilight_slope': 0.0,
                'librilight_samples': 0
            }
        
        return results
    
    def get_plot_data(self):
        """Get the plot data from the last evaluation for WandB logging"""
        return getattr(self, '_plot_data', None)


def create_paper_metrics_evaluator(mimi_encoder, interleaved_tokenizer, device: str = "cuda", config=None) -> PaperMetricsEvaluator:
    """Factory function to create paper metrics evaluator."""
    return PaperMetricsEvaluator(mimi_encoder, interleaved_tokenizer, device, config)