"""
Centralized audio encoding cache system for MIMI tokens and silence codes.

This module provides efficient caching for audio encoding operations to avoid
redundant MIMI encoding during evaluation. Supports file-based persistence
and intelligent cache invalidation based on file modification times.

Used by the paper evaluation pipeline to cache 15,000+ audio encodings
from sBLIMP, sWUGGY, and other textless NLP datasets.
"""

import logging
import os
import pickle
from typing import Optional, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from .loss import compute_loss_with_mask

logger = logging.getLogger("audio_cache")


def safe_logger_info(message: str) -> None:
    """Safe logging that falls back to print if logger not available"""
    try:
        # Check if distributed is initialized first
        import torch.distributed as dist
        if dist.is_initialized():
            from .distributed import get_rank
            if get_rank() == 0:
                logger.info(message)
        else:
            # Single process mode - just log normally
            logger.info(message)
    except (ImportError, AttributeError, ValueError, RuntimeError):
        # Fall back to regular logging or print
        try:
            logger.info(message)
        except:
            print(message)


class AudioEncodingCache:
    """
    Cache for encoded audio files to avoid re-encoding identical files.
    
    Features:
    - File-based persistence with pickle
    - Modification time validation for cache invalidation
    - GPU memory optimization (stores on CPU, transfers to GPU)
    - Batch processing support
    """
    
    def __init__(self, cache_file: str = "audio_encoding_cache.pkl"):
        self.cache = {}
        self.cache_file = cache_file
        self.cache_dirty = False
        self.load_cache_from_disk()
    
    def load_cache_from_disk(self):
        """Load audio encoding cache from disk if it exists"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
                
                # Validate cache data structure
                if not isinstance(loaded_cache, dict):
                    raise ValueError("Cache file contains invalid data - not a dictionary")
                
                self.cache = loaded_cache
                cache_size_mb = os.path.getsize(self.cache_file) / (1024 * 1024)
                safe_logger_info(f"📊 Loaded {len(self.cache):,} audio encodings from cache ({cache_size_mb:.1f}MB): {self.cache_file}")
                if len(self.cache) > 10000:
                    safe_logger_info("🎯 Large cache detected - excellent coverage for sBLIMP/sWUGGY evaluation!")
            else:
                safe_logger_info(f"No audio encoding cache found at {self.cache_file}, starting with empty cache")
        except Exception as e:
            safe_logger_info(f"Failed to load audio encoding cache from {self.cache_file}: {e}")
            self.cache = {}
    
    def save_cache_to_disk(self, force: bool = False):
        """Save audio encoding cache to disk if dirty or forced"""
        if not (self.cache_dirty or force):
            return
            
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            self.cache_dirty = False
            safe_logger_info(f"Saved {len(self.cache)} audio encodings to cache file: {self.cache_file}")
        except Exception as e:
            safe_logger_info(f"Failed to save audio encoding cache to {self.cache_file}: {e}")
    
    def get_cache_key(self, audio_file: str) -> Tuple[str, float, int]:
        """Generate cache key for audio file (file path + modification time)"""
        try:
            stat = os.stat(audio_file)
            return (audio_file, stat.st_mtime, stat.st_size)
        except:
            return (audio_file, 0, 0)  # Fallback if stat fails
    
    def get_encoded_audio(self, audio_file: str, mimi_model, device) -> Optional[torch.Tensor]:
        """Get cached audio encoding or encode and cache"""
        cache_key = self.get_cache_key(audio_file)
        
        if cache_key in self.cache:
            # Return cached encoding on correct device
            cached_codes = self.cache[cache_key]
            if str(cached_codes.device) != str(device):
                cached_codes = cached_codes.to(device, non_blocking=True)
            return cached_codes
        
        # Need to encode this file
        try:
            # Check file size and use chunked encoding for large files (>30 minutes)
            import os
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            
            if file_size_mb > 50:  # ~30-45 minutes at typical quality
                safe_logger_info(f"Large audio file detected ({file_size_mb:.1f}MB), using chunked encoding: {audio_file}")
                return self._encode_audio_chunked(audio_file, mimi_model, device)
            else:
                return self._encode_audio_simple(audio_file, mimi_model, device)
                
        except Exception as e:
            logger.warning(f"Failed to encode audio {audio_file}: {e}")
            return None
    
    def _encode_audio_simple(self, audio_file: str, mimi_model, device) -> Optional[torch.Tensor]:
        """Simple encoding for smaller files (original method)"""
        try:
            waveform, sr = torchaudio.load(audio_file)
            
            # Resample if needed (Moshi expects 24kHz)
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                waveform = resampler(waveform)
            
            waveform = waveform.to(device).unsqueeze(0)
            with torch.no_grad():
                # Store on CPU to save GPU memory
                encoded = mimi_model.encode(waveform).cpu()
                cache_key = self.get_cache_key(audio_file)
                self.cache[cache_key] = encoded
                self.cache_dirty = True
            
            # Return on correct device
            return encoded.to(device, non_blocking=True)
            
        except Exception as e:
            raise e
    
    def _encode_audio_chunked(self, audio_file: str, mimi_model, device) -> Optional[torch.Tensor]:
        """Chunked encoding for large files to avoid GPU OOM"""
        try:
            waveform, sr = torchaudio.load(audio_file)
            
            # Resample if needed (Moshi expects 24kHz)
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                waveform = resampler(waveform)
            
            # Split into 10-minute chunks (240k samples at 24kHz)
            chunk_size = 24000 * 60 * 10  # 10 minutes
            total_samples = waveform.shape[1]
            
            encoded_chunks = []
            num_chunks = (total_samples + chunk_size - 1) // chunk_size
            
            safe_logger_info(f"Processing {num_chunks} chunks of ~10 minutes each")
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_samples)
                
                chunk = waveform[:, start_idx:end_idx]
                chunk = chunk.to(device).unsqueeze(0)
                
                with torch.no_grad():
                    # Clear GPU cache before each chunk
                    torch.cuda.empty_cache()
                    
                    # Encode chunk
                    encoded_chunk = mimi_model.encode(chunk).cpu()
                    encoded_chunks.append(encoded_chunk)
                    
                    # Clear chunk from GPU immediately
                    del chunk
                    torch.cuda.empty_cache()
            
            # Concatenate all chunks along the time dimension
            encoded = torch.cat(encoded_chunks, dim=-1)
            
            # Cache the result
            cache_key = self.get_cache_key(audio_file)
            self.cache[cache_key] = encoded
            self.cache_dirty = True
            
            safe_logger_info(f"Chunked encoding complete: {encoded.shape}")
            
            # Return on correct device
            return encoded.to(device, non_blocking=True)
            
        except Exception as e:
            raise e
    
    def flush_cache(self):
        """Force save cache to disk if dirty"""
        self.save_cache_to_disk(force=True)


class SilenceCodeCache:
    """
    Cache for silence codes to avoid regenerating identical silence patterns.
    
    Silence codes are used in certain evaluation configurations where one
    audio stream contains real audio and another contains silence.
    """
    
    def __init__(self, cache_file: str = "silence_codes_cache.pkl"):
        self.cache = {}
        self.cache_file = cache_file
        self.cache_dirty = False
        self.load_cache_from_disk()
    
    def load_cache_from_disk(self):
        """Load silence codes cache from disk if it exists"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
                
                # Validate cache data structure
                if not isinstance(loaded_cache, dict):
                    raise ValueError("Cache file contains invalid data - not a dictionary")
                
                # Validate cache entries
                for key, _ in loaded_cache.items():
                    if not isinstance(key, tuple) or len(key) != 2:
                        raise ValueError(f"Invalid cache key format: {key}")
                
                self.cache = loaded_cache
                cache_size_mb = os.path.getsize(self.cache_file) / (1024 * 1024)
                safe_logger_info(f"📊 Loaded {len(self.cache):,} silence patterns from cache ({cache_size_mb:.1f}MB): {self.cache_file}")
            else:
                safe_logger_info(f"No silence cache found at {self.cache_file}, starting with empty cache")
        except Exception as e:
            safe_logger_info(f"Failed to load silence cache from {self.cache_file}: {e}")
            self.cache = {}
    
    def save_cache_to_disk(self, force: bool = False):
        """Save silence codes cache to disk if dirty or forced"""
        if not (self.cache_dirty or force):
            return
            
        try:
            # Add validation of cache data
            if not isinstance(self.cache, dict):
                raise ValueError("Cache data is corrupted - not a dictionary")
                
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            self.cache_dirty = False
            safe_logger_info(f"Saved {len(self.cache)} silence codes to cache file: {self.cache_file}")
        except Exception as e:
            safe_logger_info(f"Failed to save silence cache to {self.cache_file}: {e}")
    
    def flush_cache(self):
        """Force save cache to disk if dirty"""
        self.save_cache_to_disk(force=True)
    
    def get_silence_codes(self, target_shape: Tuple[int, int, int], mimi_model, device) -> torch.Tensor:
        """Get cached silence codes or generate new ones"""
        key = (target_shape[1], target_shape[2])  # (K, T) - codebooks and time
        if key not in self.cache:
            B, K, T = target_shape
            frame_count = T
            samples_needed = frame_count * 1920  # 1920 samples per frame at 12.5 Hz
            
            silence = torch.zeros(
                (B, mimi_model.channels, samples_needed),
                device=device,
                dtype=torch.float32
            )
            
            with torch.no_grad():
                # Generate and store on CPU to save GPU memory
                silence_codes = mimi_model.encode(silence).cpu()
                self.cache[key] = silence_codes
                self.cache_dirty = True
            
            safe_logger_info(f"Generated and cached silence codes for shape (K={K}, T={T})")
        
        # Return codes on the correct device
        cached_codes = self.cache[key]
        if str(cached_codes.device) != str(device):
            cached_codes = cached_codes.to(device, non_blocking=True)
        return cached_codes


# Global cache instances for backward compatibility
_audio_encoding_cache = None
_silence_cache = None


def get_audio_cache(cache_file: str = "audio_encoding_cache.pkl") -> AudioEncodingCache:
    """Get or create global audio encoding cache instance"""
    global _audio_encoding_cache
    if _audio_encoding_cache is None:
        _audio_encoding_cache = AudioEncodingCache(cache_file)
    return _audio_encoding_cache


def get_silence_cache(cache_file: str = "silence_codes_cache.pkl") -> SilenceCodeCache:
    """Get or create global silence code cache instance"""
    global _silence_cache
    if _silence_cache is None:
        _silence_cache = SilenceCodeCache(cache_file)
    return _silence_cache


def batch_encode_audio(audio_files, mimi_model, device, cache_file: str = "audio_encoding_cache.pkl"):
    """
    Batch encode multiple audio files efficiently using persistent cache.
    
    Args:
        audio_files: List of audio file paths
        mimi_model: MIMI encoder model
        device: Target device for encodings
        cache_file: Cache file path
        
    Returns:
        Dict mapping audio file paths to encoded tensors
    """
    audio_cache = get_audio_cache(cache_file)
    safe_logger_info(f"Batch encoding {len(audio_files)} audio files...")
    
    audio_codes = {}
    cache_hits = 0
    cache_misses = 0
    
    for audio_file in audio_files:
        cache_key = audio_cache.get_cache_key(audio_file)
        was_cached = cache_key in audio_cache.cache
        
        encoded = audio_cache.get_encoded_audio(audio_file, mimi_model, device)
        if encoded is not None:
            audio_codes[audio_file] = encoded
            if was_cached:
                cache_hits += 1
            else:
                cache_misses += 1
        else:
            logger.warning(f"Failed to encode audio file: {audio_file}")
            cache_misses += 1
    
    safe_logger_info(f"Batch encoded {len(audio_codes)} audio files successfully (cache hits: {cache_hits}, misses: {cache_misses})")
    
    # Save cache periodically to avoid losing work
    if cache_misses > 0:
        audio_cache.save_cache_to_disk()
    
    return audio_codes


def compute_likelihood_fast(
    audio_codes: torch.Tensor,
    moshi_model,
    mimi_model,
    device,
    use_silence_codes: bool = False,
    silence_cache_file: str = "silence_codes_cache.pkl",
    stream_config: str | None = None,
    semantic_weight: float = 100.0,
) -> float:
    """
    Fast likelihood computation using cached silence codes.
    
    This is the paper-exact methodology from the eval_sblimp implementation.
    """
    silence_cache = get_silence_cache(silence_cache_file)
    B, K, T = audio_codes.shape
    
    # Prepare input for audio-only evaluation
    num_codebooks = moshi_model.num_codebooks
    # Optimized: Create tensor directly on device
    input_codes = torch.full(
        (B, num_codebooks, T),
        moshi_model.zero_token_id,
        device=device,
        dtype=audio_codes.dtype
    ).contiguous()
    
    # Place audio codes with optional cached silence and stream configuration
    moshi_start = moshi_model.audio_offset
    user_start = moshi_model.audio_offset + moshi_model.dep_q

    # Normalize stream_config
    mode = (stream_config or "moshi").lower()
    if mode in ("original", "user"):
        # Audio in moshi stream (matches eval_paper methodology)
        input_codes[:, moshi_start:moshi_start + K] = audio_codes
    elif mode in ("moshi_silence", "user_silence") or use_silence_codes:
        # Dual-stream with silence in the other stream
        # Prefer explicit modes; if mode==user_silence place audio in user stream
        silence_codes = silence_cache.get_silence_codes(audio_codes.shape, mimi_model, device)
        if mode == "user_silence":
            input_codes[:, user_start:user_start + K] = audio_codes
            input_codes[:, moshi_start:moshi_start + K] = silence_codes
        else:
            # Default to moshi_silence
            input_codes[:, moshi_start:moshi_start + K] = audio_codes
            input_codes[:, user_start:user_start + K] = silence_codes
    elif mode == "duplicated":
        # Audio in both streams
        input_codes[:, moshi_start:moshi_start + K] = audio_codes
        input_codes[:, user_start:user_start + K] = audio_codes
    else:
        # Default: place audio in moshi stream only
        input_codes[:, moshi_start:moshi_start + K] = audio_codes
    
    # Forward pass - use same call as eval loss with condition_tensors
    with torch.no_grad():
        # Use fp32 autocast to avoid BF16 GEMM edge cases in eval
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float32):
            # Disable dynamo compilation during audio encoding to prevent FX tracing conflicts
            with torch._dynamo.disable():
                output = moshi_model(codes=input_codes, condition_tensors=None)
    
    # Compute loss using canonical helper (aligns with training/eval logic)
    K_eff = min(K, moshi_model.dep_q)
    loss = compute_loss_with_mask(
        output.logits[:, :K_eff],
        audio_codes[:, :K_eff],
        output.mask[:, :K_eff],
        mode="audio",
        first_codebook_weight_multiplier=float(semantic_weight),
        text_padding_weight=0.0,
        text_padding_ids=set(),
    )
    return float(loss.item())


def compute_likelihood_fast_batch(
    audio_code_list: list[torch.Tensor],
    moshi_model,
    mimi_model,
    device,
    use_silence_codes: bool = False,
    silence_cache_file: str = "silence_codes_cache.pkl",
    stream_config: str | None = None,
    semantic_weight: float = 100.0,
) -> torch.Tensor:
    """Vectorized likelihood for a list of [1, K, T_i] tensors.

    Returns: Tensor of shape [B] with per-sample NLLs.
    """
    if len(audio_code_list) == 0:
        return torch.empty(0, device=device)

    silence_cache = get_silence_cache(silence_cache_file)
    K = audio_code_list[0].shape[1]
    max_T = max(int(x.shape[2]) for x in audio_code_list)
    B = len(audio_code_list)

    # Prepare input tensor padded with zero_token_id
    num_codebooks = moshi_model.num_codebooks
    input_codes = torch.full(
        (B, num_codebooks, max_T),
        moshi_model.zero_token_id,
        device=device,
        dtype=audio_code_list[0].dtype,
    ).contiguous()

    moshi_start = moshi_model.audio_offset
    user_start = moshi_model.audio_offset + moshi_model.dep_q
    mode = (stream_config or "moshi").lower()

    # Optionally get silence for max_T and slice
    silence_codes_max = None
    if mode in ("moshi_silence", "user_silence") or use_silence_codes:
        silence_codes_max = silence_cache.get_silence_codes((1, K, max_T), mimi_model, device)

    # Place per-sample codes
    for i, codes in enumerate(audio_code_list):
        T_i = codes.shape[2]
        if mode in ("original", "user"):
            input_codes[i, moshi_start:moshi_start + K, :T_i] = codes[0]
        elif mode == "user_silence":
            input_codes[i, user_start:user_start + K, :T_i] = codes[0]
            if silence_codes_max is not None:
                input_codes[i, moshi_start:moshi_start + K, :T_i] = silence_codes_max[0, :, :T_i]
        elif mode == "duplicated":
            input_codes[i, moshi_start:moshi_start + K, :T_i] = codes[0]
            input_codes[i, user_start:user_start + K, :T_i] = codes[0]
        else:  # default moshi or moshi_silence
            input_codes[i, moshi_start:moshi_start + K, :T_i] = codes[0]
            if mode == "moshi_silence" and silence_codes_max is not None:
                input_codes[i, user_start:user_start + K, :T_i] = silence_codes_max[0, :, :T_i]

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float32):
            # Disable dynamo compilation during batch audio encoding to prevent FX tracing conflicts  
            with torch._dynamo.disable():
                output = moshi_model(codes=input_codes, condition_tensors=None)

    # Extract targets from the correct stream positions where we placed audio codes
    K_eff = min(K, moshi_model.dep_q)
    
    # Determine which stream to extract targets from based on mode
    if mode in ("original", "user"):
        target_stream_start = user_start
    elif mode == "user_silence":
        target_stream_start = user_start
    elif mode == "duplicated":
        target_stream_start = moshi_start  # Use moshi stream for consistency
    else:  # default moshi or moshi_silence
        target_stream_start = moshi_start

    # Build batched targets and masks for true batching
    batch_targets = torch.full(
        (B, K_eff, max_T),
        0,  # Will be masked out anyway
        device=device,
        dtype=input_codes.dtype
    )
    batch_masks = torch.zeros((B, K_eff, max_T), device=device, dtype=torch.bool)
    
    # Fill in actual targets and masks for each sample
    for i, codes in enumerate(audio_code_list):
        T_i = codes.shape[2]
        K_i = min(codes.shape[1], K_eff)
        batch_targets[i, :K_i, :T_i] = codes[0, :K_i, :T_i]
        batch_masks[i, :K_i, :T_i] = True

    # Extract logits and masks from correct stream position
    batch_logits = output.logits[:, target_stream_start:target_stream_start + K_eff, :max_T]
    batch_output_masks = output.mask[:, target_stream_start:target_stream_start + K_eff, :max_T]
    
    # Combine target masks with output masks
    final_masks = batch_masks & batch_output_masks

    # Compute batched loss using canonical helper
    batch_loss = compute_loss_with_mask(
        batch_logits,
        batch_targets,  
        final_masks,
        mode="audio",
        first_codebook_weight_multiplier=float(semantic_weight),
        text_padding_weight=0.0,
        text_padding_ids=set(),
    )
    
    # For individual sample losses, we need to compute per-sample
    # This is necessary because the evaluator expects per-sample comparisons
    losses = []
    for i in range(B):
        sample_loss = compute_loss_with_mask(
            batch_logits[i:i+1],
            batch_targets[i:i+1],
            final_masks[i:i+1],
            mode="audio", 
            first_codebook_weight_multiplier=float(semantic_weight),
            text_padding_weight=0.0,
            text_padding_ids=set(),
        )
        losses.append(sample_loss.detach().to(device=device, dtype=torch.float32))
    
    return torch.stack(losses).view(-1)