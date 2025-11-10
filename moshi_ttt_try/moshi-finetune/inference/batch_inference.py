#!/usr/bin/env python3
"""
Batch inference wrapper for Moshi with TTT support.

This module provides non-streaming batch inference capability, which is essential
for proper evaluation of TTT models. Unlike streaming inference (token-by-token),
batch inference processes entire sequences at once.

Key differences from streaming:
- No LMGen wrapper needed
- Calls model.forward() directly instead of model.step()
- TTT layers automatically work in batch mode
- Allows larger mini_batch_size for TTT (16-32 instead of 1)

Usage:
    from inference.batch_inference import BatchInference

    # Load model
    model = load_ttt_model(checkpoint_dir, hf_repo, device)
    mimi = checkpoint_info.get_mimi(device=device)

    # Create batch inference wrapper
    batch_inf = BatchInference(model, mimi, device=device)

    # Process audio
    audio = load_audio()  # [B, 1, samples]
    output = batch_inf.forward_audio(audio)  # Returns logits
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchInference:
    """
    Non-streaming batch inference wrapper for Moshi.

    This class provides a simple interface for batch processing, bypassing
    the LMGen streaming wrapper and calling model.forward() directly.
    """

    def __init__(
        self,
        model: nn.Module,
        mimi: nn.Module,
        text_tokenizer: Optional[object] = None,
        device: str = "cuda",
        ttt_mini_batch_size: Optional[int] = None,
    ):
        """
        Initialize batch inference wrapper.

        Args:
            model: LMModel instance (with or without TTT layers)
            mimi: MimiModel codec for audio encoding/decoding
            text_tokenizer: Optional text tokenizer (for text+audio mode)
            device: Device to run inference on
            ttt_mini_batch_size: Override TTT mini_batch_size (None = keep current)
        """
        self.model = model
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.device = device

        # Ensure model is in eval mode
        self.model.eval()

        # Ensure model is NOT in streaming mode
        self._ensure_non_streaming()

        # Configure TTT mini_batch_size if requested
        if ttt_mini_batch_size is not None:
            self._configure_ttt_batch_size(ttt_mini_batch_size)

        # Get model properties
        self.dep_q = model.dep_q  # Number of audio codebooks (8)
        self.card = model.card    # Codebook cardinality

        logger.info(f"✅ BatchInference initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Audio codebooks: {self.dep_q}")
        logger.info(f"   Codebook cardinality: {self.card}")

    def _ensure_non_streaming(self):
        """
        Ensure model is not in streaming mode.

        In streaming mode, the model maintains internal state (KV cache, offsets).
        For batch mode, we need streaming state to be None.
        """
        # Check transformer streaming state
        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, '_streaming_state'):
                if self.model.transformer._streaming_state is not None:
                    logger.warning("⚠️  Transformer is in streaming mode, disabling...")
                    self.model.transformer._streaming_state = None

        # Check depformer streaming state
        if hasattr(self.model, 'depformer') and self.model.depformer is not None:
            if hasattr(self.model.depformer, '_streaming_state'):
                if self.model.depformer._streaming_state is not None:
                    logger.warning("⚠️  Depformer is in streaming mode, disabling...")
                    self.model.depformer._streaming_state = None

        logger.info("✅ Verified non-streaming mode")

    def _configure_ttt_batch_size(self, mini_batch_size: int):
        """
        Configure TTT layers to use specified mini_batch_size.

        This is critical for TTT to work properly. In streaming mode,
        mini_batch_size=1 which gives noisy gradients. In batch mode,
        we can use larger sizes (16-32) for more stable learning.

        Args:
            mini_batch_size: Mini-batch size for TTT inner loop
        """
        ttt_layers_found = 0

        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            for i, layer in enumerate(self.model.transformer.layers):
                # Check if this is a hybrid layer with TTT
                if hasattr(layer, 'seq_modeling_block'):
                    seq_block = layer.seq_modeling_block
                    if hasattr(seq_block, 'ttt_layer'):
                        ttt_layer = seq_block.ttt_layer
                        if hasattr(ttt_layer, 'mini_batch_size'):
                            old_size = ttt_layer.mini_batch_size
                            ttt_layer.mini_batch_size = mini_batch_size
                            logger.info(f"   Layer {i}: TTT mini_batch_size {old_size} → {mini_batch_size}")
                            ttt_layers_found += 1

        if ttt_layers_found > 0:
            logger.info(f"✅ Configured {ttt_layers_found} TTT layers with mini_batch_size={mini_batch_size}")
        else:
            logger.info("ℹ️  No TTT layers found (using baseline Moshi)")

    @torch.no_grad()
    def encode_audio(self, audio_waveforms: torch.Tensor, chunk_size_seconds: int = 120) -> torch.Tensor:
        """
        Encode audio waveforms to discrete codes using MIMI codec.
        For long audio, processes in chunks to avoid OOM.

        Args:
            audio_waveforms: [B, 1, samples] raw audio at 24kHz
            chunk_size_seconds: Process audio in chunks of this many seconds (default: 120s = 2 minutes)

        Returns:
            codes: [B, 8, T] discrete audio codes
                   T = samples // frame_size (frame_size ≈ 1000)
        """
        self.mimi.eval()

        # Calculate chunk size in samples
        sample_rate = 24000
        chunk_size_samples = chunk_size_seconds * sample_rate

        B, C, total_samples = audio_waveforms.shape

        # If audio is short enough, process in one go
        if total_samples <= chunk_size_samples:
            codes = self.mimi.encode(audio_waveforms.to(self.device))
            return codes

        # Process in chunks for long audio
        logger.info(f"🔄 Audio is long ({total_samples / sample_rate:.1f}s), processing in {chunk_size_seconds}s chunks...")
        all_codes = []

        for start_idx in range(0, total_samples, chunk_size_samples):
            end_idx = min(start_idx + chunk_size_samples, total_samples)
            chunk = audio_waveforms[:, :, start_idx:end_idx]

            logger.info(f"   Encoding chunk {start_idx // chunk_size_samples + 1}: {start_idx / sample_rate:.1f}s - {end_idx / sample_rate:.1f}s")
            chunk_codes = self.mimi.encode(chunk.to(self.device))
            all_codes.append(chunk_codes.cpu())  # Move to CPU to save GPU memory

            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all chunks along time dimension
        codes = torch.cat(all_codes, dim=2).to(self.device)
        logger.info(f"✅ Encoded {len(all_codes)} chunks, total codes shape: {codes.shape}")

        return codes

    @torch.no_grad()
    def decode_audio(self, codes: torch.Tensor, chunk_size_frames: int = 3000) -> torch.Tensor:
        """
        Decode discrete codes to audio waveforms using MIMI codec.
        For long sequences, processes in chunks to avoid OOM.

        Args:
            codes: [B, 8, T] discrete audio codes
            chunk_size_frames: Process in chunks of this many frames (default: 3000 frames ≈ 120s)

        Returns:
            audio_waveforms: [B, 1, T*frame_size] raw audio at 24kHz
        """
        self.mimi.eval()

        B, C, total_frames = codes.shape

        # If codes are short enough, process in one go
        if total_frames <= chunk_size_frames:
            audio = self.mimi.decode(codes.to(self.device))
            return audio

        # Process in chunks for long sequences
        logger.info(f"🔄 Codes are long ({total_frames} frames), decoding in {chunk_size_frames} frame chunks...")
        all_audio = []

        for start_idx in range(0, total_frames, chunk_size_frames):
            end_idx = min(start_idx + chunk_size_frames, total_frames)
            chunk_codes = codes[:, :, start_idx:end_idx]

            logger.info(f"   Decoding chunk {start_idx // chunk_size_frames + 1}: frames {start_idx} - {end_idx}")
            chunk_audio = self.mimi.decode(chunk_codes.to(self.device))
            all_audio.append(chunk_audio.cpu())  # Move to CPU to save GPU memory

            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all chunks along time dimension
        audio = torch.cat(all_audio, dim=2).to(self.device)
        logger.info(f"✅ Decoded {len(all_audio)} chunks, total audio shape: {audio.shape}")

        return audio

    @torch.no_grad()
    def forward(
        self,
        codes: torch.Tensor,
        condition_tensors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch forward pass through the model.

        This is the core method that processes entire sequences at once,
        bypassing the streaming LMGen wrapper.

        Args:
            codes: [B, K, T] input codes where:
                   - K = 17 codebooks (1 text + 8 input audio + 8 output audio)
                   - T = sequence length
                   Typically, for evaluation:
                   - codes[:, 0, :] = text tokens
                   - codes[:, 1:9, :] = input audio (user)
                   - codes[:, 9:17, :] = output audio (model generates these)
            condition_tensors: Optional conditioning information

        Returns:
            Dictionary with:
                'logits': [B, dep_q, T, card] - audio logits
                'text_logits': [B, 1, T, text_card] - text logits
                'mask': [B, dep_q, T] - valid positions
                'text_mask': [B, 1, T] - valid text positions
        """
        self.model.eval()

        # Ensure codes are on correct device
        codes = codes.to(self.device)
        if condition_tensors is not None:
            condition_tensors = condition_tensors.to(self.device)

        # Call model.forward() directly
        # This is the key difference from streaming: we process all T timesteps at once
        output = self.model(
            codes=codes,
            condition_tensors=condition_tensors
        )

        # Return output as dictionary for easier access
        return {
            'logits': output.logits,           # [B, dep_q, T, card]
            'text_logits': output.text_logits, # [B, 1, T, text_card]
            'mask': output.mask,               # [B, dep_q, T]
            'text_mask': output.text_mask,     # [B, 1, T]
        }

    @torch.no_grad()
    def forward_audio(
        self,
        audio_waveforms: torch.Tensor,
        include_text: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end: audio waveforms → encode → forward → logits.

        This is a convenience method that handles encoding and prepares
        the 17-codebook input format automatically.

        Args:
            audio_waveforms: [B, 1, samples] raw audio at 24kHz
            include_text: If True, include text logits in output

        Returns:
            Dictionary with logits and masks (same as forward())
        """
        # Step 1: Encode audio to codes
        audio_codes = self.encode_audio(audio_waveforms)  # [B, 8, T]

        B, K_audio, T = audio_codes.shape
        assert K_audio == self.dep_q, f"Expected {self.dep_q} audio codebooks, got {K_audio}"

        # Step 2: Prepare 17-codebook input
        # Moshi expects: [B, 17, T]
        # - Codebook 0: text tokens (we'll use zeros for audio-only)
        # - Codebooks 1-8: input audio (user) - this is what we have
        # - Codebooks 9-16: output audio (model) - zeros (model doesn't see these during training)
        codes = torch.zeros(B, 17, T, device=self.device, dtype=audio_codes.dtype)
        codes[:, 1:9, :] = audio_codes  # Put input audio in codebooks 1-8

        # Step 3: Forward pass
        output = self.forward(codes)

        # Step 4: Return relevant parts
        result = {
            'logits': output['logits'],  # [B, dep_q, T, card]
            'mask': output['mask'],      # [B, dep_q, T]
        }

        if include_text:
            result['text_logits'] = output['text_logits']
            result['text_mask'] = output['text_mask']

        return result

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        codebook_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for audio generation.

        Args:
            logits: [B, dep_q, T, card] predictions
            targets: [B, dep_q, T] target codes
            mask: [B, dep_q, T] valid positions (optional)
            reduction: 'mean', 'sum', or 'none'
            codebook_weights: [dep_q] weights per codebook (first codebook typically weighted higher)

        Returns:
            loss: scalar tensor if reduction != 'none', else [B, dep_q, T]
        """
        B, dep_q, T, card = logits.shape

        # Reshape for cross_entropy
        logits_flat = logits.reshape(B * dep_q * T, card)  # [B*dep_q*T, card]
        targets_flat = targets.reshape(B * dep_q * T)      # [B*dep_q*T]

        # Compute loss (per-token)
        loss_flat = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none'
        )  # [B*dep_q*T]

        # Reshape back
        loss = loss_flat.reshape(B, dep_q, T)  # [B, dep_q, T]

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask

        # Apply codebook weights if provided
        if codebook_weights is not None:
            # codebook_weights: [dep_q] → [1, dep_q, 1]
            weights = codebook_weights.view(1, dep_q, 1).to(loss.device)
            loss = loss * weights

        # Apply reduction
        if reduction == 'mean':
            if mask is not None:
                # Mean over valid positions only
                return loss.sum() / mask.sum().clamp(min=1)
            else:
                return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def evaluate_perplexity(
        self,
        audio_waveforms: torch.Tensor,
        target_codes: Optional[torch.Tensor] = None,
        codebook_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate perplexity on audio sequence.

        Args:
            audio_waveforms: [B, 1, samples] input audio
            target_codes: [B, dep_q, T] target codes (if None, will encode audio)
            codebook_weights: [dep_q] weights per codebook

        Returns:
            Dictionary with:
                'perplexity': Overall perplexity
                'loss': Average loss
                'per_codebook_perplexity': [dep_q] perplexity per codebook
        """
        # Encode audio if target_codes not provided
        if target_codes is None:
            target_codes = self.encode_audio(audio_waveforms)

        # Forward pass
        output = self.forward_audio(audio_waveforms)
        logits = output['logits']  # [B, dep_q, T, card]
        mask = output['mask']      # [B, dep_q, T]

        # Compute loss
        loss = self.compute_loss(
            logits,
            target_codes,
            mask=mask,
            reduction='mean',
            codebook_weights=codebook_weights,
        )

        # Compute perplexity
        perplexity = torch.exp(loss).item()

        # Per-codebook perplexity
        per_codebook_loss = []
        for q in range(self.dep_q):
            loss_q = self.compute_loss(
                logits[:, q:q+1, :, :],
                target_codes[:, q:q+1, :],
                mask=mask[:, q:q+1, :] if mask is not None else None,
                reduction='mean',
            )
            per_codebook_loss.append(loss_q.item())

        per_codebook_perplexity = [torch.exp(torch.tensor(l)).item() for l in per_codebook_loss]

        return {
            'perplexity': perplexity,
            'loss': loss.item(),
            'per_codebook_perplexity': per_codebook_perplexity,
        }


def load_batch_inference(
    checkpoint_dir: Path,
    hf_repo: str,
    device: str = "cuda",
    ttt_mini_batch_size: Optional[int] = None,
) -> BatchInference:
    """
    Convenience function to load model and create BatchInference wrapper.

    Args:
        checkpoint_dir: Path to consolidated checkpoint directory
        hf_repo: HuggingFace repository ID for base Moshi model
        device: Device to load model on
        ttt_mini_batch_size: TTT mini_batch_size override (None = keep default)

    Returns:
        BatchInference instance ready to use
    """
    # Import here to avoid circular dependency
    from inference.run_inference_with_ttt import load_ttt_model
    from moshi.models.loaders import CheckpointInfo

    # Load model with TTT
    logger.info("Loading model with TTT...")
    model = load_ttt_model(checkpoint_dir, hf_repo, device)

    # Load MIMI codec
    logger.info("Loading MIMI codec...")
    checkpoint_info = CheckpointInfo.from_hf_repo(hf_repo)
    mimi = checkpoint_info.get_mimi(device=device)

    # Create BatchInference wrapper
    logger.info("Creating BatchInference wrapper...")
    batch_inf = BatchInference(
        model=model,
        mimi=mimi,
        device=device,
        ttt_mini_batch_size=ttt_mini_batch_size,
    )

    logger.info("✅ BatchInference ready!")
    return batch_inf
