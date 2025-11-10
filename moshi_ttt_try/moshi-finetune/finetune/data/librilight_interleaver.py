"""
LibriLight-compatible interleaver for audio-only training.

LibriLight dataset has:
- Mono audio (1 channel)
- No text transcripts
- Long audio segments (~1000-1500 seconds)

This module handles:
- Converting mono → stereo (duplicate channel)
- Zero-filled text stream (no text training)
- Standard Mimi audio encoding
"""

import math
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioOnlySample:
    """
    Sample for audio-only training (no text).

    Attributes:
        codes: Token tensor [1, num_codebooks+1, T]
               First row is text (all zeros), rest are audio
        condition_attributes: Always None for LibriLight
        file_id: Source file path for continuous RoPE tracking
        chunk_index: Chunk position within file
    """
    codes: torch.Tensor
    condition_attributes: None = None
    file_id: str = None
    chunk_index: int = None


class LibriLightInterleaver:
    """
    Interleaver for LibriLight dataset (audio-only, no text).

    Creates token streams with:
    - Text stream: all zeros (no text)
    - Audio streams: normal Mimi encoding
    """

    def __init__(self, text_padding: int = 3):
        """
        Initialize LibriLight interleaver.

        Args:
            text_padding: Token ID for text padding (typically 3)
                         Used to fill text stream for audio-only training
        """
        import logging
        logger = logging.getLogger("dataset")
        logger.info(f"🔍 LibriLightInterleaver initialized with text_padding={text_padding}")
        logger.info(f"🔍 This value will be used for the text stream (audio-only training)")
        self.text_padding = text_padding

    def prepare_item(self, num_audio_frames: int, device: str = "cuda") -> torch.Tensor:
        """
        Create text token stream (all text padding for audio-only training).

        Args:
            num_audio_frames: Number of audio frames
            device: Device to create tensor on

        Returns:
            Text tokens [1, 1, T] filled with text_padding (token ID 3)
        """
        text_tokens = torch.full(
            (1, 1, num_audio_frames),
            self.text_padding,
            dtype=torch.long,
            device=device
        )
        return text_tokens


class LibriLightTokenizer:
    """
    Tokenizer for LibriLight that processes audio-only (no text).

    This tokenizer:
    1. Takes mono audio and duplicates to stereo
    2. Encodes audio with Mimi codec
    3. Creates zero-filled text stream
    4. Returns combined token stream for training
    """

    def __init__(self, mimi, interleaver: LibriLightInterleaver, duration_sec: float):
        """
        Initialize LibriLight tokenizer.

        Args:
            mimi: Mimi audio codec
            interleaver: LibriLight interleaver
            duration_sec: Sequence duration in seconds
        """
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        # Track chunks for continuous RoPE
        self._file_chunk_map = {}  # Map file_id -> chunk_index

    def __call__(self, wav: np.ndarray, start_sec: float, path: str) -> AudioOnlySample | None:
        """
        Process audio segment for audio-only training.

        Args:
            wav: Audio array [channels, samples]
            start_sec: Start time (unused for LibriLight, kept for compatibility)
            path: File path (unused, kept for compatibility)

        Returns:
            AudioOnlySample with codes [1, num_codebooks+1, T] where:
            - First row: text tokens (all zeros)
            - Remaining rows: audio tokens from Mimi
            Or None if processing fails
        """
        with torch.no_grad():
            try:
                # Convert numpy to tensor and move to GPU
                audio_tensor = torch.from_numpy(wav).float().cuda()

                # If mono, duplicate to stereo for Moshi's stereo input
                # This creates "self-supervised" training where model continues its own audio
                if audio_tensor.shape[0] == 1:
                    audio_tensor = audio_tensor.repeat(2, 1)  # [1, samples] → [2, samples]

                # Encode with Mimi codec
                # Input: [2, samples]
                # Output: [2, num_codebooks, T] where T = num_audio_frames
                audio_tokens = self.mimi.encode(audio_tensor[:, None])
            except Exception as e:
                print(f"⚠️  Skipping problematic LibriLight audio file {path}: {e}")
                return None
            audio_tokens = audio_tokens[..., :self.num_audio_frames]

            this_num_audio_frames = audio_tokens.shape[-1]

            # Pad if needed (last segment might be shorter)
            # Note: Audio padding uses text_padding value for consistency
            if this_num_audio_frames < self.num_audio_frames:
                pad_size = self.num_audio_frames - this_num_audio_frames
                audio_tokens = torch.nn.functional.pad(
                    audio_tokens,
                    (0, pad_size),
                    value=self.interleaver.text_padding
                )

            # Reshape: [2, num_codebooks, T] → [1, 2*num_codebooks, T]
            # This combines left and right channels into single batch
            # After padding, shape should be self.num_audio_frames
            audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)

            # Create text stream (all text padding = no text)
            # CRITICAL: Use this_num_audio_frames (BEFORE padding), then pad separately
            # This matches the working DailyTalk interleaver pattern
            text_tokens = self.interleaver.prepare_item(this_num_audio_frames, device="cuda")
            text_tokens = torch.nn.functional.pad(
                text_tokens,
                (0, self.num_audio_frames - text_tokens.shape[-1]),
                value=self.interleaver.text_padding,
            )

            # Concatenate: [1, 1+2*num_codebooks, T]
            # First row: text (zeros)
            # Remaining rows: audio (encoded)
            codes = torch.cat([text_tokens, audio_tokens], dim=1)

            # Track file_id and chunk_index for continuous RoPE
            file_id = path  # Use path as unique file identifier
            chunk_index = int(start_sec / self.duration_sec)  # Calculate chunk position

            # Debug logging (first call only to avoid spam)
            if not hasattr(self, '_debug_logged'):
                import logging
                logger = logging.getLogger("dataset")
                logger.info(f"🔍 LibriLight tokenization output:")
                logger.info(f"   codes shape: {codes.shape}")
                logger.info(f"   text stream (row 0): unique values = {torch.unique(codes[0, 0, :]).cpu().tolist()}")
                logger.info(f"   audio stream (row 1): min={codes[0, 1, :].min().item()}, max={codes[0, 1, :].max().item()}")
                logger.info(f"   Expected: text stream should all be 3 (text_padding_token_id)")
                self._debug_logged = True

            return AudioOnlySample(codes, condition_attributes=None, file_id=file_id, chunk_index=chunk_index)
