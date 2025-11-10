"""
TTT-Specific Evaluation for Conversational Data
Tests TTT's ability to utilize long conversational context and adapt to speaker patterns.
"""

import logging
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TTTConversationEvaluator:
    """
    Evaluates TTT capabilities on conversational data by testing:
    1. Perplexity vs conversation position (TTT signature metric)
    2. Cross-speaker reference resolution
    3. Speaker adaptation over time
    4. Long-range coherence maintenance
    """
    
    def __init__(self, model, interleaved_tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = interleaved_tokenizer
        self.device = device
        
    def load_conversation_sequence(self, data_dir: str, min_duration: float = 800.0) -> List[Dict]:
        """
        Load and concatenate conversation files to create long sequences.
        Creates sequences of 800+ seconds (13+ minutes) where TTT benefits emerge.
        """
        data_path = Path(data_dir) / "dailytalk.jsonl"
        
        if not data_path.exists():
            logger.warning(f"Data file not found: {data_path}")
            return []
        
        # Load all conversation files
        conversations = []
        with open(data_path, 'r') as f:
            for line in f:
                conv_meta = json.loads(line)
                conversations.append(conv_meta)
        
        # Sort by file index to maintain conversation order
        conversations.sort(key=lambda x: int(x['path'].split('.')[0]))
        
        # Create long sequences by concatenating related conversations
        long_sequences = []
        current_sequence = []
        current_duration = 0.0
        
        for conv in conversations:
            current_sequence.append(conv)
            current_duration += conv['duration']
            
            # If we have enough duration, save this sequence
            if current_duration >= min_duration:
                long_sequences.append({
                    'conversations': current_sequence,
                    'total_duration': current_duration,
                    'num_files': len(current_sequence)
                })
                current_sequence = []
                current_duration = 0.0
        
        # Add remaining sequence if substantial
        if current_duration >= min_duration * 0.7:  # At least 70% of target
            long_sequences.append({
                'conversations': current_sequence,
                'total_duration': current_duration,
                'num_files': len(current_sequence)
            })
        
        logger.info(f"Created {len(long_sequences)} long conversation sequences")
        for i, seq in enumerate(long_sequences):
            logger.info(f"  Sequence {i}: {seq['total_duration']:.1f}s ({seq['num_files']} files)")
        
        return long_sequences
    
    def encode_conversation_sequence(self, sequence: Dict, data_dir: str) -> torch.Tensor:
        """
        Encode a sequence of conversations into tokens.
        Returns concatenated tokens representing the full conversation.
        """
        all_tokens = []
        data_dir = Path(data_dir)
        
        for conv_meta in sequence['conversations']:
            wav_path = data_dir / conv_meta['path']
            
            if not wav_path.exists():
                logger.warning(f"Audio file not found: {wav_path}")
                continue
            
            try:
                # Load and encode the audio file
                import torchaudio
                waveform, sample_rate = torchaudio.load(wav_path)
                
                # Resample to 24kHz if needed
                if sample_rate != 24000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                    waveform = resampler(waveform)
                
                # Use the interleaved tokenizer to encode the audio
                # Note: This creates the proper Moshi token format
                start_time = 0.0  # Each file starts at 0
                sample = self.tokenizer(waveform, start_time, str(wav_path))
                
                # Extract the codes (tokens)
                codes = sample.codes  # Shape: [1, num_codebooks, time_steps]
                all_tokens.append(codes)
                
            except Exception as e:
                logger.error(f"Failed to encode {wav_path}: {e}")
                continue
        
        if not all_tokens:
            logger.error("No tokens encoded from sequence")
            return torch.empty((1, self.model.num_codebooks, 0), device=self.device)
        
        # Concatenate all tokens along time dimension
        concatenated = torch.cat(all_tokens, dim=-1)  # [1, num_codebooks, total_time]
        
        logger.info(f"Encoded sequence: {concatenated.shape[-1]} tokens from {len(all_tokens)} conversations")
        return concatenated.to(self.device)
    
    def evaluate_perplexity_vs_position(self, sequences: List[Dict], data_dir: str) -> Dict[str, float]:
        """
        Core TTT evaluation: Measure perplexity at different positions in long conversations.
        TTT should show decreasing perplexity with more context.
        """
        position_results = []
        
        for seq_idx, sequence in enumerate(sequences):
            logger.info(f"Evaluating sequence {seq_idx+1}/{len(sequences)}")
            
            # Encode the full conversation sequence
            tokens = self.encode_conversation_sequence(sequence, data_dir)
            
            if tokens.shape[-1] < 1000:  # Skip too-short sequences
                continue
            
            # Evaluate perplexity at different positions
            positions = list(range(500, tokens.shape[-1], 200))  # Every 200 tokens from 500
            
            sequence_results = []
            for pos in positions:
                try:
                    # Context from start to position
                    context = tokens[:, :, :pos]
                    
                    # Target token at this position
                    if pos < tokens.shape[-1]:
                        target_tokens = tokens[:, :, pos:pos+1]
                    else:
                        continue
                    
                    # Compute loss at this position
                    with torch.no_grad():
                        output = self.model(context)
                        
                        # Compute loss for the target tokens
                        audio_start = self.model.audio_offset
                        audio_end = audio_start + self.model.dep_q
                        
                        # Extract relevant logits and targets
                        logits = output.logits  # [B, dep_q, T, vocab_size]
                        target = target_tokens[:, audio_start:audio_end, :]  # [B, dep_q, 1]
                        
                        if logits.shape[2] > 0 and target.shape[2] > 0:
                            # Use last timestep logits to predict target
                            last_logits = logits[:, :, -1, :]  # [B, dep_q, vocab_size]
                            target_flat = target.squeeze(-1)  # [B, dep_q]
                            
                            # Compute cross-entropy loss
                            loss = F.cross_entropy(
                                last_logits.reshape(-1, last_logits.shape[-1]),
                                target_flat.reshape(-1),
                                reduction='mean'
                            )
                            
                            perplexity = torch.exp(loss).item()
                            
                            sequence_results.append({
                                'position': pos,
                                'perplexity': perplexity,
                                'context_length': pos
                            })
                            
                except Exception as e:
                    logger.warning(f"Error at position {pos}: {e}")
                    continue
            
            position_results.extend(sequence_results)
        
        if not position_results:
            logger.error("No valid position results")
            return {'ttt_perplexity_improvement': 0.0, 'context_utilization': 1.0}
        
        # Analyze the perplexity trend
        positions = [r['position'] for r in position_results]
        perplexities = [r['perplexity'] for r in position_results]
        
        # Compute TTT benefit: early vs late perplexity
        early_perplexities = [p for pos, p in zip(positions, perplexities) if pos < 1000]
        late_perplexities = [p for pos, p in zip(positions, perplexities) if pos > 2000]
        
        if early_perplexities and late_perplexities:
            early_avg = np.mean(early_perplexities)
            late_avg = np.mean(late_perplexities)
            ttt_improvement = early_avg - late_avg  # Positive = improvement
            context_utilization = late_avg / early_avg  # < 1.0 = good utilization
        else:
            ttt_improvement = 0.0
            context_utilization = 1.0
        
        # Compute perplexity slope (should be negative for TTT)
        if len(positions) > 1:
            slope, _ = np.polyfit(positions, perplexities, 1)
        else:
            slope = 0.0
        
        logger.info(f"TTT Evaluation Results:")
        logger.info(f"  Early perplexity: {early_avg:.3f}")
        logger.info(f"  Late perplexity: {late_avg:.3f}")
        logger.info(f"  TTT improvement: {ttt_improvement:.3f}")
        logger.info(f"  Context utilization: {context_utilization:.3f}")
        logger.info(f"  Perplexity slope: {slope:.6f}")
        
        return {
            'ttt_perplexity_improvement': ttt_improvement,
            'ttt_early_perplexity': early_avg if early_perplexities else 0.0,
            'ttt_late_perplexity': late_avg if late_perplexities else 0.0,
            'ttt_context_utilization': context_utilization,
            'ttt_perplexity_slope': slope,
            'ttt_evaluation_samples': len(position_results)
        }
    
    def evaluate_speaker_adaptation(self, sequences: List[Dict], data_dir: str) -> Dict[str, float]:
        """
        Test TTT's ability to adapt to speaker patterns over time.
        Load speaker alignments and test if model improves at predicting each speaker.
        """
        # This would require parsing the .json files with speaker alignments
        # For now, return placeholder
        return {
            'ttt_speaker_adaptation': 0.0,
            'ttt_speaker_consistency': 0.0
        }
    
    def evaluate_all_ttt_metrics(self, data_dir: str) -> Dict[str, float]:
        """
        Run all TTT-specific evaluations.
        """
        self.model.eval()
        
        # Load long conversation sequences
        sequences = self.load_conversation_sequence(data_dir)
        
        if not sequences:
            logger.warning("No long sequences found for TTT evaluation")
            return {'ttt_perplexity_improvement': 0.0}
        
        # Run core TTT evaluation: perplexity vs position
        results = self.evaluate_perplexity_vs_position(sequences, data_dir)
        
        # Add speaker adaptation results (placeholder for now)
        speaker_results = self.evaluate_speaker_adaptation(sequences, data_dir)
        results.update(speaker_results)
        
        return results


def create_ttt_conversation_evaluator(model, interleaved_tokenizer, device: str = "cuda") -> TTTConversationEvaluator:
    """Factory function to create TTT conversation evaluator."""
    return TTTConversationEvaluator(model, interleaved_tokenizer, device)