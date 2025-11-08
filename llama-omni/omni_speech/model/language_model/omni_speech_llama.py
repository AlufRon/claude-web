# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM


class OmniSpeechConfig(LlamaConfig):
    model_type = "omni_speech_llama"

    def __init__(
        self,
        # TTT Configuration - all optional, defaults maintain standard Llama behavior
        use_ttt: bool = False,  # Whether to use TTT layers instead of standard attention
        ttt_layer_type: str = "ttt_linear",  # "ttt_linear" or "ttt_mlp"
        ttt_layer_indices: Optional[List[int]] = None,  # Which layers to replace with TTT (None = top 8 layers)
        ttt_mini_batch_size: int = 64,  # Mini-batch size for TTT updates (CRITICAL for stable gradients)
        ttt_base_lr: float = 1.0,  # Base learning rate for TTT inner loop
        ttt_num_heads: Optional[int] = None,  # Number of TTT heads (None = use model num_attention_heads)

        # State Management - CRITICAL for conversation-level persistence
        ttt_state_dtype: str = "float32",  # MUST be float32 for numerical stability
        ttt_reset_on_new_conversation: bool = True,  # Reset TTT state when conversation changes

        # Logging Configuration
        ttt_enable_logging: bool = True,  # Enable TTT-specific logging
        ttt_log_level: str = "INFO",  # "DEBUG", "INFO", "WARNING", "ERROR"
        ttt_log_interval: int = 100,  # Log TTT stats every N steps
        ttt_csv_log_path: Optional[str] = None,  # Path to CSV file for TTT state tracking (None = auto-generate)

        # Performance Options
        ttt_use_triton_kernel: bool = False,  # Use Triton kernel (faster but requires compilation)
        ttt_checkpoint_group_size: int = 4,  # Gradient checkpointing group size for memory efficiency

        **kwargs
    ):
        super().__init__(**kwargs)

        # TTT Configuration
        self.use_ttt = use_ttt
        self.ttt_layer_type = ttt_layer_type
        self.ttt_layer_indices = ttt_layer_indices
        self.ttt_mini_batch_size = ttt_mini_batch_size
        self.ttt_base_lr = ttt_base_lr
        self.ttt_num_heads = ttt_num_heads

        # State Management
        self.ttt_state_dtype = ttt_state_dtype
        self.ttt_reset_on_new_conversation = ttt_reset_on_new_conversation

        # Logging
        self.ttt_enable_logging = ttt_enable_logging
        self.ttt_log_level = ttt_log_level
        self.ttt_log_interval = ttt_log_interval
        self.ttt_csv_log_path = ttt_csv_log_path

        # Performance
        self.ttt_use_triton_kernel = ttt_use_triton_kernel
        self.ttt_checkpoint_group_size = ttt_checkpoint_group_size

        # Validation
        if self.use_ttt:
            assert self.ttt_state_dtype == "float32", (
                "TTT inner states MUST use float32 for numerical stability. "
                "Using float16/bfloat16 will cause accumulation errors after ~3750 updates."
            )
            assert self.ttt_mini_batch_size > 0 and self.ttt_mini_batch_size <= 128, (
                f"ttt_mini_batch_size must be between 1 and 128, got {self.ttt_mini_batch_size}"
            )
            assert self.ttt_layer_type in ["ttt_linear", "ttt_mlp"], (
                f"ttt_layer_type must be 'ttt_linear' or 'ttt_mlp', got {self.ttt_layer_type}"
            )


class OmniSpeechLlamaModel(OmniSpeechMetaModel, LlamaModel):
    config_class = OmniSpeechConfig

    def __init__(self, config: LlamaConfig):
        super(OmniSpeechLlamaModel, self).__init__(config)

        # Integrate TTT if enabled in config
        if getattr(config, 'use_ttt', False):
            from ..ttt.integration import integrate_ttt_into_model, verify_ttt_integration
            import logging

            logger = logging.getLogger(__name__)
            logger.info("[OmniSpeechLlamaModel] TTT enabled, integrating into model...")

            # Integrate TTT layers
            integrate_ttt_into_model(self, config)

            # Verify integration
            verification = verify_ttt_integration(self)

            if not verification["integration_successful"]:
                logger.error(f"[OmniSpeechLlamaModel] TTT integration failed: {verification['errors']}")
                raise RuntimeError(f"TTT integration failed: {verification['errors']}")

            logger.info(
                f"[OmniSpeechLlamaModel] TTT integration successful! "
                f"{len(verification['ttt_layers'])} layers using TTT"
            )


class OmniSpeechLlamaForCausalLM(LlamaForCausalLM, OmniSpeechMetaForCausalLM):
    config_class = OmniSpeechConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = OmniSpeechLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs

AutoConfig.register("omni_speech_llama", OmniSpeechConfig)
AutoModelForCausalLM.register(OmniSpeechConfig, OmniSpeechLlamaForCausalLM)
