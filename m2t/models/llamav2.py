# Copyright 2023 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from m2t.models import AudioEncoderConfig
from m2t.special_tokens import (
    DEFAULT_AUDIO_END_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
)

# from m2t.llava.model.mpt.modeling_mpt import MPTConfig, MPTForCausalLM, MPTModel


class WrappedLlamav2Config(LlamaConfig):
    """Config container class for the Llamav2-based model."""

    model_type = "wrapped_llamav2"
    mm_hidden_size: int = 4800  # size of the jukebox embeddings with temporal averaging


class WrappedLlamav2Model(LlamaModel):
    """Llamav2-based LLark model.

    This is the main model used in our paper.
    """

    config_class = WrappedLlamav2Config

    def __init__(self, config: LlamaConfig):
        super(WrappedLlamav2Model, self).__init__(config)

        self.audio_encoder_config = AudioEncoderConfig()

    # compare to LlavaMPTModel.initialize_vision_modules()
    def initialize_adapter_modules(
        self,
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=None,
        fsdp: bool = None,
    ):
        """
        Initialize the adapter modules.

        args:
            pretrain_mm_mlp_adapter: optional path to pretrained weights to load.
            tune_mm_mlp_adapter: unused parameter provided for compatibility.
            fsdp: unused parameter provided for compatibility.
        """
        print("[INFO] ignoring parameter fsdp")
        del fsdp
        self.config.use_mm_proj = True

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
            self.mm_projector.load_state_dict(
                {
                    k.split(".")[-1]: v
                    for k, v in mm_projector_weights.items()
                    if "mm_projector" in k
                }
            )

        return dict(
            audio_config=AudioEncoderConfig(),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.ByteTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_encodings: Optional[torch.Tensor] = None,
    ):
        """
        Implements the forward pass.

        args:
            input_ids: Tensor of shape [batch_size, sequence_len]
            past_key_values: past key values; passed to model.forward().
            attention_mask: attention mask of shape [batch_size, sequence_length]
            return_dict: passed to model.forward().
            output_attentions: passed to model.forward().
            output_hidden_states: passed to model.forward().
            use_cache: passed to model.forward().
            audio_encodings: audio encoding tensor of shape [batch_size, *audio_encodings_dim]
        """

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)

        inputs_embeds = self.embed_tokens(input_ids)

        # audio_features has shape [batch_size, model_dim].
        if audio_encodings is not None and self.config.use_mm_proj:
            if isinstance(audio_encodings, list):
                audio_features = [
                    self.mm_projector(audio_feature) for audio_feature in audio_encodings
                ]
            else:
                audio_features = self.mm_projector(audio_encodings)

            # For each element in the batch, construct the full input in embedding space
            # by concatenating blocks of text and audio tokens.

            new_input_embeds = []
            cur_audio_idx = 0

            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if self.audio_encoder_config.use_audio_start_end:
                    # Case: this is a model that uses audio start/end tokens.
                    cur_audio_features = audio_features[cur_audio_idx]
                    num_frames = cur_audio_features.shape[0]
                    if (cur_input_ids == self.audio_encoder_config.audio_start_token).sum() != (
                        cur_input_ids == self.audio_encoder_config.audio_end_token
                    ).sum():
                        raise ValueError(
                            "The number of image start tokens and image end tokens "
                            "should be the same."
                        )
                    audio_start_tokens = torch.where(
                        cur_input_ids == self.audio_encoder_config.audio_start_token
                    )[0]
                    if not len(audio_start_tokens) and (past_key_values is None):
                        logging.warning(
                            "no audio start tokens detected and there are no past_key_values;"
                            "if this is a multimodal model this could be a problem."
                        )
                    if len(audio_start_tokens):
                        # Case: there are audio start tokens; build the inputs from
                        # the appropriate elements of token embeddings + audio embeddings.
                        for audio_start_token_pos in audio_start_tokens:
                            cur_audio_features = audio_features[cur_audio_idx].to(
                                device=cur_input_embeds.device
                            )
                            num_frames = cur_audio_features.shape[0]
                            if (
                                cur_input_ids[audio_start_token_pos + num_frames + 1]
                                != self.audio_encoder_config.audio_end_token
                            ):
                                raise ValueError(
                                    "The image end token should follow the image start token."
                                )
                            if orig_embeds_params is not None:
                                cur_new_input_embeds = torch.cat(
                                    (
                                        # any tokens preceding the audio start token
                                        cur_input_embeds[:audio_start_token_pos].detach(),
                                        # the audio start token
                                        cur_input_embeds[
                                            audio_start_token_pos : audio_start_token_pos + 1
                                        ],
                                        cur_audio_features,
                                        cur_input_embeds[
                                            audio_start_token_pos
                                            + num_frames
                                            + 1 : audio_start_token_pos
                                            + num_frames
                                            + 2
                                        ],
                                        cur_input_embeds[
                                            audio_start_token_pos + num_frames + 2 :
                                        ].detach(),
                                    ),
                                    dim=0,
                                )
                            else:
                                cur_new_input_embeds = torch.cat(
                                    (
                                        cur_input_embeds[: audio_start_token_pos + 1],
                                        cur_audio_features,
                                        cur_input_embeds[audio_start_token_pos + num_frames + 1 :],
                                    ),
                                    dim=0,
                                )
                            cur_audio_idx += 1
                        new_input_embeds.append(cur_new_input_embeds)
                    else:
                        # Case: there are no audio start tokens (this can happen
                        # when calling .generate(), because the audio tokens are
                        # already incorporated via past_key_values). Just take
                        # the vanilla input emmbedding.
                        new_input_embeds.append(cur_input_embeds)

                else:
                    # Case: this is a model that does not use audio start/end tokens.
                    raise NotImplementedError(
                        "audio_encoder_config.use_audio_start_end=False is not implemented."
                    )
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(WrappedLlamav2Model, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class WrappedLlamav2ForCausalLM(LlamaForCausalLM):
    """Llamav2-based wrapper for causal language modeling."""

    config_class = WrappedLlamav2Config
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = WrappedLlamav2Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, WrappedLlamav2Model):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_encodings=None,
    ):
        """Implements the forward pass.

        Most of the logic for the foward pass happens in the call to self.model;
            see WrappedLlamav2Model for details.

        args:
            input_ids: Tensor of shape [batch_size, sequence_len]
            attention_mask: attention mask of shape [batch_size, sequence_length]
            position_ids: provided for compatibility.
            past_key_values: past key values; passed to model.forward().
            labels: labels tensor of shape [batch_size, sequence_len].
            use_cache: passed to model.forward().
            output_attentions: passed to model.forward().
            output_hidden_states: passed to model.forward().
            return_dict: passed to model.forward().
            audio_encodings: audio encoding tensor of shape [batch_size, *audio_encodings_dim]
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            audio_encodings=audio_encodings,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "audio_encodings": kwargs.get("audio_encodings", None),
            }
        )

        return model_inputs

    def initialize_audio_tokenizer(
        self,
        mm_use_audio_start_end,
        tokenizer,
        device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    ):
        """Set up the tokenizer to handle the various audio tokens."""
        del pretrain_mm_mlp_adapter

        audio_encoder_config = self.get_model().audio_encoder_config
        audio_encoder_config.use_audio_start_end = mm_use_audio_start_end
        tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_audio_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN],
                special_tokens=True,
            )
            self.resize_token_embeddings(len(tokenizer))
            (
                audio_encoder_config.audio_start_token,
                audio_encoder_config.audio_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN]
            )

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)
                ]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        audio_encoder_config.audio_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_AUDIO_PATCH_TOKEN]
        )[0]


AutoConfig.register("wrapped_llamav2", WrappedLlamav2Config)
AutoModelForCausalLM.register(WrappedLlamav2Config, WrappedLlamav2ForCausalLM)
