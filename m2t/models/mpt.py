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
import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from m2t.llava.model.mpt.modeling_mpt import MPTConfig, MPTForCausalLM, MPTModel
from m2t.models import AudioEncoderConfig
from m2t.special_tokens import (
    DEFAULT_AUDIO_END_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
)


class WrappedMPTConfig(MPTConfig):
    model_type = "wrapped_mpt"
    mm_hidden_size: int = 4800  # size of the jukebox embeddings with temporal averaging


class WrappedMPTModel(MPTModel):
    config_class = WrappedMPTConfig

    def __init__(self, config: MPTConfig):
        super(WrappedMPTModel, self).__init__(config)

        self.audio_encoder_config = AudioEncoderConfig()

    # compare to LlavaMPTModel.initialize_vision_modules()
    def initialize_adapter_modules(
        self,
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=False,
        fsdp: bool = None,
    ):
        print("[INFO] ignoring parameter fsdp")
        del fsdp
        self.config.use_mm_proj = True

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(self.config.mm_hidden_size, self.config.d_model)

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
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        audio_encodings=None,
    ):
        """
        args:
            input_ids: Tensor of shape [batch_size, sequence_len]
            past_key_values: past key values; passed to model.forward().
            attention_mask: attention mask of shape [batch_size, sequence_length]
            prefix_mask:
            sequence_id:
            return_dict: passed to model.forward().
            output_attentions: passed to model.forward().
            output_hidden_states: passed to model.forward().
            use_cache: passed to model.forward().
            audio_encodings: audio encoding tensor of shape [batch_size, *audio_encodings_dim]
        """
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        inputs_embeds = self.wte(input_ids)

        # audio_features has shape [batch_size, model_dim]
        # where model_dim is 2048 for MPT models.
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
                    if not len(audio_start_tokens):
                        logging.warning(
                            "no audio start tokens detected; if this is "
                            "a multimodal model this could be a problem."
                        )
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
                    cur_audio_features = audio_features[cur_audio_idx]
                    num_frames = cur_audio_features.shape[0]
                    if (
                        cur_input_ids == self.audio_encoder_config.audio_patch_token
                    ).sum() != num_frames:
                        raise ValueError(
                            "The number of audio patch tokens should be the same as the "
                            "number of audio frames."
                        )
                    masked_indices = torch.where(
                        cur_input_ids == self.audio_encoder_config.audio_patch_token
                    )[0]
                    mask_index_start = masked_indices[0]
                    if (
                        masked_indices
                        != torch.arange(
                            mask_index_start,
                            mask_index_start + num_frames,
                            device=masked_indices.device,
                            dtype=masked_indices.dtype,
                        )
                    ).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_audio_features,
                                cur_input_embeds[mask_index_start + num_frames :].detach(),
                            ),
                            dim=0,
                        )
                    else:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start],
                                cur_audio_features,
                                cur_input_embeds[mask_index_start + num_frames :],
                            ),
                            dim=0,
                        )
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(WrappedMPTModel, self).forward(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            tok_emb=inputs_embeds,
        )


class WrappedMPTForCausalLM(MPTForCausalLM):
    config_class = WrappedMPTConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(MPTForCausalLM, self).__init__(config)

        if not config.tie_word_embeddings:
            raise ValueError("MPTForCausalLM only supports tied word embeddings")
        self.transformer = WrappedMPTModel(config)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == "inv_sqrt_d_model":
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"logit_scale={logit_scale!r} is not recognized as an option; "
                        "use numeric value or 'inv_sqrt_d_model'."
                    )
            self.logit_scale = logit_scale

    @property
    def model(self):
        """Alias for self.transformer, to match multimodal Llama model interface."""
        return self.transformer

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, WrappedMPTModel):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        audio_encodings=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            audio_encodings=audio_encodings,
        )
        logits = F.linear(outputs.last_hidden_state, self.transformer.wte.weight)
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f"Multiplying logits by self.logit_scale={self.logit_scale!r}. "
                    "This will produce uniform (uninformative) outputs."
                )
            logits *= self.logit_scale
        loss = None
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.to(logits.device).view(-1),
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds is not implemented for MPT yet")
        attention_mask = kwargs["attention_mask"].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError("MPT does not support generation with right padding.")
        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if self.transformer.prefix_lm:
            prefix_mask = torch.ones_like(attention_mask)
            if kwargs.get("use_cache") is False:
                raise NotImplementedError(
                    "MPT with prefix_lm=True does not support use_cache=False."
                )
        else:
            prefix_mask = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prefix_mask": prefix_mask,
            "sequence_id": sequence_id,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "audio_encodings": kwargs.get("audio_encodings", None),
        }

    def initialize_audio_tokenizer(
        self,
        mm_use_audio_start_end,
        tokenizer,
        device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    ):
        """Set up the tokenizer to handle the various audio tokens."""

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

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["transformer.wte.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        "Unexpected embed_tokens_weight shape. Pretrained: "
                        + "{embed_tokens_weight.shape}. Current: {input_embeddings.shape}. "
                        + "Numer of new tokens: {num_new_tokens}."
                    )

        audio_encoder_config.audio_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_AUDIO_PATCH_TOKEN]
        )[0]


AutoConfig.register("wrapped_mpt", WrappedMPTConfig)
AutoModelForCausalLM.register(WrappedMPTConfig, WrappedMPTForCausalLM)
