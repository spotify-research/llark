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

import gc
import glob
import json
import os
from typing import Tuple

import torch
import transformers
from transformers import AutoTokenizer
from transformers.trainer import WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from m2t.models.llamav2 import WrappedLlamav2ForCausalLM
from m2t.models.mpt import WrappedMPTForCausalLM
from m2t.special_tokens import DEFAULT_AUDIO_END_TOKEN, DEFAULT_AUDIO_START_TOKEN


def load_sharded_mm_projector_weights(model, folder):
    """Load sharded mm_projector weights from folder."""
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    if not os.path.isfile(index_file):
        raise ValueError(f"Can't find a checkpoint index ({WEIGHTS_INDEX_NAME}) in {folder}.")

    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    for shard_file in shard_files:
        state_dict = torch.load(os.path.join(folder, shard_file), map_location="cpu")
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if "mm_projector" not in k},
            strict=False,  # strict is handled above prior to loading
        )

        if any("mm_projector" in x for x in state_dict.keys()):
            print(f"loading mm_projector params from ckpt file {shard_file}")
            model.get_model().mm_projector.load_state_dict(
                {k.split(".")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
            )


def load_sharded_checkpoint(model, folder, strict=True):
    """
    Override of transformers.modeling_utils.load_sharded_checkpoint() but which also
    loads the mm_adapter params.
    This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in
    RAM and deleted after being
    loaded in the model.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`):
            A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the
            keys in the sharded checkpoint.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    if not os.path.isfile(index_file):
        raise ValueError(f"Can't find a checkpoint index ({WEIGHTS_INDEX_NAME}) in {folder}.")

    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
    #         raise RuntimeError(error_message)

    for shard_file in shard_files:
        state_dict = torch.load(os.path.join(folder, shard_file), map_location="cpu")
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if "mm_projector" not in k},
            strict=False,  # strict is handled above prior to loading
        )

        if any("mm_projector" in x for x in state_dict.keys()):
            print(f"loading mm_projector params from ckpt file {shard_file}")
            model.get_model().mm_projector.load_state_dict(
                {k.split(".")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
            )

        # Make sure memory is fred before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PyTorch load_state_dict function.
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)


def load_pretrained_model(
    model_name: str,
    ckpt_num: int,
    torch_dtype=torch.float16,
    mm_use_audio_start_end=True,
    device="cuda:0",
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    ckpt_dir = os.path.join(model_name, f"checkpoint-{ckpt_num}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    # Determine whether the checkpoint is sharded or not.
    if os.path.exists(os.path.join(ckpt_dir, WEIGHTS_NAME)):
        ckpt_file = os.path.join(ckpt_dir, WEIGHTS_NAME)
        sharded_ckpt = False
    else:
        ckpt_glob = os.path.join(ckpt_dir, "pytorch_model-*of*.bin")
        ckpt_files = glob.glob(ckpt_glob)
        assert len(ckpt_files), f"no files found matching {ckpt_glob}"
        print(f"got checkpoint files {ckpt_files}")
        sharded_ckpt = True

    # Handle instantiation of each supported model class.
    if "mosaicml/mpt" in model_name:
        model = WrappedMPTForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch_dtype,
        )
        model.get_model().initialize_adapter_modules(
            tune_mm_mlp_adapter=False,
            pretrain_mm_mlp_adapter=ckpt_file,
            fsdp=None,
        )

        # load other stuff
        print(f"[DEBUG] loading {ckpt_file} weights manually")
        model_weights = torch.load(ckpt_file)
        model.load_state_dict(model_weights)
        model.config.mm_use_audio_start_end = True
        model_attr = "transformer"
        (
            getattr(model, model_attr).audio_encoder_config.audio_start_token,
            getattr(model, model_attr).audio_encoder_config.audio_end_token,
        ) = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN])

    elif "meta-llama/Llama-2" in model_name:
        model = WrappedLlamav2ForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch_dtype,
        )
        # this will NOT load the adapter weights; it just
        # initializes the module so that they can be loaded later.
        model.get_model().initialize_adapter_modules(tune_mm_mlp_adapter=False, fsdp=None)

    if sharded_ckpt:
        # Case: for sharded checkpoints, we need to manually load the
        # projector weights from the shards.
        print("[DEBUG] loading mm projector parameters")
        load_sharded_mm_projector_weights(model, ckpt_dir)

    model.cuda()
    model.get_model().mm_projector.cuda()
    model.config.mm_use_audio_start_end = mm_use_audio_start_end

    if "mosaicml/mpt" not in model_name:
        model.initialize_audio_tokenizer(
            mm_use_audio_start_end=model.config.mm_use_audio_start_end,
            tokenizer=tokenizer,
            device=device,
            tune_mm_mlp_adapter=model.config.tune_mm_mlp_adapter,
            pretrain_mm_mlp_adapter=None if "mosaicml/mpt" not in model_name else ckpt_file,
        )

    return model, tokenizer
