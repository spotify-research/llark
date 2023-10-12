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

import os
from typing import Optional

import torch
from transformers import Trainer
from transformers.trainer import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PretrainedConfig,
    __version__,
    is_sagemaker_mp_enabled,
    logger,
)

from m2t.llava.train.llava_trainer import unwrap_model
from m2t.models.mpt import WrappedMPTForCausalLM
from m2t.models.utils import load_sharded_checkpoint


class WrappedTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ["mm_projector", "embed_tokens", "embed_in"]
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save,
                    os.path.join(output_dir, "mm_projector.bin"),
                )

        super(WrappedTrainer, self)._save(output_dir, state_dict)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Forked version of transformers.Trainer._load_from_checkpoint() that also
        loads the MM projector weights."""
        if model is None:
            model = self.model

        if not os.path.isfile(
            os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        ) and not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(
                os.path.join(resume_from_checkpoint, CONFIG_NAME)
            )
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained "
                    f"with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. "
                    "This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                raise NotImplementedError()
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(
                    os.path.join(resume_from_checkpoint, WEIGHTS_NAME),
                    map_location="cpu",
                )
                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(
                    {k: v for k, v in state_dict.items() if "mm_projector" not in k},
                    False,
                )
                logger.info(f"Loading mm_projector weights from {resume_from_checkpoint}.")
                if isinstance(model, WrappedMPTForCausalLM):
                    model.transformer.mm_projector.load_state_dict(
                        {k.split(".")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
                    )
                else:
                    raise NotImplementedError(
                        f"mm project weight loading for model type {type(model)} "
                        + "not implemented yet."
                    )
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
