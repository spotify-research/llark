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
import os
import pathlib

import torch
import transformers

from m2t.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    get_bnb_model_args,
)
from m2t.data_modules import make_data_module
from m2t.llava.train.train import (
    find_all_linear_names,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)
from m2t.models.llamav2 import WrappedLlamav2ForCausalLM
from m2t.models.mpt import WrappedMPTForCausalLM
from m2t.models.trainer import WrappedTrainer
from m2t.special_tokens import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from m2t.tokenizer import get_tokenizer
from m2t.utils import (
    get_compute_dtype,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)

# pandas must be imported before other packages to avoid
# /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
):
    compute_dtype = get_compute_dtype(training_args)

    bnb_model_from_pretrained_args = get_bnb_model_args(training_args, compute_dtype)
    # if model_args.vision_tower is not None:
    if "mpt" in model_args.model_name_or_path:
        model = WrappedMPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            mm_hidden_size=model_args.mm_hidden_size,
            **bnb_model_from_pretrained_args,
        )
    else:
        model = WrappedLlamav2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            mm_hidden_size=model_args.mm_hidden_size,
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        logging.warning("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    else:
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)

            if training_args.fp16:
                model.to(torch.float16)

    tokenizer = get_tokenizer(model_args, training_args)

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )
    else:
        raise NotImplementedError(f"version {model_args.version} not implemented.")

    model_audio_dict = model.get_model().initialize_adapter_modules(
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
        tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
        fsdp=training_args.fsdp,
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.get_model().mm_projector.to(torch.bfloat16)
        if training_args.fp16:
            model.get_model().mm_projector.to(torch.float16)

    audio_config = model_audio_dict["audio_config"]

    assert data_args.is_multimodal

    model.config.tune_mm_mlp_adapter = (
        training_args.tune_mm_mlp_adapter
    ) = model_args.tune_mm_mlp_adapter
    if model_args.freeze_backbone:
        print("[INFO] freezing backbone LM weights.")
        model.requires_grad_(False)
    else:
        print("[INFO] training mm backbone LLM weights.")

    # These two flags appear to do the same thing; leaving them to maintain
    #  compatibility with LLaVA but they should be set in a consistent manner.
    assert not (model_args.tune_mm_mlp_adapter and training_args.freeze_mm_mlp_adapter)
    if model_args.tune_mm_mlp_adapter:
        print("[INFO] training MM MLP adapter weights")
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        print("[INFO] freezing mm projector weights.")
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_audio_start_end = (
        data_args.mm_use_audio_start_end
    ) = model_args.mm_use_audio_start_end
    audio_config.use_audio_start_end = (
        training_args.use_audio_start_end
    ) = model_args.mm_use_audio_start_end

    model.initialize_audio_tokenizer(
        mm_use_audio_start_end=model_args.mm_use_audio_start_end,
        tokenizer=tokenizer,
        device=training_args.device,
        tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
    )

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print(
                    "[WARNING] Attempting to use FSDP while {} ".format(len(params_no_grad))
                    + "parameters do not require gradients: {}".format(params_no_grad)
                )
            else:
                print(
                    "[WARNING] Attempting to use FSDP while {} ".format(len(params_no_grad))
                    + "parameters do not require gradients: {}...(omitted)".format(
                        ", ".join(params_no_grad[:10])
                    )
                )
            print(
                "[WARNING] Attempting to use FSDP with partially frozen paramters, "
                + "this is experimental."
            )
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build. "
                + " See here for details: "
                + "github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining"
            )

            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                FullyShardedDataParallel as FSDP,
            )

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop("use_orig_params", True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_data_module(tokenizer=tokenizer, data_args=data_args)

    # Print some sample batch info for debugging and to raise any
    # dataloading errors before training starts.
    logging.warning("data module initialized; fetching sample batch")
    sample_batch = next(iter(data_module["train_dataset"]))
    sample_batch_collated = data_module["data_collator"]([sample_batch])

    info = {
        k: f"shape: {getattr(v, 'shape', None)} dtype {getattr(v, 'dtype', None)}"
        for k, v in sample_batch.items()
    }
    logging.warning(f"sample batch info: {info}")
    info = {
        k: f"shape: {getattr(v, 'shape', None)} dtype {getattr(v, 'dtype', None)}"
        for k, v in sample_batch_collated.items()
    }
    logging.warning(f"sample batch collated info: {info}")

    trainer = WrappedTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    return dict(trainer=trainer, tokenizer=tokenizer, data_module=data_module)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args=model_args, training_args=training_args, data_args=data_args)
