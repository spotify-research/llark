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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(
        default=False, metadata={"help": "Whether to freeze the LM parameters."}
    )
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to pretrained multimodal MLP weights."},
    )
    mm_use_audio_start_end: bool = field(
        default=False,
        metadata={"help": "whether to use a token for audio start/end; suggest to set True."},
    )
    mm_hidden_size: int = field(
        default=4800,
        metadata={
            "help": "the size of the multimodal embeddings at each time frame "
            + "(i.e. 4800 for Jukebox; 512 for CLAP)"
        },
    )


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(
        default=None, metadata={"help": "(Optional) path to the eval data."}
    )
    eval_data_subsample: float = field(
        default=None,
        metadata={
            "help": "Fraction of full eval dataset to take."
            "This reduces evaluation time and can be useful during development or when "
            + "the eval dataset is large."
        },
    )
    task_sample_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "captioning": 0.15,
            "reasoning": 0.55,
            "mir": 0.3,
        },
    )
    apply_task_sample_probs: bool = False

    is_multimodal: bool = True
    sep_audio_conv_front: bool = field(
        default=False,
        metadata={
            "help": "Whether to use special conversation format; see preprocess_multimodal()."
        },
    )
    audio_folder: Optional[str] = field(default=None)
    # image_aspect_ratio: str = "square"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded "
            + "(and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def get_bnb_model_args(training_args: TrainingArguments, compute_dtype):
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )
    return bnb_model_from_pretrained_args


def write_args_to_file(args: List[str], dir: str):
    """Write the arguments to a file."""
    if not os.path.exists(dir):
        os.makedirs(dir)
    fp = os.path.join(dir, "args.txt")
    if os.path.exists(fp):
        logging.info(f"args file already exists at {fp}; overwriting it.")
        try:
            os.remove(fp)
        except FileNotFoundError:
            # when running in parallel, multiple processes might try to remove
            # the file; this handles that potential race condition.
            pass
    logging.info(f"writing args to {fp}")
    with open(fp, "w") as f:
        for arg in args:
            if arg.startswith("-"):
                f.write(arg + " ")
            else:
                f.write(arg + " \\" + "\n")
