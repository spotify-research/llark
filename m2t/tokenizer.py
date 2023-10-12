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

from typing import List

import transformers

from m2t.arguments import ModelArguments, TrainingArguments


def get_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    return tokenizer


def get_prompt_end_token_sequence(
    tokenizer: transformers.PreTrainedTokenizer,
    model_name: str,
    prompt_end_string="\n### Assistant:",
) -> List[int]:
    """Fetch the sequence of tokens that identifies the end of the prompt
    (and the start of the model generation).

    This sequence will be used to split sequences into (prompt, response) pairs.
    """
    end_seq = tokenizer([prompt_end_string], add_special_tokens=False).input_ids[0]

    if "meta-llama/Llama-2" in model_name:
        # LLaMA tokenizer adds a padding token to the front that we remove, since it
        # does not appear when we tokenize the normal sequences (probably due to the \n
        # being sometimes part of the tokenization of the characters preceding it.
        end_seq = end_seq[1:]

    return end_seq
