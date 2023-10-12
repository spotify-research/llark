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

"""
Run inference on a dataset.

Depending on the task, the outputs from this script can be used for downstream evaluation (e.g. captioning, MIR, and reasoning tasks).

Usage:
python -m m2t.infer \
    --eval_data_path "datasets/musiccaps/preprocessed/wds/musiccaps-eval-jukebox-f10-captioning-{000000..000021}.tar" \
    --model_name_or_path checkpoints/meta-llama/Llama-2-7b-chat-hf/20234/ \
    --ckpt-num 100000 \
    --report_to none \
    --bf16 True \
    --tf32 True \
    --output_dir tmp \
    --prompt "Describe the contents of the provided audio in detail." \
    --outfile inference-results/infer_results_musiccaps_eval-captions_v3_100k_fixedprompt.csv

python -m m2t.infer \
    --eval_data_path "datasets/musicnet/preprocessed/wds/musicnet-test-jukebox-f10-captioning-000000.tar" \
    --model_name_or_path checkpoints/meta-llama/Llama-2-7b-chat-hf/20234/ \
    --ckpt-num 100000 \
    --report_to none \
    --bf16 True \
    --tf32 True \
    --output_dir tmp \
    --prompt "Describe the contents of the provided audio in detail." \
    --outfile inference-results/infer_results_musicnet_test-captions_v3_100k_fixedprompt.csv

python -m m2t.infer\
    --eval_data_path "datasets/giantsteps-key/preprocessed/wds/giantsteps-eval-jukebox-f10-key-{000000..000004}.tar" \
    --model_name_or_path checkpoints/meta-llama/Llama-2-7b-chat-hf/20234/ \
    --ckpt-num 100000 \
    --report_to none \
    --bf16 True \
    --tf32 True \
    --output_dir tmp \
    --outfile inference-results/infer_results_giantsteps_key_v3_100k.csv

python -m m2t.infer\
    --eval_data_path "datasets/giantsteps-tempo/preprocessed/wds/giantsteps-eval-jukebox-f10-tempo-{000000..000005}.tar" \
    --model_name_or_path checkpoints/meta-llama/Llama-2-7b-chat-hf/20234/ \
    --ckpt-num 100000 \
    --report_to none \
    --bf16 True \
    --tf32 True \
    --output_dir tmp \
    --outfile inference-results/infer_results_giantsteps_tempo_v3_100k.csv


python -m m2t.infer\
    --eval_data_path "datasets/gtzan/preprocessed/wds/gtzan-jukebox-f10-genre-{000000..000001}.tar" \
    --model_name_or_path checkpoints/meta-llama/Llama-2-7b-chat-hf/20234/ \
    --ckpt-num 100000 \
    --report_to none \
    --bf16 True \
    --tf32 True \
    --output_dir tmp \
    --outfile inference-results/infer_results_gtzan_v3_100k.csv

python -m m2t.infer\
    --eval_data_path "datasets/musicnet/preprocessed/wds/musicnet-test-jukebox-f10-captioning-000000.tar" \
    --model_name_or_path checkpoints/meta-llama/Llama-2-7b-chat-hf/7510 \
    --ckpt-num 100000 \
    --report_to none \
    --bf16 True \
    --tf32 True \
    --output_dir tmp \
    --prompt "What instruments do you hear in the provided audio?" \
    --outfile inference-results/infer_results_musicnet_instruments_v5_100k.csv
"""
from typing import Any, Dict, Optional, Sequence

import torch
import transformers

from m2t.conversation_utils import extract_prompt_tokens
from m2t.data_modules import (
    DEFAULT_CONVERSATION_HEADER,
    concat_audio_token_and_prompt,
    preprocess_for_lm_mappable,
    preprocess_multimodal_mappable,
)
from m2t.generate import KeywordsStoppingCriteria


def infer_with_prompt(
    prompt_text: str,
    model,
    audio_encoding,
    end_seq: Sequence[int],
    multimodal_cfg: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
    example_id: Optional[str] = None,
    audio_first: bool = False,
    header: str = DEFAULT_CONVERSATION_HEADER,
    **generation_kwargs,
):
    # Apply the same preprocessing as in to_conversation()
    prompt_text = concat_audio_token_and_prompt(prompt_text, audio_first)

    elem = {
        "audio_encoding": audio_encoding,
        "audio_encoding_shape": list(audio_encoding.shape),
        "example_id": example_id,
        "id": example_id,
        "conversations": [
            {"from": "human", "value": prompt_text},
            {"from": "gpt", "value": "<empty>"},
        ],
    }

    elem = preprocess_multimodal_mappable(elem, multimodal_cfg)

    elem = preprocess_for_lm_mappable(elem, tokenizer=tokenizer, header=header)

    audio_encoding = elem.pop("audio_encoding")
    if len(audio_encoding.shape) < 3:
        audio_encoding = torch.unsqueeze(audio_encoding, 0)

    input_ids = elem.pop("input_ids")
    input_ids = extract_prompt_tokens(input_ids, end_seq)
    if len(input_ids.shape) < 2:
        input_ids = torch.unsqueeze(input_ids, 0)

    stopping_criteria = KeywordsStoppingCriteria(
        keywords=[
            "###",
        ],
        tokenizer=tokenizer,
        input_ids=input_ids,
    )

    return model.generate(
        **elem,
        **generation_kwargs,
        input_ids=input_ids.cuda(),
        audio_encodings=audio_encoding.cuda(),
        stopping_criteria=[stopping_criteria],
    )
