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

Depending on the task, the outputs from this script can be used for downstream evaluation
(e.g. captioning, MIR, and reasoning tasks).

Usage:
python -m m2t.infer \
    --eval_data_path "path/to/musiccaps-webdataset-{000000..000021}.tar" \
    --model_name_or_path checkpoints/meta-llama/Llama-2-7b-chat-hf/20234/ \
    --ckpt-num 100000 \
    --report_to none \
    --bf16 True \
    --tf32 True \
    --output_dir tmp \
    --prompt "Describe the contents of the provided audio in detail." \
    --outfile inference-results/infer_results_musiccaps_eval-captions.csv

"""
import os
from typing import Optional

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from m2t.arguments import DataArguments, ModelArguments, TrainingArguments
from m2t.conversation_utils import extract_prompt_tokens, extract_response_tokens
from m2t.data_modules import make_data_module, make_mm_config
from m2t.generate import KeywordsStoppingCriteria
from m2t.infer import infer_with_prompt
from m2t.models.utils import load_pretrained_model
from m2t.tokenizer import get_prompt_end_token_sequence
from m2t.utils import get_autocast_type


def main(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    outfile: str,
    ckpt_num: int,
    max_samples: int = None,
    prompt: Optional[str] = None,
    max_new_tokens: int = 2048,
):
    prompt_is_provided = prompt is not None
    assert data_args.is_multimodal
    print("loading model and data...")

    model, tokenizer = load_pretrained_model(model_args.model_name_or_path, ckpt_num=ckpt_num)

    data_args.mm_use_audio_start_end = True

    data_module = make_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = data_module["eval_dataset"]

    end_seq = get_prompt_end_token_sequence(tokenizer, model_args.model_name_or_path)

    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    model.cuda()

    multimodal_cfg = make_mm_config(data_args)

    outputs = []
    with torch.autocast(device_type="cuda", dtype=get_autocast_type(training_args)):
        with torch.inference_mode():
            for i, ex in tqdm(enumerate(dataset), total=max_samples):
                if not prompt_is_provided:
                    input_ids = ex["input_ids"]
                    prompt = extract_prompt_tokens(input_ids, end_seq)
                    prompt = torch.unsqueeze(prompt, 0).cuda()

                    stopping_criteria = KeywordsStoppingCriteria(
                        keywords=[
                            "###",
                        ],
                        tokenizer=tokenizer,
                        input_ids=prompt,
                    )

                    outputs_i = model.generate(
                        # Add a batch dimension to both the input_ids and the audio_encodings.
                        input_ids=prompt,
                        audio_encodings=torch.unsqueeze(ex["audio_encoding"], 0).cuda(),
                        # do_sample=True,
                        # temperature=0.8,
                        # use_cache=True,
                        max_new_tokens=max_new_tokens,
                        stopping_criteria=[stopping_criteria],
                    )
                    prompt_text = tokenizer.decode(prompt[0])
                else:
                    print(f"[DEBUG] inferring with fixed prompt: {prompt}")
                    outputs_i = infer_with_prompt(
                        prompt,
                        model=model,
                        audio_encoding=ex["audio_encoding"],
                        multimodal_cfg=multimodal_cfg,
                        end_seq=end_seq,
                        tokenizer=tokenizer,
                        audio_first=True,
                        max_new_tokens=max_new_tokens,
                    )
                    prompt_text = prompt

                print("[PROMPT]")
                print(prompt_text)

                print("[MODEL COMPLETION]")
                # input_and_model_completion_text = tokenizer.decode(outputs_i[0])
                model_completion_ids = extract_response_tokens(outputs_i[0], end_seq)
                model_completion_text = tokenizer.decode(model_completion_ids)
                print(model_completion_text)

                print("[ORIGINAL/GROUND TRUTH COMPLETION]")
                orig_completion_ids = extract_response_tokens(ex["input_ids"], end_seq)
                orig_completion_text = tokenizer.decode(orig_completion_ids)
                print(orig_completion_text)

                output_dict = {
                    "example_id": ex["example_id"],
                    "prompt_text": prompt_text,
                    "original_completion_text": orig_completion_text,
                    "model_completion_text": model_completion_text,
                }

                outputs.append(output_dict)

                print("%" * 40)
                if max_samples and (i >= max_samples):
                    break

    print(f"writing {len(outputs)} results to {outfile}")
    pd.DataFrame(outputs).to_csv(outfile, index=False)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument(
        "--ckpt-num",
        type=int,
        help="Step number of the trained checkpoint.",
        required=True,
    )
    parser.add_argument("--max-samples", default=None, type=int, help="max eval samples to use.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to use. If set, this will override the prompt in all examples. "
        "Do not add conversation headers (e.g. 'ASSISTANT:') or other formatting"
        "to the prompt; these are added automatically under the hood.",
    )
    parser.add_argument(
        "--outfile",
        default="infer_results.csv",
        help="path to csv file to write results.",
    )
    (
        model_args,
        data_args,
        training_args,
        other_args,
    ) = parser.parse_args_into_dataclasses()

    main(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        **vars(other_args),
    )
