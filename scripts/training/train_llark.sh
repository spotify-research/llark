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

# llama, 4GPU (global batch size 32)
ulimit -n 32768 && \
HF_HOME='/tmp' TOKENIZERS_PARALLELISM='true' \
 TRANSFORMERS_VERBOSITY=debug \
 WANDB_API_KEY="" \
 HUGGING_FACE_HUB_TOKEN="" \
 python -m torch.distributed.launch \
 --nproc_per_node 4 \
 -m m2t.train \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--output_dir checkpoints/meta-llama/Llama-2-7b-chat-hf/$RANDOM \
--gradient_checkpointing True \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-5 \
--freeze_backbone False \
--tune_mm_mlp_adapter True \
--save_total_limit 1 \
--mm_use_audio_start_end \
--report_to wandb \
--bf16 True \
--tf32 True \
--lr_scheduler_type "cosine" \
--warmup_ratio 0.03 \
--weight_decay 0. \
--max_steps 100000 \
--model_max_length 2048 \
--save_strategy steps \
--save_steps 5000 \
--logging_steps 1 \
--ddp_find_unused_parameters False \
--dataloader_num_workers 8 \
--train_data_path \
"gs://bucket/path/to/data-{000000..000999}.tar,"\
"gs://bucket/path/to/moredata-{000000..000999}.tar" \
--evaluation_strategy "no"