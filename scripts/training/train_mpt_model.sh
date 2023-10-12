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

# MPT, 4GPU, webdataset
ulimit -n 32768 && \
 HF_HOME='/tmp' TOKENIZERS_PARALLELISM='true' \
 WANDB_API_KEY="" \
 TRANSFORMERS_VERBOSITY=debug \
 python -m torch.distributed.launch \
 --nproc_per_node 4 \
 -m m2t.train \
--model_name_or_path mosaicml/mpt-1b-redpajama-200b-dolly \
--output_dir checkpoints/mosaicml/mpt-1b-redpajama-200b-dolly/$RANDOM \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--learning_rate 5e-5 \
--freeze_backbone False \
--tune_mm_mlp_adapter True \
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
--save_steps 25000 \
--save_total_limit 1 \
--logging_steps 1 \
--ddp_find_unused_parameters False \
--dataloader_num_workers 8 \
--dataloader_pin_memory False \
--train_data_path \
"gs://bucket/path/to/data-{000000..000999}.tar,"\
"gs://bucket/path/to/moredata-{000000..000999}.tar" \
--evaluation_strategy "no"