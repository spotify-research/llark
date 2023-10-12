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
Utilities for loading and preprocessing data.
"""

import copy
import json
import logging
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

import braceexpand
import msgspec
import numpy as np
import torch
import transformers
import webdataset as wds
from datasets import IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node

from m2t.llava import conversation as conversation_lib
from m2t.special_tokens import (
    DEFAULT_AUDIO_END_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
    DEFAULT_AUDIO_TOKEN,
    IGNORE_INDEX,
)

SHARD_SHUFFLE_SEED = 936629
DEFAULT_CONVERSATION_HEADER = f"{conversation_lib.default_conversation.system}\n\n"


def read_jsonl_data(file) -> List[Dict[Any, Any]]:
    """Read and parse a jsonl file line by line and return the resulting list."""
    logging.debug(f"reading {file}")
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def sentences_to_formatted_conversation(header, source, get_conversation=True) -> str:
    """Add speaker and start/end signal on each round and concat to a sentence."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[Dict[str, str]],
    multimodal_cfg: dict,
    encoded_shapes: Sequence[Sequence[int]],
) -> Dict:
    """
    Add start/end/patch tokens for multimodal inputs and optionally reformat the conversation.
    """
    is_multimodal = multimodal_cfg["is_multimodal"]
    if not is_multimodal:
        return sources

    for encoded_shape, source in zip(encoded_shapes, sources):
        audio_token_len, audio_token_dim = encoded_shape
        if multimodal_cfg["sep_audio_conv_front"]:
            assert DEFAULT_AUDIO_TOKEN in source[0]["value"]
            source[0]["value"] = source[0]["value"].replace(DEFAULT_AUDIO_TOKEN, "").strip()
            source[0]["value"] = (
                DEFAULT_AUDIO_TOKEN
                + conversation_lib.default_conversation.sep
                + conversation_lib.default_conversation.roles[0]
                + ": "
                + source[0]["value"]
            )
        replace_token = DEFAULT_AUDIO_PATCH_TOKEN * audio_token_len
        if multimodal_cfg["use_audio_start_end"]:
            replace_token = DEFAULT_AUDIO_START_TOKEN + replace_token + DEFAULT_AUDIO_END_TOKEN

        for sentence in source:
            sentence["value"] = sentence["value"].replace(DEFAULT_AUDIO_TOKEN, replace_token)

    return sources


def preprocess_for_lm(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocess examples for training with a language modeling objective.
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    assert conversation_lib.default_conversation.version not in ("v1", "mpt")

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = DEFAULT_CONVERSATION_HEADER
        conversation = sentences_to_formatted_conversation(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)[
            "input_ids_lens"
        ]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_encodings(audio_encoding, audio_encoding_shape: List[int]) -> torch.Tensor:
    """Reshape audio encodings and squeeze out extra third dim if present."""
    encoding = torch.Tensor(audio_encoding)
    encoding = encoding.reshape(audio_encoding_shape)
    if encoding.ndim == 3:
        encoding = torch.squeeze(encoding, 0)
    return encoding


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "audio_encoding" in instances[0]:
            encodings = [instance["audio_encoding"] for instance in instances]
            if all(x is not None and x.shape == encodings[0].shape for x in encodings):
                batch["audio_encodings"] = torch.stack(encodings)
            else:
                batch["audio_encodings"] = encodings
        else:
            logging.warning("key `audio_encoding` not detected in data collator inputs.")

        return batch


def _get_audio_token_len(enc_shape: List[int]) -> int:
    """Get the audio encoding length."""
    if len(enc_shape) == 3 and enc_shape[0] == 1:
        audio_token_len = enc_shape[1]
    else:
        audio_token_len = enc_shape[0]
    return audio_token_len


def preprocess_multimodal_mappable(e: Dict[str, Any], multimodal_cfg: Dict[str, Any]):
    """Vectorized version of preprocess_multimodal().

    Args:
        e: dictionary containing sample data.
        multimodal_cfg: Dictionary specifyig the multimodal configurations.
    """
    assert not multimodal_cfg[
        "sep_audio_conv_front"
    ], "need to implement logic for this; see datasets.preprocess_multimodal()."
    # audio_token_len, audio_token_dim = e["audio_encoding"].shape

    audio_token_len = _get_audio_token_len(e["audio_encoding_shape"])

    replace_token = DEFAULT_AUDIO_PATCH_TOKEN * audio_token_len
    source = e["conversations"]
    preprocessed_conversations = []
    if multimodal_cfg["use_audio_start_end"]:
        replace_token = DEFAULT_AUDIO_START_TOKEN + replace_token + DEFAULT_AUDIO_END_TOKEN
    for sentence in source:
        conv = copy.deepcopy(sentence)
        conv["value"] = sentence["value"].replace(DEFAULT_AUDIO_TOKEN, replace_token)
        preprocessed_conversations.append(conv)
    e["conversations"] = preprocessed_conversations
    return e


def preprocess_for_lm_mappable(
    e: Dict[str, Any], tokenizer, header: str = DEFAULT_CONVERSATION_HEADER
):
    """Vectorized version of preprocess_for_lm()."""
    source = e["conversations"]

    conversation = sentences_to_formatted_conversation(header, source)
    conversation_tokenized = _tokenize_fn([conversation], tokenizer)
    # _tokenize_fn will return a single-element list, so we access the IDs tensor
    input_ids = conversation_tokenized["input_ids"][0]
    target = copy.deepcopy(input_ids)
    tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)[
        "input_ids_lens"
    ]
    speakers = [sentence["from"] for sentence in source]
    _mask_targets(target, tokenized_lens, speakers)

    audio_encoding = preprocess_encodings(e["audio_encoding"], e["audio_encoding_shape"])
    return dict(
        input_ids=input_ids,
        labels=target,
        audio_encoding=audio_encoding,
        example_id=e["id"],
    )


def concat_audio_token_and_prompt(prompt: str, audio_first: bool) -> str:
    if audio_first:
        prompt_text = "\n".join(("<audio>", prompt))
    else:
        prompt_text = "\n".join((prompt, "<audio>"))
    return prompt_text


def webdataset_element_to_conversation(src):
    """Parse each element of the dataset into a conversation.

    Yields samples, which is compatible with Webdataset.compose().

    This takes the (multiple) responses from ChatGPT for a single observation,
        and unpacks them into individual standalone question-answer pairs, each
        containing the extra data required for training (i.e. audio_encoding and
        audio_encoding_shape).
    """
    for elem in src:
        if "json" not in elem or not elem["json"].get("response"):
            logging.warning(f"no valid json response for {elem['__key__']}; skipping")
            continue

        if not isinstance(elem["json"]["response"], list):
            logging.warning(f"invalid response for {elem['__key__']} " "(could be empty); skipping")
            continue

        try:
            audio_encoding = elem["audio_encoding.pyd"]
            audio_encoding_shape = list(audio_encoding.shape)

        except AttributeError:
            logging.warning(
                f"error reading encoding for {elem['__key__']}, potentially the file is "
                "corrupted; skipping"
            )
            continue

        for response in elem["json"]["response"]:
            question, answer = (response["question"], response["answer"])

            audio_first = random.uniform(0.0, 1.0) > 0.5
            prompt_text = concat_audio_token_and_prompt(question, audio_first)
            output = {
                "audio_encoding": audio_encoding,
                "audio_encoding_shape": audio_encoding_shape,
                "id": elem["__key__"],
                # "audio": audio_filename,
                "conversations": [
                    {"from": "human", "value": prompt_text},
                    {"from": "gpt", "value": answer},
                ],
            }
            yield output


def assert_hf_src_format(src):
    """Check the formatting of a set of sources with an assertion.

    If this function does *not* raise an assertionerror, src is a dict,
    where keys are strings and values are lists. The ith element in the
    dataset is reprented by the ith element of each of those lists.

    For example, the json for the ith element is in src['json'][i], the
    audio encoding for the ith eleemnt is in src['audio_encoding'][i], etc.
    """
    assert isinstance(src, dict)
    # print(f"[DEBUG] src has keys {src.keys()}")
    # for k in src.keys():
    #     print(f"[DEBUG] src[{k}] has type {type(src[k])} with len {len(src[k])}")

    dict_keys = list(src.keys())
    assert all(
        isinstance(src[k], list) for k in dict_keys
    ), f"expected dict of lists, got: {[(k,type(src[k])) for k in dict_keys]}"

    assert all(
        len(src[k]) == len(src[dict_keys[0]]) for k in dict_keys
    ), f"expected lists of same length, got {[(k,len(src[k])) for k in dict_keys]}"


def _to_conversation_hf(src: Dict[str, List]) -> Dict[str, List]:
    """Parse each element of the dataset into a conversation.

    This takes the (multiple) responses from ChatGPT for a single observation,
        and unpacks them into individual standalone question-answer pairs, each
        containing the extra data required for training (i.e. audio_encoding and
        audio_encoding_shape).
    """
    from collections import defaultdict

    output_dict = defaultdict(list)

    assert_hf_src_format(src)
    batch_size = len(src["__key__"])

    for i in range(batch_size):
        elem_json = src["json"][i]
        elem_id = src["__key__"][i]

        # flattened list of floats; needs to be reshaped to
        # audio_encoding_shape in final stage of pipeline.
        elem_audio_encoding = src["audio_encoding"][i]
        elem_audio_encoding_shape = src["audio_encoding_shape"][i]

        # for elem in src:
        if not elem_json.get("response"):
            logging.warning(f"no response for {src['__key__'][i]}; skipping")
            continue

        if not isinstance(elem_json["response"], list):
            logging.warning(
                f"invalid response for {src['__key__'][i]} " "(could be empty); skipping"
            )
            continue

        # Drop the response key from the json; this can be quite large.
        output_json = {k: elem_json[k] for k in elem_json.keys() if k != "response"}

        for response in elem_json["response"]:
            question, answer = (response["question"], response["answer"])

            audio_first = random.uniform(0.0, 1.0) > 0.5
            if audio_first:
                prompt_text = "\n".join(("<audio>", question))
            else:
                prompt_text = "\n".join((question, "<audio>"))
            output = {
                "audio_encoding": elem_audio_encoding,
                "audio_encoding_shape": elem_audio_encoding_shape,
                "__key__": elem_id,
                "id": elem_id,
                "json": output_json,
                "conversations": [
                    {"from": "human", "value": prompt_text},
                    {"from": "gpt", "value": answer},
                ],
            }
            for k, v in output.items():
                output_dict[k].append(v)
    return output_dict


def maybe_add_gcs_prefix(f: str) -> str:
    if f.startswith("gs://"):
        return f"pipe:gsutil cat {f}"
    return f


def expand_url_to_file_list(url: str) -> List[str]:
    urls = [file for wildcard in url.split(",") for file in braceexpand.braceexpand(wildcard)]
    return urls


def repeat_shards(urls: List[str], task_sample_probs: Optional[Dict[str, float]] = None):
    if task_sample_probs is not None:
        logging.warning(f"applying task probs {task_sample_probs}")

        def _shard_prob(shard) -> float:
            """Naive function to check if any key from task_sample_probs is in the
            shard name, and if so, return the probability."""
            for k, prob in task_sample_probs.items():
                if k in shard:
                    return prob
            raise ValueError(
                f"probability for shard {shard} not defined in probs {task_sample_probs}"
            )

        probs = [_shard_prob(s) for s in urls]
        probs = np.array(probs) / sum(probs)
    else:
        probs = None
    _REPEATS = 1024 * len(
        urls
    )  # note: must repeat data explicity when using HF Trainer with IterableDataset.
    urls = np.random.choice(urls, size=_REPEATS, replace=True, p=probs).tolist()
    return urls


def read_webdataset(
    url: str,
    multimodal_cfg: Dict[str, Any],
    tokenizer,
    is_train: bool,
    rsample_frac=None,
    task_sample_probs: Optional[Dict[str, float]] = None,
) -> wds.WebDataset:
    _preprocess_multimodal = partial(preprocess_multimodal_mappable, multimodal_cfg=multimodal_cfg)

    _preprocess_for_lm = partial(preprocess_for_lm_mappable, tokenizer=tokenizer)

    # Basic parsing of comma-separated train data paths; to use more complex sampling
    # methods via a custom IterableDataset class, refer to https://webdataset.github.io/webdataset/sources/
    logging.warning(f"reading datasets from {url}")
    urls = expand_url_to_file_list(url)
    urls = [maybe_add_gcs_prefix(f) for f in urls]

    if is_train:
        urls = repeat_shards(urls, task_sample_probs=task_sample_probs)

    do_shuffle = is_train or (rsample_frac is not None)

    dataset = wds.WebDataset(
        urls,
        resampled=is_train,
        handler=wds.warn_and_continue,
        shardshuffle=is_train,
        nodesplitter=wds.split_by_node,
    )  # at this point, we have an iterator over the shards assigned to each worker

    if is_train:
        dataset = dataset.repeat()

    # at this point, we have an list of decompressed training samples from
    # each shard in this worker in sequence, with the audio encodings
    # decompressed to torch.Tensors.
    dataset = dataset.decode(wds.imagehandler("torchrgb"))

    # Each element in dataset has keys:
    # ['__key__', '__url__', 'audio_encoding.pyd', 'json'];
    # here we parse it to conversation format and tokenize the data.
    dataset = dataset.compose(webdataset_element_to_conversation)

    if do_shuffle:
        dataset = dataset.shuffle(100)
    if rsample_frac:
        dataset = dataset.rsample(rsample_frac)

    dataset = dataset.map(_preprocess_multimodal).map(_preprocess_for_lm)
    if is_train:
        dataset_len = 1_000_000_000

        dataset = dataset.repeat(2).with_epoch(dataset_len)
    return dataset


_SHUFFLE_BUFFER_SIZE = 1000  # this is default hf buffer size.


def gen_from_webdataset_shards(shards, multimodal_cfg, tokenizer, is_train: bool):
    _preprocess_multimodal = partial(preprocess_multimodal_mappable, multimodal_cfg=multimodal_cfg)
    _preprocess_for_lm = partial(preprocess_for_lm_mappable, tokenizer=tokenizer)

    dataset = wds.WebDataset(
        shards,
        resampled=is_train,
        handler=wds.warn_and_continue,
        shardshuffle=is_train,
    )

    # at this point, we have an list of decompressed training samples from
    # each shard in this worker in sequence, with the audio encodings
    # decompressed to torch.Tensors.
    dataset = dataset.decode(wds.imagehandler("torchrgb"))

    # Each element in dataset has keys:
    # ['__key__', '__url__', 'audio_encoding.pyd', 'json'];
    # here we parse it to conversation format and tokenize the data.
    dataset = dataset.compose(webdataset_element_to_conversation)

    # TODO(jpgard): shuffle here.
    dataset = dataset.map(_preprocess_multimodal).map(_preprocess_for_lm)

    if is_train:
        dataset_len = 1_000_000_000

        # uncomment below to very quickly hit the DDP dataloading bug.
        # dataset = dataset.repeat().with_epoch(512).with_length(512)

        dataset = dataset.repeat(2).with_epoch(dataset_len)

    for sample in dataset:
        yield sample


def read_hf_webdataset(
    url: str,
    multimodal_cfg: Dict[str, Any],
    tokenizer,
    is_train: bool,
    rsample_frac=None,
):
    """Read a dataset in webdataset format, as a HF dataset."""
    # TODO(jpgard): if we stick with this design, instead, we should spin the URLs
    # into separate datasets with fixed sampling probabilities (or, use the sampling
    # weights parameter in np.random.choice() to set it there.)
    urls = expand_url_to_file_list(url)
    if is_train:
        urls = repeat_shards(urls)

    assert urls[0].endswith(".tar")

    dataset = IterableDataset.from_generator(
        gen_from_webdataset_shards,
        gen_kwargs={
            "shards": urls,
            "multimodal_cfg": multimodal_cfg,
            "tokenizer": tokenizer,
            "is_train": is_train,
        },
    )

    if os.environ.get("WORLD_SIZE") and int(os.environ["WORLD_SIZE"]) > 1:
        dataset = split_dataset_by_node(
            dataset,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )

    return dataset


def gen_from_msgpack_shards(shards):
    for shard in shards:
        if (not os.path.exists(shard)) and ("gs://" not in shard):
            logging.warning(f"skipping nonexistent shard {shard}")
            continue
        try:
            with open(shard, "rb") as f:
                decoded = msgspec.msgpack.decode(f.read())
        except Exception as e:
            logging.warning(f"error decoding data from shard {shard}: {e}; skipping.")
            continue
        # decoded should be a LIST of examples
        assert isinstance(
            decoded, list
        ), f"expected list after decoding {shard}; got {type(decoded)}"
        for sample in decoded:
            yield sample


def hf_preprocess_encodings(src: Dict[str, List]) -> Dict[str, List]:
    """Preprocess encodings in place."""
    enc = preprocess_encodings(src["audio_encoding"], src["audio_encoding_shape"])
    src["audio_encoding"] = enc
    return src


def read_hf_dataset(
    url: str,
    multimodal_cfg: Dict[str, Any],
    tokenizer,
    is_train: bool,
    rsample_frac=None,
):
    _preprocess_multimodal = partial(preprocess_multimodal_mappable, multimodal_cfg=multimodal_cfg)
    _preprocess_for_lm = partial(preprocess_for_lm_mappable, tokenizer=tokenizer)

    # TODO(jpgard): if we stick with this design, instead, we should spin the URLs
    # into separate datasets with fixed sampling probabilities (or, use the sampling
    # weights parameter in np.random.choice() to set it there.)
    urls = expand_url_to_file_list(url)
    if is_train:
        urls = repeat_shards(urls)

    if urls[0].endswith(".jsonl") or urls[0].endswith(".json"):
        dataset = load_dataset("json", data_files=urls, split="train", streaming=True)
    elif urls[0].endswith(".msgpack"):
        dataset = IterableDataset.from_generator(
            gen_from_msgpack_shards, gen_kwargs={"shards": urls}
        )
    dataset = split_dataset_by_node(
        dataset,
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )

    # Expand/flatten data so that each example contains a single QA pair,
    # pplus audio encodings and metadata (i.e., shape).
    dataset = dataset.map(
        _to_conversation_hf, batched=True, batch_size=8
    )  # TODO(jpgard): tune batch size. Must stay batched in order for FlatMap-type behavior.
    # At this point, each element is a dict mapping {str:List} where the values
    # are lists of length batch_size.

    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE)

    dataset = dataset.map(_preprocess_multimodal, batched=False)
    dataset = dataset.map(_preprocess_for_lm, batched=False)
    dataset = dataset.map(hf_preprocess_encodings, batched=False)

    return dataset


def make_mm_config(data_args):
    return dict(
        is_multimodal=data_args.is_multimodal,
        sep_audio_conv_front=data_args.sep_audio_conv_front,
        audio_folder=data_args.audio_folder,
        use_audio_start_end=getattr(data_args, "mm_use_audio_start_end", False),
        audio_processor=getattr(data_args, "audio_encoding_processor", None),
    )


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    data_collator_cls=DataCollatorForSupervisedDataset,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    multimodal_cfg = make_mm_config(data_args)

    train_dataset = None
    eval_dataset = None

    if data_args.train_data_path:
        train_dataset = read_webdataset(
            data_args.train_data_path,
            multimodal_cfg=multimodal_cfg,
            tokenizer=tokenizer,
            is_train=True,
            task_sample_probs=data_args.task_sample_probs
            if data_args.apply_task_sample_probs
            else None,
        )

    if data_args.eval_data_path:
        eval_dataset = read_webdataset(
            data_args.eval_data_path,
            multimodal_cfg=multimodal_cfg,
            tokenizer=tokenizer,
            is_train=False,
            rsample_frac=data_args.eval_data_subsample,
        )

    data_collator = data_collator_cls(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
