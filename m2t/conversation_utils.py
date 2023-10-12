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

from typing import Any, Dict, List

import numpy as np


def make_example(
    id: str,
    audio: str,
    audio_encoding: np.ndarray,
    audio_encoding_shape: List[int],
    prompt_question: str,
    response: str,
) -> Dict[str, Any]:
    return {
        "id": id,
        "audio": audio,
        "audio_encoding": audio_encoding,
        "audio_encoding_shape": audio_encoding_shape,
        "response": [{"question": prompt_question, "answer": response}],
    }


def subsequence_pos(ary, subary):
    """Helper function to find start/end indices of a subsequence in a sequence."""
    assert isinstance(ary, list)
    assert isinstance(subary, list)
    s = len(subary)
    for start_idx in range(len(ary) - s):
        if ary[start_idx : start_idx + s] == subary:
            return start_idx, start_idx + s


def extract_prompt_tokens(input_ids, end_seq):
    """Extract the input_ids from the prefix (i.e., before the model's response)."""
    _, prompt_end_idx = subsequence_pos(input_ids.tolist(), end_seq)
    return input_ids[:prompt_end_idx]


def extract_response_tokens(input_ids, end_seq):
    """Extract the input_ids from the model response."""
    _, prompt_end_idx = subsequence_pos(input_ids.tolist(), end_seq)
    return input_ids[prompt_end_idx:]
