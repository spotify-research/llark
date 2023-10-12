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

MAGNATAGATUNE_TRAIN_CHUNKS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
]
MAGNATAGATUNE_VALIDATION_CHUNKS = [
    "b",
    "c",
]
MAGNATAGATUNE_TEST_CHUNKS = ["d", "e", "f"]


def extract_id_from_mp3_path(path) -> str:
    fname = os.path.basename(path)
    return fname.replace(".mp3", "")
