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

from dataclasses import dataclass

from m2t.special_tokens import (
    DEFAULT_AUDIO_END_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
)


@dataclass
class AudioEncoderConfig:
    use_audio_start_end: bool = True
    use_audio_start_end: bool = True
    audio_start_token: str = DEFAULT_AUDIO_START_TOKEN
    audio_end_token: str = DEFAULT_AUDIO_END_TOKEN
    audio_patch_token = str = DEFAULT_AUDIO_PATCH_TOKEN
