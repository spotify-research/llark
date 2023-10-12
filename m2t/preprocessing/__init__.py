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

from .jsonify import (
    DatasetJsonifier,
    FmaJsonifier,
    Fsl10kJsonifier,
    GiantStepsKeyJsonifier,
    GiantStepsTempoJsonifier,
    JamendoJsonifier,
    MagnaTagATuneJsonifier,
    MusiccapsJsonifier,
    MusicNetJsonifier,
    SlakhJsonifier,
    WavCapsJsonifier,
    YT8MMusicTextClipsJsonifier,
)

_JSONIFIERS = {
    "jamendo": JamendoJsonifier,
    "fma": FmaJsonifier,
    "fsl10k": Fsl10kJsonifier,
    "wavcaps": WavCapsJsonifier,
    "giantsteps-key": GiantStepsKeyJsonifier,
    "giantsteps-tempo": GiantStepsTempoJsonifier,
    "magnatagatune": MagnaTagATuneJsonifier,
    "yt8m-musictextclips": YT8MMusicTextClipsJsonifier,
    "musicnet": MusicNetJsonifier,
    "musiccaps": MusiccapsJsonifier,
    "slakh": SlakhJsonifier,
}


def get_jsonifier(dataset: str, *args, **kwargs) -> DatasetJsonifier:
    assert dataset in _JSONIFIERS, f"dataset {dataset} not in valid datasets {_JSONIFIERS.keys()}"
    cls = _JSONIFIERS[dataset]
    return cls(name=dataset, *args, **kwargs)
