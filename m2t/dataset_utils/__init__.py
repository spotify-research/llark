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

import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from tqdm import tqdm

from m2t.gcs_utils import read_audio_encoding
from m2t.instruct.captioning import CAPTIONING_PROMPTS


def fetch_audio_start_end(example_id: str) -> Tuple[float, float]:
    start_str = re.search("start(\d+\\.\d+)", example_id)
    if start_str is not None:
        start_str = float(start_str.group(1))
    end_str = re.search("end(\d+\\.\d+)", example_id)
    if end_str is not None:
        end_str = float(end_str.group(1))
    return start_str, end_str


def fetch_true_example_id(example_id: str) -> str:
    """Reverse the string substitution that happens in webdataset.

    For webdataset, '.' characters in keys are not supported, so we replace
    them with underscores. However, we need to recover the original UIDs
    (for example, to read the audio file). This reverses the process; for
    example, it would convert
    'mysong_start_0_000_end_30_000' --> mysong_start_0.000_end_30.000
    """
    start_str = re.search("start\d+_", example_id)
    if start_str is not None:
        start_str = start_str.group()
        example_id = example_id.replace(start_str, start_str[:-1] + ".")
    end_str = re.search("end\d+_", example_id)
    if end_str is not None:
        end_str = end_str.group()
        example_id = example_id.replace(end_str, end_str[:-1] + ".")
    return example_id


def wds_key_to_example_id(key: str, dataset_name) -> str:
    """Map the webdataset-friendly key format back to the original key format."""
    res = re.search("start\d+_\d+", key)
    if res:
        to_replace = res.group()
        replacement = to_replace.replace("_", ".")  # e.g. 'start30_000' -> 'start30.000'
        key = key.replace(to_replace, replacement)
    res = re.search("end\d+_\d+", key)
    if res:
        to_replace = res.group()
        replacement = to_replace.replace("_", ".")  # e.g. 'end60_000' -> 'end60.000'
        key = key.replace(to_replace, replacement)
    if dataset_name != "magnatagatune":
        key = key.replace("_", ".")
    return key


def make_start_end_str(start_secs: float, end_secs: float) -> str:
    """Create the string-formatted start/end string used for observation IDs.

    This string uniquely identifies the start/end time used when cropping a piece
        of audio, and can be used to match the cropped segment to the corresponding
        segment in the original, full-length audio.
    """
    return f"start{start_secs:.3f}-end{end_secs:.3f}"


def get_cropped_uri(uid: Any, start_secs: float, end_secs: float) -> str:
    """Fetch the uri for the element, cropped at the specified start/end.

    Args:
        uid: the original uid of the (uncropped) observation.
        start_secs: start time of the cropped audio, in seconds.
        end_secs: end time of the cropped audio, in seconds.
    Returns:
        The string URI for the element cropped as specified.
    """
    start_end_str = make_start_end_str(start_secs=start_secs, end_secs=end_secs)
    return f"{uid}-{start_end_str}"


@dataclass
class DatasetInfo:
    """Class to represent information about a dataset.

    By default, datasets have a unique identifying field called 'id',
    and this field is used to fetch the audio via {id}.wav. If this is
    *not* true for a dataset, then the method .id_to_filename() and id_col
    may need to be overriden.
    """

    id_col: str = "id"
    caption_col: Optional[str] = None

    def preprocess_id_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function to apply any preprocessing to the id column."""
        df[self.id_col] = df[self.id_col].astype(str)
        return df

    def id_to_filename(self, track_id: str, dirname: Optional[str] = None):
        if not isinstance(track_id, str):
            track_id = str(track_id)
        filename = str(track_id) + ".wav"
        if dirname:
            filename = os.path.join(dirname, filename)
        return filename

    @property
    def caption_prompts(self) -> Union[List[str], None]:
        return CAPTIONING_PROMPTS.get(self.name)


@dataclass
class Fsl10kDatasetInfo(DatasetInfo):
    name = "fsl10k"

    def preprocess_id_col(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.id_col] = df[self.id_col].astype(str)
        df[self.id_col] = df[self.id_col].apply(lambda x: x.replace(".wav", ""))
        return df


@dataclass
class MusicCapsDatasetInfo(DatasetInfo):
    name = "musiccaps"


@dataclass
class JamendoDatasetInfo(DatasetInfo):
    name = "mtg-jamendo"


@dataclass
class FmaDatasetInfo(DatasetInfo):
    name = "fma"

    def preprocess_id_col(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.id_col] = df[self.id_col].apply(lambda x: f"{x:06}")
        return df

    def id_to_filename(self, track_id: Union[str, int], dirname: Optional[str] = None):
        # python formatter requires int for aligned/padded numbers
        track_id = int(track_id)

        filename = f"{track_id:06}.wav"
        if dirname:
            filename = os.path.join(dirname, filename)
        return filename


@dataclass
class GiantStepsDatasetInfo(DatasetInfo):
    name = "giant_steps"
    id_col = None


@dataclass
class MusicNetDatasetInfo(DatasetInfo):
    name = "musicnet"


@dataclass
class SlakhDatasetInfo(DatasetInfo):
    name = "slakh"


@dataclass
class MagnaTagATuneDatasetInfo(DatasetInfo):
    name = "magnatagatune"


@dataclass
class YT8MMusicTextClipsDatasetInfo(DatasetInfo):
    name = "yt8m-musictextclips"


DATASET_INFO: Dict[str, DatasetInfo] = {
    "fsl10k": Fsl10kDatasetInfo(),
    "fma": FmaDatasetInfo(id_col="track.id"),
    "mtg-jamendo": JamendoDatasetInfo(),
    "magnatagatune": MagnaTagATuneDatasetInfo(id_col="example_id"),
    "musiccaps": MusicCapsDatasetInfo(
        id_col="ytid",
        caption_col="caption",
    ),
    "yt8m-musictextclips": YT8MMusicTextClipsDatasetInfo(
        id_col="video_id",
        caption_col="text",
    ),
    "musicnet": MusicNetDatasetInfo(),
    "slakh": SlakhDatasetInfo(id_col="id"),
}


def read_jsonl_data(path: str) -> pd.DataFrame:
    """Read JSONL file(s) from a wildcard path and return a DataFrame."""
    files = glob.glob(path)
    if not len(files):
        raise ValueError(f"no files found matching {path}")
    out = []
    for f in tqdm(files, desc=f"read {path}"):
        annotations = pd.read_json(path_or_buf=f, lines=True)
        out.append(annotations)

    if len(out) > 1 and not all(
        set(out[0].columns) == set(out[j].columns) for j in range(1, len(out))
    ):
        logging.warning(
            "got different sets of columns for different datasets;"
            " there may be an alignment issue with the data."
        )

    df = pd.concat(out)
    return df


def format_examples_for_model(elem: Dict[str, Any]) -> Dict[str, Any]:
    """Format a set of examples for training/inference."""
    audio_encoding = elem.pop("audio_encoding")
    audio_encoding_shape = elem.pop("audio_encoding_shape")
    key = elem.pop("id")
    return {
        "__key__": key,
        "json": elem,
        "audio_encoding": audio_encoding,
        "audio_encoding_shape": audio_encoding_shape,
    }


def maybe_trim_json(
    elem: Dict[str, Any], fields_to_keep: Sequence[str], trim_json: bool
) -> Dict[str, Any]:
    """If trim_json, remove all fields under the json key except fields_to_keep."""
    if trim_json:
        assert all([x in elem["json"] for x in fields_to_keep])
        elem["json"] = {k: elem["json"][k] for k in fields_to_keep}
    return elem


def read_and_insert_audio_encoding(
    elem: Dict[str, Any], representations_dir: str, id_colname: str
) -> Dict[str, Any]:
    """
    Read the audio encoding associated with an example, if it exists, and include it.
    """
    audio_encoding = read_audio_encoding(
        elem[id_colname], representations_dir, numpy_to_torch=False
    )
    if audio_encoding is not None:
        elem["audio_encoding_shape"] = (
            list(audio_encoding.shape) if audio_encoding is not None else None
        )
        audio_encoding = audio_encoding.flatten().tolist()
    elem["audio_encoding"] = audio_encoding
    return elem


def read_ids_file(filename: str) -> set:
    """Read a newline-delimited file containing a set of IDs.

    Note that the datatype of the elements returned from this function will
    always be Python string, since the elements are read from a text file.
    """
    assert os.path.exists(filename), f"{filename} does not exist."
    with open(filename, "r") as f:
        ids = f.readlines()
    ids = set([x.strip() for x in ids])
    return ids
