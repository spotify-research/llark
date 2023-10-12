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
Usage:

# mtg-jamendo
python scripts/preprocessing/crop_annotations.py \
    --annotations-dir datasets/mtg-jamendo/annotated/ \
    --output ./datasets/mtg-jamendo/jamendo-annotated.jsonl \
    --audio-dir gs://bucketname/datasets/mtg-jamendo/wav-crop/ \
    --dataset-name mtg-jamendo
"""
import argparse
import json
import logging
import os
from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from m2t.dataset_utils import DATASET_INFO, read_jsonl_data
from m2t.gcs_utils import list_blobs_with_prefix, split_gcs_bucket_and_filepath


def parse_cropped_filenames(filenames) -> List[Tuple[str, float, float]]:
    parsed = []
    for f in filenames:
        basename = os.path.basename(f)
        basename = basename.rsplit(".", maxsplit=1)[0]  # drop file extension
        id, start_str, end_str = basename.split("-")
        try:
            start = float(start_str.replace("start", ""))
            end = float(end_str.replace("end", ""))
        except Exception:
            logging.warning(f"error parsing filename {f}; skipping")
        parsed.append((id, start, end))
    return parsed


def crop_column(
    crop_colname: str,
    df: pd.DataFrame,
    time_start_colname: str = "start_secs",
    time_end_colname: str = "end_secs",
    max_crop_duration: Optional[float] = None,
) -> pd.DataFrame:
    """Crop a column that does not have a start/end time (a fixed-time event like a beat).

    args:
        crop_colname: the column to crop. Should be a list of Python dictionary objects,
            where every element contains a 'time' key with a float as its corresponding value.
        df: the DataFrame. It should contain columns corresponding to time_start_colname
            and time_end_colname.
        time_start_colname: the start time column; of type float.
        time_end_colname: the end time column; of type float.
    """
    for i, row in tqdm(df.iterrows(), desc=f"crop {crop_colname}"):
        start, end = row[time_start_colname], row[time_end_colname]

        if max_crop_duration:
            end = min(end, start + max_crop_duration)

        cropped = [x for x in row[crop_colname] if x["time"] >= start and x["time"] <= end]

        # renormalize the times to the segment interval, modifying them inplace
        _ = [x.update({"time": x["time"] - start}) for x in cropped]
        # sanity check the crops
        assert (all(x["time"] >= 0 and x["time"] <= end - start) for x in cropped)
        df.at[i, crop_colname] = cropped
    return df


def crop_column_with_start_end(
    crop_colname: str,
    df: pd.DataFrame,
    time_start_colname: str = "start_secs",
    time_end_colname: str = "end_secs",
    max_crop_duration: Optional[float] = None,
) -> pd.DataFrame:
    """Crop a column with a start/end time (i.e. chords).

    args:
        crop_colname: the column to crop. Should be a list of Python dictionary objects,
            where every element contains a 'start_time' key and an 'end_time' key,
            each with a float as its corresponding value.
        df: the DataFrame. It should contain columns corresponding to time_start_colname
            and time_end_colname.
        time_start_colname: the start time column; of type float.
        time_end_colname: the end time column; of type float.
        max_crop_duration: maximum duration of any crop. If the input
            (end - start) times are greater than this value, the annotation
            will be truncated.
    """
    for i, row in tqdm(df.iterrows(), desc=f"crop {crop_colname}"):
        start, end = row[time_start_colname], row[time_end_colname]

        if max_crop_duration:
            end = min(end, start + max_crop_duration)

        cropped = [
            x for x in row[crop_colname] if x["end_time"] >= start and x["start_time"] <= end
        ]

        # renormalize the times to the segment interval, modifying them inplace
        for x in cropped:
            x.update(
                {
                    "start_time": max(x["start_time"] - start, 0),
                    "end_time": min(x["end_time"] - start, end - start),
                }
            )
        # sanity check the crops
        assert (all(x["start_time"] >= 0 and x["end_time"] <= end - start) for x in cropped)
        df.at[i, crop_colname] = cropped
    return df


def crop_midi_notes_column(
    df: pd.DataFrame,
    crop_colname="notes",
    time_start_colname: str = "start_secs",
    time_end_colname: str = "end_secs",
    max_crop_duration: Optional[float] = None,
) -> pd.DataFrame:
    for i, row in tqdm(df.iterrows(), desc="crop midi", total=len(df)):
        start, end = row[time_start_colname], row[time_end_colname]
        if max_crop_duration:
            end = min(end, start + max_crop_duration)

        # MIDI notes is a dict, where the keys are instrument names and the values are
        # sequences of dictionaries representing notes.
        midi_notes = json.loads(row[crop_colname])
        midi_notes_cropped = defaultdict(list)
        for inst, inst_notes in midi_notes.items():
            notes_cropped = [x for x in inst_notes if x["end"] >= start and x["start"] <= end]
            # renormalize the times to the segment interval, modifying them inplace
            for x in notes_cropped:
                x.update(
                    {
                        "start": max(x["start"] - start, 0),
                        "end": min(x["end"] - start, end - start),
                    }
                )
            # sanity check the crops
            assert (all(x["start"] >= 0 and x["end"] <= end - start) for x in notes_cropped)
            midi_notes_cropped[inst] = notes_cropped
        df.at[i, crop_colname] = midi_notes_cropped

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations-dir",
        help="Path to a directory containing annotations JSONL files.",
    )

    parser.add_argument(
        "--annotations-file",
        help="path to a JSON file containing the annotations.",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="path to directory containing the (cropped) audio.",
    )
    parser.add_argument("--output", required=True, help="Path to a JSON file for output.")
    parser.add_argument(
        "--max-crop-duration",
        type=float,
        default=25.0,
        help="The maximum duration, from start time, to use when cropping."
        "This reflects the fact that e.g. JukeBox only takes the first ~25s of the audio.",
    )
    parser.add_argument(
        "--dataset-name",
        choices=["fma", "mtg-jamendo", "musicnet", "slakh", "fsl10k"],
        required=True,
    )

    args = parser.parse_args()
    assert (args.annotations_dir or args.annotations_file) and not (
        args.annotations_dir and args.annotations_file
    ), "Specify one of either --annotations-dir or --anotations-file, and not both."
    assert args.output.endswith(".jsonl"), "outfile must end with .jsonl."
    assert args.audio_dir.endswith(
        "/"
    ), "use a trailing slash for audio dir to avoid returning the contents of subdirectories."
    dataset_info = DATASET_INFO[args.dataset_name]

    # read the annotations into a dataframe
    if args.annotations_dir:
        df = read_jsonl_data(os.path.join(args.annotations_dir, "*.jsonl"))
    elif args.annotations_file:
        df = pd.read_json(args.annotations_file, lines=True)
    df = dataset_info.preprocess_id_col(df)

    # for each annotated element, fetch the cropping info.
    bucket_name, prefix = split_gcs_bucket_and_filepath(args.audio_dir)
    blobs = list_blobs_with_prefix(bucket_name, prefix)
    crop_filenames = [x.name for x in blobs]
    id_start_end = parse_cropped_filenames(crop_filenames)
    crop_df = pd.DataFrame(id_start_end, columns=[dataset_info.id_col, "start_secs", "end_secs"])
    crop_df = dataset_info.preprocess_id_col(crop_df)

    df = df.merge(crop_df, on=dataset_info.id_col)
    assert len(df), "got no elements after merging; do the id columns match?"

    if "downbeats_madmom" in df.columns:
        df = crop_column("downbeats_madmom", df, max_crop_duration=args.max_crop_duration)

    if "chords" in df.columns:
        df = crop_column_with_start_end("chords", df, max_crop_duration=args.max_crop_duration)
    if "notes" in df.columns:
        df = crop_midi_notes_column(df, max_crop_duration=args.max_crop_duration)

    print(f"writing output to {args.output}")
    outdir = os.path.dirname(args.output)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    df.to_json(args.output, orient="records", lines=True)
