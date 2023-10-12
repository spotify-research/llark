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
import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import note_seq
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from m2t.dataset_utils.magnatagatune import (
    MAGNATAGATUNE_TEST_CHUNKS,
    MAGNATAGATUNE_TRAIN_CHUNKS,
    MAGNATAGATUNE_VALIDATION_CHUNKS,
    extract_id_from_mp3_path,
)
from m2t.dataset_utils.midi import get_formatted_notes_list, pitch_to_note_labels
from m2t.dataset_utils.mtg_jamendo import mtg_jamendo_read_file
from m2t.dataset_utils.slakh2100_redux import (
    DRUM_PITCH_TO_NAME,
    MIDI_PROGRAM_TO_SLAKH_CLASSES,
    TEST_TRACKS,
    TRAIN_TRACKS,
)
from m2t.dataset_utils.wavcaps import (
    JSON_TO_DATASET_NAME,
    keyword_filter,
    length_filter,
)


def extract_text_from_html(html):
    """Experimental function to parse HTML and return only its text contents."""
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text


@dataclass
class DatasetJsonifier(ABC):
    input_dir: str
    name: str
    split: str
    data: Sequence[Any] = None

    @abstractmethod
    def load_raw_data(self):
        """Loads the dataset."""
        raise

    def export_to_json(self, output_dir, examples_per_shard: Optional[int] = None):
        if not self.data:
            print("[WARNING] no data to write; returning.")
            return
        if self.split:
            fp = os.path.join(output_dir, self.name + f"-{self.split}.json")
        else:
            fp = os.path.join(output_dir, self.name + ".json")

        print(f"[INFO] writing {len(self.data)} records to {fp}")
        with open(fp, "w") as f:
            for elem in self.data:
                f.write(json.dumps(elem) + "\n")
        return


@dataclass
class WavCapsJsonifier(DatasetJsonifier):
    use_keyword_filter: bool = True
    use_length_filter: bool = True
    minimum_caption_length: int = 99  # set to a large number to ensure it is overridden

    def _apply_filter(self, data: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        def filter_fn(caption) -> bool:
            caption = caption.lower()
            kw_filter_result = (not self.use_keyword_filter) or keyword_filter(caption)
            len_filter_result = (not self.use_length_filter) or length_filter(
                caption, self.minimum_caption_length
            )
            return kw_filter_result and len_filter_result

        return [x for x in data if filter_fn(x["caption"])]

    def load_raw_data(self):
        wavcaps_data = {}
        for filepath in glob.glob(os.path.join(self.input_dir, "*.json")):
            print(f"processing {filepath}")
            with open(filepath, "r") as f:
                raw_data = json.load(f)["data"]
            filtered_data = self._apply_filter(raw_data)
            print(f"kept {len(filtered_data)} of {len(raw_data)} elements after filtering")

            wavcaps_data[os.path.basename(filepath)] = filtered_data

        for dataset_json, v in wavcaps_data.items():
            if len(v):
                for elem in v:
                    elem["id"] = "::".join((elem["id"], JSON_TO_DATASET_NAME[dataset_json]))
                    del elem["wav_path"]

        self.data = [x for y in wavcaps_data.values() for x in y]


# ZERO_INDEX_TRANSCRIBERS = [
#     "Segundo G. Yogore",  # piano only
#     "Martin Charles Bucknall",  # piano only
#     "Jeruen Espino Dery",  # solo piano
#     "www.bachcentral.com",  # piano only
#     "piano-midi.de",  # piano only
# ]

# ONE_INDEX_TRANSCRIBERS = [
#     "http://tirolmusic.blogspot.com/",
#     "harfesoft.de",
#     "Gunter R. Findenegg",
#     "Reinier B. Bakels",
#     "Oliver Seely",
#     "Michael Iscenko",
#     "Benjamin R. Tubb",
#     "Masahiro Ishii",
#     "Andrew D. Lawson",
#     "Eric L. Schissel",
#     "clarinetinstitute",
#     "David Rothschild",
#     "Gregory Richardson",
#     "suzumidi",
#     "David J. Grossman",
#     "suzumedia",
# ]


@dataclass
class MusicNetJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        meta_df = pd.read_csv(
            os.path.join(self.input_dir, "musicnet_metadata.csv"),
            dtype="object",
        )
        midi_dir = os.path.join(self.input_dir, "musicnet_em", "musicnet_em")
        midi_files = glob.glob(os.path.join(midi_dir, "*.mid"))
        # maping of filename: NoteSequence
        print(f"[INFO] reading MIDI data from {midi_dir}")
        midi_data = {f: note_seq.midi_file_to_note_sequence(f) for f in midi_files}
        print("[INFO] preprocessing MIDI data to string")
        midi_data = {
            k: get_formatted_notes_list(
                v,
                no_synth=True,
                midi_program_is_zero_indexed=False,
                use_musicnet_program_corrections=True,
            )
            for k, v in midi_data.items()
        }

        midi_data = {k: json.dumps(v) for k, v in midi_data.items()}
        midi_data = {os.path.basename(k).replace(".mid", ""): v for k, v in midi_data.items()}
        midi_df = pd.DataFrame.from_dict(midi_data, orient="index", columns=["notes"]).reset_index(
            names="id"
        )
        # Note that there appears to be a convention in MusicNet where "Pizzicato Strings"
        # is used to indicate a second violin (i.e. in a string quartet). We leave it
        # but it might be worth correcting.
        data = meta_df.merge(midi_df, how="inner")

        self.data = data[
            ["id", "composer", "composition", "movement", "ensemble", "notes"]
        ].to_dict("records")


class GiantStepsKeyJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        key_files_dir = os.path.join(self.input_dir, "annotations", "key")
        files = os.listdir(key_files_dir)
        data = [
            {
                "id": f.replace(".key", ""),
                "giantsteps_key": open(os.path.join(key_files_dir, f)).read(),
            }
            for f in files
            if f.endswith(".key")
        ]
        self.data = data


class GiantStepsTempoJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        key_files_dir = os.path.join(self.input_dir, "annotations_v2", "tempo")
        files = os.listdir(key_files_dir)
        data = [
            {
                "id": f.replace(".bpm", ""),
                "giantsteps_tempo": open(os.path.join(key_files_dir, f)).read(),
            }
            for f in files
            if f.endswith(".bpm")
        ]
        self.data = data


def format_slakh_notes_list(
    ns: note_seq.NoteSequence,
) -> Dict[str, List[OrderedDict]]:
    per_instrument_sequences = defaultdict(list)
    for x in ns.notes:
        if x.is_drum:
            inst = "Drums"
            if isinstance(x.pitch, str) and not x.pitch.isnumeric():
                print(
                    f"[DEBUG] using drum pitch from MIDI {x.pitch}; if this is not a"
                    " human-readable drum name you should check the data."
                )
                pitch = x.pitch
            elif x.pitch in DRUM_PITCH_TO_NAME:
                pitch = DRUM_PITCH_TO_NAME[x.pitch]
            else:
                # Skip unknown drums; we prefer to have missing notes than wrong notes.
                print(f"[WARNING] got unknown drum pitch {x.pitch}; skipping.")
                continue
        else:
            inst = MIDI_PROGRAM_TO_SLAKH_CLASSES[x.program]["name"]
            pitch = f"{pitch_to_note_labels[x.pitch % 12]}{x.pitch // 12}"
        note_info = OrderedDict(
            {
                "start": round(x.start_time, 2),
                "end": round(x.end_time, 2),
                "pitch": pitch,
            }
        )
        per_instrument_sequences[inst].append(note_info)
    return per_instrument_sequences


class SlakhJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        if self.split == "train":
            tracks = TRAIN_TRACKS
        elif self.split == "test":
            tracks = TEST_TRACKS
        else:
            raise ValueError("unknown split")
        data = []
        for track in tqdm(tracks):
            elem = {"id": track}
            ns = note_seq.midi_file_to_note_sequence(
                os.path.join(self.input_dir, "midi", self.split, track + ".mid")
            )
            ns = note_seq.apply_sustain_control_changes(ns)
            midi_data = format_slakh_notes_list(ns)
            elem["notes"] = json.dumps(midi_data)
            data.append(elem)

        self.data = data


_MAGNATAGATUNE_SPLITS = {
    "train": MAGNATAGATUNE_TRAIN_CHUNKS,
    "validation": MAGNATAGATUNE_VALIDATION_CHUNKS,
    "test": MAGNATAGATUNE_TEST_CHUNKS,
}


class MusiccapsJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        assert self.split in ("train", "eval")
        df = pd.read_csv(os.path.join(self.input_dir, "musiccaps-public.csv"))
        if self.split == "eval":
            df = df[df.is_audioset_eval is True]
        else:
            df = df[df.is_audioset_eval is not True]

        self.data = df.to_dict("records")


class YT8MMusicTextClipsJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        assert self.split in ("train", "test", "all")
        if self.split == "all":
            train_df = pd.read_csv(os.path.join(self.input_dir, "train.csv"))
            test_df = pd.read_csv(os.path.join(self.input_dir, "test.csv"))
            df = pd.concat((train_df, test_df))
        elif self.split == "train":
            df = pd.read_csv(os.path.join(self.input_dir, "train.csv"))
        elif self.split == "test":
            df = pd.read_csv(os.path.join(self.input_dir, "test.csv"))

        self.data = df.to_dict("records")


class MagnaTagATuneJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        clip_info = pd.read_csv(os.path.join(self.input_dir, "clip_info_final.csv"), delimiter="\t")
        tags = pd.read_csv(
            os.path.join(self.input_dir, "annotations_final.csv"),
            delimiter="\t",
        )

        data = clip_info.merge(tags, on=["clip_id", "mp3_path"])
        data["chunk"] = data["mp3_path"].apply(lambda x: x.split("/")[0])
        data["example_id"] = data["mp3_path"].apply(extract_id_from_mp3_path)
        split_chunks = _MAGNATAGATUNE_SPLITS[self.split]
        split_idxs = np.isin(data["chunk"], split_chunks)
        split_data = data[split_idxs]
        self.data = split_data.to_dict("records")


class JamendoJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        assert self.split, "is split implemented for this dataset?"
        fields_to_use = ("genre", "instrument", "mood/theme")
        data = []
        tsv_file = os.path.join(self.input_dir, "autotagging.tsv")
        tracks, tags, extra = mtg_jamendo_read_file(tsv_file)
        for track_id, track_annotations in tqdm(tracks.items(), total=len(tracks)):
            track_data = {k: list(track_annotations[k]) for k in fields_to_use}
            track_data["id"] = str(track_id)
            data.append(track_data)

        self.data = data
        print(f"[INFO] loaded {len(data)} tracks")
        return


def postprocess_fsl10k_annotations(
    annotations: Dict[str, Any],
    keys_to_drop=(
        "save_for_later",
        "well_cut",
        "discard",
        "comments",
        "username",
        "num_ratings",
        "num_downloads",
        "license",
        "avg_rating",
        "preview_url",
        "type",  # file type, e.g. 'wav'
        "pack",
        "image",
    ),
) -> Dict[str, Any]:
    # Drop any unwanted keys
    annotations = {k: v for k, v in annotations.items() if k not in keys_to_drop}
    # Handle key/mode by reformatting
    key = annotations.pop("key")
    mode = annotations.pop("mode")
    annotations["key_fsl10k"] = f"{key} {mode}"

    # remove nested instrumentation structure
    instrumentation = annotations.pop("instrumentation")
    for k, v in instrumentation.items():
        annotations["instrumentation_" + k] = v

    annotations["time_signature"] = annotations.pop("signature")

    return annotations


class Fsl10kJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        files = glob.glob(os.path.join(self.input_dir, "FSL10K", "audio", "wav", "*.wav"))
        annotations = {}
        invalid_annotations = 0
        for file in tqdm(files, desc="finding annotations"):
            fsid = os.path.basename(file).split("_")[0]
            try:
                # Read the annotation
                annotation = glob.glob(
                    os.path.join(self.input_dir, "annotations", "*", f"sound-{fsid}.json")
                )[0]
                with open(annotation, "r") as f:
                    metadata = json.load(f)
                if metadata["discard"]:
                    # Ignore any files marked 'discard'
                    continue
                # Read the FSD data
                fsd_analysis = os.path.join(self.input_dir, "FSL10K", "fs_analysis", fsid + ".json")
                with open(fsd_analysis, "r") as f:
                    fsd_data = json.load(f)
                metadata.update(fsd_data)
                filename = os.path.basename(file)
                metadata.update(
                    {
                        "filename": filename,
                        # create the ID such that the audio files can be found
                        # via {id}.wav, like all other datasets.
                        "id": filename.rsplit(".wav", maxsplit=1)[0],
                    }
                )
                annotations[fsid] = metadata

            except Exception:
                invalid_annotations += 1
        print(
            f"got {len(annotations)} valid annotations; "
            f"no annotations for {invalid_annotations} FSIDs."
        )
        annotations = {k: postprocess_fsl10k_annotations(v) for k, v in annotations.items()}

        self.data = list(annotations.values())

        return


class FmaJsonifier(DatasetJsonifier):
    def load_raw_data(self):
        genres = pd.read_csv(os.path.join(self.input_dir, "genres.csv"))
        genre_map = {x["genre_id"]: x["title"] for _, x in genres.iterrows()}

        # Via https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
        language_map = {
            "en": "English",
            "fi": "Finnish",
            "pt": "Portuguese",
            "tr": "Turkish",
            "sw": "Swahili",
            "el": "Greek",
            "ar": "Arabic",
            "pl": "Polish",
            "es": "Spanish",
            "id": "Indonesian",
            "tw": "Twi",
            "eu": "Basque",
            "ms": "Malay",
            "fr": "French",
            "ty": "Tahitian",
            "hi": "Hindi",
            "vi": "Vietnamese",
            "ja": "Japanese",
            "tl": "Tagalog",
            "it": "Italian",
            "my": "Burmese",
            "gu": "Gujarati",
            "zh": "Chinese",
            "az": "Azerbaijani",
            "hy": "Armenian",
            "sr": "Serbian",
            "lt": "Lithuanian",
            "th": "Thai",
            "bg": "Bulgarian",
            "de": "German",
            "ko": "Korean",
            "uz": "Uzbek",
            "ka": "Georgian",
            "ha": "Hausa",
            "sk": "Slovak",
            "nl": "Dutch",
            "bm": "Bambara",
            "ru": "Russian",
            "he": "Hebrew",
            "cs": "Czech",
            "la": "Latin",
            "ee": "Ewe",
            "Unknown": "Unknown",
        }
        tracks = pd.read_csv(
            os.path.join(self.input_dir, "tracks.csv"),
            names=[
                "track.id",
                "album.comments",
                "album.date_created",
                "album.date_released",
                "album.engineer",
                "album.favorites",
                "album.id",
                "album.information",
                "album.listens",
                "album.producer",
                "album.tags",
                "album.title",
                "album.tracks",
                "album.type",
                "artist.active_year_begin",
                "artist.active_year_end",
                "artist.associated_labels",
                "artist.bio",
                "artist.comments",
                "artist.date_created",
                "artist.favorites",
                "artist.id",
                "artist.latitude",
                "artist.location",
                "artist.1longitude",
                "artist.members",
                "artist.name",
                "artist.1related_projects",
                "artist.tags",
                "artist.website",
                "artist.wikipedia_page",
                "set.split",
                "set.subset",
                "track.bit_rate",
                "track.comments",
                "track.composer",
                "track.date_created",
                "track.date_recorded",
                "track.duration",
                "track.favorites",
                "track.genre_top",
                "track.genres",
                "track.genres_all",
                "track.information",
                "track.interest",
                "track.language_code",
                "track.license",
                "track.listens",
                "track.lyricist",
                "track.number",
                "track.publisher",
                "track.tags",
                "track.title",
            ],
            skiprows=3,
        )

        # drop columns with count features or attributes we don't want included in
        # instruction-tuning data
        drop_cols = [
            "album.comments",
            "album.date_created",
            "album.date_released",
            "album.engineer",
            "album.favorites",
            "album.id",
            #  'album.information',
            "album.listens",
            "album.producer",
            #  'album.tags',
            "album.title",
            "album.tracks",
            "album.type",
            "artist.active_year_begin",
            "artist.active_year_end",
            "artist.associated_labels",
            "artist.bio",  # TODO(jpgard): look into whether we actually want this column later.
            "artist.comments",
            "artist.date_created",
            "artist.favorites",
            "artist.id",
            "artist.latitude",
            "artist.location",
            "artist.1longitude",
            "artist.members",
            "artist.name",
            "artist.1related_projects",
            #    'artist.tags',
            "artist.website",
            "artist.wikipedia_page",
            # "set.split",
            # "set.subset",
            "track.bit_rate",
            "track.comments",
            "track.composer",
            "track.date_created",
            "track.date_recorded",
            "track.duration",
            "track.favorites",
            # "track.genre_top",
            "track.genres",
            # "track.genres_all",
            # "track.information",
            "track.interest",
            # "track.language_code",
            "track.license",
            "track.listens",
            "track.lyricist",
            "track.number",
            "track.publisher",
            "track.tags",
            "track.title",
        ]
        tracks.drop(columns=drop_cols, inplace=True)
        # Split the data.
        if self.split == "train":
            # We allow split to be specified as 'train' for consistency with other datasets,
            # but in FMA this split is actually called 'training'.
            split = "training"
        else:
            split = self.split
        tracks = tracks[tracks["set.split"] == split].drop(columns=["set.split", "set.subset"])
        # Map numeric genre IDs to string names
        tracks["track.genres_all"] = tracks["track.genres_all"].apply(
            lambda x: [genre_map[i] for i in json.loads(x)]
        )
        # Map language abbreviations to string names
        tracks["track.language_code"].fillna("Unknown", inplace=True)
        tracks["track.language_code"] = tracks["track.language_code"].map(language_map)

        print("[INFO] parsing HTML-like fields; this can take a few minutes...")
        tracks["album.information"] = tracks["album.information"].apply(
            # lambda x: re.sub("(<p>|</p>)", "", str(x))
            lambda x: extract_text_from_html(str(x))
        )
        tracks["track.information"] = tracks["track.information"].apply(
            lambda x: extract_text_from_html(str(x))
        )
        print("[INFO] parsing HTML-like fields complete.")

        self.data = tracks.to_dict("records")
