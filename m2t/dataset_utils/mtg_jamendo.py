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
Some functions here are from https://github.com/MTG/mtg-jamendo-dataset/blob/master/scripts/commons.py.
"""
import csv
from collections import defaultdict

CATEGORIES = ["genre", "instrument", "mood/theme"]
TAG_HYPHEN = "---"
METADATA_DESCRIPTION = (
    "TSV file with such columns: TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION, TAGS"
)


def get_id(value):
    return int(value.split("_")[1])


def get_length(values):
    return len(str(max(values)))


def mtg_jamendo_read_file(tsv_file):
    tracks = {}
    tags = defaultdict(dict)

    # For statistics
    artist_ids = set()
    albums_ids = set()

    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            track_id = get_id(row[0])
            tracks[track_id] = {
                "artist_id": get_id(row[1]),
                "album_id": get_id(row[2]),
                "path": row[3],
                "duration": float(row[4]),
                "tags": row[5:],  # raw tags, not sure if will be used
            }
            tracks[track_id].update({category: set() for category in CATEGORIES})

            artist_ids.add(get_id(row[1]))
            albums_ids.add(get_id(row[2]))

            for tag_str in row[5:]:
                category, tag = tag_str.split(TAG_HYPHEN)

                if tag not in tags[category]:
                    tags[category][tag] = set()

                tags[category][tag].add(track_id)

                if category not in tracks[track_id]:
                    tracks[track_id][category] = set()

                tracks[track_id][category].update(set(tag.split(",")))

    print(
        "Reading: {} tracks, {} albums, {} artists".format(
            len(tracks), len(albums_ids), len(artist_ids)
        )
    )

    extra = {
        "track_id_length": get_length(tracks.keys()),
        "artist_id_length": get_length(artist_ids),
        "album_id_length": get_length(albums_ids),
    }
    return tracks, tags, extra
