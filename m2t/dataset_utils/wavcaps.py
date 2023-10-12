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

JSON_TO_DATASET_NAME = {
    "as_final.json": "audioset",
    "sb_final.json": "soundbible",
    "fsd_final.json": "freesound",
    "bbc_final.json": "bbc_sound_effects",
}

KEYWORDS = {
    "music": [
        "music",
        "song",
        "singer",
        "band",
        "instrument",
        "chord",
        "melody",
        "melodic",
        "jingle",
    ],
    "keyed": [
        "piano",
        "harpsichord",
        "clavinet",
        "celesta",
        "glockenspiel",
        "vibraphone",
        "marimba",
        "xylophone",
        "bells",
        "dulcimer",
        "santoor",
        "organ",
        "drawbar",
        "accordion",
    ],
    "guitar": [
        "guitar",
        "stratocaster",
    ],
    "orchestral": [
        "violin",
        "viola",
        "cello",
        "contrabass",
        "strings",
        "tremolo",
        "pizzicato",
        "orchestra",
        "timpani",
        "ensemble",
        "choir",
    ],
    "wind": [
        "trumpet",
        "trombone",
        "tuba",
        "french horn",
        "brass",
        "sax",
        "alto",
        "tenor",
        "baritone",
        "oboe",
        "bassoon",
        "clarinet",
        "piccolo",
        "flute",
        "shakuhachi",
        "ocarina",
    ],
    "synth": [
        "synth",
        "sawtooth",
        "sine",
        "polyphon",
    ],
    "other midi": [
        "harmonica",
        "bandoneon",
        "bowed",
        "sitar",
        "banjo",
        "shamisen",
        "koto",
        "kalimba",
        "bag pipe",
        "bagpipe",
        "fiddle",
        "shanai",
        "cowbell",
        "cow bell",
        "steel drum",
        "taiko drum",
        "cymbal",
    ],
    "midi percussion": [
        "hi hat",
        "hihat",
        "hi-hat",
        "drum",
        "cymbal",
        "drumstick",
        "drum stick",
        "snare",
        "low tom",
        "floor tom",
        "mid tom",
        "high tom",
        "drum set",
        "drumset",
        "bell",
        "tambourine",
        "bongo",
        "conga",
        "timbale",
        "agogo",
        "cabasa",
        "maraca",
        "guiro",
        "clave",
        "triangle",
        "shaker",
        "chime",
        "castanet",
        "surdo",
        "tam-tam",
        "tamtam",
    ],
    # genre labels that do not have common non-musical meanings
    "genre": [
        "jazz",
        "rock",
        "country",
        "hip hop",
        "hiphop",
        "techno",
        "punk",
        "electronica",
        "soundtrack",
        "folk",
        "rnb",
        "classical",
        "funk",
    ],
}


def keyword_filter(caption) -> bool:
    keywords = [kw for kws in KEYWORDS.values() for kw in kws]
    return any(x in caption for x in keywords)


def length_filter(caption, length) -> bool:
    return len(caption.split(" ")) >= length
