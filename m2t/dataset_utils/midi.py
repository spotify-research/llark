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

from collections import OrderedDict, defaultdict
from typing import Dict, List

import note_seq

midi_program_to_instrument: Dict[int, str] = {
    0: "Piano",
    1: "Piano",
    2: "Bright Acoustic Piano",
    3: "Electric Grand Piano",
    4: "Honky-tonk Piano",
    5: "Electric Piano 1 (usually a Rhodes piano)",
    6: "Electric Piano 2 (usually an FM piano patch)",
    7: "Harpsichord",
    8: "Clavinet",
    9: "Celesta",
    10: "Glockenspiel",
    11: "Music Box",
    12: "Vibraphone",
    13: "Marimba",
    14: "Xylophone",
    15: "Tubular Bells",
    16: "Dulcimer or Santoor",
    17: "Drawbar Organ or Organ 1",
    18: "Percussive Organ or Organ 2",
    19: "Rock Organ or Organ 3",
    20: "Church Organ",
    21: "Reed Organ",
    22: "Accordion",
    23: "Harmonica",
    24: "Bandoneon or Tango Accordion",
    25: "Acoustic Guitar (nylon)",
    26: "Acoustic Guitar (steel)",
    27: "Electric Guitar (jazz)",
    28: "Electric Guitar (clean)",
    29: "Electric Guitar (muted)",
    30: "Electric Guitar (overdriven)",
    31: "Electric Guitar (distortion)",
    32: "Electric Guitar (harmonics)",
    33: "Acoustic Bass",
    34: "Electric Bass (finger)",
    35: "Electric Bass (picked)",
    36: "Electric Bass (fretless)",
    37: "Slap Bass 1",
    38: "Slap Bass 2",
    39: "Synth Bass 1",
    40: "Synth Bass 2",
    41: "Violin",
    42: "Viola",
    43: "Cello",
    44: "Contrabass",
    45: "Tremolo Strings",
    46: "Pizzicato Strings",
    47: "Orchestral Harp",
    48: "Timpani",
    49: "String Ensemble 1",
    50: "String Ensemble 2",
    51: "Synth Strings 1",
    52: "Synth Strings 2",
    53: "Choir Aahs",
    54: "Voice Oohs (or Doos)",
    55: "Synth Voice or Synth Choir",
    56: "Orchestra Hit",
    57: "Trumpet",
    58: "Trombone",
    59: "Tuba",
    60: "Muted Trumpet",
    61: "French Horn",
    62: "Brass Section",
    63: "Synth Brass 1",
    64: "Synth Brass 2",
    65: "Soprano Sax",
    66: "Alto Sax",
    67: "Tenor Sax",
    68: "Baritone Sax",
    69: "Oboe",
    70: "English Horn",
    71: "Bassoon",
    72: "Clarinet",
    73: "Piccolo",
    74: "Flute",
    75: "Recorder",
    76: "Pan Flute",
    77: "Blown bottle",
    78: "Shakuhachi",
    79: "Whistle",
    80: "Ocarina",
    81: "Lead 1 (square, often chorused)",
    82: "Lead 2 (sawtooth, often chorused)",
    83: "Lead 3 (triangle, or calliope, usually resembling a woodwind)",
    84: "Lead 4 (sine, or chiff)",
    85: "Lead 5 (charang, a guitar-like lead)",
    86: "Lead 6 (voice)",
    87: "Lead 7 (fifths)",
    88: "Lead 8 (bass and lead or solo lead)",
    89: "Pad 1 (new age, pad stacked with a bell)",
    90: "Pad 2 (warm, a mellower saw pad)",
    91: "Pad 3 (polysynth or poly, a saw-like percussive pad resembling an early 1980s polyphonic synthesizer)",
    92: 'Pad 4 (choir, similar to "synth voice")',
    93: "Pad 5 (bowed glass or glass harmonica sound)",
    94: "Pad 6 (metallic sound)",
    95: "Pad 7 (halo, choir-like pad)",
    96: 'Pad 8 (sweep, pad with a pronounced "wah" filter effect)',
    97: "FX 1 (rain, a bright pluck with echoing pulses)",
    98: "FX 2 (soundtrack, a bright perfect fifth pad)",
    99: "FX 3 (crystal, a synthesized bell sound)",
    100: "FX 4 (atmosphere, usually a classical guitar-like sound)",
    101: "FX 5 (brightness, a fast-attack stacked pad with choir or bell)",
    102: "FX 6 (goblins, a slow-attack pad with chirping or murmuring sounds)",
    103: 'FX 7 (echoes or echo drops, similar to "rain")',
    104: "FX 8 (sci-fi or star theme, usually an electric guitar-like pad)",
    105: "Sitar",
    106: "Banjo",
    107: "Shamisen",
    108: "Koto",
    109: "Kalimba",
    110: "Bag pipe",
    111: "Fiddle",
    112: "Shanai",
    113: "Tinkle Bell",
    114: "Agogô or cowbell",
    115: "Steel Drums",
    116: "Woodblock",
    117: "Taiko Drum",
    118: "Melodic Tom or 808 Toms",
    119: "Synth Drum",
    120: "Reverse Cymbal",
    121: "Guitar Fret Noise",
    122: "Breath Noise",
    123: "Seashore",
    124: "Bird Tweet",
    125: "Telephone Ring",
    126: "Helicopter",
    127: "Applause",
    128: "Gunshot",
}

pitch_to_note_labels: Dict[int, str] = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}


def get_formatted_notes_list(
    ns: note_seq.NoteSequence,
    include_velocity=False,
    no_synth=False,
    midi_program_is_zero_indexed=True,
    use_musicnet_program_corrections=False,
) -> Dict[str, List[OrderedDict]]:
    """Fetch a formatted and less-verbose string representation
    of the notes in a NoteSequence."""

    def _get_inst_name(program: int):
        if not midi_program_is_zero_indexed and program > 0:
            # Case: some datasets, like MusicNet, use program numbers
            # starting from 1 (not 0). However, MusicNet still uses '0' for piano
            # (which is insane, but perhaps a MIDI convention) so we leave
            # program number 0 alone.
            program += 1

        if use_musicnet_program_corrections and program == 46:
            # Convert from 'Pizzicato Strings' to 'Violin'. MusicNet transcribers use a
            # convention where the second violin is treated as 'Pizzicato Strings'
            # but we want it to be labeled as the correct instrument, another violin.
            print("[DEBUG] converting pizzicato strings to violin")
            program = 41

        if no_synth:
            return midi_program_to_instrument[program].replace("Synth", "").strip()
        else:
            return midi_program_to_instrument[program]

    per_instrument_sequences = defaultdict(list)

    for x in ns.notes:
        inst = f"{_get_inst_name(x.program)}{'' if x.instrument ==0 else ' '+str(x.instrument+1)}"
        note_info = OrderedDict(
            {
                "start": round(x.start_time, 2),
                "end": round(x.end_time, 2),
                "pitch": f"{pitch_to_note_labels[x.pitch % 12]}{x.pitch // 12}",
            }
        )
        per_instrument_sequences[inst].append(note_info)

    return per_instrument_sequences
