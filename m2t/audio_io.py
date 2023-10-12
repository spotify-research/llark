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

import logging
import os
from typing import Optional

import ffmpeg


def convert_to_wav(infile: str, outdir: str) -> Optional[str]:
    """
    Convert the provided audio file (infile) to a WAV file by shelling out to FFMPEG.
    TODO: Use pedalboard.io.AudioFile instead to remove the need for this.
    """
    basename, extension = os.path.basename(infile.replace("gs://", "")).rsplit(".", maxsplit=1)
    outfile = os.path.join(outdir, f"{basename}.wav")
    try:
        ffmpeg.input(infile).output(outfile, ar=44100, ac=1).overwrite_output().run(
            capture_stdout=True
        )
        return outfile
    except Exception as e:
        logging.error(f"error processing file {infile} to {outfile}: {e}")
        return None
