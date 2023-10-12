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

import base64
import io

import numpy as np
from IPython import display
from scipy.io import wavfile


def play_audio(array_of_floats, sample_rate, autoplay=False):
    """Creates an HTML5 audio widget to play a sound in Colab.

    This function should only be called from a Colab notebook.

    Args:
      array_of_floats: A 1D or 2D array-like container of float sound
        samples. Values outside of the range [-1, 1] will be clipped.
      sample_rate: Sample rate in samples per second.
      autoplay: If True, automatically start playing the sound when the
        widget is rendered.
    """
    normalizer = float(np.iinfo(np.int16).max)
    array_of_ints = np.array(np.asarray(array_of_floats) * normalizer, dtype=np.int16)
    memfile = io.BytesIO()
    wavfile.write(memfile, sample_rate, array_of_ints)
    html = """<audio controls {autoplay}>
              <source controls src="data:audio/wav;base64,{base64_wavfile}"
              type="audio/wav" />
              Your browser does not support the audio element.
            </audio>"""
    html = html.format(
        autoplay="autoplay" if autoplay else "",
        base64_wavfile=base64.b64encode(memfile.getvalue()).decode("ascii"),
    )
    memfile.close()
    display.display(display.HTML(html))
