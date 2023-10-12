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

# Prompts for extensive, formal descriptions, e.g. those generated from MIDI.
import random
from typing import Any, Dict, Sequence

LONG_CAPTION_PROMPTS = [
    "Describe the song in detail.",
    "Provide an elaborate description of the song.",
    "Break down the song with great detail.",
    "Offer a thorough explanation of the musical piece.",
    "Present an intricate account of the song that follows.",
    "Describe the details of what you hear in the musical composition.",
    "Describe the musical piece comprehensively.",
    "Analyze the intricacies of the musical audio.",
    "Paint a detailed picture of the song.",
    "Unravel the intricacies of the musical piece.",
    "Examine the audio closely and share its details.",
    "What does this music sound like? Give a detailed description.",
    "What happens in this song? Present a thorough analysis.",
    "Examine the song with meticulous attention to detail.",
    "Narrate the contents of the audio with precision.",
]

# Prompts for short, informal descriptions,
# e.g. those in MusicCaps or YT8M-MusicTextClips.
SHORT_CAPTION_PROMPTS = [
    "Give a short, informal summary of the clip.",
    "What does this music sound like? Give a short description.",
    "Provide an overview of the musical content of the clip.",
    "What does this music sound like?",
    "Briefly narrate the contents of the provided music.",
    "How would you summarize this song?",
    "What happens in this song? Provide a brief summary.",
    "Please provide a quick summary of the musical audio.",
    "Explain the contents of this song.",
    "What do you hear in this music? Give a short summary.",
    "Please provide a cursory description of the song.",
    "Give a short description of this music.",
    "What is happening in this clip? Provide a brief description.",
    "Give a short synopsis of the provided music.",
    "How would you briefly caption this audio?",
    "Offer a concise portrayal of the audio clip's musical essence.",
    "Present a succinct breakdown of the musical composition in the clip.",
    "Describe the auditory characteristics of this music in a few words.",
    "Summarize the key elements found within this musical piece.",
    "Outline the main features of the provided music in brief.",
    "Provide a succinct account of the sonic content in the clip.",
    "Give a quick rundown of what this musical excerpt entails.",
    "Elaborate briefly on the musical components present in the recording.",
    "Sum up your perception of the music in a concise manner.",
    "Deliver a short, descriptive overview of the song's auditory elements.",
    "Summarize the musical content of the audio.",
    "Give a short and clear description of the clip provided.",
    "What do you hear in the provided music excerpt?",
]

# Mappping of dataset names to caption prompts. We use 'long' captions
# for datasets with note- and instrument-level information.
CAPTIONING_PROMPTS = {
    "musiccaps": SHORT_CAPTION_PROMPTS,
    "yt8m-musictextclips": SHORT_CAPTION_PROMPTS,
    "musicnet": LONG_CAPTION_PROMPTS,
    "slakh": LONG_CAPTION_PROMPTS,
    "fsl10k": SHORT_CAPTION_PROMPTS,
}


def is_caption_resonse(elem) -> bool:
    return "caption" in elem["response"]


def insert_caption_qa(elem: Dict[str, Any], caption_prompts: Sequence[str]) -> Dict[str, Any]:
    """Randomly select a prompt from caption_prompts and insert it."""
    caption_prompt = random.choice(caption_prompts)
    caption = elem["response"]["caption"]
    elem["response"] = [{"question": caption_prompt, "answer": caption}]
    return elem
