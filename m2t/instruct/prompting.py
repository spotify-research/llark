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

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from m2t.dataset_utils import DatasetInfo
from m2t.diffusify_utils import oxford_comma, parse_almost_json
from m2t.instruct.fewshot_examples import FewShotExample
from m2t.instruct.fewshot_examples.mirqa import MIRQA_FEWSHOT_EXAMPLES
from m2t.instruct.fewshot_examples.reasoning_qa import REASONING_QA_FEWSHOT_EXAMPLES

PROMPT = open(os.path.join(os.path.dirname(__file__), "openai-chatgpt-prompt.txt")).read()

EXPECTED_FIELDS = [
    "context_activities",
    "context_cultural",
    "genre",
    "mood",
    "sound_descriptions",
    "music_descriptions",
    "music_analysis",
    "music_creation",
    "abstract",
]
OPTIONAL_FIELDS = ["language", "lyrics", "vocals", "instruments", "rhythm"]
ALLOWED_FIELDS = set(["title", "artist", "uri"] + EXPECTED_FIELDS + OPTIONAL_FIELDS)


def correct_element(input_row: Dict) -> Dict:
    """
    Apply a series of corrections to the input dictionary, to
    constrain GPT's "creativity":
    - no nested arrays (e.g.: {"languages": ["de","en",[]]} -- if
      present, flatten them)
    - check that the values in the dictionary are lists of individual
      elements (i.e., that returned values don't contain list of
      dictionaries -- in that case, ignore those dictionaries)
    - if a field (aside from uri/title/artist) is a string, make it a [string]
    - the language field is not null (rather an empty list) -- because
      the schema auto-detection guesses that language is NOT an
      optional field
    - no other fields than the ones requested (i.e., that gpt didn't invent
      a field)
    """
    output_row = {}
    # break nested return values (e.g.: "languages": ["de","en",[]]) and set
    for key, value_in in input_row.items():
        output_row[key] = unnest_list(value_in) if isinstance(value_in, list) else value_in
    # make sure each openai field is a list
    for key in EXPECTED_FIELDS + OPTIONAL_FIELDS:
        if key in output_row:
            if isinstance(output_row[key], str):
                output_row[key] = [output_row[key]]
    # make sure the language field is not null
    if output_row.get("language") is None:
        output_row["language"] = []
    # make sure there are no invented fields
    output_row = {key: value for key, value in output_row.items() if key in ALLOWED_FIELDS}
    return output_row


def unnest_list(list_in):
    # recursive unnesting / ignoring nested dictionaries
    def _unnest(a_list):
        for e in a_list:
            if isinstance(e, list):
                _unnest(e)  # recurse if list
            elif isinstance(e, dict):
                pass  # don't know how to handle nested dictionaries, ignore
            else:
                yield e

    return list(_unnest(list_in))


@dataclass
class PromptHelper(ABC):
    few_shot: bool
    prompt_text: str
    few_shot_examples: Optional[Sequence[FewShotExample]] = None

    def get_prompt_text(self) -> str:
        """Fetch the prompt text."""
        return self.prompt_text

    @abstractmethod
    def get_chatgpt_query(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the query text to provide to ChatGPT."""
        raise

    def build_messages(self, prompt_text, query) -> List[Dict[str, str]]:
        """Builds the `messages` attribute to use for openai.ChatCompletion.create()."""
        fewshot_examples_formatted = []
        if self.few_shot:
            for fewshot_example in self.few_shot_examples:
                fewshot_examples_formatted.append(
                    {
                        "role": "user",
                        "content": json.dumps(fewshot_example.user),
                    }
                )
                fewshot_examples_formatted.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(fewshot_example.assistant),
                    }
                )
        return [
            {"role": "system", "content": prompt_text},
            *fewshot_examples_formatted,
            {"role": "user", "content": json.dumps([query])},
        ]

    @abstractmethod
    def postprocess_response_text(self, text: str, query, uri) -> Dict[str, Any]:
        """Postprocess the ChatGPT response and return a (possibly cleaned) version."""
        raise

    @abstractmethod
    def check_chatgpt_response_meets_schema(
        self, response: Union[Dict, List[Dict]]
    ) -> Union[Dict, List[Dict]]:
        raise


@dataclass
class BasicPromptHelper(PromptHelper):
    """Helper for the default prompt type."""

    few_shot = False
    prompt_text = PROMPT

    def get_chatgpt_query(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the query text to provide to ChatGPT."""
        track = metadata["name"]
        artists = oxford_comma([a["name"] for a in metadata["artist"]])
        return {"title": track, "artist": artists}

    def postprocess_response_text(self, text: str, query, uri) -> Dict[str, Any]:
        """Postprocess the ChatGPT response and return a (possibly cleaned) version."""
        response = self.check_chatgpt_response_meets_schema(parse_almost_json(text)[0])
        row = dict(list(response.items()) + list(query.items()) + [("uri", uri)])
        row = correct_element(row)
        return row

    def check_chatgpt_response_meets_schema(
        self, response: Union[Dict, List[Dict]]
    ) -> Union[Dict, List[Dict]]:
        assert isinstance(response, dict)
        expected_fields = EXPECTED_FIELDS
        optional_fields = OPTIONAL_FIELDS

        for expected_field in expected_fields:
            if expected_field not in response:
                raise ValueError(f"Missing field from ChatGPT response: {expected_field}")
        for optional_field in optional_fields:
            if optional_field not in response:
                response = dict(response.items())
                response[optional_field] = []
        return response


@dataclass
class MirQAPromptHelper(PromptHelper):
    """Helper for MIR question-answering prompt."""

    few_shot_examples = MIRQA_FEWSHOT_EXAMPLES

    def get_chatgpt_query(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the query text to provide to ChatGPT."""
        return metadata

    def postprocess_response_text(self, text: str, query, uri) -> Dict[str, Any]:
        """Postprocess the ChatGPT response and return a (possibly cleaned) version."""
        # For MIR-QA, the output is a list of question/answer dicts.
        response = self.check_chatgpt_response_meets_schema(parse_almost_json(text))
        row = dict(list(query.items()) + [("uri", uri)])
        row["response"] = response
        return row

    def check_chatgpt_response_meets_schema(
        self, response: Union[Dict, List[Dict]]
    ) -> Union[Dict, List[Dict]]:
        assert isinstance(response, list)
        expected_fields = ("question", "answer")
        for elem in response:
            for expected_field in expected_fields:
                if expected_field not in elem:
                    raise ValueError(f"Missing field from ChatGPT response: {expected_field}")
        return response


class ReasoningQAPromptHelper(PromptHelper):
    few_shot_examples = REASONING_QA_FEWSHOT_EXAMPLES

    def get_chatgpt_query(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the query text to provide to ChatGPT."""
        return metadata

    def postprocess_response_text(self, text: str, query, uri) -> Dict[str, Any]:
        """Postprocess the ChatGPT response and return a (possibly cleaned) version."""
        # For Reasoning QA, the output is a list of question/answer dicts.
        response = self.check_chatgpt_response_meets_schema(parse_almost_json(text))
        row = dict(list(query.items()) + [("uri", uri)])
        row["response"] = response
        return row

    def check_chatgpt_response_meets_schema(
        self, response: Union[Dict, List[Dict]]
    ) -> Union[Dict, List[Dict]]:
        assert isinstance(response, list)
        expected_fields = ("question", "answer")
        for elem in response:
            for expected_field in expected_fields:
                if expected_field not in elem:
                    raise ValueError(f"Missing field from ChatGPT response: {expected_field}")
        return response


class CaptioningPromptHelper(PromptHelper):
    def get_chatgpt_query(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch the query text to provide to ChatGPT."""
        return metadata

    def check_chatgpt_response_meets_schema(
        self, response: Union[Dict, List[Dict]]
    ) -> Union[Dict, List[Dict]]:
        """A no-op; captions are text-only."""
        return response

    def postprocess_response_text(self, text: str, query, uri) -> Dict[str, Any]:
        """Postprocess the ChatGPT response and return a (possibly cleaned) version."""
        # For captioning, the output is just text.
        response = self.check_chatgpt_response_meets_schema(text)
        row = dict(list(query.items()) + [("uri", uri)])
        row["response"] = {"caption": response}
        return row


def get_prompt_helper(prompt_type, dataset_info: DatasetInfo, few_shot: bool) -> PromptHelper:
    # Get the prompt text.
    if prompt_type == "default":
        prompt_text = PROMPT
    else:
        prompt_file = f"{prompt_type}-{dataset_info.name}-prompt.txt"
        prompt_text = open(os.path.join(os.path.dirname(__file__), prompt_file)).read()

    # Fetch the PromptHelper class.
    if prompt_type == "default":
        if few_shot:
            logging.warning("few_shot is True but BasicPrompter is selected.")
        helper_cls = BasicPromptHelper
    elif prompt_type == "mir":
        helper_cls = MirQAPromptHelper
    elif prompt_type == "reasoning":
        helper_cls = ReasoningQAPromptHelper
    elif prompt_type == "captioning":
        helper_cls = CaptioningPromptHelper
    else:
        raise NotImplementedError(f"prompt type {prompt_type} not implemented.")

    return helper_cls(few_shot=few_shot, prompt_text=prompt_text)
