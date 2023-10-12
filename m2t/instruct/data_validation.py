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

from typing import Any, Dict

DISALLOWED_ANSWER_PHRASES = [
    "metadata",
    "is not provided",
    "based on the provided metadata",
    "based on the provided beat",
    "based on the provided chord",
    "based on the provided information",
    "based on the provided annotations",
    "no specific mood",
    "there is no mention of",
    "there is no specific mention of any",
    "As an AI assistant, I am unable to",
    "As an AI assistant, I do not",
    "it is difficult to determine",
    "it is not possible to determine",
    "no information is available about the album",
    "cannot determine",
    "violin 1",
    "violin 2",
    "violin 3",
    "viola 1",
    "viola 2",
    "viola 3",
    "pack",
]

DISALLOWED_QUESTION_PHRASES = [
    "what is the composer",
    "who is the composer",
    "tell me about the composer",
    "name of the composer",
    "who is the artist",
    "tell me about the artist",
    "what tags are associated with the artist",
    "what are the tags associated with the artist",
    "is there any information available about the album",
    "about the album",
    "name of the artist",
    "what is the name",
    "what is the movement",
    "what is the specific movement",
    "what is the title",
    "which movement is",
    "what is the length of this clip",
    "duration",
    "pack",
]


def is_invalid_qa_response(response: Dict[str, str]) -> bool:
    """Check whether a resonse contains any invalid/disallowed phrases."""
    assert isinstance(response, dict), f"expected dict, got type {type(response)}"

    if any(x.lower() in response["answer"].lower() for x in DISALLOWED_ANSWER_PHRASES):
        # Case: a disallowed phrase is in the answer text.
        print(f"[DEBUG] got invalid response answer {response['answer']}")
        return True
    if any(x.lower() in response["question"].lower() for x in DISALLOWED_QUESTION_PHRASES):
        # Case: a disallowed phrase is in the question text.
        print(f"[DEBUG] got invalid response question {response['question']}")
        return True
    return False


def drop_invalid_qa_responses(elem: Dict[str, Any]):
    input_len = len(elem["response"])
    elem["response"] = [
        x for x in elem["response"] if isinstance(x, dict) and not is_invalid_qa_response(x)
    ]
    output_len = len(elem["response"])
    print(f"[DEBUG] dropped {input_len-output_len} invalid responses.")
    return elem


def element_response_is_not_exception(elem: Dict[str, Any]) -> bool:
    return "response" in elem and "exception" not in elem


def response_format_is_valid_strict(x) -> bool:
    """Generic checking of the structure (not format) of QA response."""
    if not isinstance(x, dict):
        # Case: the example is a list, but not a list of dicts. Could
        # be an issue parsing the outputs from OpenAI.
        return False
    if not x.get("question") or not x.get("answer"):
        # Case: the example dict does not contain a valid question or answer.
        return False
    return True


def element_is_valid_strict(elem: Dict[str, Any]) -> bool:
    """
    Check if an element is properly formed and contains non-empty instruction data.
    """
    if "response" not in elem:
        # Case: no response. This can occur when all of the Q/A pairs were removed
        # ata  filtering stage.
        print(
            f"[DEBUG] element uri {elem['uri']} has no key 'response'; "
            "this is unexpected and could be a sign of a data error or bug."
        )
        return False
    if not isinstance(elem["response"], list) or not len(elem["response"]):
        # Case: empty or non-list response. This can occur when the response was
        # empty even in the initial example (which results in the JSON string '' being
        # parsed to a string, not a list as in a well-formed example) or when all
        # Q/A pairs have been removed due to filtering.
        print(
            f"[DEBUG] element uri {elem['uri']} has null response {elem['response']};"
            "this can be normal when filtering has been applied to the raw response."
        )
        return False

    for x in elem["response"]:
        if not response_format_is_valid_strict(x):
            print(
                f"[DEBUG] element uri {elem['uri']} has a response entry "
                "with invalid format: {x}"
            )
            return False
    return True
