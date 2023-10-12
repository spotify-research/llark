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
from typing import List


def oxford_comma(x: List[str]) -> str:
    if len(x) == 1:
        return x[0]
    elif len(x) == 2:
        return " and ".join(x)
    else:
        return ", ".join(x[:-1]) + ", and " + x[-1]


class LLMsArentPerfectAtGeneratingJSON(ValueError):
    pass


def parse_almost_json(response: str):
    """
    Parse a JSON object or array that should be valid, but might be missing a brace
    or bracket here or there.

    This is used when we're asking a Large Language Model to generate syntactically
    valid JSON for us. This alone is a sign that we're living in the future, but alas,
    the future still has some problems we need to deal with.

    Sometimes, the LLM misses the mark a bit and forgets to close a brace on the end,
    of a JSON object,  or adds an extra character (or three) on the end. This function
    attempts to parse the provided JSON string a bit more tolerantly.
    """
    for suffix in ("", "]", "}", "}]"):
        try:
            return json.loads(response + suffix)
        except Exception as e:
            if "extra data" in str(e):
                limit = int(str(e).split("char ")[1].split(")")[0])
                try:
                    return json.loads(response[:limit])
                except Exception:
                    pass

    # If none of the above attempts at parsing worked, try cutting the end of the string:
    for to_cut in range(0, 100):
        try:
            return json.loads(response[:-to_cut])
        except Exception:
            pass

    # If none of _those_ attempts worked, well, throw something:
    raise LLMsArentPerfectAtGeneratingJSON(
        f"OpenAI returned a JSON response that was not syntactically valid: {response!r}"
    )
