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
Convert any suported dataset to JSON format.

This allows for a unified interface of all of the downstream components
(audio annotation, etc.) by ensuring that every dataset is JSON-formatted.

Usage:

python scripts/preprocessing/jsonify_dataset.py \
    --dataset fsl10k \
    --input-dir datasets/fsl10k \
    --output-dir datasets/fsl10k \
    --split all
"""
import argparse
import json

from m2t.preprocessing import _JSONIFIERS, get_jsonifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(_JSONIFIERS.keys()),
        help="dataset to use.",
        required=True,
    )
    parser.add_argument("--input-dir", required=True, help="input dir for the dataset.")
    parser.add_argument(
        "--output-dir",
        default="./tmp",
        help="where to write the json file(s) for the dataset.",
    )
    parser.add_argument("--split", choices=["train", "test", "validation", "eval", "all"])
    parser.add_argument(
        "--dataset-kwargs",
        default=None,
        help="json-formatted string of dataset-specific kwargs; "
        + 'example: `{"minimum_caption_length": 8}` ',
    )

    args = parser.parse_args()

    jsonify_kwargs = json.loads(args.dataset_kwargs) if args.dataset_kwargs else {}

    jsonifier = get_jsonifier(
        args.dataset, input_dir=args.input_dir, split=args.split, **jsonify_kwargs
    )
    jsonifier.load_raw_data()
    jsonifier.export_to_json(args.output_dir)

    return


if __name__ == "__main__":
    main()
