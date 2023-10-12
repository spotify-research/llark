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
Script for fetching instruction-tuning data from OpenAI.

Usage:

# jamendo mir
python scripts/fetch_openai_instruct_data.py \
    --data-source datasets/mtg-jamendo/jamendo-annotated.jsonl \
    --dataset-name mtg-jamendo \
    --prompt-type mir \
    --runner DataflowRunner


"""
import argparse
import datetime
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional

import apache_beam as beam
import openai
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import storage

from m2t.dataset_utils import DATASET_INFO, read_jsonl_data
from m2t.instruct.prompting import PromptHelper, get_prompt_helper
from m2t.keys import OPENAI_KEY, OPENAI_ORGANIZATION

# set the below to match your setup
DEFAULT_MODEL = "gpt-3.5-turbo"
NUM_RETRIES = 4
GCP_PROJECT_NAME = ""
GCS_BUCKET_NAME = ""
US_CENTRAL1_REGION = "us-central1"


class StreamIntoFiles(beam.DoFn):
    """
    Write files, but don't end the pipeline - pass the written
    file paths on to the next transform.
    This transform avoids a shuffle step and avoids buffering data in memory,
    but doesn't allow control over the number of output files (num_shards).
    """

    def __init__(
        self,
        output_path: str,
        file_name_suffix=".txt",
        max_records_per_file: Optional[int] = 500,
    ):
        self.output_path = output_path
        self.file_name_suffix = file_name_suffix
        self.max_records_per_file = max_records_per_file

        if "gs://" in self.output_path:
            self.bucket_name = output_path.split("gs://")[1].split("/")[0]
            self.blob_prefix = output_path.replace(f"gs://{self.bucket_name}/", "")
            if not self.blob_prefix.endswith("/"):
                self.blob_prefix = self.blob_prefix + "/"

    def setup(self) -> None:
        self.log = logging.getLogger()
        self.storage_client = storage.Client()

    def open_file(self):
        import tensorflow.io.gfile

        self.bundle_uuid = str(uuid.uuid4()).replace("-", "")
        if "gs://" in self.output_path:
            self.bucket = self.storage_client.bucket(self.bucket_name)
            blob_name = f"{self.blob_prefix}{self.bundle_uuid}{self.file_name_suffix}"
            self.blob = self.bucket.blob(blob_name)
            self.handle = self.blob.open("wb")
            self.log.info(f"Opened: gs://{self.blob.bucket.name}/{self.blob.name}")
            self.handle.write(b"")
        else:
            self.filename = f"{self.output_path}/{self.bundle_uuid}{self.file_name_suffix}"
            tensorflow.io.gfile.makedirs(os.path.dirname(self.filename))
            self.handle = open(self.filename, "wb")
            self.log.info(f"Opened: {self.filename}")
        self.records_written = 0

    def start_bundle(self):
        self.open_file()

    def process(self, value):
        self.write_record_to_handle(value, self.handle)
        self.records_written += 1
        if self.records_written >= self.max_records_per_file:
            self.close_file()
            self.open_file()

    def write_record_to_handle(self, record, handle):
        if isinstance(record, str):
            self.handle.write(record.encode("utf-8"))
            if not record.endswith("\n"):
                self.handle.write(b"\n")
        elif isinstance(record, bytes):
            self.handle.write(record)
        else:
            raise NotImplementedError(
                f"Not sure how to write '{type(record)}' objects "
                f"to file extension {self.file_name_suffix}!"
            )

    def close_file(self):
        self.handle.close()
        if "gs://" in self.output_path:
            path = f"gs://{self.blob.bucket.name}/{self.blob.name}"
        else:
            path = self.filename
        self.log.info(f"Closed: {path}")

    def finish_bundle(self):
        self.close_file()


def response_contains_metadata_filter(response_text) -> bool:
    return "metadata" in response_text.lower()


def prompt(model: str, uri: str, metadata, prompt_helper: PromptHelper) -> Dict[str, List[str]]:
    """
    Given track metadata, ask ChatGPT to answer the prompt, and return ChatGPT's response
    as a JSON dict with the metadata added back in.
    """
    query = prompt_helper.get_chatgpt_query(metadata)
    last_exception = None
    prompt_text = prompt_helper.get_prompt_text()
    messages = prompt_helper.build_messages(prompt_text, query)

    # This needs to be done at runtime on each Dataflow worker:
    openai.organization = OPENAI_ORGANIZATION
    openai.api_key = OPENAI_KEY

    for retry in range(NUM_RETRIES):
        try:
            response_iterator = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=True,
                timeout=60,
            )
            response_chunks = []
            for chunk in response_iterator:
                if not chunk.get("choices", []):
                    continue
                if chunk["choices"][0]["finish_reason"] == "length":
                    raise ValueError(
                        "Model returned too much data before running out of available token space!"
                    )
                text = chunk["choices"][0]["delta"].get("content")
                if text:
                    response_chunks.append(text)
            text = "".join(response_chunks)
            return prompt_helper.postprocess_response_text(text, query, uri)
        except Exception as e:
            last_exception = e
            time.sleep(2**retry)
    print(f"Failed to fetch data for {uri} due to {last_exception}")
    return {"uri": uri, "exception": repr(last_exception)}


def main():
    parser = argparse.ArgumentParser(
        description=("Query OpenAI's model(s) for structured information about Spotify track URIs.")
    )

    parser.add_argument("--output-path", help="The output file to save JSONL data to.")
    parser.add_argument(
        "--runner",
        help="Which Beam runtime to use.",
        choices=["DataflowRunner", "DirectRunner"],
        default="DataflowRunner",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="The OpenAI model to use.",
        choices=["gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    )
    parser.add_argument(
        "--num-threads-per-worker",
        default=4,
        type=int,
        help=(
            "The number of parallel requests to make of OpenAI per Dataflow worker. "
            "Multiplied by --num-workers, this roughly gives the total number of simultaneous "
            "requests. Turn this number down if getting errors or OpenAI rate limits."
        ),
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="The number of Dataflow workers to spin up.",
    )
    parser.add_argument(
        "--worker-disk-size-gb",
        default=128,
        type=int,
        help="Worker disk size in GB. Note that disk size must be at least size of the "
        + "docker image.",
    )

    parser.add_argument(
        "--data-source",
        required=True,
        help="Data source to use. Should be a path (or wildcard) to a valid JSONL file(s).",
    )

    parser.add_argument(
        "--dataset-name",
        choices=list(DATASET_INFO.keys()),
        required=True,
        help="Name of the dataset being used. This is required to select "
        + "the correct prompt template.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit on number of samples to try. If the provided dataset is larger, "
        + "it will be downsampled.",
    )
    parser.add_argument(
        "--few-shot",
        default=False,
        action="store_true",
        help="Whether to use few-shot prompting to GPT. "
        "If True, use few-shot examples for prompting. See PromptHelper for more info.",
    )

    parser.add_argument(
        "--prompt-type",
        default="default",
        choices=["default", "mir", "reasoning", "captioning"],
        help="the type of prompt to use.",
    )
    parser.add_argument("--drop-columns", default=None, nargs="+")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    date_slug = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    if args.output_path is None:
        args.output_path = f"gs://{GCS_BUCKET_NAME}/openai-data/{date_slug}/{args.model}/"

    if args.runner == "DataflowRunner":
        job_name = f"music2text-{args.model.replace('.', '-')}-scraper-{date_slug}"[:128]
        print(f"🏠 Output data will be written to: {args.output_path}")
        print(f"🚀 Starting a Dataflow job named: {job_name}...")

        pipeline_options = {
            "runner": "DataflowRunner",
            "project": GCP_PROJECT_NAME,
            "job_name": job_name,
            "region": US_CENTRAL1_REGION,
            "machine_type": "n1-highcpu-4",
            "max_num_workers": args.num_workers,
            "worker_disk_type": "pd-ssd",
            "disk_size_gb": args.worker_disk_size_gb,
            "experiments": ["use_runner_v2", "beam_fn_api"],
            # Control concurrency here to avoid DDOS'ing OpenAI:
            "number_of_worker_harness_threads": args.num_threads_per_worker,
            "save_main_session": True,
            "sdk_container_image": "gcr.io/path/to/m2t-preprocess:latest",
            "temp_location": f"gs://{GCS_BUCKET_NAME}/dataflow-tmp",
        }
    else:
        pipeline_options = {
            "runner": args.runner,
            "project": GCP_PROJECT_NAME,
            "temp_location": f"gs://{GCS_BUCKET_NAME}/dataflow-tmp",
        }

    pipeline_options = PipelineOptions(**pipeline_options)

    dataset_info = DATASET_INFO[args.dataset_name]
    uri_key = dataset_info.id_col
    prompt_helper = get_prompt_helper(args.prompt_type, dataset_info, few_shot=args.few_shot)
    with beam.Pipeline(options=pipeline_options) as p:
        df = read_jsonl_data(args.data_source)
        if args.drop_columns:
            df.drop(columns=args.drop_columns, inplace=True)

        if args.max_samples is not None and args.max_samples < len(df):
            logging.warning(f"downsampling data from size {len(df)} to {args.max_samples}")
            df = df.sample(n=args.max_samples)
        uris_and_metadata = (
            p
            | "Create from df" >> beam.Create(df.to_dict("records"))
            | "AddURIKey" >> beam.Map(lambda x: (x[uri_key], x))
        )

        (
            uris_and_metadata
            | f"Ask {args.model}"
            >> beam.MapTuple(lambda uri, metadata: prompt(args.model, uri, metadata, prompt_helper))
            | "Convert to JSONL" >> beam.Map(lambda dict: json.dumps(dict))
            | "Write Batches"
            >> beam.ParDo(
                StreamIntoFiles(
                    args.output_path,
                    file_name_suffix=".jsonl",
                    max_records_per_file=50,
                )
            )
        )


if __name__ == "__main__":
    main()
