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
Usage:

# run the docker environment needed to execute the code
docker run -it --entrypoint=/bin/bash --rm -v $(pwd):/tmp -v \
    ~/.config/gcloud:/root/.config/gcloud m2t-preprocess:latest
cd /tmp


python scripts/annotate_dataset.py \

    --audio-dir datasets/fma/full/wav \
    --output-dir datasets/fma/full \
    --runner DirectRunner \
    --max-audio-duration-seconds 360

python scripts/annotate_dataset.py \
    --dataset-name slakh \
    --input-file 'gs://bucketname/datasets/slakh2100/slakh-test.json' \
    --audio-dir 'gs://bucketname/datasets/slakh2100/wav/test' \
    --output-dir 'gs://bucketname/datasets/slakh2100/annotated/test' \
    --num-workers 512 \
    --runner DataflowRunner


"""
import argparse
import json
import logging
import os
import time
from functools import partial
from typing import Any, Dict, Optional

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from m2t.annotation import (
    ExtractMadmomChordEstimates,
    ExtractMadmomDownbeatFeatures,
    ExtractMadmomKeyEstimates,
    ExtractMadmomTempoFeatures,
)
from m2t.dataset_utils import DATASET_INFO, DatasetInfo
from m2t.gcs_utils import (
    GCP_PROJECT_NAME,
    GCS_BUCKET_NAME,
    US_CENTRAL1_REGION,
    file_exists,
    read_wav,
)


def read_element_wav(
    elem: Dict[str, Any],
    audio_dir,
    dataset_info: DatasetInfo,
    target_sr=44100,
    duration: Optional[float] = None,
) -> Dict[str, Any]:
    track_id = elem[dataset_info.id_col]
    filepath = dataset_info.id_to_filename(track_id, audio_dir)
    samples, sr = read_wav(filepath=filepath, target_sr=target_sr, duration=duration)
    elem["audio"] = samples
    elem["audio_sample_rate"] = sr
    return elem


def drop_audio_features(elem: Dict[str, Any]) -> Dict[str, Any]:
    if "audio" in elem:
        del elem["audio"]
    if "audio_sample_rate" in elem:
        del elem["audio_sample_rate"]
    return elem


def audio_file_exists(elem: Dict[str, Any], audio_dir: str, dataset_info: DatasetInfo) -> bool:
    track_id = elem[dataset_info.id_col]
    filepath = dataset_info.id_to_filename(track_id, audio_dir)
    return file_exists(filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="path or wildcard to input files")
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="path to directory containing wav audio.",
    )
    parser.add_argument(
        "--max-audio-duration-seconds",
        default=360,
        type=float,
        help="Maximum audio duration to load (in seconds). Helps limit memory use and "
        + "avoid OOM during feature extraction.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        choices=list(DATASET_INFO.keys()),
    )
    parser.add_argument(
        "--replace-files",
        default=False,
        action="store_true",
        help="If true, overwrite each preexisting file with the newly-featurized "
        + "data at the same location.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory to output files, if replace-files is False.",
    )
    parser.add_argument(
        "--runner",
        default="DirectRunner",
        choices=["DirectRunner", "DataflowRunner"],
    )
    parser.add_argument("--job-name", default="music2text-annotate-dataset")
    parser.add_argument("--num-workers", default=32, help="max workers", type=int)
    parser.add_argument(
        "--worker-disk-size-gb",
        default=32,
        type=int,
        help="Worker disk size in GB. Note that disk size must be at least size "
        + "of the docker image.",
    )
    parser.add_argument(
        "--machine-type",
        default="n1-highmem-4",
        help="Worker machine type to use.",
    )
    args = parser.parse_args()
    job_name = f"{args.job_name}-{int(time.time())}"

    if args.runner == "DirectRunner":
        pipeline_options = {
            "runner": args.runner,
            "project": GCP_PROJECT_NAME,
            "temp_location": f"gs://{GCS_BUCKET_NAME}/dataflow-tmp",
        }
    else:
        pipeline_options = {
            "runner": args.runner,
            "project": GCP_PROJECT_NAME,
            "temp_location": f"gs://{GCS_BUCKET_NAME}/dataflow-tmp",
            "job_name": job_name,
            "region": US_CENTRAL1_REGION,
            "max_num_workers": args.num_workers,
            "worker_disk_type": "pd-ssd",
            "disk_size_gb": args.worker_disk_size_gb,
            "machine_type": args.machine_type,
            "save_main_session": True,
            "experiments": [
                "use_runner_v2",
                "beam_fn_api",
                "no_use_multiple_sdk_containers",
            ],
            "sdk_container_image": "gcr.io/path/to/m2t-preprocess:latest",
        }
        assert args.input_file.startswith("gs://"), "Must use GCS files for DataflowRunner."

    assert (args.replace_files or args.output_dir) and not (
        args.replace_files and args.output_dir
    ), "Must set one of --replace-files or --output-dir, but not both."

    if args.max_audio_duration_seconds > 360.0:
        logging.warning(
            f"max_audio_duration_seconds is set to {args.max_audio_duration_seconds}; "
            + "very long audio (>>45mins) can cause OOM for madmom."
        )

    pipeline_options = PipelineOptions(**pipeline_options)
    dataset_info = DATASET_INFO[args.dataset_name]

    # Read the wav audio and sample rate. Note that we do NOT allow to adjust
    # the sampple rate (and instead fix it at 44100) because this value is also
    # hard-coded in the Madmom code
    # (e.g. https://github.com/CPJKU/madmom/blob/3bc8334099feb310acfce884ebdb76a28e01670d/
    #           madmom/features/beats.py#L92)
    _read_wav_fn = partial(
        read_element_wav,
        audio_dir=args.audio_dir,
        duration=args.max_audio_duration_seconds,
        dataset_info=dataset_info,
    )

    with beam.Pipeline(options=pipeline_options) as p:
        p |= (
            "ReadInput" >> beam.io.ReadFromText(args.input_file)
            | "ParseJSON" >> beam.Map(json.loads)
            | "FilterInputs"
            >> beam.Filter(
                audio_file_exists,
                audio_dir=args.audio_dir,
                dataset_info=dataset_info,
            )
            | "ReadAudio" >> beam.Map(_read_wav_fn)
            | "FilterNonEmptyAudio" >> beam.Filter(lambda x: len(x["audio"]))
            | "ExtractMadmomTempo" >> beam.ParDo(ExtractMadmomTempoFeatures())
            | "ExtractMadmomDownbeats" >> beam.ParDo(ExtractMadmomDownbeatFeatures())
            | "ExtractMadmomChords" >> beam.ParDo(ExtractMadmomChordEstimates())
            | "ExtractMadmomKey" >> beam.ParDo(ExtractMadmomKeyEstimates())
            | "RemoveAudio" >> beam.Map(drop_audio_features)
            | "SerializeToText" >> beam.Map(json.dumps)
            # | "PrintOutput" >> beam.Map(print)
            | "WriteOutput"
            >> beam.io.WriteToText(
                file_path_prefix=os.path.join(args.output_dir, job_name),
                file_name_suffix=".jsonl",
            )
        )
    return


if __name__ == "__main__":
    main()
