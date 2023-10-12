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
Convert many audio files to wav in parallel using ffmpeg.

Usage:

# on a small local sample
python scripts/convert_audio_to_wav.py \
    --input-dir datasets/testdata \
    --input-extension .mp3 \
    --output-dir datasets/tmp \
    --runner DirectRunner

# on a small GCS sample
python scripts/convert_audio_to_wav.py \
    --input-dir gs://bucket/testdata-mp3/ \
    --input-extension .mp3 \
    --output-dir gs://bucket/testoutputs-mp3towav/ \
    --runner DirectRunner


"""

import argparse
import logging
import os
import tempfile
import time

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from m2t.audio_io import convert_to_wav
from m2t.gcs_utils import (
    GCP_PROJECT_NAME,
    GCS_BUCKET_NAME,
    US_CENTRAL1_REGION,
    gcs_bucket,
    list_files_with_extension,
    move_file,
    split_gcs_bucket_and_filepath,
)


def process_file(infile: str, output_dir: str):
    if infile.startswith("gs://"):
        # Case: file is on GCS; download it first to use FFMPEG.

        bucket_src, filepath_src = split_gcs_bucket_and_filepath(infile)
        gcs_bucket_obj = gcs_bucket(bucket_src)
        filename = os.path.basename(filepath_src)
        blob_src = gcs_bucket_obj.blob(filepath_src)

        # Download the file to a temorary directory to process it with ffmpeg.
        with tempfile.TemporaryDirectory() as tmpdir:
            destination_file_path = os.path.join(tmpdir, filename)
            blob_src.download_to_filename(destination_file_path)

            converted_local_filename = convert_to_wav(destination_file_path, tmpdir)
            converted_dest_filename = os.path.join(
                output_dir, os.path.basename(converted_local_filename)
            )
            if not converted_local_filename:
                logging.warning(f"got no converted file for {filepath_src}")
                return

            move_file(converted_local_filename, converted_dest_filename)
    else:
        convert_to_wav(infile, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="path or wildcard to input files")
    parser.add_argument(
        "--input-extension",
        default=".mp3",
        help=(
            "The extension of the files to process (e.g. '.mp3'). "
            "Files without this extension will be ignored. "
            "Must be a file type that ffmpeg can read.",
        ),
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
    parser.add_argument("--job-name", default="music2text-convert-audio")
    parser.add_argument("--num-workers", default=32, help="max workers", type=int)
    parser.add_argument(
        "--worker-disk-size-gb",
        default=32,
        type=int,
        help=(
            "Worker disk size in GB. Note that disk size must be at least size of the docker image."
        ),
    )
    parser.add_argument(
        "--machine-type", default="n1-standard-2", help="Worker machine type to use."
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

    # Read the wav audio and sample rate. Note that we do NOT allow to adjust
    # the sample rate (and instead fix it at 44100) because this value is also
    # hard-coded in the Madmom code
    # (e.g. https://github.com/CPJKU/madmom/blob/3bc8334099feb310acfce884ebdb76a28e01670d/madmom/features/beats.py#L92)
    input_paths = list_files_with_extension(args.input_dir, extension=args.input_extension)
    print(f"processing {len(input_paths):,} files.")

    with beam.Pipeline(options=PipelineOptions(**pipeline_options)) as p:
        (
            p
            | "CreatePColl" >> beam.Create(input_paths)
            | "ProcessAudioFile" >> beam.Map(process_file, output_dir=args.output_dir)
        )


if __name__ == "__main__":
    main()
