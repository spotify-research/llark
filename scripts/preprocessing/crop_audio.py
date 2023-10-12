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

docker run -it --entrypoint=/bin/bash \
    --rm -v $(pwd):/tmp \
    -v ~/.config/gcloud:/root/.config/gcloud \
    annotate-dataset-dataflow
cd /tmp

# run on a small local test dataset
python scripts/preprocessing/crop_audio.py \
    --input-dir datasets/jamendo/wav \
        --output-dir datasets/testdata/ \
            --runner DirectRunner

# run on dataflow on mtg-jamendo
python scripts/preprocessing/crop_audio.py \
    --input-dir gs://bucketname/datasets/mtg-jamendo/wav \
    --output-dir gs://bucketname/datasets/mtg-jamendo/wav-crop \
    --runner DataflowRunner \
    --num-workers 512

# giantsteps-key
python scripts/preprocessing/crop_audio.py \
    --input-dir gs://bucketname/datasets/giantsteps-key/wav \
    --output-dir gs://bucketname/datasets/giantsteps-key/wav-crop \
    --runner DataflowRunner \
    --num-workers 512  \
    --p 0.0

# musicnet (with multicrop)
python scripts/preprocessing/crop_audio.py \
    --input-dir gs://bucketname/datasets/musicnet/wav \
    --output-dir gs://bucketname/datasets/musicnet/wav-crop \
    --runner DataflowRunner \
    --num-workers 512 \
    --multicrop


# medleydb
python scripts/preprocessing/crop_audio.py \
    --input-dir gs://bucketname/datasets/medleydb/wav \
    --output-dir gs://bucketname/datasets/medleydb/wav-crop \
    --runner DataflowRunner \
    --num-workers 32 \
    --p 0.0
"""
import argparse
import copy
import io
import os
import time
from typing import Any, Dict, Optional, Sequence

import apache_beam as beam
import numpy as np
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import storage
from scipy.io import wavfile

from m2t.dataset_utils import make_start_end_str
from m2t.gcs_utils import (
    GCP_PROJECT_NAME,
    GCS_BUCKET_NAME,
    US_CENTRAL1_REGION,
    list_files_with_extension,
    read_wav,
)


def id_to_filename(track_id: str, audio_dir: str):
    return os.path.join(audio_dir, str(track_id) + ".wav")


def _read_wav(
    filepath,
    target_sr=44100,
    duration: Optional[float] = 61,
) -> Dict[str, Any]:
    samples, sr = read_wav(filepath=filepath, target_sr=target_sr, duration=duration)
    return {"samples": samples, "audio_sample_rate": sr, "filepath": filepath}


def crop_samples(samples: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    assert end > start, f"cannot crop samples with start {start} and end {end}"
    return samples[int(start * sr) : int(end * sr)]


def crop_elem_audio_single(elem: Dict[str, Any], p: float = 0.2):
    """Crop audio, producing a single 30s crop for elem.

    This function can be applied to a PCollection of songs via a beam.Map
    transformation.

    If a track is less than 60s, we take the first 30s (or whatever audio
    exists).
    If a track is at least 60s, we take the first 30s with probability p,
    and the second 30s with probability 1-p.
    """
    samples_len = len(elem["samples"])
    sr = elem["audio_sample_rate"]
    if ((samples_len / sr) < 60) or (np.random.uniform() < p):
        # If cropped audio is less than 60secs; take the first 30s or whatever
        # audio it contains.
        # Also, take the first 30s for a random subset of samples with
        # probability p.
        cropped_samples = elem["samples"][: 30 * sr]
        audio_start_seconds = 0.0
        audio_end_seconds = len(cropped_samples) / sr
        print(f"[DEBUG] taking first {len(cropped_samples)/sr}sec of audio")
    else:
        # cropped_samples = elem["samples"][30 * sr : 60 * sr]
        cropped_samples = crop_samples(elem["samples"], sr, start=30.0, end=60.0)
        audio_start_seconds = 30.0
        audio_end_seconds = 30.0 + len(cropped_samples) / sr
        print("[DEBUG] taking from times 30:60sec of audio")

    elem["samples"] = cropped_samples.astype(np.float32)
    elem["audio_start_seconds"] = audio_start_seconds
    elem["audio_end_seconds"] = audio_end_seconds
    return elem


def crop_elem_audio_multi(elem: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
    """Crop audio, producing one 30s crop for every 30s chunk in elem.

    This function can be applied to a PCollection of songs via a beam.FlatMap transformation.
    """
    samples_len = len(elem["samples"])
    sr = elem["audio_sample_rate"]
    samples_len_seconds = samples_len / sr

    cropped_elems = []
    for i in range(0, int(samples_len_seconds // 30)):
        segment_start = i * 30
        segment_end = (i + 1) * 30
        cropped_elem = copy.deepcopy(elem)
        cropped_samples = crop_samples(cropped_elem["samples"], sr, segment_start, segment_end)
        cropped_elem["samples"] = cropped_samples.astype(np.float32)
        cropped_elem["audio_start_seconds"] = segment_start
        cropped_elem["audio_end_seconds"] = segment_end
        cropped_elems.append(cropped_elem)
    return cropped_elems


def write_output(elem: Dict[str, Any], output_dir: str):
    # output files with the same name
    filename = os.path.basename(elem["filepath"].replace("gs://", ""))
    output_filepath = os.path.join(output_dir, filename)

    # Record the start and end times of the crop in the filename.
    start_secs = round(elem["audio_start_seconds"], 3)
    end_secs = round(elem["audio_end_seconds"], 3)

    start_end_str = make_start_end_str(start_secs=start_secs, end_secs=end_secs)
    output_filepath = output_filepath.replace(".wav", f"-{start_end_str}.wav")
    print(f"[DEBUG] writing to {output_filepath}")

    if output_filepath.startswith("gs://"):
        # write the output to GCS
        gcs = storage.Client()
        bucket, file_name = output_filepath.replace("gs://", "").split("/", maxsplit=1)
        gcs_bucket_obj = gcs.get_bucket(bucket)
        blob = gcs_bucket_obj.blob(file_name)

        # wavfile can't write directly to GCS file handle, so we write the bytes
        # to a buffer first and then stream the buffer data to GCS.
        buf = io.BytesIO()
        wavfile.write(buf, rate=elem["audio_sample_rate"], data=elem["samples"])
        blob.upload_from_string(buf.read(), content_type="audio/x-wav")

    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # write the output to local files
        wavfile.write(
            output_filepath,
            rate=elem["audio_sample_rate"],
            data=elem["samples"],
        )
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="path to directory containing wav audio.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.2,
        help="probability of selecting first 30s for audio that is > 60s."
        "See process_audio() for more information.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory to output files.",
    )
    parser.add_argument(
        "--runner",
        default="DirectRunner",
        choices=["DirectRunner", "DataflowRunner"],
    )
    parser.add_argument(
        "--multicrop",
        action="store_true",
        default=False,
        help="If True, crop the entire audio into individual 30s chunks."
        "Any partial chunks are dropped (only full 30s chunks are kept).",
    )
    parser.add_argument("--job-name", default="music2text-crop-audio")
    parser.add_argument("--num-workers", default=32, help="max workers", type=int)
    parser.add_argument(
        "--worker-disk-size-gb",
        default=32,
        type=int,
        help="Worker disk size in GB. Note that disk size must be at least "
        + "size of the docker image.",
    )
    parser.add_argument(
        "--machine-type",
        default="n1-standard-2",
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
            "sdk_container_image": "gcr.io/bucketname/m2t-preprocess:latest",
        }

    pipeline_options = PipelineOptions(**pipeline_options)

    input_paths = list_files_with_extension(args.input_dir, extension=".wav")
    print(f"[INFO] processing files {input_paths}")

    with beam.Pipeline(options=pipeline_options) as p:
        p = p | "CreateData" >> beam.Create(input_paths)

        if not args.multicrop:
            p = (
                p
                | "ReadAudio" >> beam.Map(_read_wav)
                | "ProcessAudio" >> beam.Map(crop_elem_audio_single, p=args.p)
            )
        else:
            p = (
                p
                | "ReadAudio" >> beam.Map(_read_wav, duration=None)
                | "ProcessAudioMulticrop" >> beam.FlatMap(crop_elem_audio_multi)
            )

        p |= "WriteOutput" >> beam.Map(write_output, output_dir=args.output_dir)

    return


if __name__ == "__main__":
    main()
