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
Fetch the CLAP embeddings for a set of audio files and write them to numpy.

Usage (should be run from inside m2t-preprocess docker environment; see docker directory):

python scripts/clap/clap_embeddings.py \
    --input-dir tests/data/ \
    --output-dir tmp/clap-test \
    --ckpt-file checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt \
    --runner DirectRunner

python scripts/clap/clap_embeddings.py \
    --input-dir gs://bucketname//wav/ \
    --output-dir gs://bucketname//representations/clap \
    --ckpt-file gs://bucketname/checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt \
    --runner DataflowRunner

"""
import argparse
import os
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import apache_beam as beam
import laion_clap
import numpy as np
import torch
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions
from laion_clap.training.data import (
    float32_to_int16,
    get_audio_features,
    int16_to_float32,
)

from m2t.gcs_utils import (
    GCP_PROJECT_NAME,
    GCS_BUCKET_NAME,
    US_CENTRAL1_REGION,
    US_CENTRAL1_SUBNETWORK,
    download_blob,
    list_files_with_extension,
    read_wav,
    split_gcs_bucket_and_filepath,
    write_npy,
)


class ClapModelHandler(ModelHandler[str, PredictionResult, laion_clap.CLAP_Module]):
    def __init__(self, ckpt_file: str):
        self.ckpt_file = ckpt_file

    def load_model(self) -> laion_clap.CLAP_Module:
        """Loads and initializes a model for processing."""
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")

        if self.ckpt_file.startswith("gs://"):
            bucket_name, blob_name = split_gcs_bucket_and_filepath(self.ckpt_file)
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = os.path.basename(blob_name)
                tmp_filepath = os.path.join(tmpdir, filename)
                download_blob(bucket_name, blob_name, tmp_filepath)
                model.load_ckpt(tmp_filepath)
        else:
            model.load_ckpt(self.ckpt_file)

        return model

    def run_inference(
        self,
        batch: Sequence[List[Dict[str, Any]]],
        model: laion_clap.CLAP_Module,
        inference_args: Optional[Dict[str, Any]] = None,
    ) -> Iterable[PredictionResult]:
        """Runs inferences on a batch of text strings representing a filename.

        Args:
          batch: A sequence of examples as text strings.
          model: A laion_clap.CLAP_Module
          inference_args: Any additional arguments for an inference.

        Returns:
          An Iterable of type PredictionResult.
        """
        # Loop each text string, and use a tuple to store the inference results.
        predictions = []
        for elem in batch:
            audio_features = elem["audio_features"]
            with torch.no_grad():
                outputs = model.model.get_audio_embedding(audio_features)
            outputs = outputs.numpy()
            predictions.append(outputs)
        return [PredictionResult(x, y) for x, y in zip(batch, predictions)]


# config for the checkoint music_audioset_epoch_15_esc_90.14.pt
# with enable_fusion=False, amodel= 'HTSAT-base'
CLAP_MODEL_CFG = {
    "audio_length": 1024,
    "clip_samples": 480000,
    "mel_bins": 64,
    "sample_rate": 48000,
    "window_size": 1024,
    "hop_size": 480,
    "fmin": 50,
    "fmax": 14000,
    "class_num": 527,
    "model_type": "HTSAT",
    "model_name": "base",
}


def load_audio_input(
    elem: Dict[str, Any],
    model_cfg=CLAP_MODEL_CFG,
    enable_fusion=False,
    target_sr=48_000,
) -> Dict[str, Any]:
    """Adapted from laion_clap.hook.py"""
    # load the waveform of the shape (T,), should resample to 48000
    f = elem["file"]
    audio_waveform, _ = read_wav(f, target_sr=target_sr)

    # quantize
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()
    audio_features_dict = {}
    audio_features_dict = get_audio_features(
        audio_features_dict,
        audio_waveform,
        target_sr,
        data_truncating="fusion" if enable_fusion else "rand_trunc",
        data_filling="repeatpad",
        audio_cfg=model_cfg,
        require_grad=audio_waveform.requires_grad,
    )

    elem["audio_features"] = [audio_features_dict]
    return elem


def write_output(prediction_result: PredictionResult, output_dir: str) -> PredictionResult:
    elem = prediction_result.example
    embed = prediction_result.inference

    key = elem["file"]
    key = os.path.basename(key)
    print(f"[DEBUG] writing embedding for key {key}")
    print(f"[DEBUG] embedding for key {key} is of type {type(embed)}")

    output_path = os.path.join(output_dir, key.replace(".wav", ".npy"))
    assert isinstance(embed, np.ndarray), f"expected ndarray, got {type(embed)}"
    write_npy(output_path, embed)
    return prediction_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="path to directory containing wav audio.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory to output files.",
    )
    parser.add_argument(
        "--ckpt-file",
        required=True,
        help="Path to a CLAP checkpoint file (should end with .pt).",
    )
    parser.add_argument(
        "--runner",
        default="DirectRunner",
        choices=["DirectRunner", "DataflowRunner"],
    )
    parser.add_argument("--job-name", default="music2text-clap-embed")
    parser.add_argument("--num-workers", default=128, help="max workers", type=int)
    parser.add_argument(
        "--worker-disk-size-gb",
        default=32,
        type=int,
        help="Worker disk size in GB. Note that disk size must be at least size of the "
        + "docker image.",
    )
    parser.add_argument(
        "--machine-type",
        default="n1-highmem-2",
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
            "subnetwork": US_CENTRAL1_SUBNETWORK,
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
    pipeline_options = PipelineOptions(**pipeline_options)

    input_paths = list_files_with_extension(args.input_dir, extension=".wav")
    print(f"[INFO] processing files {input_paths}")
    clap_model_handler = ClapModelHandler(ckpt_file=args.ckpt_file)

    with beam.Pipeline(options=pipeline_options) as p:
        p = (
            p
            | "CreateData" >> beam.Create(input_paths)
            | "ToDict" >> beam.Map(lambda x: {"file": x})
            | "LoadAudioInput" >> beam.Map(load_audio_input)
            | "FilterInvalidAudio" >> beam.Filter(lambda x: x["audio_features"] is not None)
            | "RunClapInference" >> RunInference(clap_model_handler)
            | "FilterInvalidOutputs" >> beam.Filter(lambda x: x.inference is not None)
            # | beam.Map(print)
            | "WriteOutput" >> beam.Map(write_output, output_dir=args.output_dir)
        )
