"""
Note: due to the different python version used in the Jukebox docker
    image, this script should be launched from inside the jukemir
    conda environment.

usage:
python dataflow_inference.py \
    --input "gs://<my-bucket>/datasets/gtzan/wav/" \
    --output "gs://<my-bucket>/datasets/gtzan/representations/jukebox/f10" \
    --runner DataflowRunner \
    --accelerator-type nvidia-tesla-v100 \
    --num-workers 128

"""
import argparse
import io
import logging
import os
import pathlib
import time
from functools import lru_cache, partial
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import apache_beam as beam
import numpy as np
import torch
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import storage

# Constants; set these to match your GCP configuration, or set them as environment variables.
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME") or ""
GCP_REGION = "us-central1"

# To avoid warnings, set the project.
os.environ["GOOGLE_CLOUD_PROJECT"] = GOOGLE_CLOUD_PROJECT

if not GOOGLE_CLOUD_PROJECT:
    raise ValueError(
        f"Please set the GOOGLE_CLOUD_PROJECT variable in {__file__} "
        "(or as an environment variable)"
    )

if not GCS_BUCKET_NAME:
    raise ValueError(
        f"Please set the GCS_BUCKET_NAME variable in {__file__} (or as an environment variable)"
    )


@lru_cache(None)
def gcs_client():
    return storage.Client()


@lru_cache(None)
def gcs_bucket(bucket_name: str):
    return gcs_client().get_bucket(bucket_name)


def split_gcs_bucket_and_filepath(filepath: str) -> Tuple[str, str]:
    """Return a (bucketname, filepath) tuple."""
    return filepath.replace("gs://", "").split("/", maxsplit=1)


def read_wav_bytes(filepath: str) -> bytes:
    """Read the bytes for a wav file from GCS."""
    assert filepath.startswith("gs://"), f"Expected a file path on GCS, but got: {filepath!r}"
    bucket_name, file_name = split_gcs_bucket_and_filepath(filepath)
    return gcs_bucket(bucket_name).blob(file_name).download_as_string()


class JukeboxModelWrapper:
    """Wrapper for a Jukebox embedding model."""

    def __init__(self, model_name: str, device) -> None:
        from jukebox.hparams import Hyperparams, setup_hparams
        from jukebox.make_models import MODELS, make_prior, make_vqvae

        self.model = model_name
        self.device = device
        self.hps = Hyperparams()
        self.hps.sr = 44100
        self.hps.n_samples = 3 if self.model == "5b_lyrics" else 8
        self.hps.name = "samples"
        self.chunk_size = 16 if self.model == "5b_lyrics" else 32
        self.max_batch_size = 3 if self.model == "5b_lyrics" else 16
        self.hps.levels = 3
        self.hps.hop_fraction = [0.5, 0.5, 0.125]
        vqvae, *priors = MODELS[self.model]
        self.vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=1048576)), self.device)

        # Set up language self.model
        self.hparams = setup_hparams(priors[-1], dict())
        self.hparams["prior_depth"] = 36
        self.top_prior = make_prior(self.hparams, self.vqvae, self.device)

    def __call__(self, input_path: str, *args: Any, **kwds: Any) -> np.ndarray:
        from main import EmptyFileError, get_acts_from_file  # TODO: rename 'main'.

        print(f"processing file {input_path} on device {self.device}")
        wav_bytes = read_wav_bytes(input_path)
        with torch.no_grad():
            try:
                representation = get_acts_from_file(
                    io.BytesIO(wav_bytes),
                    self.hps,
                    self.vqvae,
                    self.top_prior,
                    meanpool=True,
                    pool_frames_per_second=10,
                )
                return representation
            except EmptyFileError:
                return None


class JukeboxModelHandler(ModelHandler[str, PredictionResult, JukeboxModelWrapper]):
    def __init__(self, model_name: str = "5b"):
        self.model_name = model_name

    def load_model(self) -> JukeboxModelWrapper:
        """Loads and initializes a model for processing."""
        # Set up MPI
        from jukebox.utils.dist_utils import setup_dist_from_mpi

        rank, local_rank, device = setup_dist_from_mpi()

        import torch

        print(f"cuda is available: {torch.cuda.is_available()}")
        print(f"device count is {torch.cuda.device_count()}")
        print(f"device name: {torch.cuda.get_device_name(0)}")
        assert torch.cuda.is_available()
        return JukeboxModelWrapper(self.model_name, device)

    def run_inference(
        self,
        batch: Sequence[str],
        model: JukeboxModelWrapper,
        inference_args: Optional[Dict[str, Any]] = None,
    ) -> Iterable[PredictionResult]:
        """Runs inferences on a batch of text strings representing a filename.

        Args:
          batch: A sequence of examples as text strings.
          model: A JukeboxModelWrapper model
          inference_args: Any additional arguments for an inference.

        Returns:
          An Iterable of type PredictionResult.
        """
        # Loop each text string, and use a tuple to store the inference results.
        predictions = []
        for filepath in batch:
            outputs = model(filepath)
            predictions.append([outputs])
        return [PredictionResult(x, y) for x, y in zip(batch, predictions)]


def get_input_file_list(input_dir: str, extension: str = ".wav") -> Sequence[str]:
    if input_dir.startswith("gs://"):
        bucket, prefix = split_gcs_bucket_and_filepath(input_dir)
        blobs = [x for x in gcs_client().list_blobs(bucket, prefix=prefix)]
        input_paths = [f"gs://{x._bucket.name}/{x.name}" for x in blobs]
    else:
        input_dir = pathlib.Path(input_dir)
        input_paths = sorted(list(input_dir.iterdir()))

    return [x for x in input_paths if x.endswith(extension)]


def write_prediction_result(prediction_result: PredictionResult, output_dir) -> None:
    """Write a prediction result to GCS or local path as a numpy array."""
    # input_path contains the full absolute path to the input file;
    # an example input path: gs://bucketname/datasets/testdata/zw5dkiklbhE.wav
    input_path = prediction_result.example
    representation = prediction_result.inference

    input_filename = os.path.basename(input_path.replace("gs://", ""))
    output_filename = input_filename.replace(".wav", ".npy")

    if output_dir.startswith("gs://"):
        # Case: GCS output; write bytes directly to GCS file handle.

        bucket_name, blob_path = split_gcs_bucket_and_filepath(output_dir)
        output_path = os.path.join(blob_path, output_filename)
        print(f"[DEBUG] writing output to gs://{bucket_name}/{output_path}")
        logging.warning(f"[DEBUG] writing output to gs://{bucket_name}/{output_path}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(output_path)

        with blob.open("wb") as f:
            np.save(f, representation)

    else:
        # Case: local file; save to local file path.
        output_dir = pathlib.Path(args.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, representation)
    return


def main_method():
    logging.getLogger().setLevel(logging.WARN)
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", default="/input", help="path to inputs")
    parser.add_argument("--output-dir", default="/output", help="path to outputs")
    parser.add_argument(
        "--runner",
        help="Which Beam runtime to use.",
        choices=["DataflowRunner", "DirectRunner"],
        default="DirectRunner",
    )
    parser.add_argument(
        "--job-name",
        default="music2text-jukebox-embed-pipeline",
        help="Job name prefix to use in DataFlow.",
    )
    parser.add_argument("--accelerator-type", default="nvidia-tesla-v100")
    parser.add_argument("--num-workers", default=16, type=int, help="maximum number of workers.")
    parser.add_argument(
        "--accelerator-count",
        default=1,
        type=int,
        help="Count of accelerators per worker.",
    )

    args = parser.parse_args()

    pipeline_options = {
        "runner": args.runner,
        "project": GOOGLE_CLOUD_PROJECT,
    }

    if args.runner == "DataflowRunner":
        pipeline_options.update(
            {
                "job_name": f"{args.job_name}-{int(time.time())}",
                "region": GCP_REGION,
                "machine_type": "n1-highmem-8",
                "worker_disk_type": "pd-ssd",
                "experiments": [
                    "use_runner_v2",
                    "no_use_multiple_sdk_containers",
                    "disable_worker_container_image_prepull",  # see https://cloud.google.com/dataflow/docs/guides/using-custom-containers#usage
                ],
                "temp_location": f"gs://{GCS_BUCKET_NAME}/dataflow-tmp",
                "dataflow_service_options": [
                    f"worker_accelerator=type:{args.accelerator_type};count:{args.accelerator_count};install-nvidia-driver"
                ],
                "max_num_workers": args.num_workers,
                "save_main_session": True,
                "sdk_container_image": "gcr.io/bucketname/music2text-dataflow:latest",
                "disk_size_gb": 128,  # TODO: try decreasing later
                "number_of_worker_harness_threads": 1,
            }
        )

    input_paths = get_input_file_list(args.input_dir)
    print(f"[INFO] processing files {input_paths}")

    with beam.Pipeline(options=PipelineOptions(**pipeline_options)) as p:
        (
            p
            | "Input Data" >> beam.Create(input_paths)
            | "Run Jukebox Inference" >> RunInference(JukeboxModelHandler(model_name="5b"))
            | "Filter Invalid Outputs" >> beam.Filter(lambda x: x.inference is not None)
            | "Write Output"
            >> beam.Map(partial(write_prediction_result, output_dir=args.output_dir))
        )


if __name__ == "__main__":
    main_method()
