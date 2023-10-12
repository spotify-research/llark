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

import io
import logging
import math
import os
import shutil
import tempfile
from typing import Optional, Sequence, Tuple, Union
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf
import torch
from google.cloud import storage

# Constants; set these to match your GCP configuration, or set them as environment variables.
M2T_BUCKET_NAME = ""

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
    return storage.Client(project=GOOGLE_CLOUD_PROJECT)


@lru_cache(None)
def gcs_bucket(bucket_name: str):
    return gcs_client().get_bucket(bucket_name)


def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """Download a blob from the bucket to the provided filename"""
    gcs_bucket(bucket_name).blob(source_blob_name).download_to_filename(destination_file_name)
    print(
        f"Downloaded storage object {source_blob_name!r} from bucket "
        f"{bucket_name!r} to local file {destination_file_name!r}."
    )


def move_file(src: str, dest: str):
    """Move a local file from src to dest, where dest can be a GCS path."""
    assert os.path.exists(src), f"source file {src} does not exist."

    if dest.startswith("gs://"):
        bucket_dest, filepath_dest = split_gcs_bucket_and_filepath(dest)
        gcs_bucket(bucket_dest).blob(filepath_dest).upload_from_filename(src)
    else:
        shutil.move(src, dest)


def split_gcs_bucket_and_filepath(filepath: str) -> Tuple[str, str]:
    """Return a (bucketname, filepath) tuple."""
    return filepath.replace("gs://", "").split("/", maxsplit=1)


def file_exists(filepath):
    """Check if a file exists (handles both local and GCS filepaths)."""
    if filepath.startswith("gs://"):
        bucket_name, file_name = split_gcs_bucket_and_filepath(filepath)
        bucket = gcs_bucket(bucket_name)
        return bucket.blob(file_name).exists()
    else:
        return os.path.exists(filepath)


def read_wav(
    filepath: str, target_sr: int = 44100, duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """Read a wav file, either local on on GCS."""

    print(f"reading audio from {filepath}")
    if filepath.startswith("gs://"):
        # Case: file located on GCS

        # Create a Cloud Storage client and parse the path.
        gcs = storage.Client(project=GOOGLE_CLOUD_PROJECT)
        bucket, file_name = filepath.replace("gs://", "").split("/", maxsplit=1)
        gcs_bucket_obj = gcs.get_bucket(bucket)

        # read the file data
        blob = gcs_bucket_obj.blob(file_name)
        bytes_as_string = blob.download_as_string()

    else:
        # Case: local file
        with open(filepath, "rb") as f:
            bytes_as_string = f.read()

    # Samplerate does not allow to specify sr when reading; if desired,
    # the audio will need to be resampled in a postprocessing step.
    # For some reason, librosa fails to read due to an issue with
    # lazy-loading of modules when executed within a beam pipeline.
    samples, audio_sr = sf.read(
        io.BytesIO(bytes_as_string),
        frames=math.floor(target_sr * duration) if duration is not None else -1,
    )
    print(
        f"finished reading audio from {filepath} with sr {audio_sr} "
        f"with duration {round(len(samples)/audio_sr,2)}secs"
    )

    if audio_sr != target_sr:
        print(f"resampling audio input {filepath} from {audio_sr} to {target_sr}")
        samples = librosa.resample(samples, orig_sr=audio_sr, target_sr=target_sr)

    assert np.issubdtype(
        samples.dtype, float
    ), f"exected floating-point audio; got type {samples.dtype}"

    return samples, target_sr


def list_blobs_with_prefix(bucket_name, prefix):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned.
    """
    logging.info(f"reading blobs for gs://{bucket_name}/{prefix}")
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # consume the iterator to actually make the requests to GCS
    return [x for x in blobs]


def list_files_with_extension(input_dir: str, extension: str) -> Sequence[str]:
    """List all files in input_dir matching extension, where input_dir can be a
    local or GCS path."""
    if input_dir.startswith("gs://"):
        client = storage.Client()
        bucket, prefix = split_gcs_bucket_and_filepath(input_dir)
        blobs = [x for x in client.list_blobs(bucket, prefix=prefix)]
        input_paths = [f"gs://{x._bucket.name}/{x.name}" for x in blobs]
    else:
        input_paths = [os.path.join(input_dir, fp) for fp in os.listdir(input_dir)]
    return sorted([x for x in input_paths if x.endswith(extension)])


def write_npy(filepath: str, ary: np.ndarray) -> None:
    assert filepath.endswith(".npy")
    assert isinstance(ary, np.ndarray)
    if filepath.startswith("gs://"):
        # Case: GCS output; write bytes directly to GCS file handle.

        bucket_name, output_path = split_gcs_bucket_and_filepath(filepath)
        print(f"[DEBUG] writing output to gs://{bucket_name}/{output_path}")
        logging.warning(f"[DEBUG] writing output to gs://{bucket_name}/{output_path}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(output_path)

        with blob.open("wb") as f:
            np.save(f, ary)

    else:
        # Case: local file; save to local file path.
        output_dir = os.path.dirname(filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(filepath, ary)
    return


def read_audio_encoding(
    uri: str, representations_dir: str, numpy_to_torch=True
) -> Union[None, torch.Tensor, np.ndarray]:
    if not isinstance(uri, str):
        logging.debug(f"casting uri {uri} of type {type(uri)} to string {str(uri)}")
        uri = str(uri)
    audio_filename = uri + ".wav"

    encoding_fp = os.path.join(representations_dir, uri + ".npy")
    audio_encoding = None

    if encoding_fp.startswith("gs://"):
        # Case: file located on GCS

        # Create a Cloud Storage client and parse the path.
        gcs = storage.Client()
        bucket, file_name = encoding_fp.replace("gs://", "").split("/", maxsplit=1)
        gcs_bucket_obj = gcs.get_bucket(bucket)

        # Download the file locally and read it
        blob = gcs_bucket_obj.blob(file_name)

        if not blob.exists():
            # Case: encoding does not exist on GCS.
            logging.warning(f"no encodings found for {encoding_fp}; skipping")
            audio_encoding = None

        else:
            # Case: encoding exists on GCS; load it.
            with tempfile.TemporaryDirectory() as tmp:
                encoding_fp_local = os.path.join(tmp, audio_filename)
                logging.info(f"downloading {encoding_fp} to {encoding_fp_local}")
                blob.download_to_filename(encoding_fp_local)
                logging.info(f"loading downloaded file from {encoding_fp_local}")
                audio_encoding = np.load(encoding_fp_local)

    else:
        # Case: file is local.
        try:
            logging.debug(f"reading local encoding file from {encoding_fp}")
            audio_encoding = np.load(encoding_fp)
        except FileNotFoundError:
            logging.warning(f"no encodings found for {encoding_fp}; skipping")

    if audio_encoding is not None and numpy_to_torch:
        return torch.from_numpy(audio_encoding)
    else:
        return audio_encoding
