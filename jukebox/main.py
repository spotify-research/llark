import logging
import os
from math import floor

import librosa as lr
import numpy as np
import torch
from tqdm import tqdm

JUKEBOX_SAMPLE_RATE = 44100
T = 8192  # time dimension of jukebox activations

# Sample length of 5B prior (Table 6, Top-level prior
# hyperparameters of https://arxiv.org/pdf/2005.00341.pdf)
JUKEBOX_EXPECTED_SAMPLES_LEN = 1048576

JUKEBOX_SAMPLE_SECONDS = JUKEBOX_EXPECTED_SAMPLES_LEN / JUKEBOX_SAMPLE_RATE

# Activations sample rate is approximately 345.654.
# Note that this also matches number of "approximately 345"
# in jukemir paper.
ACTS_SAMPLE_RATE = T / JUKEBOX_SAMPLE_SECONDS


class EmptyFileError(ValueError):
    pass


def load_audio_from_file(fpath: str) -> np.ndarray:
    try:
        audio, _ = lr.load(fpath, sr=JUKEBOX_SAMPLE_RATE)
    except ValueError as ve:
        raise EmptyFileError(
            f"file {fpath} failed to read with exception {ve!r}; it is probably empty."
        )
    if audio.ndim == 1:
        audio = audio[np.newaxis]
    audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    return audio.flatten()


def maybe_pad_audio_to_max_len(audio: np.ndarray) -> np.ndarray:
    if len(audio) < JUKEBOX_EXPECTED_SAMPLES_LEN:
        audio = np.pad(audio, (0, JUKEBOX_EXPECTED_SAMPLES_LEN - len(audio)))
    return audio


def get_z(audio, vqvae):
    # don't compute unnecessary discrete encodings
    assert (
        len(audio) >= JUKEBOX_EXPECTED_SAMPLES_LEN
    ), f"expected samples with shape {JUKEBOX_EXPECTED_SAMPLES_LEN}; got shape {audio.shape}."
    audio = audio[:JUKEBOX_EXPECTED_SAMPLES_LEN]

    zs = vqvae.encode(torch.cuda.FloatTensor(audio[np.newaxis, :, np.newaxis]))

    z = zs[-1].flatten()[np.newaxis, :]

    if z.shape[-1] < T:
        raise ValueError("Audio file is not long enough")

    return z


def get_cond(hps, top_prior):
    sample_length_in_seconds = 62

    hps.sample_length = (
        int(sample_length_in_seconds * hps.sr) // top_prior.raw_to_tokens
    ) * top_prior.raw_to_tokens

    # NOTE: the 'lyrics' parameter is required, which is why it is included,
    # but it doesn't actually change anything about the `x_cond`, `y_cond`,
    # nor the `prime` variables
    metas = [
        dict(
            artist="unknown",
            genre="unknown",
            total_length=hps.sample_length,
            offset=0,
            lyrics="""lyrics go here!!!""",
        ),
    ] * hps.n_samples

    labels = [None, None, top_prior.labeller.get_batch_labels(metas, "cuda")]

    x_cond, y_cond, prime = top_prior.get_cond(None, top_prior.get_y(labels[-1], 0))

    x_cond = x_cond[0, :T][np.newaxis, ...]
    y_cond = y_cond[0][np.newaxis, ...]

    return x_cond, y_cond


def get_final_activations(z, x_cond, y_cond, top_prior):
    x = z[:, :T]

    # make sure that we get the activations
    top_prior.prior.only_encode = True

    # encoder_kv and fp16 are set to the defaults, but explicitly so
    out = top_prior.prior.forward(x, x_cond=x_cond, y_cond=y_cond, encoder_kv=None, fp16=False)

    return out


def windowed_average(acts, frame_len: int, ceil_mode=False):
    """Compute the windowed average of a signal.
    args:
        z: signal array of shape [num_samples,]
        frame_len: length of frames. This is equivalent to the
            kernel size and stride of average pooling.
        ceil_mode: argument passed to torch.nn.AvgPool1D.
    returns:
        samples array of shape floor(len(z) / f) when ceil_mode is false,
        and array of shape ceil(len(z) / f) when ceil_mode is true.
    """
    assert acts.ndim == 2, "expected 2d inputs"
    assert acts.shape[1] == 4800
    acts = torch.unsqueeze(acts, 0)  # [T, 4800] --> [1, T, 4800]
    acts = torch.transpose(acts, 1, 2)  # [1, T, 4800] --> [1, 4800, T]
    pool = torch.nn.AvgPool1d(frame_len, stride=frame_len, ceil_mode=ceil_mode)
    acts = pool(acts)
    return torch.transpose(acts, 1, 2)  # [1, 4800, T/frame_len] --> [1, T/frame_len, 4800]


def get_acts_from_file(fpath, hps, vqvae, top_prior, meanpool=True, pool_frames_per_second=None):
    audio = load_audio_from_file(fpath)
    input_audio_len = len(audio)
    latent_audio_len = floor(T * input_audio_len / JUKEBOX_EXPECTED_SAMPLES_LEN)
    audio = maybe_pad_audio_to_max_len(audio)

    # run vq-vae on the audio; z has shape [1, T]
    z = get_z(audio, vqvae)

    # get conditioning info
    x_cond, y_cond = get_cond(hps, top_prior)

    # get the activations from the LM
    acts = get_final_activations(z, x_cond, y_cond, top_prior)
    acts = acts.squeeze().type(torch.float32)

    # acts has shape [T, 4800]. We truncate acts proportional to
    # original inputs len to remove the embedddings of padded samples.
    # For an input audio of shape [JUKEBOX_EXPECTED_SAMPLES_LEN,],
    # the activations have shape [T, 4800]. For shorter audio, the
    # activations have length proportional to len(audio)/JUKEBOX_EXPECTED_SAMPLES_LEN
    acts = acts[:latent_audio_len, :]

    # postprocessing
    if meanpool:
        logging.warning(f"mean pooling at f={pool_frames_per_second}")
        if not pool_frames_per_second:
            acts = acts.mean(dim=0)
        else:
            frame_len = floor(ACTS_SAMPLE_RATE / pool_frames_per_second)

            # After windowed averaging (and squeeze), acts has
            # shape [latent_audio_len // frame_len, 4800]
            acts = windowed_average(acts, frame_len)
            acts = torch.squeeze(acts, 0)

    acts = np.array(acts.cpu())

    logging.info(f"acts after pooling has shape {acts.shape}")

    return acts


def load_model(model="5b"):
    from jukebox.hparams import Hyperparams, setup_hparams
    from jukebox.make_models import MODELS, make_prior, make_vqvae
    from jukebox.utils.dist_utils import setup_dist_from_mpi

    # Set up MPI
    rank, local_rank, device = setup_dist_from_mpi()

    # Set up VQVAE
    hps = Hyperparams()
    hps.sr = JUKEBOX_SAMPLE_RATE
    hps.n_samples = 3 if model == "5b_lyrics" else 8
    hps.name = "samples"
    # chunk_size = 16 if model == "5b_lyrics" else 32
    # max_batch_size = 3 if model == "5b_lyrics" else 16
    hps.levels = 3
    hps.hop_fraction = [0.5, 0.5, 0.125]
    vqvae, *priors = MODELS[model]
    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=1048576)), device)

    # Set up language model
    hparams = setup_hparams(priors[-1], dict())
    hparams["prior_depth"] = 36
    top_prior = make_prior(hparams, vqvae, device)
    return hps, vqvae, top_prior


def main():
    import pathlib
    from argparse import ArgumentParser

    # imports and set up Jukebox's multi-GPU parallelization

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--batch_idx", type=int, default=None)
    parser.add_argument("--input_dir", default="/input", help="path to inputs")
    parser.add_argument("--output_dir", default="/output", help="path to outputs")
    parser.add_argument(
        "--pool-frames-per-second",
        default=10,
        type=int,
        help="Frames per second for pooling. Set to zero to pool over all timesteps.",
    )

    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    input_paths = sorted(list(input_dir.iterdir()))
    if args.batch_size is not None and args.batch_idx is not None:
        batch_starts = list(range(0, len(input_paths), args.batch_size))
        if args.batch_idx >= len(batch_starts):
            raise ValueError("Invalid batch index")
        batch_start = batch_starts[args.batch_idx]
        input_paths = input_paths[batch_start : batch_start + args.batch_size]

    loaded = False
    for input_path in tqdm(input_paths):
        if not loaded:
            hps, vqvae, top_prior = load_model()

            loaded = True

        # Decode, resample, convert to mono, and normalize audio
        with torch.no_grad():
            representation = get_acts_from_file(
                input_path,
                hps,
                vqvae,
                top_prior,
                meanpool=True,
                pool_frames_per_second=args.pool_frames_per_second,
            )
        input_filename = os.path.basename(input_path)
        output_filename = input_filename.replace(".wav", ".npy")
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, representation)


if __name__ == "__main__":
    main()
