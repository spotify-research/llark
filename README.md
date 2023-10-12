# LLark: A Multimodal Foundation Model for Music

![Static Badge](https://img.shields.io/badge/status-experimental-green)


This is the code release associated with the paper:

> LLark: A Multimodal Foundation Model for Music<br />
> Josh Gardner, Simon Durand, Daniel Stoller, Rachel Bittner.<br />
> Under Review at ICLR 2024

This repository contains the code used to build the training dataset, preprocess existing open-source music datasets, train the model, and run inference. **Note that this paper is not accompanied with any trained models.**

For more details about this work, you can read the [preprint of the paper](https://arxiv.org/abs/2310.07160), read the [Spotify Research blog post] about this work, and [listen to demos on the companion site](https://storage.googleapis.com/music2text-public/index.html).

**This is not an officially supported Spotify product.**


## Citing This Repository

If you use this code in a publication, please cite our paper:

```
@article{gardner2023llark,
  title={LLark: A Multimodal Foundation Model for Music},
  author={Gardner, Josh and Durand, Simon and Stoller, Daniel and Bittner, Rachel},
  journal={arXiv preprint arXiv:2310.07160},
  year={2023}
}
```

# Using The Code

## Docker Environments

All of the code in this repo should be run in one of the provided Docker containers (see `docker`). There are three Dockerfiles for separate use cases:
* `m2t-train.dockerfile` is for model training and inference
* `m2t-preprocess.dockerfile` is for data preprocessing (i.e. running Beam pipelines)
* `jukebox-embed.dockerfile` is for extracting Jukebox embeddings

> **Warning**
> 
> Note that the data preprocessing pipeline can be run locally or on [Google Cloud Dataflow](https://cloud.google.com/dataflow), a cloud-based data processing system that allows the code to be sped up by running in parallel across potentially thousands of machines. **Using these pipelines on Dataflow may incur costs against your own Google Cloud project.**
> 
> Running these pipelines on Google Cloud Dataflow may also require pushing one or more Docker images to [Google Artifact Registry](https://cloud.google.com/artifact-registry), properly setting up permissions to use these Docker images from within Google Cloud Dataflow, and replacing various hard-coded Docker image paths with paths to your uploaded Docker images. **This repository contains no code to aid with this process.**

## Preprocess Your Own Audio

We provide some utilitities for converting, cropping, and annotating audio datasets using the process described in our paper. These utilities are located in `scripts/preprocessing`; see the header of each file in that directory for usage examples.

All of the preprocessing scripts use Apache Beam (on Google DataFlow). In order to run these code examples, we recommend using the Docker environment defined in `docker/m2t-preprocess.dockerfile`.

## Generate Instruction-Tuning Data

The file `scripts/openai/fetch_openai_instruct_data.py` can be adapted to generate instruction-tuning data. The prompts for each dataset are located in the `m2t/instruct` directory.

## Extract Jukebox Embeddings

The Jukebox embedding pipeline uses Apache Beam (on Google Cloud Dataflow). In order to run these code examples, we recommend using the Docker environment defined in `docker/jukebox-embed.dockerfile`. Embedding a set of around 100k audio files takes less than 1 hour using the default parameters in that script.

For using CLAP, we provide a similar set of utilities in the `scripts/clap` subdirectory; the CLAP embedding script can be executed from the `m2t-preprocess.dockerfile` Docker environment and does not require its own separate Docker environment.

## Training

This repo does not officially support training. However, we provide scripts that could be adapted to train a model (mostly as a way to describe the exact training hyperparameters used in training) in the `scripts/training` subdirectory. This includes the main LLark model, along with models based on MPT-1B and CLAP.


## Contributing

We feel that a welcoming community is important and we ask that you follow Spotify's
[Open Source Code of Conduct](https://github.com/spotify/code-of-conduct/blob/main/code-of-conduct.md)
in all interactions with the community.

## Authors

* Josh Gardner <jpgard@cs.washington.edu>
* Peter Sobot <psobot@spotify.com>

Follow [@SpotifyResearch](https://twitter.com/SpotifyResearch) on Twitter for updates.


## License

Copyright 2023 Spotify, Inc.

Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

The `m2t/llava` directory contains a subset of the code from [LLaVA](https://github.com/haotian-liu/LLaVA). We do not make substantial modifications to this code but include it in order to make the repository self-contained and remove other external dependencies.

The `jukebox` directory contains code adapted from [`jukemir`](https://github.com/p-lambda/jukemir).

## Security Issues?

Please report sensitive security issues via Spotify's bug-bounty program (https://hackerone.com/spotify) rather than GitHub.
