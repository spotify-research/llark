This folder contains code adapted from the [`jukemir`](https://github.com/p-lambda/jukemir) library, which is released under an [MIT License](https://github.com/p-lambda/jukemir/blob/main/LICENSE). The code was part of the work:

```latex
@inproceedings{castellon2021calm,
  title={Codified audio language modeling learns useful representations for music information retrieval},
  author={Castellon, Rodrigo and Donahue, Chris and Liang, Percy},
  booktitle={ISMIR},
  year={2021}
}
```

We thank the authors of the jukemir library for making their code available!


# Run locally via Dataflow `DirectRunner`
```bash
docker run -it --rm \
    -v ~/.config/gcloud:/root/.config/gcloud \
    --entrypoint=/bin/bash \
    gcr.io/bucketname/music2text-dataflow:latest
```


# Launch a Google Cloud Dataflow job (from within the docker container)

```bash
python dataflow_inference.py \
  --input-dir "gs://bucketname/datasets/medleydb/wav-crop" \
  --output-dir "gs://bucketname/datasets/medleydb/representations/jukebox/f10/" \
  --runner "DataflowRunner" \
  --accelerator-type nvidia-tesla-v100 \
  --num-workers 128
```
