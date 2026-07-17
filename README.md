<h2 align="center">
  <img src="logo.png" alt="log" width="100" height="100"/>
  <br>
  allumette
</h2>

`guimauve`, French for marshmallow, is an inference server built on top of [burn](https://github.com/tracel-ai/burn).

## Quickstart

This is a small tutorial to run an inference server on an English to Portuguese translation model.

Download the datasets from `https://web.archive.org/web/20240301220426if_/http://www.phontron.com/data/qi18naacl-dataset.tar.gz`
and extract them. We only care about the `pt_to_en` folder.

### Train tokenizers

From the `python-onnx-example` folder, run:

```bash
uv run \
    -m train.train_tokenizer \
    -i datasets/pt_to_en/pt.dev \
        datasets/pt_to_en/pt.test \
        datasets/pt_to_en/pt.train \
        datasets/pt_to_en/pt.train.r0.125 \
        datasets/pt_to_en/pt.train.r0.25 \
        datasets/pt_to_en/pt.train.r0.5 \
    -o pt_tokenizer.json \
    --max-seq-len 128
```

This will create a `pt_tokenizer.json` file.

Same thing for the English data which will create a `en_tokenizer.json` file.


Build the Docker image, `ARTIFACTS_DIR` should contain the tokenizer files built previously. 

```bash
docker build \
    --build-arg CRATE_NAME=onnx-example \
    --build-arg ARTIFACTS_DIR=models/pt_to_en \
    -t onnx-example .
```

```bash
docker run -p 3000:3000 --name onnx-example onnx-example
```

```
CRATE_NAME=onnx-example \
  ARTIFACTS_DIR=models/pt_to_en \
  docker compose up --force-recreate --remove-orphans --detach --build guimauve
```
