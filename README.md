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
