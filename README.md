<h2 align="center">
  <img src="logo.png" alt="log" width="100" height="100"/>
  <br>
  allumette
</h2>

`guimauve`, French for marshmallow, is an inference server built on top of [burn](https://github.com/tracel-ai/burn).

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
