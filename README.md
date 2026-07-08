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
