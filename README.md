```bash
docker build \
    --build-arg CRATE_NAME=onnx-example \
    --build-arg ARTIFACTS_DIR=models/pt_to_en \
    -t onnx-example .
```
