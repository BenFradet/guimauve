This is a small CLI to convert PyTorch models saved with MLflow to ONNX.

# Usage

Clone this project:

```bash
git clone https://github.com/BenFradet/guimauve
```

```bash
Usage: databricks-to-onnx [OPTIONS]

Options:
  -m, --model-location TEXT       Unity Catalog model URI, e.g. "models:/catal
                                  og.schema.model_name@champion", or local
                                  directory, e.g. "path/to/champion/"
                                  [required]
  -i, --input-tensor-schema TEXT  Input tensor schema as
                                  "name:dtype:dim1,dim2". Repeatable.
                                  [required]
  -o, --output-path TEXT          Output path for the .onnx file.  [required]
  --help                          Show this message and exit.
```

## Loading a model from databricks

First, get a Databricks PAT from
`https://{workspace-id}.cloud.databricks.com/settings/user/developer`.

Then, run the cli:

```bash
DATABRICKS_HOST=https://{workspace-id}.cloud.databricks.com \
DATABRICKS_TOKEN={PAT} \
uv run databricks-to-onnx \
  -m "models:/catalog.schema.model_name@alias" \
  -i "a:float32:1,100" \
  -i "b:int64:10,50,3" \
  -o "model.onnx"
```

## Loading a model locally

First, move all MLflow model artifacts to a folder, e.g. `models/model-v1/`.

Then, run the cli:

```bash
uv run databricks-to-onnx \
  -m "models/model-v1/" \
  -i "a:float32:1,100" \
  -i "b:int64:10,50,3" \
  -o "model.onnx"
```

## External references

If the above fails with an import error and/or if the model was saved via pickle with
references to external classes (e.g. a custom `torch.nn.Module`), loading it requires
those classes to be importable.

As a result, you'll need to install the cli directly where the model definition is:

```bash
uv add --dev --editable {clone-dir}/guimauve/databricks-to-onnx
```

And repeat the above.
