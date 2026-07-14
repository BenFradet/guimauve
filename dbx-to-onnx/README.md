This is a small CLI to convert PyTorch models saved with MLflow to ONNX.

# Usage

```bash
Usage: dbx-to-onnx [OPTIONS]

Options:
  -m, --model-location TEXT       Unity Catalog model URI, e.g. "models:/catalog.schema.model_name@champion", or local directory, e.g. "path/to/champion/"
                                  [required]
  -i, --input-tensor-schema TEXT  Input tensor schema as "name:dtype:dim1,dim2".
                                  Repeatable.
                                  [required]
  -d, --dict TEXT                 Extract a dict you want to get out of the model in dotted path format, e.g. "maps.vocab".
                                  Will be written out as "maps.vocab.json" in the output directory.
                                  Repeatable.
  -e, --embedding TEXT            Extract embeddings from a nn.Module or nn.ModuleDict as a safetensors file, in dotted path format, e.g. "embeddings.input".
                                  Will be written out as "embeddings.input.safetensors" in the output directory.
                                  Repeatable.
  -o, --output-dir TEXT           Output dir for the model.onnx file.
```

## Loading a model from databricks

First, get a Databricks PAT from
`https://{workspace-id}.cloud.databricks.com/settings/user/developer`.

Then, run the cli:

```bash
DATABRICKS_HOST=https://{workspace-id}.cloud.databricks.com \
DATABRICKS_TOKEN={PAT} \
uv run dbx-to-onnx \
  -m "models:/catalog.schema.model_name@alias" \
  -i "a:float32:1,100" \
  -i "b:int64:10,50,3" \
  -o "model.onnx"
```

## Loading a model locally

First, move all MLflow model artifacts to a folder, e.g. `models/model-v1/`.

Then, run the cli:

```bash
uv run dbx-to-onnx \
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
uv add --dev --editable {clone-dir}/guimauve/dbx-to-onnx
```

And repeat the above.
