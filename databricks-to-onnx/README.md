# Loading a model from databricks

## To get a Databricks PAT

https://{workspace-id}.cloud.databricks.com/settings/user/developer

## Usage

DATABRICKS_HOST=https://{workspace-id}.cloud.databricks.com \
DATABRICKS_TOKEN={PAT} \
uv run databricks-to-onnx \
  -m "models:/catalog.schema.model_name@alias" \
  -i "input:float32:1,100" \
  -o model.onnx
